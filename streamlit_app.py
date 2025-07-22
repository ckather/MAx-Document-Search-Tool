import streamlit as st
from openai import OpenAI
import tempfile
import os
import fitz  # PyMuPDF
import docx
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Legal Doc Chatbot", layout="wide")
st.title("📄🔍 Legal Document Semantic Search Chatbot")

# Step 1: Ask for OpenAI API key
openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please enter your OpenAI API key to continue.", icon="🔐")
    st.stop()
client = OpenAI(api_key=openai_api_key)

# Step 2: Upload documents and tagging
uploaded_files = st.file_uploader("Upload legal documents (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
selected_tags = st.multiselect("Select relevant tags to include", ["Medical", "Pharmacy", "Insurance", "Contract Law"])

# Step 3: Parse documents and extract text
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file) -> List[str]:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            chunks.append((f"Page {page_num + 1}", text))
    return chunks

@st.cache_data(show_spinner=False)
def extract_text_from_docx(file) -> List[str]:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    temp.write(file.read())
    temp.close()
    doc = docx.Document(temp.name)
    chunks = []
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip():
            chunks.append((f"Paragraph {i + 1}", para.text.strip()))
    os.unlink(temp.name)
    return chunks

all_chunks = []
metadata = []

if uploaded_files and selected_tags:
    for file in uploaded_files:
        ext = file.name.split(".")[-1].lower()
        if ext == "pdf":
            text_chunks = extract_text_from_pdf(file)
        elif ext == "docx":
            text_chunks = extract_text_from_docx(file)
        else:
            continue
        for loc, chunk in text_chunks:
            all_chunks.append(chunk)
            metadata.append({"document": file.name, "location": loc, "tags": selected_tags})

    st.success(f"Loaded {len(all_chunks)} text chunks from {len(uploaded_files)} files.")

# Step 4: Build embeddings and FAISS index
    embeddings = embedder.encode(all_chunks, convert_to_tensor=False, show_progress_bar=True)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

# Step 5: Ask question via chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a legal question about these documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Embed the question and search
        question_embedding = embedder.encode([prompt])[0]
        D, I = index.search(np.array([question_embedding]), k=3)

        matched_chunks = [all_chunks[i] for i in I[0]]
        matched_meta = [metadata[i] for i in I[0]]

        response_text = ""
        for i in range(len(matched_chunks)):
            meta = matched_meta[i]
            response_text += f"\n**📄 {meta['document']}**\n- Location: *{meta['location']}*\n- Tags: `{', '.join(meta['tags'])}`\n- Match: {matched_chunks[i][:500]}...\n"

        # Generate LLM answer
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ] + [
                {"role": "system", "content": f"Here are the relevant legal document excerpts:\n{response_text}"}
            ],
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(chat_completion)

        st.session_state.messages.append({"role": "assistant", "content": response})

