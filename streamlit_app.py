import os
import re
from pathlib import Path

import chromadb
import requests
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ======================================================
# Load Environment Variables
# ======================================================

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
    OPENROUTER_MODEL = st.secrets["OPENROUTER_MODEL"]
except Exception:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

# ======================================================
# Load Documents
# ======================================================

def load_documents(folder="documents"):
    text = ""

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder, file)
            reader = PdfReader(pdf_path)

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

    return text


# ======================================================
# Preprocessing
# ======================================================

def preprocess_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ======================================================
# Chunking
# ======================================================

def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


# ======================================================
# Load Embedding Model
# ======================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ======================================================
# Connect to ChromaDB
# ======================================================

client = chromadb.PersistentClient(path="chroma_db")


# ======================================================
# Build Chroma Database
# ======================================================

def build_database():

    documents = load_documents()
    cleaned_text = preprocess_text(documents)
    chunks = chunk_text(cleaned_text)

    embeddings = model.encode(chunks)

    collection = client.get_or_create_collection("rag_documents")

    if collection.count() == 0:
        collection.add(
            ids=[str(i) for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=[
                {"source": "Student RAG Project Instructions.pdf"}
                for _ in chunks
            ]
        )

    return collection


collection = build_database()


# ======================================================
# Retrieve Relevant Context
# ======================================================

def retrieve_context(query, n_results=2):

    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )

    return results


# ======================================================
# Ask OpenRouter LLM
# ======================================================

def ask_llm(question, context):

    prompt = f"""
Answer the question ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )

    if response.status_code != 200:
        st.error("OpenRouter request failed.")
        st.code(response.text)
        return None

    return response.json()["choices"][0]["message"]["content"]


# ======================================================
# Streamlit Interface
# ======================================================

st.set_page_config(
    page_title="Student RAG Assistant",
    page_icon="📚"
)

st.title("📚 Student RAG Assistant")

question = st.text_input("Ask a question about your documents:")

if st.button("Ask"):

    if question.strip():

        with st.spinner("Searching documents..."):

            results = retrieve_context(question)

            context = "\n\n".join(results["documents"][0])

            answer = ask_llm(question, context)

        if answer:
            st.subheader("Answer")
            st.write(answer)

        st.subheader("Retrieved Sources")

        shown_sources = set()

        for metadata in results["metadatas"][0]:
            source = metadata["source"]

            if source not in shown_sources:
                st.write(f"📄 {source}")
                shown_sources.add(source)

    else:
        st.warning("Please enter a question.")