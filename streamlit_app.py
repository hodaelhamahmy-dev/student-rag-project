import os
from pathlib import Path

import chromadb
import requests
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("rag_documents")


def retrieve_context(query, n_results=2):
    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )

    return results


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

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Student RAG Assistant")

st.title("📚 Student RAG Assistant")

question = st.text_input("Ask a question about your documents:")

if st.button("Ask"):

    if question:

        with st.spinner("Searching documents..."):

            results = retrieve_context(question)

            context = "\n\n".join(results["documents"][0])

            answer = ask_llm(question, context)

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Retrieved Sources")

        for meta in results["metadatas"][0]:
            st.write("📄", meta["source"])