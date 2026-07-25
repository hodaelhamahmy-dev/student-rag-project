import os
from pathlib import Path
import re
from urllib import response
from pypdf import PdfReader
import chromadb
import requests
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load .env
env_path = Path(__file__).parent / ".env"

st.write("Current file:", __file__)
st.write("Loading .env from:", env_path.resolve())

load_dotenv(dotenv_path=env_path, override=True)

st.write("Loading .env from:", env_path)
st.write("Exists:", env_path.exists())

try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
    OPENROUTER_MODEL = st.secrets["OPENROUTER_MODEL"]
    st.write("Using Streamlit Secrets")
except Exception:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
    st.write("Using .env")

st.write("API key loaded:", OPENROUTER_API_KEY is not None)
st.write("Model:", OPENROUTER_MODEL)

if OPENROUTER_API_KEY:
    st.write("Key starts with:", OPENROUTER_API_KEY[:15] + "...")
else:
    st.write("No API key found!")
# -------------------------
# Build ChromaDB if needed
# -------------------------

def load_documents(folder="documents"):
    text = ""

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, file))

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

    return text


def preprocess_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks

def build_database():

    documents = load_documents()

    cleaned = preprocess_text(documents)

    chunks = chunk_text(cleaned)

    embeddings = model.encode(chunks)

    collection = client.get_or_create_collection("rag_documents")

    existing = collection.count()

    if existing == 0:

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
# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = build_database()

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

    if response.status_code != 200:
       st.error(f"Status Code: {response.status_code}")
       st.code(response.text)
       return "OpenRouter request failed."
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