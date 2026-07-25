import os
from pathlib import Path
import chromadb
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

env_path = Path(__file__).parent / ".env"

print("Looking for .env:", env_path)
print("File exists:", env_path.exists())

load_dotenv(dotenv_path=env_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

print("API Key:", OPENROUTER_API_KEY[:15] + "..." if OPENROUTER_API_KEY else "None")
print("Model:", OPENROUTER_MODEL)

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

    return "\n\n".join(results["documents"][0])


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


if __name__ == "__main__":

    question = input("Ask a question: ")

    context = retrieve_context(question)

    answer = ask_llm(question, context)

    print("\n" + "=" * 60)
    print("Answer:\n")
    print(answer)