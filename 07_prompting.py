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


def retrieve_context(query, n_results=2, max_distance=1.0, min_length=20):
    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=["documents", "distances"]
    )

    docs = results["documents"][0]
    distances = results["distances"][0]

    # Keep only chunks close enough to the query
    filtered = [doc for doc, dist in zip(docs, distances) if dist <= max_distance]

    if not filtered:
        return None

    context = "\n\n".join(filtered)

    if len(context.strip()) < min_length:
        return None

    return context


def ask_llm(question, context):

    system_prompt = """You are a strict document-based Q&A assistant.
You must answer ONLY using the provided context.
Never use outside knowledge or make assumptions beyond the text.
If the answer is not in the context, say exactly: "I cannot answer this from the provided document."
"""

    user_prompt = f"""Context:
{context}

Question:
{question}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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

    if context is None:
        print("\nI cannot answer this — no relevant content found in the document.")
    else:
        answer = ask_llm(question, context)
        print("\n" + "=" * 60)
        print("Answer:\n")
        print(answer)