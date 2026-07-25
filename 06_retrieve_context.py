import chromadb
from sentence_transformers import SentenceTransformer

# Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to the existing Chroma database
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection("rag_documents")


def retrieve_context(query, n_results=2):
    """
    Retrieve the most relevant chunks for a user query.
    """

    # Convert the question into an embedding
    query_embedding = model.encode(query)

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )

    return results


if __name__ == "__main__":

    question = input("Ask a question: ")

    results = retrieve_context(question)

    print("\n" + "=" * 60)

    print("Retrieved Context:\n")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        print(f"Result {i}")
        print(f"Source: {meta['source']}")
        print(doc)
        print("-" * 60)