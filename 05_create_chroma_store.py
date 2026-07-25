import re
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def load_documents(folder_path="documents"):
    documents = []

    folder = Path(folder_path)

    for pdf_file in folder.glob("*.pdf"):
        reader = PdfReader(pdf_file)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        documents.append({
            "file_name": pdf_file.name,
            "text": text
        })

    return documents


def preprocess_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create persistent Chroma database
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="rag_documents"
)


if __name__ == "__main__":

    docs = load_documents()

    chunk_id = 0

    for doc in docs:

        clean_text = preprocess_text(doc["text"])

        chunks = chunk_text(clean_text)

        embeddings = model.encode(chunks)

        for chunk, embedding in zip(chunks, embeddings):

            collection.add(
                ids=[str(chunk_id)],
                documents=[chunk],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "source": doc["file_name"]
                }]
            )

            chunk_id += 1

    print("=" * 50)
    print("Database created successfully!")
    print(f"Total stored chunks: {chunk_id}")