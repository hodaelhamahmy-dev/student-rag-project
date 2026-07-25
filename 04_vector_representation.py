import re
from pathlib import Path
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


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


if __name__ == "__main__":

    docs = load_documents()

    for doc in docs:

        clean_text = preprocess_text(doc["text"])

        chunks = chunk_text(clean_text)

        embeddings = model.encode(chunks)

        print("=" * 60)
        print("File:", doc["file_name"])
        print(f"Total Chunks: {len(chunks)}")
        print(f"Embedding Shape: {embeddings.shape}")

        print("\nFirst Chunk:")
        print(chunks[0])

        print("\nFirst Embedding (first 10 values):")
        print(embeddings[0][:10])