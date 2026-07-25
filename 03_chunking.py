import re
from pathlib import Path
from pypdf import PdfReader


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
    """
    Split text into chunks of fixed size.
    """

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":

    docs = load_documents()

    for doc in docs:

        clean_text = preprocess_text(doc["text"])

        chunks = chunk_text(clean_text)

        print("=" * 60)
        print("File:", doc["file_name"])
        print(f"Total Chunks: {len(chunks)}\n")

        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}")
            print(chunk)
            print("-" * 60)