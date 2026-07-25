import re
from pathlib import Path
from pypdf import PdfReader


def load_documents(folder_path="documents"):
    """
    Load all PDF files and extract text.
    """
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
    """
    Clean the extracted text.
    """

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


if __name__ == "__main__":

    docs = load_documents()

    print(f"Found {len(docs)} document(s).\n")

    for doc in docs:

        cleaned_text = preprocess_text(doc["text"])

        print("=" * 60)
        print("File:", doc["file_name"])
        print()
        print(cleaned_text[:700])
        print()