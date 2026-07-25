from pathlib import Path
from pypdf import PdfReader


def load_documents(folder_path="documents"):
    """
    Load all PDF files from the documents folder
    and extract their text.
    """
    documents = []

    folder = Path(folder_path)

    print("Current folder:", Path.cwd())
    print("Documents folder exists:", folder.exists())
    print("PDF files found:", list(folder.glob("*.pdf")))

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


if __name__ == "__main__":
    docs = load_documents()

    print(f"Found {len(docs)} document(s).\n")

    for doc in docs:
        print("=" * 50)
        print(f"File: {doc['file_name']}")
        print(doc["text"][:500])   # Display the first 500 characters
        print()