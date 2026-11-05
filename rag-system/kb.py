from pathlib import Path

import chromadb
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_pdf(pdf_path: Path) -> list[tuple[str, int]]:
    """Read a PDF and return its text grouped by page number."""
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            pages.append((text, page_number))

    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap

    return chunks


def build_knowledge_base() -> int:
    """Load all PDFs from data/, chunk them, and store them in ChromaDB."""
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    documents = []
    ids = []
    metadatas = []

    for pdf_path in pdf_files:
        for page_text, page_number in load_pdf(pdf_path):
            for chunk_number, chunk in enumerate(chunk_text(page_text)):
                chunk_id = f"{pdf_path.stem}-p{page_number}-c{chunk_number}"
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_number,
                })

    if not documents:
        raise ValueError("The PDF files did not contain extractable text")

    collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
    print(f"Stored {len(documents)} chunks from {len(pdf_files)} PDF file(s) in {CHROMA_DIR}")
    return len(documents)


if __name__ == "__main__":
    build_knowledge_base()
