"""
Document ingestion pipeline.

Usage:
    python -m app.ingest

Reads all PDFs from data/documents/, chunks them, embeds them,
and writes a FAISS index + metadata to data/faiss_index/.
"""

import json
import fitz  # PyMuPDF
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from app.config import (
    DOCUMENTS_DIR,
    FAISS_INDEX_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from a PDF, returning a list of {page, text} dicts."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_num, "text": text, "source": pdf_path.name})
    doc.close()
    return pages


def chunk_text(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split page texts into overlapping chunks of approximately `chunk_size` words.
    Each chunk retains metadata about its source file and page number.
    """
    chunks = []
    for page_data in pages:
        words = page_data["text"].split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "text": chunk_text,
                "source": page_data["source"],
                "page": page_data["page"],
                "chunk_index": len(chunks),
            })

            # Move forward by (chunk_size - overlap)
            start += chunk_size - overlap

    return chunks


def build_faiss_index(chunks: list[dict], model: SentenceTransformer) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Embed all chunks and build a FAISS inner-product index."""
    texts = [c["text"] for c in chunks]

    print(f"  Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Inner product on normalized vectors = cosine similarity
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"  FAISS index built: {index.ntotal} vectors, dim={dimension}")
    return index, chunks


def save_index(index: faiss.IndexFlatIP, metadata: list[dict]):
    """Persist FAISS index and chunk metadata to disk."""
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(FAISS_INDEX_DIR / "index.faiss"))

    with open(FAISS_INDEX_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved index and metadata to {FAISS_INDEX_DIR}")


def run_ingestion():
    """Main ingestion entrypoint."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DOCUMENTS_DIR}. Add some documents and try again.")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {DOCUMENTS_DIR}")

    # 1. Extract text from all PDFs
    all_pages = []
    for pdf_path in pdf_files:
        print(f"  Extracting: {pdf_path.name}")
        pages = extract_text_from_pdf(pdf_path)
        all_pages.extend(pages)
        print(f"    → {len(pages)} pages")

    # 2. Chunk the text
    print("Chunking text...")
    chunks = chunk_text(all_pages)
    print(f"  → {len(chunks)} chunks")

    # 3. Load embedding model and build index
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Building FAISS index...")
    index, metadata = build_faiss_index(chunks, model)

    # 4. Save to disk
    save_index(index, metadata)

    print("\nIngestion complete!")
    print(f"  Documents: {len(pdf_files)}")
    print(f"  Pages: {len(all_pages)}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Index location: {FAISS_INDEX_DIR}")


if __name__ == "__main__":
    run_ingestion()
