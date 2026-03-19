"""
FAISS retriever: loads the pre-built index and performs similarity search.
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import FAISS_INDEX_DIR, EMBEDDING_MODEL, TOP_K


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.metadata = []
        self._load_index()

    def _load_index(self):
        index_path = FAISS_INDEX_DIR / "index.faiss"
        metadata_path = FAISS_INDEX_DIR / "metadata.json"

        if not index_path.exists():
            print("WARNING: No FAISS index found. Run `python -m app.ingest` first.")
            return

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        print(f"Loaded FAISS index: {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Embed the query and return the top-k most similar chunks.

        Returns a list of dicts, each with:
            - text: the chunk content
            - source: original filename
            - page: page number
            - score: cosine similarity score
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Embed and normalize the query
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        # Search FAISS
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            chunk = self.metadata[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0
