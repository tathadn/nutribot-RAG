"""
Hybrid retriever: combines FAISS semantic search with BM25 keyword search.
Reciprocal Rank Fusion (RRF) merges the two ranked lists for better retrieval.
"""

import json
import math
import faiss
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from app.config import FAISS_INDEX_DIR, EMBEDDING_MODEL, TOP_K, MIN_SCORE_THRESHOLD


# ---------------------------------------------------------------------------
# Lightweight BM25 implementation (no external dependency)
# ---------------------------------------------------------------------------

class BM25:
    """Okapi BM25 ranking over pre-tokenised documents."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / self.N if self.N else 1

        # document frequency for each term
        self.df: dict[str, int] = defaultdict(int)
        for doc in corpus:
            seen = set()
            for token in doc:
                if token not in seen:
                    self.df[token] += 1
                    seen.add(token)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        doc = self.corpus[doc_idx]
        dl = len(doc)
        tf_map: dict[str, int] = defaultdict(int)
        for t in doc:
            tf_map[t] += 1

        s = 0.0
        for qt in query_tokens:
            tf = tf_map.get(qt, 0)
            idf = self._idf(qt)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * numerator / denominator
        return s

    def search(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        scores = [(i, self.score(query_tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Simple tokeniser
# ---------------------------------------------------------------------------

import re

def tokenize(text: str) -> list[str]:
    """Lowercase split on non-alphanumeric chars, drop short tokens."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 2]


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Two-stage retriever:
      1. FAISS (semantic)  +  BM25 (keyword)
      2. Reciprocal Rank Fusion to merge
    """

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.metadata: list[dict] = []
        self.bm25: BM25 | None = None
        self._load()

    # -- loading ----------------------------------------------------------

    def _load(self):
        index_path = FAISS_INDEX_DIR / "index.faiss"
        metadata_path = FAISS_INDEX_DIR / "metadata.json"

        if not index_path.exists():
            print("WARNING: No FAISS index found. Run `python -m app.ingest` first.")
            return

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Build BM25 index from chunk texts
        corpus_tokens = [tokenize(chunk["text"]) for chunk in self.metadata]
        self.bm25 = BM25(corpus_tokens)

        print(f"Loaded hybrid index: {self.index.ntotal} vectors, BM25 over {len(corpus_tokens)} docs")

    # -- search -----------------------------------------------------------

    def _faiss_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return list of (chunk_index, score) from FAISS."""
        if self.index is None or self.index.ntotal == 0:
            return []
        emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(emb, top_k)
        return [
            (int(idx), float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return list of (chunk_index, score) from BM25."""
        if self.bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        return self.bm25.search(tokens, top_k)

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[tuple[int, float]]],
        k: int = 60,
    ) -> list[tuple[int, float]]:
        """
        Reciprocal Rank Fusion (Cormack et al. 2009).
        Merges multiple ranked lists into one by summing 1/(k + rank).
        """
        fused: dict[int, float] = defaultdict(float)
        for ranked in ranked_lists:
            for rank, (doc_id, _score) in enumerate(ranked, start=1):
                fused[doc_id] += 1.0 / (k + rank)
        # Sort by fused score descending
        result = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return result

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Hybrid search: FAISS + BM25 merged via RRF.
        Returns list of chunk dicts with a 'score' field.
        """
        if self.index is None:
            return []

        # Retrieve more candidates from each, then fuse
        faiss_results = self._faiss_search(query, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)

        fused = self._reciprocal_rank_fusion([faiss_results, bm25_results])

        results = []
        for doc_id, rrf_score in fused[:top_k]:
            if doc_id >= len(self.metadata):
                continue
            chunk = self.metadata[doc_id].copy()
            chunk["score"] = rrf_score
            # Also store the raw FAISS score for threshold filtering
            faiss_score = dict(faiss_results).get(doc_id, 0.0)
            chunk["semantic_score"] = faiss_score
            results.append(chunk)

        return results

    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0
