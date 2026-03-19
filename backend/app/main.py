"""
FastAPI application — NutriBot backend (v2 with hybrid search + caching).

Endpoints:
    POST /api/ask         — Ask a question, get a cited answer
    POST /api/upload       — Upload a PDF document
    GET  /api/status       — Check if the index is loaded
    POST /api/reindex      — Re-run ingestion pipeline
"""

import hashlib
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import CORS_ORIGINS, DOCUMENTS_DIR, MIN_SCORE_THRESHOLD, ENABLE_CACHE
from app.hybrid_retriever import HybridRetriever
from app.llm import generate_answer
from app.ingest import run_ingestion

app = FastAPI(title="NutriBot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize hybrid retriever at startup
retriever = HybridRetriever()

# Simple in-memory answer cache
answer_cache: dict[str, dict] = {}


# --- Request / Response Models ---

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    cached: bool = False


class StatusResponse(BaseModel):
    ready: bool
    num_chunks: int
    documents_dir: str
    cache_size: int


# --- Helpers ---

def cache_key(question: str) -> str:
    return hashlib.md5(question.strip().lower().encode()).hexdigest()


# --- Endpoints ---

@app.get("/api/status", response_model=StatusResponse)
def get_status():
    return StatusResponse(
        ready=retriever.is_ready(),
        num_chunks=retriever.index.ntotal if retriever.index else 0,
        documents_dir=str(DOCUMENTS_DIR),
        cache_size=len(answer_cache),
    )


@app.post("/api/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not retriever.is_ready():
        raise HTTPException(
            status_code=503,
            detail="No documents indexed yet. Upload PDFs and run ingestion first.",
        )

    # 1. Check cache
    key = cache_key(req.question)
    if ENABLE_CACHE and key in answer_cache:
        cached = answer_cache[key]
        return AskResponse(answer=cached["answer"], sources=cached["sources"], cached=True)

    # 2. Retrieve relevant chunks (hybrid: FAISS + BM25)
    chunks = retriever.search(req.question)

    # 3. Cost gate: skip API if no chunk is relevant enough
    if not chunks or chunks[0].get("semantic_score", 0) < MIN_SCORE_THRESHOLD:
        fallback = (
            "I couldn't find relevant information in the indexed nutrition papers for this question. "
            "Try asking about gut microbiome, omega-3 fatty acids, the Mediterranean diet, "
            "probiotics/prebiotics, or nutrition and immunity."
        )
        return AskResponse(answer=fallback, sources=[], cached=False)

    # 4. Filter out low-scoring chunks to reduce tokens sent to Claude
    relevant_chunks = [c for c in chunks if c.get("semantic_score", 0) > 0.2]
    if not relevant_chunks:
        relevant_chunks = chunks[:1]  # at least send the top chunk

    # 5. Generate answer with Claude
    result = generate_answer(req.question, relevant_chunks)

    # 6. Cache the result
    if ENABLE_CACHE:
        answer_cache[key] = result

    return AskResponse(answer=result["answer"], sources=result["sources"], cached=False)


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOCUMENTS_DIR / file.filename

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"message": f"Uploaded {file.filename}", "path": str(dest)}


@app.post("/api/reindex")
def reindex():
    """Re-run the ingestion pipeline and reload the hybrid index."""
    global retriever, answer_cache
    run_ingestion()
    retriever = HybridRetriever()
    answer_cache.clear()  # invalidate cache after reindex
    return {
        "message": "Reindexing complete",
        "num_chunks": retriever.index.ntotal if retriever.index else 0,
    }
