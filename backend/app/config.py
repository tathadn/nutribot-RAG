import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# --- Anthropic ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# --- Embeddings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Chunking ---
CHUNK_SIZE = 400        # smaller chunks = more precise retrieval
CHUNK_OVERLAP = 40      # token overlap between consecutive chunks

# --- Retrieval (cost-optimized) ---
TOP_K = 3                     # retrieve top 3 chunks (down from 5)
MIN_SCORE_THRESHOLD = 0.3     # skip API call if best semantic score below this
ENABLE_CACHE = True           # cache answers to avoid repeat API calls
CLAUDE_MAX_TOKENS = 512       # shorter answers = lower cost

# --- API ---
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
]
