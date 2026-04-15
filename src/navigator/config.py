"""Configuration for the Navigator application."""

import os
from pathlib import Path

# Paths — override with NAVIGATOR_DATA_DIR env var for container deployments
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path(os.environ["NAVIGATOR_DATA_DIR"]) if "NAVIGATOR_DATA_DIR" in os.environ else PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db"
FPL_DIR = DATA_DIR / "fpl_tables"
PROGRAMS_DIR = DATA_DIR / "programs"
TRAINING_DIR = DATA_DIR / "training"

# RunPod detection — RunPod auto-sets RUNPOD_POD_ID in pod environments
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID")
IS_RUNPOD = RUNPOD_POD_ID is not None


def get_runpod_public_url(port: int = 7860) -> str | None:
    """Return the RunPod proxy URL for the given port, or None if not on RunPod."""
    if RUNPOD_POD_ID is None:
        return None
    return f"https://{RUNPOD_POD_ID}-{port}.proxy.runpod.net"


# Ollama — env override for Docker Compose (OLLAMA_BASE_URL) or RunPod (localhost)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "navigator"

# RunPod: persist Ollama models on /workspace volume
if IS_RUNPOD and "OLLAMA_MODELS" not in os.environ:
    os.environ["OLLAMA_MODELS"] = "/workspace/.ollama/models"

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ChromaDB
CHROMA_COLLECTION = "benefits_kb"

# RAG
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_VECTOR = 10
TOP_K_BM25 = 10
TOP_K_FINAL = 8
BM25_WEIGHT = 0.4  # alpha for hybrid: (1-alpha)*vector + alpha*bm25

# Readability
TARGET_READING_LEVELS = {
    "simple": 5.0,    # 5th grade Flesch-Kincaid
    "standard": 8.0,  # 8th grade
    "detailed": 12.0, # 12th grade (no simplification)
}

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "hmn": "Hmong",
    "so": "Somali",
}

# Ensure data directories exist
for d in [RAW_DIR, PROCESSED_DIR, CHROMA_DIR, FPL_DIR, PROGRAMS_DIR, TRAINING_DIR]:
    d.mkdir(parents=True, exist_ok=True)
