import os
from dotenv import load_dotenv

load_dotenv()

# === Models & paths ===
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "microsoft/Phi-4-mini-instruct")
LOAD_4BIT = os.getenv("LOAD_4BIT", "false").lower() == "true" # requires bitsandbytes
DEVICE_MAP = os.getenv("DEVICE_MAP", "auto") # "auto" | "cuda" | "cpu"
ATTN_IMPL = os.getenv("ATTN_IMPL", "eager") # "auto" -> let HF pick; or "eager" if flash-attn not supported

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_DIR = os.getenv("INDEX_DIR", "index")
META_PATH = os.path.join(INDEX_DIR, "meta.jsonl")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")

# === Retrieval ===
TOP_K = int(os.getenv("TOP_K", 12))
RERANK_K = int(os.getenv("RERANK_K", 50))
FINAL_K = int(os.getenv("FINAL_K", 6))
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", 380))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 60))
MAX_QUOTE_CHARS = int(os.getenv("MAX_QUOTE_CHARS", 320))

# === Generation params ===
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
TOP_P = float(os.getenv("TOP_P", 0.9))

# Safety / policy
DISPLAY_DISCLAIMER = os.getenv("DISPLAY_DISCLAIMER", "true").lower() == "true"