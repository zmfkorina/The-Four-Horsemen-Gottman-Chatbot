import os, re, json, math, glob
from typing import List, Dict, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.config import (
    DATA_DIR, INDEX_DIR, META_PATH, FAISS_PATH, EMBEDDING_MODEL,
    CHUNK_WORDS, CHUNK_OVERLAP
)

os.makedirs(INDEX_DIR, exist_ok=True)

# ---------- text extraction ----------

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_md(path: str) -> str:
    return read_txt(path)

def read_pdf(path: str) -> str:
    # pypdf simple, robust for text-based PDFs
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts)

def read_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html5lib")
    # Remove script/style/nav
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.extract()
    text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()

EXT_READERS = {
    ".txt": read_txt,
    ".md": read_md,
    ".pdf": read_pdf,
    ".html": read_html,
    ".htm": read_html,
}

# ---------- chunking ----------

def word_chunks(text: str, chunk_words: int, overlap: int) -> List[str]:
    words = re.findall(r"\S+", text)
    chunks = []
    step = max(1, chunk_words - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+chunk_words])
        if chunk:
            chunks.append(chunk)
    return chunks

# ---------- embedding ----------

e5 = SentenceTransformer(EMBEDDING_MODEL, device="mps")

def embed_passages(passages: List[str]) -> np.ndarray:
    # E5 expects "passage: " prefix for documents
    prefixed = [f"passage: {p}" for p in passages]
    # normalize_embeddings=True gives unit vectors so we can use inner product
    vecs = e5.encode(prefixed, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")

# ---------- ingest pipeline ----------

def iter_files(data_dir: str) -> Iterable[str]:
    exts = tuple(EXT_READERS.keys())
    for path in glob.glob(os.path.join(data_dir, "**", "*"), recursive=True):
        if os.path.isfile(path) and path.lower().endswith(exts):
            yield path


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    reader = EXT_READERS.get(ext)
    if not reader:
        return ""
    try:
        return reader(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return ""


def main():
    docs_meta = []
    all_chunks = []

    for path in iter_files(DATA_DIR):
        text = extract_text(path)
        if not text or len(text.strip()) < 50:
            continue
        title = os.path.basename(path)
        chunks = word_chunks(text, CHUNK_WORDS, CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            meta = {
                "doc_id": f"{title}#{idx}",
                "title": title,
                "source_path": path,
                "chunk_index": idx,
                "text": chunk,
            }
            docs_meta.append(meta)
            all_chunks.append(chunk)

    if not all_chunks:
        print("No chunks found. Put files under ./data and try again.")
        return

    vecs = embed_passages(all_chunks)
    dim = vecs.shape[1]

    # Inner-product FAISS on normalized vectors ≈ cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in docs_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Indexed {len(all_chunks)} chunks → {FAISS_PATH}\nMetadata → {META_PATH}")

if __name__ == "__main__":
    main()