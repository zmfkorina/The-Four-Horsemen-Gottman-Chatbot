from typing import List, Dict, Any
import os, json, re
import torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # quieter & safer on macOS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.config import (
    FAISS_PATH, META_PATH, EMBEDDING_MODEL,
    USE_RERANKER, RERANKER_MODEL,
    TOP_K, RERANK_K, FINAL_K, MAX_QUOTE_CHARS,
    HF_MODEL_ID, LOAD_4BIT, DEVICE_MAP,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P
)
from app.prompts import SYSTEM_PROMPT, DISCLAIMER, CRISIS_KEYWORDS

# Optional reranker
try:
    from FlagEmbedding import FlagReranker  # pip install -U FlagEmbedding
    _HAS_RERANKER = True
except Exception:
    _HAS_RERANKER = False

# === Hugging Face Transformers LLM ===
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_tokenizer = None
_model = None

def _load_llm():
    global _tokenizer, _model
    if _model is not None:
        return

    _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=True)
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id

    # Load on CPU in half precision, then move to MPS and use eager attention (more stable on Mac)
    _model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _model.to("mps")


# === Vector index + metadata ===
e5 = SentenceTransformer(EMBEDDING_MODEL)
_index = faiss.read_index(FAISS_PATH)
_meta: List[Dict[str, Any]] = [json.loads(l) for l in open(META_PATH, "r", encoding="utf-8")]

_reranker = None
if USE_RERANKER and _HAS_RERANKER:
    _reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)


def embed_query(q: str):
    vec = e5.encode([f"query: {q}"], normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")


def search(q: str) -> List[Dict[str, Any]]:
    qv = embed_query(q)
    D, I = _index.search(qv, max(RERANK_K if _reranker else TOP_K, TOP_K))

    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        m = _meta[idx]
        hits.append({
            "score": float(score),
            "text": m["text"],
            "title": m["title"],
            "doc_id": m["doc_id"],
            "source_path": m["source_path"],
        })

    if _reranker:
        pairs = [[q, h["text"]] for h in hits[:RERANK_K]]
        scores = _reranker.compute_score(pairs, normalize=True)
        for h, s in zip(hits[:RERANK_K], scores):
            h["rerank"] = float(s)
        hits = sorted(hits[:RERANK_K], key=lambda x: x.get("rerank", x["score"]), reverse=True)

    return hits[:FINAL_K]


def clamp_quotes(text: str, max_chars: int) -> str:
    def _shorten(m):
        body = m.group(1)
        if len(body) <= max_chars:
            return f'"{body}"'
        return f'"{body[:max_chars]}…"'
    return re.sub(r'"([^"]{'+str(max_chars+50)+r',})"', _shorten, text)


# --- replace your build_messages() ---
def build_messages(q: str, contexts: List[Dict[str, Any]]):
    bullets = []
    for i, c in enumerate(contexts, 1):
        snippet = c["text"].replace("\n", " ")           # <- FIX: only strip newlines
        if len(snippet) > 1200:
            snippet = snippet[:1200] + " …"
        bullets.append(f"[{i}] Title: {c['title']} | DocID: {c['doc_id']}\n{snippet}")

    ctx = "\n\n".join(bullets)
    user = (
        f"CONTEXT\n{ctx}\n\n"
        f"USER QUESTION\n{q}\n\n"
        "TASK: Using ONLY the context snippets [1..N], answer with: "
        "(1) a 3–6 step plan, (2) concrete phrases to try, (3) short rationale. "
        "Cite sources as [#] by DocID/Title. If context is insufficient, say so."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(max_quote=MAX_QUOTE_CHARS)},
        {"role": "user", "content": user},
    ]



def generate_answer(messages) -> str:
    _load_llm()
    # Use chat template for correct formatting
    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,  # avoid runaway prompts
        max_length=8192  # safe cap; Phi-4-mini supports far larger contexts but 8k is plenty for RAG
    )
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = _tokenizer.decode(gen_ids, skip_special_tokens=True)
    text = clamp_quotes(text, MAX_QUOTE_CHARS)
    return text


def maybe_disclaimer(text: str) -> str:
    if any(kw.lower() in text.lower() for kw in CRISIS_KEYWORDS):
        return text + "**Important:** " + DISCLAIMER
    return text + ("" + DISCLAIMER if os.getenv("APPEND_DISCLAIMER", "true").lower()=="true" else "")


def ask(q: str) -> Dict[str, Any]:
    docs = search(q)
    messages = build_messages(q, docs)
    answer = generate_answer(messages)
    answer = maybe_disclaimer(answer)
    cites = [{"title": d["title"], "doc_id": d["doc_id"]} for d in docs]
    return {"answer": answer, "citations": cites}