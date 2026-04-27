"""
embeddings.py — sentence-transformers + FAISS vector store for HeyPDF 2.0

Model: all-MiniLM-L6-v2 (~90MB, downloaded once and cached by sentence-transformers)
Index: One FAISS IndexFlatIP per uploaded PDF, stored at backend/storage/{pdf_id}/

Why IndexFlatIP?
  Embeddings are L2-normalized, so inner product == cosine similarity.
  IndexFlatIP is exact (no ANN approximation) and fast enough for session-scale data.
"""

import os
import pickle
import shutil

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Storage directory ───────────────────────────────────────────────────────
# Stored relative to this file so it works regardless of cwd
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# ── Load model once at import time ──────────────────────────────────────────
# sentence-transformers caches the model in ~/.cache/huggingface after first download
print("[Embeddings] Loading sentence-transformers model (all-MiniLM-L6-v2)...")
_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[Embeddings] Model ready.")

EMBEDDING_DIM = 384  # Output dimension of all-MiniLM-L6-v2


# ── Path helpers ─────────────────────────────────────────────────────────────

def _storage_path(pdf_id: str) -> str:
    """Return (and create) the storage directory for a given PDF."""
    path = os.path.join(STORAGE_DIR, pdf_id)
    os.makedirs(path, exist_ok=True)
    return path


# ── Public API ───────────────────────────────────────────────────────────────

def embed_and_store(chunks: list[dict], pdf_id: str) -> None:
    """
    Embed all chunks and persist the FAISS index + chunk metadata to disk.

    Args:
        chunks: List of chunk dicts from pdf_processor (must have 'text' key)
        pdf_id: Unique PDF identifier used as the storage directory name
    """
    texts = [c["text"] for c in chunks]

    # Generate normalized embeddings (shape: [n_chunks, 384])
    print(f"[Embeddings] Embedding {len(texts)} chunks for pdf_id={pdf_id}...")
    embeddings = _model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,  # L2-normalize for cosine similarity
        batch_size=64,
    )

    # Build and populate FAISS index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(np.array(embeddings, dtype=np.float32))

    # Persist both the index and the raw chunk metadata
    path = _storage_path(pdf_id)
    faiss.write_index(index, os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"[Embeddings] Stored {len(chunks)} chunks for pdf_id={pdf_id}.")


def search_chunks(query: str, pdf_ids: list[str], top_k: int = 5) -> list[dict]:
    """
    Search across multiple PDF FAISS indexes for the most relevant chunks.

    Args:
        query: The user's question
        pdf_ids: PDF IDs whose indexes to search
        top_k: Maximum number of results to return (across all PDFs combined)

    Returns:
        List of chunk dicts sorted by descending cosine similarity,
        with an additional 'score' field added.
    """
    if not pdf_ids:
        return []

    # Embed the query (normalized for cosine similarity)
    query_vec = _model.encode([query], normalize_embeddings=True)
    query_arr = np.array(query_vec, dtype=np.float32)

    all_results = []

    for pdf_id in pdf_ids:
        path = _storage_path(pdf_id)
        index_file = os.path.join(path, "index.faiss")
        chunks_file = os.path.join(path, "chunks.pkl")

        if not (os.path.exists(index_file) and os.path.exists(chunks_file)):
            print(f"[Embeddings] Warning: No index for pdf_id={pdf_id}, skipping.")
            continue

        # Load index and chunk metadata
        index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)

        # Search: retrieve top_k results per PDF
        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_arr, k)

        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for unused slots
                chunk = chunks[idx].copy()
                chunk["score"] = float(score)
                all_results.append(chunk)

    # Merge and return global top_k by score
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


def delete_pdf_index(pdf_id: str) -> None:
    """
    Remove the FAISS index and chunk metadata for a PDF from disk.

    Args:
        pdf_id: The PDF whose storage directory should be deleted
    """
    path = _storage_path(pdf_id)
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[Embeddings] Deleted index for pdf_id={pdf_id}.")
