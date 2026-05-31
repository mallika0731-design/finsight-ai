"""
core/embedder.py
----------------
Sentence embeddings via all-MiniLM-L6-v2.

Why MiniLM-L6-v2?
  - 22 MB model, fast on CPU
  - 384-dim dense embeddings
  - State-of-the-art on STS benchmarks for its size
  - Native support for ``normalize_embeddings=True`` → L2-normalised vectors
    so dot-product == cosine similarity (required for FAISS IndexFlatIP)

The module uses a module-level singleton so the model is loaded once per
process and never reloaded across Streamlit reruns.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singleton — loaded lazily on first call
_model: SentenceTransformer = None  # type: ignore[assignment]


def _get_model() -> SentenceTransformer:
    """Return the cached model, loading it if necessary."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ── Public API ────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised float32 embeddings.

    Parameters
    ----------
    texts      : Non-empty list of strings to embed.
    batch_size : Encoding mini-batch size (tune to available RAM).

    Returns
    -------
    np.ndarray of shape (len(texts), 384), dtype=float32, L2-normalised.

    Raises
    ------
    ValueError – if ``texts`` is empty.
    """
    if not texts:
        raise ValueError("texts must not be empty")

    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise → dot-product == cosine sim
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns
    -------
    np.ndarray of shape (1, 384), dtype=float32, L2-normalised.
    The 2-D shape is what FAISS ``index.search`` expects.

    Raises
    ------
    ValueError – if ``query`` is blank.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must not be blank")

    # embed_texts returns (1, 384); keep the batch dimension for FAISS
    return embed_texts([query])
