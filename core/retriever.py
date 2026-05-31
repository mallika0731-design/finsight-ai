"""
core/retriever.py
-----------------
FAISS-based cosine similarity retriever.

Design choice — IndexFlatIP:
  Because embed_texts / embed_query produce L2-normalised vectors,
  the inner product between any two vectors equals their cosine similarity.
  IndexFlatIP is exact (no approximation), which is correct for documents
  numbering in the hundreds to low thousands (typical for one PDF).

  NexusIQ used IndexFlatL2, which computes Euclidean distance — a different
  metric from cosine similarity and incorrect for normalised embeddings.
  That bug is fixed here.
"""

import numpy as np
import faiss


class FAISSRetriever:
    """
    Exact cosine-similarity retriever over a set of text chunks.

    Usage
    -----
    retriever = FAISSRetriever()
    retriever.build(chunks, embeddings)
    results   = retriever.search(query_embedding, top_k=5)
    """

    def __init__(self, dimension: int = 384):
        """
        Parameters
        ----------
        dimension : Embedding dimension. Must match embed_texts output (384).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)   # inner product on L2-normed vecs
        self.chunks: list = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: list, embeddings: np.ndarray) -> None:
        """
        Index a set of chunks and their pre-computed embeddings.

        Parameters
        ----------
        chunks     : List of chunk dicts from ``chunker.chunk_pages``.
        embeddings : float32 array of shape (n_chunks, dimension).

        Raises
        ------
        ValueError – on empty input or shape mismatch.
        """
        if not chunks:
            raise ValueError("chunks must not be empty")

        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D, got shape {embeddings.shape}"
            )
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Shape mismatch: {len(chunks)} chunks but "
                f"{embeddings.shape[0]} embedding rows"
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} != "
                f"index dim {self.dimension}"
            )

        self.chunks = list(chunks)          # defensive copy
        self.index.reset()                  # clear any previous index
        self.index.add(embeddings)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list:
        """
        Return the top-k most similar chunks for a query.

        Parameters
        ----------
        query_embedding : float32 array of shape (1, dimension), L2-normalised.
        top_k           : Maximum number of results.

        Returns
        -------
        list of dict — [{ chunk: dict, score: float }]
        Sorted by descending cosine similarity score.
        Scores are in [-1, 1]; higher is more similar.

        Raises
        ------
        RuntimeError – if build() has not been called yet.
        """
        if self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Call build() before search().")

        if query_embedding.ndim != 2 or query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"query_embedding must be shape (1, {self.dimension}), "
                f"got {query_embedding.shape}"
            )

        k = min(top_k, self.index.ntotal)   # can't return more than indexed
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:                     # FAISS signals "not found" with -1
                continue
            results.append(
                {
                    "chunk": self.chunks[idx],
                    "score": float(score),  # cosine similarity ∈ [-1, 1]
                }
            )

        # Already sorted by FAISS (descending score), but make it explicit
        results.sort(key=lambda r: r["score"], reverse=True)
        return results
