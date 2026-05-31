"""
core/chunker.py
---------------
Sliding-window chunker that splits text at word boundaries.

Why word-level rather than character-level?
  Character splits can cut mid-word. Word-boundary splits preserve semantic
  units and make embeddings more meaningful.

chunk_size  ≈ 400 chars  →  fits comfortably in MiniLM's 256-token limit
overlap     ≈  80 chars  →  ~20% overlap keeps cross-chunk context
"""

from typing import Optional


# ── Internal helper ───────────────────────────────────────────────────────────

def _char_len(words: list) -> int:
    """Character length of a list of words joined by single spaces."""
    if not words:
        return 0
    return sum(len(w) for w in words) + (len(words) - 1)


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
    page_num: Optional[int] = None,
) -> list:
    """
    Split *text* into overlapping chunks at word boundaries.

    Parameters
    ----------
    text       : Input string.
    chunk_size : Target character count per chunk.
    overlap    : Character overlap between consecutive chunks.
    page_num   : Source page number stored in every returned dict.

    Returns
    -------
    list of dict
        Each dict: { chunk_id (int), text (str), page_num (int|None) }

    Raises
    ------
    ValueError – if overlap >= chunk_size.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    words = text.split()
    if not words:
        return []

    # Edge case: entire text fits in one chunk
    if _char_len(words) <= chunk_size:
        return [{"chunk_id": 0, "text": " ".join(words), "page_num": page_num}]

    step_size = chunk_size - overlap   # how many characters to advance per chunk
    chunks = []
    chunk_id = 0
    word_start = 0                     # word-level pointer into `words`

    while word_start < len(words):
        # ── Build one chunk ──────────────────────────────────────────────────
        chunk_words = []
        char_count = 0

        i = word_start
        while i < len(words):
            w = words[i]
            # +1 for the separating space (not added before first word)
            addition = len(w) + (1 if chunk_words else 0)
            if chunk_words and char_count + addition > chunk_size:
                break
            chunk_words.append(w)
            char_count += addition
            i += 1

        # Safety: a single word longer than chunk_size must still be its own chunk
        if not chunk_words:
            chunk_words = [words[word_start]]
            i = word_start + 1

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": " ".join(chunk_words),
                "page_num": page_num,
            }
        )
        chunk_id += 1

        # ── Advance word_start by step_size characters ───────────────────────
        advanced_chars = 0
        advanced_words = 0
        while word_start + advanced_words < len(words) and advanced_chars < step_size:
            advanced_chars += len(words[word_start + advanced_words]) + 1
            advanced_words += 1

        word_start += max(1, advanced_words)  # always advance at least one word

    return chunks


def chunk_pages(
    pages: list,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list:
    """
    Chunk all pages output by ``parser.parse_pdf`` and assign global IDs.

    Parameters
    ----------
    pages      : List of page dicts from ``parse_pdf``.
    chunk_size : Passed through to ``chunk_text``.
    overlap    : Passed through to ``chunk_text``.

    Returns
    -------
    Flat list of chunk dicts with monotonically increasing chunk_id values.
    """
    all_chunks = []
    global_id = 0

    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            chunk_size=chunk_size,
            overlap=overlap,
            page_num=page["page_num"],
        )
        for chunk in page_chunks:
            chunk["chunk_id"] = global_id
            global_id += 1
            all_chunks.append(chunk)

    return all_chunks
