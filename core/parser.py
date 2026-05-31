"""
core/parser.py
--------------
Extracts text and table data from PDF files using pdfplumber.
Returns a list of page dicts that the chunker consumes.
"""

import re
import pdfplumber
from pathlib import Path
from typing import Union


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Normalise whitespace and strip non-printable characters.
    Preserves newlines so sentence boundaries survive.
    """
    # Drop characters that are neither printable ASCII nor common Unicode
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)
    # Collapse runs of spaces / tabs (but keep newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _table_to_text(table: list) -> str:
    """
    Flatten a pdfplumber table (list-of-lists) into pipe-delimited rows.
    Skips rows that are entirely empty.
    """
    rows = []
    for row in table:
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        if any(cells):                        # skip blank rows
            rows.append(" | ".join(cells))
    return "\n".join(rows)


# ── Public API ────────────────────────────────────────────────────────────────

def parse_pdf(file_path: Union[str, Path]) -> list:
    """
    Open a PDF and extract text + tables page-by-page.

    Parameters
    ----------
    file_path : str or Path
        Absolute path to the PDF file.

    Returns
    -------
    list of dict
        Each dict: { page_num (int), text (str), has_tables (bool) }
        Pages with no extractable text are omitted.

    Raises
    ------
    FileNotFoundError  – if the path does not exist.
    ValueError         – if pdfplumber cannot open the file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages_out = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # --- Main text body ---
            raw_text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""

            # --- Tables (optional; never crash the pipeline if they fail) ---
            table_parts = []
            try:
                for tbl in page.extract_tables() or []:
                    t = _table_to_text(tbl)
                    if t:
                        table_parts.append(t)
            except Exception:
                pass  # Tables are supplementary; swallow extraction errors

            # Combine body + tables
            combined = raw_text
            if table_parts:
                combined += "\n\n[TABLE DATA]\n" + "\n\n".join(table_parts)

            cleaned = _clean_text(combined)
            if cleaned:
                pages_out.append(
                    {
                        "page_num": page_num,
                        "text": cleaned,
                        "has_tables": bool(table_parts),
                    }
                )

    return pages_out


def pages_to_full_text(pages: list) -> str:
    """
    Concatenate all page texts into one string with page markers.
    Useful for regex-based financial ratio extraction.
    """
    return "\n\n".join(f"[Page {p['page_num']}]\n{p['text']}" for p in pages)
