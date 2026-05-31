"""
ui/upload_view.py
-----------------
Streamlit view that handles the full ingestion pipeline:
  PDF upload → parse → chunk → embed → FAISS index → session_state
"""

import os
import tempfile
import streamlit as st

from core.parser import parse_pdf, pages_to_full_text
from core.chunker import chunk_pages
from core.embedder import embed_texts
from core.retriever import FAISSRetriever

_MAX_FILE_MB = 20


def run_upload_view() -> None:
    st.markdown("### 📄 Upload a Financial Document")
    st.markdown(
        "Supported formats: balance sheets, P&L statements, GST filings, "
        "bank statements — any **text-based PDF** up to 20 MB."
    )

    uploaded_file = st.file_uploader(
        label="Choose a PDF",
        type=["pdf"],
        help="Scanned/image-only PDFs will not extract correctly. Use text-based PDFs.",
    )

    if uploaded_file is None:
        # Show status if a document is already loaded
        if "uploaded_filename" in st.session_state:
            st.info(
                f"Currently indexed: **{st.session_state['uploaded_filename']}**  \n"
                "Navigate to **Q&A** or **Insights** in the sidebar."
            )
        return

    # ── Guard: file size ─────────────────────────────────────────────────────
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > _MAX_FILE_MB:
        st.error(f"File is {size_mb:.1f} MB — limit is {_MAX_FILE_MB} MB.")
        return

    # ── Guard: avoid re-processing the same file ─────────────────────────────
    if st.session_state.get("uploaded_filename") == uploaded_file.name:
        st.success(
            f"✅ **{uploaded_file.name}** is already indexed.  \n"
            "Use **Q&A** or **Insights** from the sidebar."
        )
        return

    # ── Step 1: Parse ─────────────────────────────────────────────────────────
    with st.spinner("Parsing PDF…"):
        # pdfplumber needs a real file path, not a BytesIO buffer
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                fh.write(uploaded_file.read())
            pages = parse_pdf(tmp_path)
        finally:
            # Always remove the temp file, even if parsing fails
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if not pages:
        st.error(
            "Could not extract any text from this PDF.  \n"
            "Make sure the file is a **text-based** PDF (not a scanned image)."
        )
        return

    st.info(f"Parsed **{len(pages)} page(s)**.")

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    with st.spinner("Chunking text…"):
        chunks = chunk_pages(pages, chunk_size=400, overlap=80)

    if not chunks:
        st.error("No text chunks could be created. The document may be empty.")
        return

    st.info(f"Created **{len(chunks)} text chunks**.")

    # ── Step 3: Embed + Index ─────────────────────────────────────────────────
    with st.spinner(
        "Building semantic index — this takes ~30 s on first run "
        "(model downloads once, then stays cached)…"
    ):
        chunk_texts_list = [c["text"] for c in chunks]
        embeddings = embed_texts(chunk_texts_list)

        retriever = FAISSRetriever(dimension=embeddings.shape[1])
        retriever.build(chunks, embeddings)

    # ── Persist to session_state ──────────────────────────────────────────────
    st.session_state["retriever"]         = retriever
    st.session_state["chunks"]            = chunks
    st.session_state["pages"]             = pages
    st.session_state["full_text"]         = pages_to_full_text(pages)
    st.session_state["uploaded_filename"] = uploaded_file.name

    # Clear any stale results from a previous document
    for key in ("last_qa_result",):
        st.session_state.pop(key, None)

    st.success(
        f"✅ **{uploaded_file.name}** indexed successfully!  \n"
        "Go to **Q&A** to ask questions or **Insights** for the loan score."
    )
