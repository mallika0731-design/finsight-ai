"""
ui/qa_view.py
-------------
Streamlit view for asking natural-language questions about the loaded document.

Two subtle Streamlit patterns handled correctly here:

1. Example-question buttons that update the text input:
   In Streamlit, st.text_input's ``value`` parameter is only the *initial*
   default. Once the widget has a key, it owns its own state. To update it
   from an external button, write directly to st.session_state[key] and
   call st.rerun().

2. Caching Q&A results across reruns:
   We store the last result in st.session_state so it survives reruns
   caused by slider / other widget interactions.
"""

import streamlit as st

from core.embedder import embed_query
from core.generator import generate_answer
from eval.faithfulness import score_answer

_QA_INPUT_KEY = "qa_question_input"

_EXAMPLE_QUESTIONS = [
    "What is the total revenue?",
    "What is the net profit?",
    "What are the main liabilities?",
    "Is the business profitable?",
    "What are the total assets?",
]

_FAITH_COLORS = {
    "ENTAILED":     ("#bbf7d0", "#14532d"),   # (background, text)
    "NEUTRAL":      ("#fef9c3", "#713f12"),
    "CONTRADICTED": ("#fee2e2", "#7f1d1d"),
}


def _faithfulness_badge(score: float, label: str) -> str:
    """Return inline HTML badge for the faithfulness label."""
    bg, fg = _FAITH_COLORS.get(label, ("#e2e8f0", "#0f172a"))
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 12px;'
        f'border-radius:12px;font-size:12px;font-weight:600;">'
        f"{label} &nbsp;·&nbsp; {score:.0%}"
        f"</span>"
    )


def run_qa_view() -> None:
    st.markdown("### 💬 Ask Questions About Your Document")

    if "retriever" not in st.session_state:
        st.warning("Please upload a document first (use the **Upload** page).")
        return

    st.caption(f"Document: {st.session_state.get('uploaded_filename', '—')}")

    # ── Example question buttons ──────────────────────────────────────────────
    st.markdown("**Quick questions — click to fill:**")
    cols = st.columns(len(_EXAMPLE_QUESTIONS))
    for col, q in zip(cols, _EXAMPLE_QUESTIONS):
        if col.button(q, key=f"example__{q}", use_container_width=True):
            # Write directly to session state so the text_input picks it up
            st.session_state[_QA_INPUT_KEY] = q
            st.rerun()

    # ── Text input ────────────────────────────────────────────────────────────
    question = st.text_input(
        label="Your question",
        placeholder="e.g. What is the net profit margin?",
        key=_QA_INPUT_KEY,
    )

    top_k = st.slider(
        "Context chunks to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="More chunks = broader context; fewer = tighter focus.",
    )

    run_btn = st.button("Get Answer", type="primary", use_container_width=True)

    if run_btn:
        if not question or not question.strip():
            st.warning("Please type a question or click one of the examples above.")
            return

        retriever = st.session_state["retriever"]

        with st.spinner("Retrieving relevant context…"):
            q_emb   = embed_query(question)
            results = retriever.search(q_emb, top_k=top_k)

        with st.spinner("Generating answer…"):
            qa_out  = generate_answer(question, results)

        with st.spinner("Scoring faithfulness…"):
            faith   = score_answer(qa_out["answer"], results)

        # Persist so the result survives subsequent widget interactions
        st.session_state["last_qa_result"] = {
            "question":    question,
            "qa_out":      qa_out,
            "faithfulness": faith,
        }

    # ── Display result ────────────────────────────────────────────────────────
    if "last_qa_result" not in st.session_state:
        return

    saved = st.session_state["last_qa_result"]
    qa    = saved["qa_out"]
    faith = saved["faithfulness"]

    st.markdown("---")

    st.markdown(f"**Q:** {saved['question']}")
    st.markdown(f"**A:** {qa['answer']}")

    st.markdown(
        f"**Faithfulness:** {_faithfulness_badge(faith['entailment'], faith['label'])}",
        unsafe_allow_html=True,
    )

    # Faithfulness explanation tooltip
    with st.expander("What does this faithfulness score mean?"):
        st.markdown(
            "The score comes from a dedicated NLI (Natural Language Inference) model "
            "that checks whether the generated answer is *entailed* by the retrieved "
            "context.\n\n"
            "- **ENTAILED** — the context logically supports the answer ✅\n"
            "- **NEUTRAL** — the answer may be correct but isn't clearly in the context ⚠️\n"
            "- **CONTRADICTED** — the answer conflicts with the retrieved text ❌\n\n"
            f"Entailment probability: **{faith['entailment']:.1%}**  \n"
            f"Contradiction probability: **{faith['contradiction']:.1%}**  \n"
            f"Neutral probability: **{faith['neutral']:.1%}**"
        )

    # Retrieved context chunks
    with st.expander("View retrieved context chunks"):
        for i, r in enumerate(qa["context_used"], start=1):
            chunk = r["chunk"]
            st.markdown(
                f"**Chunk {i}** · Page {chunk.get('page_num', '?')} · "
                f"Cosine similarity: `{r['score']:.4f}`"
            )
            st.text(chunk["text"])
            if i < len(qa["context_used"]):
                st.markdown("---")
