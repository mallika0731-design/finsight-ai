"""
core/generator.py
-----------------
Grounded answer generation using google/flan-t5-base.

Why Flan-T5?
  - Instruction-fine-tuned → follows "answer only from context" instructions
  - 248 M parameters, runs well on CPU
  - No proprietary API needed
  - Unlike the NexusIQ approach (Python f-string template), this is a real
    neural generation step — answers will vary based on the retrieved context

Generation strategy: beam search (num_beams=4, do_sample=False)
  - Deterministic and reproducible — important for demos
  - no_repeat_ngram_size=3 prevents copy-paste repetition from context
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_TOKENS = 512       # Flan-T5-base maximum context window
MAX_NEW_TOKENS = 200         # cap output to avoid runaway generation

# Module-level singletons — loaded once per process
_tokenizer = None
_model = None


def _get_model():
    """Return cached (tokenizer, model), loading on first call."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _model.eval()   # inference-only: disables dropout
    return _tokenizer, _model


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(question: str, retrieved: list) -> str:
    """
    Build an instruction-style prompt for Flan-T5.

    Flan-T5 was fine-tuned on templates like:
      "Answer the question based on the passage: ... Question: ... Answer:"
    so we follow that pattern precisely.

    Parameters
    ----------
    question  : User's natural-language question.
    retrieved : Output from FAISSRetriever.search().
    """
    context_parts = []
    for i, r in enumerate(retrieved, start=1):
        context_parts.append(f"[{i}] {r['chunk']['text']}")
    context = "\n".join(context_parts)

    return (
        "Answer the following question using only the information in the context below. "
        "If the answer cannot be found in the context, respond with exactly: "
        "'I could not find this information in the document.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def generate_answer(question: str, retrieved: list) -> dict:
    """
    Generate a grounded answer from retrieved context chunks.

    Parameters
    ----------
    question  : User's question string.
    retrieved : List of result dicts from FAISSRetriever.search().

    Returns
    -------
    dict with keys:
        answer       (str)  – generated answer
        prompt       (str)  – full prompt sent to the model
        context_used (list) – the retrieved results used for generation
    """
    if not retrieved:
        return {
            "answer": "No relevant context found in the document.",
            "prompt": "",
            "context_used": [],
        }

    tokenizer, model = _get_model()
    prompt = _build_prompt(question, retrieved)

    # Tokenise — truncate to model's max if the prompt is very long
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        padding=False,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            do_sample=False,            # beam search is deterministic
        )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    return {
        "answer": answer or "The model returned an empty response.",
        "prompt": prompt,
        "context_used": retrieved,
    }
