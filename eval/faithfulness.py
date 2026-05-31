"""
eval/faithfulness.py
--------------------
NLI-based faithfulness scoring.

What is faithfulness?
  A generated answer is *faithful* if it is entailed by the retrieved context —
  i.e., the context contains enough information to logically support the answer.
  A low faithfulness score means the model hallucinated.

Model: cross-encoder/nli-MiniLM2-L6-H768
  - Trained on SNLI + MultiNLI
  - 3 output classes: contradiction (0), entailment (1), neutral (2)
  - CrossEncoder.predict() returns raw logits → we apply softmax for probabilities

Why this beats NexusIQ's confidence score:
  NexusIQ: confidence = 1 - (distances[0][0] / np.max(distances[0]))
    - Mathematically undefined when all distances are equal (0 / 0 = NaN)
    - L2 distance does not map to a probability in [0, 1]
    - Has no relationship to whether the answer is actually grounded

  FinSight: P(entailment | context, answer) via NLI
    - True probability from a calibrated classifier
    - Directly answers "is this answer supported by the retrieved text?"
    - Explainable to any judge
"""

import numpy as np
from scipy.special import softmax
from sentence_transformers.cross_encoder import CrossEncoder

NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"

# Label indices for this model (from its config.json):
#   0 → contradiction, 1 → entailment, 2 → neutral
_LABEL_CONTRADICTION = 0
_LABEL_ENTAILMENT = 1
_LABEL_NEUTRAL = 2

# Module-level singleton
_nli_model: CrossEncoder = None  # type: ignore[assignment]


def _get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder(NLI_MODEL_NAME)
    return _nli_model


# ── Internal scorer ───────────────────────────────────────────────────────────

def _score_pair(premise: str, hypothesis: str) -> dict:
    """
    Score one (premise, hypothesis) pair.

    For faithfulness:
      premise   = the retrieved context (what the model was given)
      hypothesis = the generated answer (what we want to verify)

    Returns probabilities for all three NLI classes.
    """
    if not premise.strip() or not hypothesis.strip():
        return {
            "entailment": 0.0,
            "contradiction": 0.0,
            "neutral": 1.0,
            "label": "NEUTRAL",
        }

    model = _get_nli_model()

    # predict() on a list of pairs → shape (n_pairs, 3) raw logits
    logits = model.predict([(premise, hypothesis)])   # (1, 3)

    # Softmax converts logits to a proper probability distribution
    probs = softmax(logits[0]).tolist()

    p_contradiction = probs[_LABEL_CONTRADICTION]
    p_entailment    = probs[_LABEL_ENTAILMENT]
    p_neutral       = probs[_LABEL_NEUTRAL]

    # Assign label to the highest-probability class
    if p_entailment >= p_neutral and p_entailment >= p_contradiction:
        label = "ENTAILED"
    elif p_contradiction >= p_neutral:
        label = "CONTRADICTED"
    else:
        label = "NEUTRAL"

    return {
        "entailment":    round(p_entailment,    4),
        "contradiction": round(p_contradiction, 4),
        "neutral":       round(p_neutral,       4),
        "label":         label,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def score_answer(answer: str, retrieved: list) -> dict:
    """
    Score how faithfully *answer* is supported by *retrieved* context chunks.

    We concatenate the top-3 retrieved chunks as the NLI premise so the
    model sees the same evidence the generator saw.

    Parameters
    ----------
    answer    : Generated answer string from ``generator.generate_answer``.
    retrieved : List of result dicts from ``FAISSRetriever.search``.

    Returns
    -------
    dict with keys:
        entailment    (float) – P(answer is entailed by context) ∈ [0, 1]
        contradiction (float) – P(answer contradicts context)
        neutral       (float) – P(answer is unrelated)
        label         (str)   – "ENTAILED" | "NEUTRAL" | "CONTRADICTED"
    """
    if not retrieved:
        return {
            "entailment": 0.0,
            "contradiction": 0.0,
            "neutral": 1.0,
            "label": "NEUTRAL",
        }

    # Use at most the top-3 chunks to keep the NLI input tractable
    context = " ".join(r["chunk"]["text"] for r in retrieved[:3])
    return _score_pair(premise=context, hypothesis=answer)
