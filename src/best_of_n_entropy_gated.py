"""
best_of_n_entropy_gated.py
Two-level hallucination reduction:

  Level 1 (token): SLED + entropy gate  — corrects individual uncertain tokens
  Level 2 (question): Self-consistency best-of-N — when the model is uncertain
                       about the ENTIRE question, generate N answers and pick
                       the most self-consistent one.

The novel contribution: question-level entropy gates whether to run the
expensive N-sample consistency check. Easy questions skip Level 2 entirely.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from generate import model, tokenizer, get_layer_logits_cached, compute_entropy, gated_generate


scorer = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity in MiniLM embedding space."""
    ea = scorer.encode(text_a, convert_to_tensor=True, device="cpu")
    eb = scorer.encode(text_b, convert_to_tensor=True, device="cpu")
    return float(util.cos_sim(ea, eb).item())


def question_level_entropy(prompt: str) -> float:
    """
    First-token entropy — how uncertain is the model about where to even start.
    High → model is globally confused about this question (Level 2 fires).
    Low  → model knows what it's saying (skip Level 2, trust Level 1 output).
    """
    input_ids        = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    layer_logits, _  = get_layer_logits_cached(input_ids, None)
    return compute_entropy(layer_logits[-1])


def entropy_gated_best_of_n(
    prompt: str,
    question_entropy_threshold: float = 3.5,   # Level-2 gate
    n_samples: int = 3,                          # N candidates when gate fires
    token_entropy_threshold: float = 3.5,        # Level-1 SLED gate
    contrast_alpha: float = 0.3,
    repetition_penalty: float = 1.3,
    max_new_tokens: int = 80,
) -> dict:
    """
    Two-level pipeline:
      Always: Level 1 SLED+gate → primary answer
      If q_entropy > question_entropy_threshold:
        Level 2: generate N-1 more candidates, pick most self-consistent
    """
    # ── Level 1: SLED + token-entropy gate ───────────────────────────────────
    primary_result = gated_generate(
        prompt,
        max_new_tokens=max_new_tokens,
        curve_threshold=0.0,
        entropy_threshold=token_entropy_threshold,
        contrast_alpha=contrast_alpha,
        repetition_penalty=repetition_penalty,
        gate_mode="sled_entropy",
    )
    primary_answer = primary_result["text"]

    # ── Level 2: Question-entropy gate ───────────────────────────────────────
    q_entropy = question_level_entropy(prompt)

    if q_entropy <= question_entropy_threshold:
        return {
            "text":             primary_answer,
            "strategy":         "sled_only",
            "question_entropy": round(q_entropy, 3),
            "n_generated":      1,
            "gate_fire_rate":   primary_result["gate_fire_rate"],
        }

    # High question entropy → generate N-1 additional candidates
    candidates = [primary_answer]

    for _ in range(n_samples - 1):
        sample_result = gated_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            curve_threshold=0.0,
            entropy_threshold=token_entropy_threshold + 0.5,  # slightly more lenient
            contrast_alpha=contrast_alpha * 0.8,              # lighter correction
            repetition_penalty=repetition_penalty,
            gate_mode="sled_entropy",
        )
        if len(sample_result["text"].strip()) > 5:
            candidates.append(sample_result["text"])

    if len(candidates) == 1:
        return {
            "text":             primary_answer,
            "strategy":         "sled_only_fallback",
            "question_entropy": round(q_entropy, 3),
            "n_generated":      1,
            "gate_fire_rate":   primary_result["gate_fire_rate"],
        }

    # ── Pick most self-consistent candidate ──────────────────────────────────
    n      = len(candidates)
    scores = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if i != j:
                scores[i] += compute_semantic_similarity(candidates[i], candidates[j])
        scores[i] /= (n - 1)

    best_idx    = int(np.argmax(scores))
    best_answer = candidates[best_idx]

    return {
        "text":               best_answer,
        "strategy":           "sled_plus_consistency",
        "question_entropy":   round(q_entropy, 3),
        "n_generated":        len(candidates),
        "best_candidate_idx": best_idx,
        "consistency_scores": [round(float(s), 3) for s in scores],
        "gate_fire_rate":     primary_result["gate_fire_rate"],
    }
