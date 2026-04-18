#!/usr/bin/env python3
# CRITICAL BUG HISTORY: tau_kappa was originally 0.08 and tau_ECR was 0.10.
# These values passed essentially no questions through Gate 2 because:
#   measured mean_kappa ≈ 0.597 >> 0.08 (gate requires kappa_q < tau_kappa)
#   measured mean_ECR   ≈ 0.031–0.076, often < 0.10 (gate requires ECR_q > tau_ECR)
# Fixed to tau_kappa=0.70 and tau_ECR=0.04 in v2. See configs/router_thresholds.json.
# Old CURED (broken gates): ~greedy parity on TruthfulQA. Fixed CURED v2: ~+10 pp over greedy.
"""
CURED: Complete Unified Routing and Evaluation for Decoding
===========================================================

Single-file script and importable module for multi-model hallucination
mitigation experiments.  See the ``cured/`` package for the organized
Python API surface.

Architecture Overview
---------------------
CURED implements a 5-gate principled router that selects a decoding
strategy per question using three trajectory features:

  R²   — mean R² of late-layer logit growth (measures ALTA viability)
  κ    — quadratic gain fraction (curvature of logit trajectory)
  ECR  — Entropy Compression Ratio: H_final / H_peak

Gate flow (CUREDRouterV2; first match wins in order below):
  Gate 1  H_final < τ_H_easy (0.5) and (for ≤14B) SC_q ≥ τ_SC_easy — requires ``--compute-sc``
          → greedy_confident.  Canonical Phase 4 omits ``--compute-sc``, so Gate 1 is inactive at 3B/8B.
  Scale   profile_mean_r2 ≥ 0.55, not medical, H_final > τ_H_easy → alta_global_viable
  Gate 2  R²_q > τ_R2, κ_q < τ_κ, ECR_q > τ_ECR → continue toward ALTA / ITI / CoVe (see Gates 3–5)
  Gate 3  medical + ITI available → iti_medical_gate3
  Gate 4  composite ALTA score S_ALTA > 0.5 → alta_gate4
  Gate 5  medical + SC > 0.5 (when SC computed) → cove_gate5_medical; else → greedy_gate5

5-Phase Experiment Pipeline
---------------------------
  Phase 1  Measure logit linearity R² per model (compute_logit_linearity.py)
  Phase 2  Protocol ablations: greedy / ALTA / CoVe / ITI  (run_phase2_ablations.sh)
  Phase 3  Calibrate router thresholds  (calibrate_router.py)
  Phase 4  Main CURED v2 evaluation n=500  (run_all_experiments.sh)
  Phase 5  Statistics + R²-stratified analysis  (compute_final_stats.py)

Features
--------
- Loads HuggingFace causal LMs (Llama/Mistral/Qwen/Gemma/GPT-like families)
- Runs standalone protocols: greedy, ALTA, DeLTa+DoLa, CoVe, ITI, SelfCheck
- Runs CURED router (v1 d2H-based; v2 trajectory-feature-based)
- Calibrates model-specific R², d2H, ECR thresholds (cached per model)
- Evaluates on TruthfulQA, MedHallu, StrategyQA, or custom CSV
- Saves compact summary and per-question logs with routing decisions

Example
-------
  python cured.py --model meta-llama/Llama-3.2-3B-Instruct --benchmark both --n 50 \\
      --protocols greedy,alta,cove,cured --skip-iti
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Phase 1: Constants and defaults
# ---------------------------------------------------------------------------

ALL_PROTOCOLS = ["greedy", "alta", "delta_dola", "cove", "iti", "selfcheck", "cured"]
API_PROTOCOLS = ["greedy", "cove", "cured_api"]
DEFAULT_LOCAL_PROTOCOLS = "greedy,alta,cove,cured"
DEFAULT_API_PROTOCOLS = "greedy,cove,cured_api"

DEFAULT_SCORER = "all-MiniLM-L6-v2"
DEFAULT_CACHE_ROOT = Path.home() / ".cured_cache"

DEFAULT_COSINE_THRESHOLD = 0.65

# ALTA defaults (from local experiments)
ALTA_R2_CUTOFF = 0.50
ALTA_EARLY_IDX = 7
ALTA_MID_IDX = 14
ALTA_TOP_K = 200
ALTA_ALPHA_C = 0.3
ALTA_ALPHA_E = 0.3

# ITI defaults
ITI_TOP_K_HEADS = 20
ITI_ALPHA = 0.5

# Calibration defaults
DEFAULT_R2_QUESTIONS = 15
DEFAULT_D2H_QUESTIONS = 50

SYSTEM_PROMPT = "You are a helpful, honest assistant. Answer questions accurately and concisely."


# API pacing state (used to reduce provider rate-limit spikes).
_LAST_API_CALL_TS = 0.0
_CLOUDFLARE_NEXT_INDEX = 0


MEDICAL_KEYWORDS = {
    "disease",
    "drug",
    "symptom",
    "treatment",
    "diagnosis",
    "patient",
    "clinical",
    "therapy",
    "cancer",
    "hospital",
    "doctor",
    "surgery",
    "medicine",
    "health",
    "infection",
    "tumor",
    "cardiac",
    "pathology",
    "prognosis",
    "dosage",
    "pharmaceutical",
    "neurological",
    "pulmonary",
    "gene",
    "protein",
    "cell",
    "tissue",
    "dna",
    "rna",
    "mutation",
    "antibody",
    "receptor",
    "enzyme",
    "inhibitor",
    "peptide",
    "serum",
    "glucose",
    "insulin",
    "cytokine",
    "lymphocyte",
    "inflammatory",
    "biomarker",
    "vagus",
    "steatohepatitis",
    "keratoprosthesis",
    "interictal",
    "epileptic",
    "esophagus",
    "pouchitis",
    "microbleed",
    "hepatic",
    "renal",
    "gastric",
    "colonic",
    "ophthalmol",
    "orthoped",
    "psychiatr",
    "barrett",
    "autoimmune",
    "lobar",
    "antibiotic",
    "refractory",
    "placebo",
    "trial",
    "cohort",
    "randomized",
    "randomised",
    "controlled",
    "intervention",
    "outcome",
    "participants",
    "biopsy",
    "prevalence",
    "incidence",
    "mortality",
    "morbidity",
    "smoking",
    "cessation",
    "obesity",
    "vaccination",
    "immunization",
    "pathogen",
    "virus",
    "bacterial",
    "anemia",
    "diabetes",
    "hypertension",
}

MED_PATTERN = re.compile(
    r"\b(odds ratio|hazard ratio|confidence interval|"
    r"p\s*[<=]\s*0\.\d+|significantly\s+(?:associated|reduced|increased|"
    r"improved|higher|lower)|risk of|efficacy of|association between|"
    r"randomized controlled|double.blind|meta.analys|systematic review|"
    r"mg/(?:dl|kg|ml)|mmhg|clinical trial|adverse event|side effect|"
    r"pharmacokinetic|serum level|biomarker)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Phase 1: Utilities
# ---------------------------------------------------------------------------


def p(msg: str) -> None:
    print(msg, flush=True)


class APIFatalError(RuntimeError):
    """Non-recoverable API error that should stop the current benchmark run."""


def safe_model_name(model_name: str) -> str:
    return (
        model_name.replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def get_model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def detect_domain(question: str) -> str:
    q = question.lower()
    if any(k in q for k in MEDICAL_KEYWORDS):
        return "medical"
    if MED_PATTERN.search(question):
        return "medical"
    return "general"


def format_prompt(tokenizer: Any, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"Question: {question}\nAnswer:"


def entropy(logits: np.ndarray) -> float:
    shifted = logits - np.max(logits)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs)
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0))))


def apply_repetition_penalty(logits: np.ndarray, generated_ids: list[int], penalty: float = 1.3) -> np.ndarray:
    out = logits.copy()
    for token_id in set(generated_ids):
        if out[token_id] > 0:
            out[token_id] = out[token_id] / penalty
        else:
            out[token_id] = out[token_id] * penalty
    return out


def has_repetition(text: str, window: int = 5) -> bool:
    words = text.lower().split()
    if len(words) < max(window * 2, 20):
        return False
    ngrams = [tuple(words[i : i + window]) for i in range(len(words) - window + 1)]
    return len(ngrams) != len(set(ngrams))


def scorer_device_str(scorer: SentenceTransformer) -> str:
    d = getattr(scorer, "device", None)
    if d is not None:
        return str(d)
    td = getattr(scorer, "_target_device", None)
    if td is not None:
        return str(td)
    return "cpu"


def cosine_match(scorer: SentenceTransformer, generated: str, reference: str, threshold: float) -> bool:
    if not generated.strip() or not reference.strip():
        return False
    dev = scorer_device_str(scorer)
    eg = scorer.encode(generated, convert_to_tensor=True, device=dev)
    er = scorer.encode(reference, convert_to_tensor=True, device=dev)
    return float(util.cos_sim(eg, er).item()) >= threshold


def _normalize_for_match(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _extract_binary_label(text: str) -> str | None:
    t = _normalize_for_match(text)
    if not t:
        return None
    m = re.search(r"\b(yes|no|maybe|true|false|affirmative|negative|uncertain)\b", t)
    if not m:
        return None
    token = m.group(1)
    if token in {"yes", "true", "affirmative"}:
        return "yes"
    if token in {"no", "false", "negative"}:
        return "no"
    if token in {"maybe", "uncertain"}:
        return "maybe"
    return None


def letter_match(generated: str, reference: str) -> bool:
    ref = (reference or "").strip().lower()
    if len(ref) != 1 or ref not in {"a", "b", "c", "d", "e"}:
        return False
    pred = _extract_option_letter(generated)
    return pred == ref


def yesno_match(generated: str, reference: str) -> bool:
    ref = (reference or "").strip().lower()
    if ref not in {"yes", "no", "maybe"}:
        return False
    pred = _extract_binary_label(generated)
    return pred == ref


def _to_binary_labels(labels: list[Any]) -> np.ndarray:
    out: list[float] = []
    for label in labels:
        if isinstance(label, (bool, np.bool_)):
            out.append(1.0 if bool(label) else 0.0)
            continue

        try:
            out.append(1.0 if int(label) == 1 else 0.0)
            continue
        except Exception:
            pass

        s = str(label).strip().lower()
        out.append(1.0 if s in {"1", "true", "yes"} else 0.0)

    return np.asarray(out, dtype=np.float32)




def _average_choice_log_prob_alta(
    model: Any,
    tokenizer: Any,
    question: str,
    choice: str,
    early_idx: int = ALTA_EARLY_IDX,
    mid_idx: int = ALTA_MID_IDX,
    top_k: int = ALTA_TOP_K,
    alpha_c: float = ALTA_ALPHA_C,
    alpha_e: float = ALTA_ALPHA_E,
) -> float:
    """Like _average_choice_log_prob but applies ALTA logit correction at each answer token."""
    dev = get_model_device(model)
    prompt_text = format_prompt(tokenizer, question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": str(choice)},
    ]
    try:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        full_text = f"Question: {question}\nAnswer: {choice}"
    try:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    except Exception:
        prompt_ids = tokenizer.encode(prompt_text)
        full_ids = tokenizer.encode(full_text)

    if len(full_ids) <= len(prompt_ids):
        try:
            answer_ids = tokenizer.encode(" " + str(choice).strip(), add_special_tokens=False)
        except Exception:
            answer_ids = tokenizer.encode(" " + str(choice).strip())
        full_ids = list(prompt_ids) + list(answer_ids)

    prompt_len = len(prompt_ids)
    if prompt_len <= 0 or len(full_ids) <= prompt_len:
        return -999.0

    input_ids = torch.tensor([full_ids], device=dev)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    hidden_states = getattr(out, "hidden_states", None)
    if hidden_states is None or len(hidden_states) < 2:
        # Fallback: no hidden states available, use baseline scorer
        return _average_choice_log_prob(model, tokenizer, question, choice)

    hidden_states = hidden_states[1:]  # skip embedding layer
    norm, lm_head = get_final_norm_and_lm_head(model)

    total_lp = 0.0
    n_tokens = 0
    for offset in range(prompt_len, len(full_ids) - 1):
        layer_logits_list = []
        for h in hidden_states:
            hs = h[:, offset, :]
            logits = lm_head(norm(hs)).squeeze(0).detach().to(torch.float32).cpu().numpy()
            layer_logits_list.append(logits)
        layer_logits_np = np.asarray(layer_logits_list, dtype=np.float32)

        corrected, _, _ = alta_logits(
            layer_logits_np,
            early_idx=early_idx,
            mid_idx=mid_idx,
            top_k=top_k,
            alpha_contrast=alpha_c,
            alpha_extrap=alpha_e,
        )
        shifted = corrected - np.max(corrected)
        log_probs = shifted - np.log(np.sum(np.exp(shifted)))
        tok_id = int(full_ids[offset + 1])
        total_lp += float(log_probs[tok_id])
        n_tokens += 1

    return total_lp / max(n_tokens, 1)

def _average_choice_log_prob(model: Any, tokenizer: Any, question: str, choice: str) -> float:
    dev = get_model_device(model)
    prompt_text = format_prompt(tokenizer, question)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": str(choice)},
    ]
    try:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        full_text = f"Question: {question}\nAnswer: {choice}"

    try:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    except Exception:
        prompt_ids = tokenizer.encode(prompt_text)
        full_ids = tokenizer.encode(full_text)

    if len(full_ids) <= len(prompt_ids):
        try:
            answer_ids = tokenizer.encode(" " + str(choice).strip(), add_special_tokens=False)
        except Exception:
            answer_ids = tokenizer.encode(" " + str(choice).strip())
        full_ids = list(prompt_ids) + list(answer_ids)

    prompt_len = len(prompt_ids)
    if prompt_len <= 0 or len(full_ids) <= prompt_len:
        return -999.0

    input_ids = torch.tensor([full_ids], device=dev)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)

    shift_logits = out.logits[0, prompt_len - 1 : -1, :]
    shift_labels = input_ids[0, prompt_len:]

    if shift_logits.shape[0] == 0 or shift_labels.numel() == 0:
        return -999.0

    if shift_logits.shape[0] != shift_labels.shape[0]:
        n = min(int(shift_logits.shape[0]), int(shift_labels.shape[0]))
        if n <= 0:
            return -999.0
        shift_logits = shift_logits[:n, :]
        shift_labels = shift_labels[:n]

    lp = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_lps = lp.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
    return float(token_lps.mean().detach().cpu())


def mc_score_sample(
    model: Any,
    tokenizer: Any,
    question: str,
    choices: list[str],
    labels: list[int],
    choices_mc2: list[str] | None = None,
    labels_mc2: list[int] | None = None,
    mc_protocol: str = "greedy",
) -> dict[str, float]:
    """TruthfulQA MC1/MC2 scoring using candidate log-likelihoods.

    Computes MC1 (single-best answer log-prob) and MC2 (normalized probability
    mass over all correct answers) for a multiple-choice question.

    MC1: 1.0 if the highest-log-prob choice is correct, else 0.0.
    MC2: correct_prob / (correct_prob + wrong_prob), where probabilities are
         computed from softmax-normalised log-probs.

    Args:
        model:        HuggingFace CausalLM (already loaded, on GPU).
        tokenizer:    Matching tokenizer.
        question:     Question text used as the prompt prefix.
        choices:      List of answer strings for MC1.
        labels:       Binary label list (1=correct, 0=incorrect) for MC1.
        choices_mc2:  Answer strings for MC2 (falls back to choices if None).
        labels_mc2:   Binary labels for MC2 (falls back to labels if None).
        mc_protocol:  "greedy" → standard log-probs;
                      "alta"   → ALTA-weighted log-probs.

    Returns:
        dict:
            "mc1" (float): 1.0 if highest-prob choice is correct, else 0.0.
            "mc2" (float): Fraction of probability mass on correct answers.
    """
    if not choices or not labels:
        return {"mc1": 0.0, "mc2": 0.0}

    _mc_score_fn = _average_choice_log_prob_alta if mc_protocol == "alta" else _average_choice_log_prob
    mc1_log_probs = np.asarray(
        [_mc_score_fn(model, tokenizer, question, c) for c in choices],
        dtype=np.float32,
    )
    mc1_labels = _to_binary_labels(labels)

    if mc1_log_probs.shape[0] != mc1_labels.shape[0]:
        n = min(int(mc1_log_probs.shape[0]), int(mc1_labels.shape[0]))
        mc1_log_probs = mc1_log_probs[:n]
        mc1_labels = mc1_labels[:n]

    if mc1_log_probs.size == 0 or mc1_labels.size == 0:
        return {"mc1": 0.0, "mc2": 0.0}

    best_idx = int(np.argmax(mc1_log_probs))
    mc1 = float(mc1_labels[best_idx] == 1.0)

    use_mc2_choices = choices_mc2 if choices_mc2 else choices
    use_mc2_labels = labels_mc2 if labels_mc2 else labels
    mc2_log_probs = np.asarray(
        [_mc_score_fn(model, tokenizer, question, c) for c in use_mc2_choices],
        dtype=np.float32,
    )
    mc2_labels = _to_binary_labels(use_mc2_labels)

    if mc2_log_probs.shape[0] != mc2_labels.shape[0]:
        n = min(int(mc2_log_probs.shape[0]), int(mc2_labels.shape[0]))
        mc2_log_probs = mc2_log_probs[:n]
        mc2_labels = mc2_labels[:n]

    if mc2_log_probs.size == 0 or mc2_labels.size == 0:
        return {"mc1": mc1, "mc2": 0.0}

    probs = np.exp(mc2_log_probs - np.max(mc2_log_probs))
    denom = float(np.sum(probs))
    if denom <= 0.0:
        return {"mc1": mc1, "mc2": 0.0}
    probs = probs / denom

    correct_prob = float(np.sum(probs * mc2_labels))
    wrong_prob = float(np.sum(probs * (1.0 - mc2_labels)))
    total_prob = correct_prob + wrong_prob
    mc2 = correct_prob / total_prob if total_prob > 0 else 0.0

    return {"mc1": mc1, "mc2": float(mc2)}


def _extract_option_letter(text: str) -> str | None:
    if not (text or "").strip():
        return None

    # Common short-form answers: "A", "(B)", "Answer: C", "Option D".
    m = re.search(
        r"^\s*(?:the\s+)?(?:correct\s+)?(?:answer|option)?\s*[:\-]?\s*\(?([A-E])\)?\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).lower()

    m = re.search(r"\(([A-E])\)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    return None


def _extract_question_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}

    # Multiline form: "A. ..." / "B) ..."
    for m in re.finditer(
        r"(?:^|\n)\s*([A-E])[\)\.:]\s*(.+?)(?=(?:\n\s*[A-E][\)\.:])|\Z)",
        question,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        label = m.group(1).lower()
        text = re.sub(r"\s+", " ", m.group(2)).strip()
        if text:
            options[label] = text

    # Inline form: "(A) ... (B) ..."
    if not options:
        for m in re.finditer(
            r"\(([A-E])\)\s*([^\(\)]+?)(?=\s*\([A-E]\)|$)",
            question,
            flags=re.IGNORECASE,
        ):
            label = m.group(1).lower()
            text = re.sub(r"\s+", " ", m.group(2)).strip()
            if text:
                options[label] = text

    return options


def _expected_option_letter(reference: str, question_options: dict[str, str]) -> str | None:
    ref_norm = _normalize_for_match(reference)
    if not ref_norm:
        return None

    # Exact text match first.
    for label, opt_text in question_options.items():
        if _normalize_for_match(opt_text) == ref_norm:
            return label

    # Fuzzy fallback for minor punctuation/wording differences.
    ref_tokens = set(ref_norm.split())
    if not ref_tokens:
        return None

    best_label = None
    best_overlap = 0.0
    for label, opt_text in question_options.items():
        opt_norm = _normalize_for_match(opt_text)
        opt_tokens = set(opt_norm.split())
        if not opt_tokens:
            continue
        overlap = len(ref_tokens & opt_tokens) / max(len(ref_tokens), 1)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label

    if best_overlap >= 0.8:
        return best_label
    return None


def custom_reference_match(
    scorer: SentenceTransformer,
    question: str,
    generated: str,
    reference: str,
    threshold: float,
) -> bool:
    if not generated.strip() or not reference.strip():
        return False

    # Fix 1: yes/no tasks should be graded by polarity, not embedding similarity.
    expected_binary = _extract_binary_label(reference)
    if expected_binary is not None:
        predicted_binary = _extract_binary_label(generated)
        if predicted_binary is not None:
            return predicted_binary == expected_binary

    # Fix 2: multiple-choice tasks should support letter-based outputs.
    question_options = _extract_question_options(question)
    if question_options:
        expected_letter = _expected_option_letter(reference, question_options)
        predicted_letter = _extract_option_letter(generated)
        if expected_letter is not None and predicted_letter is not None:
            return predicted_letter == expected_letter

    # Fast lexical checks before semantic similarity.
    gen_norm = _normalize_for_match(generated)
    ref_norm = _normalize_for_match(reference)
    if gen_norm and ref_norm:
        if f" {ref_norm} " in f" {gen_norm} ":
            return True
        if len(gen_norm.split()) >= 4 and f" {gen_norm} " in f" {ref_norm} ":
            return True

    return cosine_match(scorer, generated, reference, threshold=threshold)


def reference_match(
    scorer: SentenceTransformer,
    sample: dict[str, Any],
    generated: str,
    reference: str,
    threshold: float,
    scoring: str = "cosine",
) -> bool:
    if scoring == "letter":
        return letter_match(generated, reference)
    if scoring == "yesno":
        return yesno_match(generated, reference)

    if str(sample.get("dataset", "")) == "custom_csv":
        return custom_reference_match(
            scorer=scorer,
            question=str(sample.get("question", "")),
            generated=generated,
            reference=reference,
            threshold=threshold,
        )
    return cosine_match(scorer, generated, reference, threshold=threshold)


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _split_multi_items(raw: str) -> list[str]:
    normalized = raw.replace(";", ",").replace("\n", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def _parse_cloudflare_credential_pair(raw: str) -> tuple[str, str] | None:
    s = raw.strip()
    if not s:
        return None

    for sep in ("@", "|", ":"):
        if sep in s:
            token, account = s.split(sep, 1)
            token = token.strip()
            account = account.strip()
            if token and account:
                return token, account

    parts = s.split()
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[0].strip(), parts[1].strip()

    return None


def _collect_cloudflare_credentials() -> list[tuple[str, str]]:
    creds: list[tuple[str, str]] = []

    primary_token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip() or os.environ.get("CF_API_TOKEN", "").strip()
    primary_account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip() or os.environ.get("CF_ACCOUNT_ID", "").strip()
    if primary_token and primary_account:
        creds.append((primary_token, primary_account))

    pair_sources = (
        os.environ.get("CLOUDFLARE_API_CREDENTIALS", "").strip(),
        os.environ.get("CF_API_CREDENTIALS", "").strip(),
    )
    for src in pair_sources:
        if not src:
            continue
        for item in _split_multi_items(src):
            parsed = _parse_cloudflare_credential_pair(item)
            if parsed is not None:
                creds.append(parsed)

    token_list_raw = (
        os.environ.get("CLOUDFLARE_API_TOKENS", "").strip()
        or os.environ.get("CF_API_TOKENS", "").strip()
    )
    account_list_raw = (
        os.environ.get("CLOUDFLARE_ACCOUNT_IDS", "").strip()
        or os.environ.get("CF_ACCOUNT_IDS", "").strip()
    )
    if token_list_raw and account_list_raw:
        tokens = _dedupe_strings(_split_multi_items(token_list_raw))
        accounts = _dedupe_strings(_split_multi_items(account_list_raw))
        n_pairs = min(len(tokens), len(accounts))
        if len(tokens) != len(accounts):
            p(
                "  Warning: Cloudflare token/account list length mismatch; "
                f"using first {n_pairs} pairs"
            )
        for idx in range(n_pairs):
            creds.append((tokens[idx], accounts[idx]))

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in creds:
        if pair in seen:
            continue
        deduped.append(pair)
        seen.add(pair)

    return deduped


def _http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_sec: int = 120) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"HTTP {exc.code} calling {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error calling {url}: {exc}") from exc


def _http_post_json_retry(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_sec: int = 120,
    max_retries: int = 5,
    backoff_base_sec: float = 2.0,
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)

            if exc.code in retry_statuses and attempt < max_retries:
                retry_after = None
                try:
                    retry_after_raw = exc.headers.get("Retry-After")
                    if retry_after_raw is not None:
                        retry_after = float(retry_after_raw)
                except Exception:
                    retry_after = None

                wait_s = float(backoff_base_sec) * (2 ** attempt)
                if retry_after is not None and retry_after > 0:
                    wait_s = max(wait_s, retry_after)
                p(
                    f"  HTTP {exc.code} from API (attempt {attempt+1}/{max_retries+1}); "
                    f"retrying in {wait_s:.1f}s"
                )
                time.sleep(wait_s)
                continue

            raise RuntimeError(f"HTTP {exc.code} calling {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt < max_retries:
                wait_s = float(backoff_base_sec) * (2 ** attempt)
                p(
                    f"  Network error from API (attempt {attempt+1}/{max_retries+1}: {exc}); "
                    f"retrying in {wait_s:.1f}s"
                )
                time.sleep(wait_s)
                continue
            raise RuntimeError(f"Network error calling {url}: {exc}") from exc

    raise RuntimeError("Unexpected retry loop termination in _http_post_json_retry")


def _apply_api_rate_limit(api_mode: str) -> None:
    global _LAST_API_CALL_TS

    default_interval = "1.0" if api_mode == "cloudflare" else "0"
    raw = os.environ.get("API_MIN_INTERVAL_SEC", default_interval)
    try:
        min_interval = max(0.0, float(raw))
    except Exception:
        min_interval = 0.0

    if min_interval <= 0.0:
        return

    now = time.time()
    elapsed = now - _LAST_API_CALL_TS
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _LAST_API_CALL_TS = time.time()


def groq_generate(
    api_key: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_new_tokens),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    out = _http_post_json(url=url, payload=payload, headers=headers)
    try:
        text = out["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected Groq response shape: {out}") from exc
    return str(text).strip()


def gemini_generate(
    api_key: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    safe_model = urllib.parse.quote(model_name, safe="")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{safe_model}:generateContent"
        f"?key={urllib.parse.quote(api_key, safe='')}"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{prompt}"}],
            }
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_new_tokens),
        },
    }
    headers = {"Content-Type": "application/json"}
    out = _http_post_json(url=url, payload=payload, headers=headers)
    try:
        parts = out["candidates"][0]["content"]["parts"]
        text = "".join(str(part.get("text", "")) for part in parts)
    except Exception as exc:
        raise RuntimeError(f"Unexpected Gemini response shape: {out}") from exc
    return text.strip()


def openrouter_generate(
    api_key: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_new_tokens),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Optional but recommended by OpenRouter; keeps provider routing happier.
        "HTTP-Referer": "https://local-benchmark",
        "X-Title": "LLM_Hallucination_Benchmark",
    }
    out = _http_post_json(url=url, payload=payload, headers=headers)
    try:
        content = out["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
            if parts:
                return "".join(parts).strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {out}") from exc
    raise RuntimeError(f"Unexpected OpenRouter response shape: {out}")


def _collect_openrouter_keys() -> list[str]:
    keys: list[str] = []

    single_primary = os.environ.get("OPENROUTER_API_KEY", "").strip()
    single_alt = os.environ.get("OPEN_ROUTER_API_KEY", "").strip()
    multi_primary = os.environ.get("OPENROUTER_API_KEYS", "").strip()
    multi_alt = os.environ.get("OPEN_ROUTER_API_KEYS", "").strip()

    for raw in (single_primary, single_alt):
        if raw:
            keys.append(raw)

    multi_raw = multi_primary or multi_alt
    if multi_raw:
        normalized = multi_raw.replace(";", ",").replace("\n", ",")
        for part in normalized.split(","):
            token = part.strip()
            if token:
                keys.append(token)

    return _dedupe_strings(keys)


def foundry_generate(
    api_key: str,
    base_url: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    base = base_url.rstrip("/")
    url = f"{base}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_new_tokens),
    }
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    out = _http_post_json(url=url, payload=payload, headers=headers)
    try:
        content = out["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
            if parts:
                return "".join(parts).strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected Foundry response shape: {out}") from exc
    raise RuntimeError(f"Unexpected Foundry response shape: {out}")


def cloudflare_generate(
    api_token: str,
    account_id: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    safe_account = urllib.parse.quote(account_id, safe="")
    safe_model = urllib.parse.quote(model_name, safe="@/-._")
    url = f"https://api.cloudflare.com/client/v4/accounts/{safe_account}/ai/run/{safe_model}"

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Connection": "close",
    }
    max_retries = int(os.environ.get("CLOUDFLARE_API_MAX_RETRIES", "6"))
    out: dict[str, Any] | None = None
    for attempt in range(max_retries + 1):
        out = _http_post_json_retry(
            url=url,
            payload=payload,
            headers=headers,
            timeout_sec=30,
            max_retries=max_retries,
            backoff_base_sec=2.0,
        )

        if out.get("success", True):
            break

        errors = out.get("errors")
        errors_text = json.dumps(errors, ensure_ascii=False).lower() if errors is not None else ""
        transient_markers = (
            "temporarily unavailable",
            "temporarily",
            "overloaded",
            "rate limit",
            "too many requests",
        )
        is_transient = any(marker in errors_text for marker in transient_markers)

        if is_transient and attempt < max_retries:
            wait_s = 2.0 * (2 ** attempt)
            p(
                "  Cloudflare model unavailable/rate-limited "
                f"(attempt {attempt+1}/{max_retries+1}); retrying in {wait_s:.1f}s"
            )
            time.sleep(wait_s)
            continue

        if isinstance(errors, list):
            for err in errors:
                if not isinstance(err, dict):
                    continue
                code = err.get("code")
                msg = str(err.get("message", ""))
                if code in (4006, 10000):
                    # 4006 = daily free-neuron quota exhausted, 10000 = auth error.
                    raise APIFatalError(f"Cloudflare fatal API error {code}: {msg}")

        raise RuntimeError(f"Cloudflare API error: {out}")

    if out is None:
        raise RuntimeError("Cloudflare API returned no response")

    result = out.get("result", {})
    if isinstance(result, dict):
        if isinstance(result.get("response"), str):
            return result["response"].strip()
        if isinstance(result.get("output_text"), str):
            return result["output_text"].strip()
        if isinstance(result.get("text"), str):
            return result["text"].strip()

        content = result.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "".join(parts).strip()

        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"].strip()

    raise RuntimeError(f"Unexpected Cloudflare response shape: {out}")


def api_generate(
    api_mode: str,
    api_model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    _apply_api_rate_limit(api_mode)

    if api_mode == "groq":
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set in environment.")
        return groq_generate(key, api_model, prompt, max_new_tokens, temperature)

    if api_mode == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set in environment.")
        return gemini_generate(key, api_model, prompt, max_new_tokens, temperature)

    if api_mode == "openrouter":
        keys = _collect_openrouter_keys()
        if not keys:
            raise RuntimeError(
                "OPENROUTER_API_KEY / OPEN_ROUTER_API_KEY is not set. "
                "You can also set OPENROUTER_API_KEYS as a comma-separated list for backup keys."
            )

        last_exc: Exception | None = None
        total_keys = len(keys)
        for idx, key in enumerate(keys, start=1):
            try:
                return openrouter_generate(key, api_model, prompt, max_new_tokens, temperature)
            except Exception as exc:
                last_exc = exc
                if idx < total_keys:
                    p(f"  OpenRouter key {idx}/{total_keys} failed; trying backup key")
                    continue

                # Optional failover: when OpenRouter quota is exhausted, try Foundry/Azure.
                msg = str(exc).lower()
                quota_markers = (
                    "insufficient",
                    "quota",
                    "credit",
                    "payment",
                    "rate limit",
                    "429",
                )
                should_failover = any(marker in msg for marker in quota_markers)
                foundry_key = os.environ.get("FOUNDRY_API_KEY", "").strip() or os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
                foundry_base = os.environ.get("FOUNDRY_BASE_URL", "").strip() or os.environ.get("AZURE_OPENAI_BASE_URL", "").strip()
                foundry_model = os.environ.get("FOUNDRY_MODEL", "").strip() or os.environ.get("AZURE_OPENAI_MODEL", "").strip()
                if should_failover and foundry_key and foundry_base and foundry_model:
                    p("  OpenRouter failed; attempting Foundry fallback")
                    return foundry_generate(
                        api_key=foundry_key,
                        base_url=foundry_base,
                        model_name=foundry_model,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("OpenRouter generation failed without an explicit error.")

    if api_mode == "foundry":
        key = os.environ.get("FOUNDRY_API_KEY", "").strip() or os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        base_url = os.environ.get("FOUNDRY_BASE_URL", "").strip() or os.environ.get("AZURE_OPENAI_BASE_URL", "").strip()
        if not key:
            raise RuntimeError("FOUNDRY_API_KEY (or AZURE_OPENAI_API_KEY) is not set in environment.")
        if not base_url:
            raise RuntimeError("FOUNDRY_BASE_URL (or AZURE_OPENAI_BASE_URL) is not set in environment.")

        model_name = api_model.strip() or os.environ.get("FOUNDRY_MODEL", "").strip() or os.environ.get("AZURE_OPENAI_MODEL", "").strip()
        if not model_name:
            raise RuntimeError("Foundry model is missing. Set --api-model or FOUNDRY_MODEL/AZURE_OPENAI_MODEL.")

        return foundry_generate(
            api_key=key,
            base_url=base_url,
            model_name=model_name,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    if api_mode == "cloudflare":
        creds = _collect_cloudflare_credentials()
        if not creds:
            raise RuntimeError(
                "Cloudflare credentials are missing. Set one of: "
                "CLOUDFLARE_API_TOKEN+CLOUDFLARE_ACCOUNT_ID, "
                "CLOUDFLARE_API_CREDENTIALS, or "
                "CLOUDFLARE_API_TOKENS+CLOUDFLARE_ACCOUNT_IDS."
            )

        global _CLOUDFLARE_NEXT_INDEX
        start_idx = _CLOUDFLARE_NEXT_INDEX % len(creds)
        last_exc: Exception | None = None

        for attempt in range(len(creds)):
            idx = (start_idx + attempt) % len(creds)
            token, account_id = creds[idx]
            try:
                text = cloudflare_generate(token, account_id, api_model, prompt, max_new_tokens, temperature)
                _CLOUDFLARE_NEXT_INDEX = (idx + 1) % len(creds)
                return text
            except Exception as exc:
                last_exc = exc
                if attempt < len(creds) - 1:
                    p(
                        f"  Cloudflare credential {idx+1}/{len(creds)} failed "
                        f"({type(exc).__name__}); trying backup credential"
                    )

        fallback_model = os.environ.get("CLOUDFLARE_OPENROUTER_FALLBACK_MODEL", "").strip()
        if fallback_model:
            keys = _collect_openrouter_keys()
            if keys:
                p("  Cloudflare credentials exhausted; attempting OpenRouter fallback")
                for i, key in enumerate(keys, start=1):
                    try:
                        return openrouter_generate(key, fallback_model, prompt, max_new_tokens, temperature)
                    except Exception as exc:
                        last_exc = exc
                        if i < len(keys):
                            p(f"  OpenRouter fallback key {i}/{len(keys)} failed; trying next key")

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Cloudflare generation failed without an explicit error.")

    raise ValueError(f"Unsupported api_mode: {api_mode}")


# ---------------------------------------------------------------------------
# Phase 1: Model/architecture helpers
# ---------------------------------------------------------------------------


def configure_torch_backends_for_inference() -> None:
    """Enable TF32 / matmul settings that speed Ampere+ GPUs without changing numerics much."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def resolve_scorer_device(
    scorer_device_arg: str, load_in_4bit: bool, model_params_b: float
) -> str:
    """
    Place MiniLM scorer on GPU when VRAM is likely safe; keep CPU for 4-bit or large LMs.
    Override with env CURED_SCORER_DEVICE=cpu|cuda or --scorer-device.
    """
    env = os.environ.get("CURED_SCORER_DEVICE", "").strip().lower()
    if env in ("cpu", "cuda"):
        scorer_device_arg = env
    if scorer_device_arg == "cpu":
        return "cpu"
    if scorer_device_arg == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    if load_in_4bit:
        return "cpu"
    if model_params_b >= 12.0:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_name: str, load_in_4bit: bool) -> tuple[Any, Any]:
    p(f"\nLoading model: {model_name}")
    kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            p("  Quantization: 4-bit (bitsandbytes)")
        except Exception as exc:
            p(f"  4-bit setup failed ({type(exc).__name__}: {exc}); falling back to standard dtype")
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_sdpa = os.environ.get("CURED_DISABLE_SDPA", "").strip() != "1"
    model = None
    if use_sdpa:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation="sdpa", **kwargs
            )
        except Exception as exc:
            p(f"  attn_implementation=sdpa failed ({type(exc).__name__}: {exc}); using default attention")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    dev = get_model_device(model)
    n_layers = getattr(model.config, "num_hidden_layers", "?")
    vocab = getattr(model.config, "vocab_size", "?")
    p(f"  Ready: device={dev} layers={n_layers} vocab={vocab}")
    return model, tokenizer


def get_transformer_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    return None


def get_final_norm_and_lm_head(model: Any) -> tuple[Any, Any]:
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Model has no lm_head; this script supports causal LMs with lm_head.")

    if hasattr(model, "model"):
        m = model.model
        norm = getattr(m, "norm", None)
        if norm is None:
            norm = getattr(m, "final_layernorm", None)
        if norm is None:
            norm = getattr(m, "ln_f", None)
        if norm is None:
            norm = torch.nn.Identity()
        return norm, lm_head

    if hasattr(model, "transformer"):
        t = model.transformer
        norm = getattr(t, "ln_f", None)
        if norm is None:
            norm = torch.nn.Identity()
        return norm, lm_head

    return torch.nn.Identity(), lm_head


def get_arch(model: Any) -> dict[str, Any]:
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None)
    n_heads = getattr(cfg, "num_attention_heads", None)
    n_kv_heads = getattr(cfg, "num_key_value_heads", None)
    hidden_size = getattr(cfg, "hidden_size", None)

    layers_obj = get_transformer_layers(model)
    if n_layers is None and layers_obj is not None:
        n_layers = len(layers_obj)
    if n_kv_heads is None:
        n_kv_heads = n_heads

    head_dim = None
    if hidden_size is not None and n_heads:
        head_dim = int(hidden_size) // int(n_heads)

    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
    }


def get_layer_logits_cached(model: Any, input_ids: torch.Tensor, past_key_values: Any = None) -> tuple[np.ndarray, Any]:
    """
    Returns per-layer next-token logits (skipping embedding layer):
      [n_layers, vocab_size], past_key_values
    """
    norm, lm_head = get_final_norm_and_lm_head(model)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None or len(hidden_states) < 2:
        raise RuntimeError("Model did not return hidden_states. Cannot run ALTA/DeLTa logic.")

    all_logits: list[np.ndarray] = []
    for h in hidden_states[1:]:
        last = h[:, -1, :]
        logits_t = lm_head(norm(last))
        logits_np = logits_t.squeeze(0).detach().to(torch.float32).cpu().numpy()
        all_logits.append(logits_np)

    return np.asarray(all_logits, dtype=np.float32), getattr(outputs, "past_key_values", None)


def get_final_logits_cached(
    model: Any, input_ids: torch.Tensor, past_key_values: Any = None
) -> tuple[np.ndarray, Any]:
    """
    Next-token logits from the final LM head only (no per-layer hidden_states).
    Matches layer_logits[-1] from get_layer_logits_cached but avoids O(n_layers) extra work per step.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            output_hidden_states=False,
            use_cache=True,
        )
    logits_t = outputs.logits[:, -1, :]
    logits_np = logits_t.squeeze(0).detach().to(torch.float32).cpu().numpy()
    return logits_np, getattr(outputs, "past_key_values", None)


def compute_d2h_features(model: Any, tokenizer: Any, prompt: str) -> dict[str, float]:
    dev = get_model_device(model)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)
    layer_logits, _ = get_layer_logits_cached(model, input_ids, None)

    n = len(layer_logits)
    i1 = max(0, n // 4)
    i2 = max(0, n // 2)
    i3 = max(0, 3 * n // 4)
    i4 = n - 1

    h1 = entropy(layer_logits[i1])
    h2 = entropy(layer_logits[i2])
    h3 = entropy(layer_logits[i3])
    h4 = entropy(layer_logits[i4])

    return {
        "H_final": float(h4),
        "dH": float(h4 - h1),
        "d2H": float(h4 - 2 * h2 + h1),
        "dH_late": float(h4 - h3),
    }


# ---------------------------------------------------------------------------
# Phase 2: Protocol implementations (greedy, ALTA, CoVe, ITI, SelfCheck, DoLa)
# ---------------------------------------------------------------------------


def greedy_generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 80) -> str:
    """Single-prompt greedy decoding via HF model.generate() (CUDA-optimised, no Python loop)."""
    dev = get_model_device(model)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


def _auto_batch_size(model_params_b: float, load_in_4bit: bool = False) -> int:
    """Return a safe generation batch size for greedy/cove protocols.

    Conservative: accounts for KV-cache growth during generation. 4-bit and
    large models stay at 1 to avoid VRAM spikes.
    """
    if not torch.cuda.is_available():
        return 1
    if load_in_4bit:
        return 1
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        vram_gb = 24.0
    # 40GB+ A800/A100 class: push batches harder.
    # KV-cache per seq is small relative to weights; safe headroom verified.
    if model_params_b >= 30.0:
        return 1
    if model_params_b >= 12.0:
        return 2 if vram_gb >= 35 else 1
    if model_params_b >= 7.0:
        return 4 if vram_gb >= 35 else 2
    return 8 if vram_gb >= 35 else 2


def batch_greedy_generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 80,
) -> list[str]:
    """Generate greedy outputs for B prompts in a single GPU call (left-padded batch).

    Produces the same text as calling greedy_generate() individually:
    identical do_sample=False argmax decoding with the same repetition penalty.
    """
    if not prompts:
        return []
    if len(prompts) == 1:
        return [greedy_generate(model, tokenizer, prompts[0], max_new_tokens)]

    dev = get_model_device(model)
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    enc = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(dev)
    attn_mask = enc["attention_mask"].to(dev)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    tokenizer.padding_side = orig_padding_side
    return [
        tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True)
        for i in range(out.shape[0])
    ]


def compute_delta_dola_logits(
    layer_logits: np.ndarray,
    alpha1: float = 0.3,
    alpha2: float = 0.3,
    early_layer_idx: int = ALTA_EARLY_IDX,
    mid_layer_idx: int = ALTA_MID_IDX,
    top_k: int = ALTA_TOP_K,
) -> np.ndarray:
    n_layers = int(layer_logits.shape[0])
    z_final = layer_logits[-1].astype(np.float32, copy=False)

    early_idx = min(max(int(early_layer_idx), 0), n_layers - 1)
    z_early = layer_logits[early_idx].astype(np.float32, copy=False)
    z_dola = z_final - z_early

    reg_start = min(max(int(mid_layer_idx), 0), max(n_layers - 2, 0))
    reg_layers = np.arange(reg_start, n_layers, dtype=np.int32)

    z_delta = z_final.copy()
    if len(reg_layers) >= 2:
        k = min(max(int(top_k), 1), z_final.shape[0])
        top_idx = np.argpartition(z_final, -k)[-k:]

        y = layer_logits[reg_layers][:, top_idx].astype(np.float32, copy=False)
        x = np.arange(len(reg_layers), dtype=np.float32)
        x_n = (x - float(x.mean())) / max(float(x.std()), 1e-8)
        denom = float(np.sum(x_n**2))

        if denom > 1e-8:
            y_m = y.mean(axis=0)
            b1 = np.sum(x_n[:, None] * (y - y_m), axis=0) / denom
            b0 = y_m - b1 * float(x_n.mean())
            x_virt = float(x_n[-1] + (x_n[1] - x_n[0]))
            z_delta[top_idx] = b0 + b1 * x_virt

    # Matches existing repo's DeLTa+DoLa blend formulation.
    return z_final + alpha1 * (z_delta - z_final) + alpha2 * (z_dola - z_final)


def delta_dola_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 80,
    alpha1: float = 0.3,
    alpha2: float = 0.3,
) -> str:
    dev = get_model_device(model)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)
    layer_logits, past_kv = get_layer_logits_cached(model, input_ids, None)

    generated: list[int] = []
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        hybrid = compute_delta_dola_logits(layer_logits, alpha1=alpha1, alpha2=alpha2)
        logits = apply_repetition_penalty(hybrid, generated)
        next_id = int(np.argmax(logits))
        generated.append(next_id)

        if eos_id is not None and next_id == eos_id:
            break
        step_ids = torch.tensor([[next_id]], device=dev)
        layer_logits, past_kv = get_layer_logits_cached(model, step_ids, past_kv)

    return tokenizer.decode(generated, skip_special_tokens=True)


def alta_logits(
    layer_logits: np.ndarray,
    early_idx: int = ALTA_EARLY_IDX,
    mid_idx: int = ALTA_MID_IDX,
    top_k: int = ALTA_TOP_K,
    alpha_contrast: float = ALTA_ALPHA_C,
    alpha_extrap: float = ALTA_ALPHA_E,
) -> tuple[np.ndarray, float, float]:
    n_layers = int(layer_logits.shape[0])
    z_final = layer_logits[-1].astype(np.float32, copy=False)

    h_final = float(entropy(z_final))
    gate = float(np.clip(h_final / 3.0, 0.0, 1.0))

    z_early = layer_logits[min(max(int(early_idx), 0), n_layers - 1)].astype(np.float32, copy=False)
    z_dola = z_final - z_early

    reg_start = min(max(int(mid_idx), 0), max(n_layers - 2, 0))
    reg_layers = np.arange(reg_start, n_layers, dtype=np.int32)
    z_delta = z_final.copy()

    if len(reg_layers) >= 2:
        k = min(max(int(top_k), 1), z_final.shape[0])
        top_idx = np.argpartition(z_final, -k)[-k:]
        y = layer_logits[reg_layers][:, top_idx].astype(np.float32, copy=False)
        x = np.arange(len(reg_layers), dtype=np.float32)
        x_n = (x - float(x.mean())) / max(float(x.std()), 1e-8)
        denom = float(np.sum(x_n**2))

        if denom > 1e-8:
            ym = y.mean(axis=0)
            b1 = np.sum(x_n[:, None] * (y - ym), axis=0) / denom
            b0 = ym - b1 * float(x_n.mean())
            x_virt = float(x_n[-1] + (x_n[1] - x_n[0]))
            z_delta[top_idx] = b0 + b1 * x_virt

    correction = alpha_contrast * z_dola + alpha_extrap * (z_delta - z_final)
    return z_final + gate * correction, gate, h_final


def alta_generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 80) -> dict[str, Any]:
    dev = get_model_device(model)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)
    layer_logits, past_kv = get_layer_logits_cached(model, input_ids, None)

    generated: list[int] = []
    gate_weights: list[float] = []
    first_entropy = None
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        corrected, gate, h = alta_logits(layer_logits)
        if first_entropy is None:
            first_entropy = h
        gate_weights.append(gate)

        logits = apply_repetition_penalty(corrected, generated)
        next_id = int(np.argmax(logits))
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break

        step_ids = torch.tensor([[next_id]], device=dev)
        layer_logits, past_kv = get_layer_logits_cached(model, step_ids, past_kv)

    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "mean_gate": float(np.mean(gate_weights)) if gate_weights else 0.0,
        "first_entropy": float(first_entropy) if first_entropy is not None else 0.0,
    }


def extract_questions(plan_text: str, max_q: int = 2) -> list[str]:
    lines: list[str] = []
    for raw in plan_text.splitlines():
        line = raw.strip().lstrip("-*").strip()
        while line and (line[0].isdigit() or line[0] in ".)"):
            line = line[1:].strip()
        if len(line) > 5 and "?" in line:
            lines.append(line)
        if len(lines) >= max_q:
            break
    return lines


def cove_generate(model: Any, tokenizer: Any, question: str, max_new_tokens: int = 80) -> str:
    draft = greedy_generate(model, tokenizer, format_prompt(tokenizer, question), max_new_tokens=60)

    plan_prompt = format_prompt(
        tokenizer,
        (
            f"I answered the question '{question}' with:\n"
            f"'{draft}'\n\n"
            "Write 2 short factual questions to fact-check this answer. "
            "Output only the questions, one per line, each ending with '?'."
        ),
    )
    plan_text = greedy_generate(model, tokenizer, plan_prompt, max_new_tokens=80)
    vqs = extract_questions(plan_text, max_q=2)

    if not vqs:
        return draft

    checks: list[str] = []
    for vq in vqs:
        ans = greedy_generate(model, tokenizer, format_prompt(tokenizer, vq), max_new_tokens=50)
        checks.append(f"Check: {vq}\nAnswer: {ans}")

    refine_prompt = format_prompt(
        tokenizer,
        (
            f"Original question: {question}\n"
            f"Initial answer: {draft}\n\n"
            f"Fact-checks:\n{chr(10).join(checks)}\n\n"
            "Based on these fact checks, write the accurate final answer:"
        ),
    )
    final = greedy_generate(model, tokenizer, refine_prompt, max_new_tokens=max_new_tokens)
    return final if final.strip() else draft


def api_cove_generate(
    api_mode: str,
    api_model: str,
    question: str,
    max_new_tokens: int = 80,
    api_temperature: float = 0.0,
) -> str:
    draft = api_generate(
        api_mode=api_mode,
        api_model=api_model,
        prompt=question,
        max_new_tokens=min(60, max_new_tokens),
        temperature=api_temperature,
    )

    plan_prompt = (
        f"I answered the question '{question}' with:\n"
        f"'{draft}'\n\n"
        "Write 2 short factual questions to fact-check this answer. "
        "Output only the questions, one per line, each ending with '?'."
    )
    try:
        plan_text = api_generate(
            api_mode=api_mode,
            api_model=api_model,
            prompt=plan_prompt,
            max_new_tokens=120,
            temperature=api_temperature,
        )
    except Exception:
        return draft

    vqs = extract_questions(plan_text, max_q=2)
    if not vqs:
        return draft

    checks: list[str] = []
    for vq in vqs:
        try:
            ans = api_generate(
                api_mode=api_mode,
                api_model=api_model,
                prompt=vq,
                max_new_tokens=60,
                temperature=api_temperature,
            )
        except Exception:
            continue
        checks.append(f"Check: {vq}\nAnswer: {ans}")

    if not checks:
        return draft

    refine_prompt = (
        f"Original question: {question}\n"
        f"Initial answer: {draft}\n\n"
        f"Fact-checks:\n{chr(10).join(checks)}\n\n"
        "Based on these fact checks, write the accurate final answer:"
    )
    try:
        final = api_generate(
            api_mode=api_mode,
            api_model=api_model,
            prompt=refine_prompt,
            max_new_tokens=max_new_tokens,
            temperature=api_temperature,
        )
    except Exception:
        return draft

    return final if final.strip() else draft


class CUREDAPIRouter:
    def __init__(self, api_mode: str, api_model: str, api_temperature: float = 0.0) -> None:
        self.api_mode = api_mode
        self.api_model = api_model
        self.api_temperature = float(api_temperature)

    def route(self, question: str, max_new_tokens: int = 80, scoring: str = "cosine") -> dict[str, Any]:
        domain = detect_domain(question)

        if scoring in ("letter", "yesno"):
            text = api_generate(
                api_mode=self.api_mode,
                api_model=self.api_model,
                prompt=question,
                max_new_tokens=max_new_tokens,
                temperature=self.api_temperature,
            )
            return {
                "text": text,
                "strategy": f"api_greedy_{domain}_structured",
                "domain": domain,
            }

        if domain == "general":
            text = api_generate(
                api_mode=self.api_mode,
                api_model=self.api_model,
                prompt=question,
                max_new_tokens=max_new_tokens,
                temperature=self.api_temperature,
            )
            return {
                "text": text,
                "strategy": "api_greedy_general",
                "domain": domain,
            }

        try:
            text = api_cove_generate(
                api_mode=self.api_mode,
                api_model=self.api_model,
                question=question,
                max_new_tokens=max_new_tokens,
                api_temperature=self.api_temperature,
            )
            return {
                "text": text,
                "strategy": "api_cove_medical",
                "domain": domain,
            }
        except Exception as exc:
            text = api_generate(
                api_mode=self.api_mode,
                api_model=self.api_model,
                prompt=question,
                max_new_tokens=max_new_tokens,
                temperature=self.api_temperature,
            )
            return {
                "text": text,
                "strategy": "api_greedy_medical_fallback",
                "domain": domain,
                "fallback_from": "api_cove_medical",
                "fallback_error": str(exc)[:300],
            }


def get_attn_module(layer: Any) -> Any:
    for name in ("self_attn", "attn", "self_attention", "attention"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def iti_generate(
    model: Any,
    tokenizer: Any,
    arch: dict[str, Any],
    top_heads: np.ndarray,
    head_vectors: np.ndarray,
    prompt: str,
    alpha: float = ITI_ALPHA,
    max_new_tokens: int = 80,
) -> str:
    layers = get_transformer_layers(model)
    if layers is None:
        p("  ITI fallback: could not locate transformer layers.")
        return greedy_generate(model, tokenizer, prompt, max_new_tokens)

    n_heads = arch.get("n_heads")
    head_dim = arch.get("head_dim")
    if not n_heads or not head_dim:
        p("  ITI fallback: missing n_heads/head_dim.")
        return greedy_generate(model, tokenizer, prompt, max_new_tokens)

    dev = get_model_device(model)
    first_input = tokenizer.encode(prompt, return_tensors="pt").to(dev)
    step_input = first_input
    past_kv = None
    generated: list[int] = []
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        hooks = []

        def make_hook(layer_idx: int):
            def hook_fn(_module: Any, _inp: Any, output: Any) -> Any:
                if isinstance(output, tuple):
                    attn_out = output[0]
                    tail = output[1:]
                else:
                    attn_out = output
                    tail = None

                if not torch.is_tensor(attn_out) or attn_out.ndim < 3:
                    return output

                for l, h in top_heads:
                    li = int(l)
                    hi = int(h)
                    if li != layer_idx or hi < 0 or hi >= int(n_heads):
                        continue
                    start = hi * int(head_dim)
                    end = start + int(head_dim)
                    if end > attn_out.shape[-1]:
                        continue
                    direction = torch.as_tensor(
                        head_vectors[li, hi], dtype=attn_out.dtype, device=attn_out.device
                    )
                    attn_out[:, -1, start:end] = attn_out[:, -1, start:end] + float(alpha) * direction

                if tail is None:
                    return attn_out
                return (attn_out,) + tail

            return hook_fn

        for idx, layer in enumerate(layers):
            attn = get_attn_module(layer)
            if attn is None:
                continue
            hooks.append(attn.register_forward_hook(make_hook(idx)))

        try:
            with torch.no_grad():
                outputs = model(input_ids=step_input, past_key_values=past_kv, use_cache=True)
        finally:
            for h in hooks:
                h.remove()

        logits = outputs.logits[0, -1, :].detach().to(torch.float32).cpu().numpy()
        past_kv = getattr(outputs, "past_key_values", None)

        logits = apply_repetition_penalty(logits, generated)
        next_id = int(np.argmax(logits))
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break

        step_input = torch.tensor([[next_id]], device=dev)

    return tokenizer.decode(generated, skip_special_tokens=True)


def selfcheck_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    scorer: SentenceTransformer,
    max_new_tokens: int = 80,
    k_samples: int = 4,
    temperature: float = 0.7,
) -> dict[str, Any]:
    draft = greedy_generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    dev = get_model_device(model)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)
    eos_id = tokenizer.eos_token_id

    samples: list[str] = []
    for _ in range(max(1, k_samples)):
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.3,
                pad_token_id=eos_id,
            )
        text = tokenizer.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True)
        if text.strip():
            samples.append(text)

    consistency = 0.0
    if draft.strip() and samples:
        embs = scorer.encode(
            [draft] + samples, convert_to_tensor=True, device=scorer_device_str(scorer)
        )
        sims = [float(util.cos_sim(embs[0], embs[i]).item()) for i in range(1, len(embs))]
        consistency = float(np.mean(sims)) if sims else 0.0

    return {"text": draft, "consistency": consistency, "n_samples": len(samples)}


# ---------------------------------------------------------------------------
# Phase 3: Calibration (measure_r2, compute_ecr, train_iti_probes, calibrate_d2h)
# ---------------------------------------------------------------------------


def measure_r2(model: Any, tokenizer: Any, n_questions: int = DEFAULT_R2_QUESTIONS) -> float:
    p(f"\nCalibrating late-layer R2 on TruthfulQA (n={n_questions})...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    norm, lm_head = get_final_norm_and_lm_head(model)
    dev = get_model_device(model)

    r2_values: list[float] = []
    n_use = min(int(n_questions), len(ds))
    for i, sample in enumerate(ds.select(range(n_use))):
        prompt = format_prompt(tokenizer, sample["question"])
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)

        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

        hs = out.hidden_states[1:]
        n_layers = len(hs)
        start = max(0, n_layers // 2)

        reg_logits = []
        for h in hs[start:]:
            logits = lm_head(norm(h[:, -1, :])).squeeze(0).detach().to(torch.float32).cpu().numpy()
            reg_logits.append(logits)
        reg_logits = np.asarray(reg_logits, dtype=np.float32)

        if reg_logits.shape[0] < 2:
            continue

        top_idx = np.argsort(reg_logits[-1])[-50:]
        x = np.arange(reg_logits.shape[0], dtype=np.float32)
        x = (x - x.mean()) / max(float(x.std()), 1e-8)
        x_m = float(x.mean())
        denom = float(np.sum((x - x_m) ** 2))
        if denom < 1e-12:
            continue

        per_token_r2: list[float] = []
        for tok in top_idx:
            y = reg_logits[:, tok]
            y_m = float(y.mean())
            b1 = float(np.sum((x - x_m) * (y - y_m)) / denom)
            b0 = float(y_m - b1 * x_m)
            y_hat = b0 + b1 * x
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y_m) ** 2))
            if ss_tot < 1e-12:
                continue
            per_token_r2.append(1.0 - ss_res / ss_tot)

        if per_token_r2:
            r2_values.append(float(np.mean(per_token_r2)))

        if (i + 1) % 5 == 0:
            running = float(np.mean(r2_values)) if r2_values else 0.0
            p(f"  R2 progress {i+1}/{n_use}: running_mean={running:.4f}")

    mean_r2 = float(np.mean(r2_values)) if r2_values else 0.0
    p(f"  Calibrated mean R2={mean_r2:.4f} (ALTA enabled: {mean_r2 >= ALTA_R2_CUTOFF})")
    return mean_r2


# ---------------------------------------------------------------------------
# Phase 3: Per-question trajectory features (R², κ, ECR — used by CUREDRouterV2)
# ---------------------------------------------------------------------------


def _compute_layer_features(
    hidden_states: Any,
    lm_head: Any,
    norm: Any,
    layer_start_ratio: float = 0.7,
    top_k: int = 50,
) -> tuple[float, float, float]:
    """Single-pass computation of (mean_r2, var_r2, kappa) from late-layer logits.

    Builds the (L, vocab) logit matrix once and reuses it for both R² and κ,
    avoiding the 2× redundant GPU→CPU transfer of computing them separately.

    R² measures linear growth of top-k token logits across the late layers.
    κ = (R²_quad - R²_lin) / (1 - R²_lin + ε) measures curvature (quadratic gain).

    Args:
        hidden_states:     Tuple of hidden state tensors (embedding layer removed;
                           i.e. out.hidden_states[1:]).
        lm_head:           Language model head (linear projection to vocabulary).
        norm:              Final layer-norm module.
        layer_start_ratio: Fraction of layers to skip from bottom (default 0.7,
                           i.e. only the top 30% of layers are used).
        top_k:             Number of highest-logit tokens to compute R²/κ over.

    Returns:
        tuple(r2_mean, r2_var, kappa):
            r2_mean (float): Mean linear R² across top_k tokens.
            r2_var  (float): Variance of R² across top_k tokens.
            kappa   (float): Mean quadratic gain fraction κ ∈ [0, 1].
    """
    n_layers = len(hidden_states)
    start = max(0, int(n_layers * layer_start_ratio))
    layer_logits_list: list[np.ndarray] = []
    for h in hidden_states[start:]:
        logits = lm_head(norm(h[:, -1, :])).squeeze(0).detach().cpu().float().numpy()
        layer_logits_list.append(logits)
    logit_matrix = np.asarray(layer_logits_list, dtype=np.float32)  # (L, vocab)
    L = int(logit_matrix.shape[0])

    if L < 2:
        return 0.0, 0.0, 0.0

    topk_idx = np.argsort(logit_matrix[-1])[-top_k:]
    x_n = np.arange(L, dtype=np.float32)
    x_n = (x_n - x_n.mean()) / max(float(x_n.std()), 1e-8)  # zero-mean, unit-std

    r2s: list[float] = []
    kappas: list[float] = []
    for tok in topk_idx:
        y = logit_matrix[:, int(tok)]
        ss_tot = float(np.var(y)) * L
        if ss_tot < 1e-8:
            continue

        # Linear fit → R²
        A_lin = np.column_stack([x_n, np.ones(L)])
        b_l, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
        ss_lin = float(np.sum((y - A_lin @ b_l) ** 2))
        r2_lin = max(0.0, 1.0 - ss_lin / ss_tot)
        r2s.append(r2_lin)

        # Quadratic fit → κ (only meaningful with ≥3 layers)
        if L >= 3:
            A_quad = np.column_stack([x_n ** 2, x_n, np.ones(L)])
            b_q, _, _, _ = np.linalg.lstsq(A_quad, y, rcond=None)
            ss_quad = float(np.sum((y - A_quad @ b_q) ** 2))
            r2_quad = max(0.0, 1.0 - ss_quad / ss_tot)
            kappas.append(max(0.0, (r2_quad - r2_lin) / (1.0 - r2_lin + 1e-8)))

    r2_arr = np.array(r2s, dtype=np.float32) if r2s else np.array([0.0], dtype=np.float32)
    kappa_val = float(np.mean(kappas)) if kappas else 0.0
    return float(r2_arr.mean()), float(r2_arr.var()), kappa_val


def compute_per_question_r2(
    hidden_states: Any,
    lm_head: Any,
    norm: Any,
    layer_start_ratio: float = 0.7,
    top_k: int = 50,
) -> tuple[float, float]:
    """Thin wrapper — returns (r2_mean, r2_var) from _compute_layer_features."""
    r2_mean, r2_var, _ = _compute_layer_features(hidden_states, lm_head, norm, layer_start_ratio, top_k)
    return r2_mean, r2_var


def compute_curvature(
    hidden_states: Any,
    lm_head: Any,
    norm: Any,
    layer_start_ratio: float = 0.7,
    top_k: int = 50,
) -> float:
    """Thin wrapper — returns κ (quadratic-gain fraction) from _compute_layer_features."""
    _, _, kappa = _compute_layer_features(hidden_states, lm_head, norm, layer_start_ratio, top_k)
    return kappa


def compute_ecr(
    hidden_states: Any,
    lm_head: Any,
    norm: Any,
) -> tuple[float, float, float]:
    """Compute the Entropy Compression Ratio (ECR) across all transformer layers.

    ECR = H_final / H_peak, where H_layer = -sum(p * log(p)) over the softmax
    of layer-projected logits.  Low ECR signals that the model has compressed
    its prediction entropy by the final layer (high confidence at output).

    Empirical profile (3B): H1=0.08, H7=10.83 (peak), H28=0.85 → ECR=0.078.
    Gate 2 ALTA feasibility includes ECR_q > tau_ECR (see ``CUREDRouterV2.route``).

    IMPORTANT: caller must pass hidden_states with the embedding layer already
    removed (e.g. out.hidden_states[1:]).  hidden_states[0] = transformer
    layer 1, hidden_states[-1] = final transformer layer.

    Args:
        hidden_states: Tuple of hidden state tensors (embedding layer removed).
        lm_head:       Language model head (linear projection to vocabulary).
        norm:          Final layer-norm module.

    Returns:
        tuple(ECR, H_final, H_peak):
            ECR     (float): H_final / H_peak ratio.
            H_final (float): Entropy at the final transformer layer.
            H_peak  (float): Maximum entropy across all transformer layers.
    """
    entropies: list[float] = []
    for h in hidden_states:
        logits = lm_head(norm(h[:, -1, :])).squeeze(0).detach().cpu().float()
        probs = torch.softmax(logits, dim=-1).numpy()
        H = float(-np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0))))
        entropies.append(H)
    H_final = entropies[-1]
    H_peak = max(entropies)
    ECR = H_final / (H_peak + 1e-8)
    return float(ECR), float(H_final), float(H_peak)


def compute_self_consistency(
    model: Any,
    tokenizer: Any,
    prompt: str,
    k: int = 3,
    temperature: float = 0.7,
    max_new_tokens: int = 40,
) -> tuple[float, str]:
    """Sample k responses and return (SC_score, modal_answer).

    SC = fraction of k samples that agree with the modal answer.
    SC=1.0 → all agree; SC=0.33 → all disagree.
    """
    from collections import Counter
    dev = get_model_device(model)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)
    responses: list[str] = []
    for _ in range(k):
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip().lower()[:50]
        responses.append(text)
    modal = Counter(responses).most_common(1)[0][0]
    sc = sum(1 for r in responses if r == modal) / k
    return float(sc), modal


def try_load_medhallu_dataset() -> tuple[Dataset, str, str | None, str]:
    candidates = [
        ("UTAustin-AIHealth/MedHallu", "pqa_artificial", "train"),
        ("UTAustin-AIHealth/MedHallu", "pqa_labeled", "train"),
        ("medhallu", None, "test"),
        ("medhallu", None, "validation"),
        ("openlifescienceai/medhallu", None, "test"),
        ("FreedomIntelligence/MedHallu", None, "test"),
        ("hirundo-io/medhallu", "default", "train"),
        ("Lizong/MedHallu", "pqa_artificial", "train"),
    ]
    errors: list[str] = []
    for dataset_id, subset, split in candidates:
        try:
            if subset:
                ds = load_dataset(dataset_id, subset, split=split)
            else:
                ds = load_dataset(dataset_id, split=split)
            return ds, dataset_id, subset, split
        except Exception as exc:
            errors.append(f"{dataset_id}:{subset}:{split} -> {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Could not auto-load MedHallu for calibration. Tried:\n" + "\n".join(errors)
    )


def medhallu_question(sample: dict[str, Any]) -> str:
    for key in ("Question", "question", "query", "prompt", "input", "instruction", "claim"):
        v = sample.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def medhallu_ground_truth(sample: dict[str, Any]) -> str:
    for key in ("Ground Truth", "ground_truth", "answer", "reference", "best_answer"):
        v = sample.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def calibrate_d2h(model: Any, tokenizer: Any, n_questions: int = DEFAULT_D2H_QUESTIONS) -> float:
    p(f"\nCalibrating d2H threshold on medical set (target n={n_questions})...")
    try:
        ds, ds_id, ds_subset, ds_split = try_load_medhallu_dataset()
        p(f"  Loaded MedHallu source: {ds_id} subset={ds_subset} split={ds_split}")
    except Exception as exc:
        p(f"  MedHallu load failed ({type(exc).__name__}: {exc})")
        p("  Falling back to default d2H threshold: -0.82")
        return -0.82

    vals: list[float] = []
    used = 0
    for sample in ds:
        if used >= int(n_questions):
            break
        question = medhallu_question(sample)
        if not question:
            continue
        prompt = format_prompt(tokenizer, question)
        feats = compute_d2h_features(model, tokenizer, prompt)
        vals.append(float(feats["d2H"]))
        used += 1
        if used % 10 == 0:
            p(f"  d2H progress: {used}/{n_questions}")

    if not vals:
        p("  No usable medical rows; using default d2H threshold: -0.82")
        return -0.82

    threshold = float(np.median(vals))
    p(f"  Calibrated d2H threshold={threshold:.4f} (std={float(np.std(vals)):.4f}, n={len(vals)})")
    return threshold


def extract_attention_activations(
    model: Any,
    tokenizer: Any,
    prompt: str,
    n_heads: int,
    head_dim: int,
) -> np.ndarray | None:
    layers = get_transformer_layers(model)
    if layers is None:
        return None

    dev = get_model_device(model)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)

    activations: list[np.ndarray] = []
    hooks = []

    def make_hook() -> Any:
        def hook_fn(_module: Any, _inp: Any, output: Any) -> Any:
            out0 = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(out0) or out0.ndim < 3:
                return output
            last = out0[0, -1, :]
            if int(last.shape[-1]) < int(n_heads) * int(head_dim):
                return output
            per_head = last[: int(n_heads) * int(head_dim)].view(int(n_heads), int(head_dim))
            activations.append(per_head.detach().to(torch.float32).cpu().numpy())
            return output

        return hook_fn

    try:
        for layer in layers:
            attn = get_attn_module(layer)
            if attn is None:
                hooks.append(None)
                continue
            hooks.append(attn.register_forward_hook(make_hook()))

        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for h in hooks:
            if h is not None:
                h.remove()

    if not activations:
        return None
    return np.asarray(activations, dtype=np.float32)


def train_iti_probes(
    model: Any,
    tokenizer: Any,
    arch: dict[str, Any],
    cache_dir: Path,
    max_questions: int,
    top_k_heads: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    n_layers = arch.get("n_layers")
    n_heads = arch.get("n_heads")
    head_dim = arch.get("head_dim")
    if not n_layers or not n_heads or not head_dim:
        p("  ITI disabled: model missing layer/head metadata.")
        return None, None

    p(f"\nTraining ITI probes (max_questions={max_questions})...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    start_idx = min(200, max(0, len(ds) - 1))
    end_idx = min(len(ds), start_idx + int(max_questions))
    probe_ds = ds.select(range(start_idx, end_idx))
    p(f"  Probe data range: {start_idx}..{end_idx-1} (n_questions={len(probe_ds)})")

    acts_all: list[np.ndarray] = []
    labels_all: list[int] = []

    for i, sample in enumerate(probe_ds):
        q = sample["question"]
        mc = sample["mc1_targets"]
        choices = mc["choices"]
        labels = mc["labels"]

        for choice, label in zip(choices, labels):
            msgs = [
                {"role": "system", "content": "You are a helpful, honest assistant."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": str(choice)},
            ]
            try:
                prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                prompt = f"Question: {q}\nAnswer: {choice}"

            acts = extract_attention_activations(model, tokenizer, prompt, int(n_heads), int(head_dim))
            if acts is None or acts.shape[0] != int(n_layers):
                continue

            acts_all.append(acts)
            labels_all.append(int(label))

        if (i + 1) % 20 == 0:
            p(f"  Activation extraction: {i+1}/{len(probe_ds)} questions")

    if not acts_all:
        p("  ITI training failed: no activations extracted.")
        return None, None

    X_all = np.asarray(acts_all, dtype=np.float32)  # [N, L, H, D]
    y_all = np.asarray(labels_all, dtype=np.int32)
    p(
        f"  Samples={len(y_all)} truthful={int(y_all.sum())} hallu={int((1-y_all).sum())} "
        f"shape={X_all.shape}"
    )

    if len(np.unique(y_all)) < 2:
        p("  ITI training failed: labels have only one class.")
        return None, None

    head_scores = np.zeros((int(n_layers), int(n_heads)), dtype=np.float32)
    head_vectors = np.zeros((int(n_layers), int(n_heads), int(head_dim)), dtype=np.float32)

    p("  Fitting probes per head...")
    for li in range(int(n_layers)):
        for hi in range(int(n_heads)):
            X = X_all[:, li, hi, :]
            try:
                probe = LogisticRegression(max_iter=400, C=0.1)
                probe.fit(X, y_all)
                head_scores[li, hi] = float(probe.score(X, y_all))
                d = probe.coef_[0]
                head_vectors[li, hi] = d / (np.linalg.norm(d) + 1e-8)
            except Exception:
                continue

        if (li + 1) % 4 == 0:
            best_h = int(np.argmax(head_scores[li]))
            p(f"  Layer {li+1}/{n_layers} best_head={best_h} acc={head_scores[li, best_h]:.3f}")

    flat = head_scores.reshape(-1)
    k = min(int(top_k_heads), len(flat))
    top_flat = np.argsort(flat)[-k:]
    top_heads = np.asarray([(idx // int(n_heads), idx % int(n_heads)) for idx in top_flat], dtype=np.int32)

    np.save(cache_dir / "iti_head_scores.npy", head_scores)
    np.save(cache_dir / "iti_head_vectors.npy", head_vectors)
    np.save(cache_dir / "iti_top_heads.npy", top_heads)
    p(f"  ITI artifacts saved in: {cache_dir}")

    return top_heads, head_vectors


# ---------------------------------------------------------------------------
# Phase 4: CURED Router — per-question routing using trajectory features
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 4: CUREDRouterV2 — 5-gate principled router (see module docstring)
# ---------------------------------------------------------------------------


def score_alta(
    r2_q: float,
    var_r2_q: float,
    kappa_q: float,
    H_final: float,
    tau_r2: float = 0.65,
    tau_H: float = 1.0,
    beta1: float = 3.0,
    beta2: float = 0.5,
    beta3: float = 5.0,
    beta4: float = 2.0,
) -> float:
    """Composite ALTA score ∈ [0,1]. >0.5 → route to ALTA.

    Signs: R² and entropy increase the score (more linearity/uncertainty → ALTA helps).
    Curvature and R² variance decrease it (non-linear trajectory → linear extrapolation unreliable).
    Betas are configurable from router_thresholds.json and updated by calibrate_router.py.
    """
    lc = (
        beta1 * (r2_q - tau_r2)
        + beta2 * (H_final - tau_H)
        - beta3 * kappa_q
        - beta4 * var_r2_q
    )
    return float(1.0 / (1.0 + np.exp(-lc)))


def score_cove(
    SC_q: float | None,
    H_final: float,
    domain_medical: int,
    model_params_B: float,
    tau_SC: float = 0.60,
    tau_H: float = 3.0,
    max_medical_params: float = 4.0,
    beta_SC: float = 2.0,
    beta_H: float = 0.5,
    beta_med_penalty: float = 3.0,
) -> float:
    """CoVe score ∈ [0,1]. >0.5 → route to CoVe (medical domain only).

    Heavy penalty for medical+large model (hallucination snowballing risk from Zhang et al. 2025).
    SC_q=None (large models) → treat as SC_q=tau_SC (neutral signal).

    NOTE: _uncertainty_gate only acts on s_cove > 0.5 when domain_medical=True.
    For non-medical domains (e.g. TruthfulQA), CoVe is disabled entirely: empirical
    finding shows CoVe degrades general/adversarial QA by 10–18 pp at all scales.
    """
    sc_val = SC_q if SC_q is not None else tau_SC
    penalty = beta_med_penalty * domain_medical * int(model_params_B > max_medical_params)
    lc = beta_SC * (tau_SC - sc_val) + beta_H * (H_final - tau_H) - penalty
    return float(1.0 / (1.0 + np.exp(-lc)))


def compute_semantic_entropy(
    model: Any,
    tokenizer: Any,
    prompt: str,
    scorer: Any,
    k: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 40,
) -> float:
    """Semantic entropy via response clustering (Farquhar et al., 2024).

    Generates k stochastic responses, clusters them by cosine similarity
    (threshold=0.80), and returns Shannon entropy over cluster probabilities (nats).

    Higher = model genuinely uncertain about the answer.
    Lower  = model confidently and consistently picks one answer.

    Cost: k forward passes (~5x greedy latency). Use sparingly (ablations only).
    Prerequisite: sentence-transformers scorer passed in (same as used for SC_q).
    """
    from collections import Counter

    dev = get_model_device(model)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)
    responses: list[str] = []

    for _ in range(k):
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        responses.append(text.strip().lower()[:60])

    if not responses:
        return 0.0

    embs = scorer.encode(responses, convert_to_tensor=True)
    cluster_id = [-1] * k
    next_cluster = 0
    for i in range(k):
        if cluster_id[i] == -1:
            cluster_id[i] = next_cluster
            for j in range(i + 1, k):
                if float(util.cos_sim(embs[i], embs[j]).item()) >= 0.80:
                    cluster_id[j] = next_cluster
            next_cluster += 1

    counts = Counter(cluster_id)
    total = sum(counts.values())
    H_sem = -sum((c / total) * np.log(c / total + 1e-10) for c in counts.values())
    return float(H_sem)


class CUREDRouterV2:
    """5-gate principled router (see module docstring for full decision order).

    Gate 1: Confidence (H_final, optional SC) → greedy_confident when active.
    Scale shortcut: profile R² viability + non-medical + H_final > τ_H_easy → ALTA.
    Gate 2: Feasibility — R²_q > τ_R2, κ_q < τ_kappa, ECR_q > τ_ECR before ALTA path.
    Gate 3: Medical + ITI available → ITI.
    Gate 4: Composite ALTA score → alta_gate4.
    Gate 5: Medical CoVe vs greedy in ``_uncertainty_gate``.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        arch: dict[str, Any],
        model_params_B: float,
        top_heads: np.ndarray | None,
        head_vectors: np.ndarray | None,
        thresholds: dict[str, Any],
        compute_sc: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.arch = arch
        self.params_B = float(model_params_B)
        self.top_heads = top_heads
        self.head_vectors = head_vectors
        self.compute_sc = compute_sc
        self.iti_available = top_heads is not None and head_vectors is not None

        self.tau_R2 = float(thresholds.get("tau_R2", 0.65))
        self.tau_kappa = float(thresholds.get("tau_kappa", 0.70))   # was 0.08; mean kappa=0.597@8B
        self.tau_ECR = float(thresholds.get("tau_ECR", 0.04))       # was 0.10; mean ECR=0.031–0.076
        self.tau_H_easy = float(thresholds.get("tau_H_easy", 0.5))
        self.tau_H_hard = float(thresholds.get("tau_H_hard", 3.0))
        self.tau_SC_easy = float(thresholds.get("tau_SC_easy", 0.90))
        self.tau_SC_hard = float(thresholds.get("tau_SC_hard", 0.60))
        self.beta1 = float(thresholds.get("beta1", 3.0))
        self.beta2 = float(thresholds.get("beta2", 0.5))
        self.beta3 = float(thresholds.get("beta3", 5.0))
        self.beta4 = float(thresholds.get("beta4", 2.0))
        # Model-level R² from profiling (profile_*.json). When ≥0.55, ALTA is
        # globally beneficial at this scale → bypass per-question gating for
        # general-domain, matching the robust behavior of the old CUREDRouter.
        self.profile_mean_r2 = float(thresholds.get("profile_mean_r2", 0.0))
        self.alta_globally_viable = self.profile_mean_r2 >= 0.55

    def _features(self, prompt: str) -> tuple[float, float, float, float, float, float | None]:
        dev = get_model_device(self.model)
        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True, use_cache=False)
        # out.hidden_states[0] = embedding layer; [1:] = transformer layers 1..N
        hs = out.hidden_states[1:]
        norm, lm_head = get_final_norm_and_lm_head(self.model)

        # Single pass over late-layer logit matrix for R², var(R²), κ
        r2_q, var_r2_q, kappa_q = _compute_layer_features(hs, lm_head, norm)
        # Full-depth pass for ECR (needs all layers, not just final 30%)
        ECR_q, H_final, _ = compute_ecr(hs, lm_head, norm)

        # SC: None for large models — 3× generate() at 32B adds ~2h for n=500
        # Using a 1.0 sentinel would silently hijack Gate 1; None is explicit.
        SC_q: float | None = None
        if self.compute_sc and self.params_B <= 14.0:
            SC_q, _ = compute_self_consistency(self.model, self.tokenizer, prompt, k=3)

        return r2_q, var_r2_q, kappa_q, ECR_q, H_final, SC_q

    def route(self, question: str, max_new_tokens: int = 80, scoring: str = "cosine") -> dict[str, Any]:
        domain = detect_domain(question)
        prompt = format_prompt(self.tokenizer, question)
        r2_q, var_r2_q, kappa_q, ECR_q, H_final, SC_q = self._features(prompt)

        routing_log: dict[str, Any] = {
            "r2_q": round(r2_q, 4),
            "var_r2_q": round(var_r2_q, 4),
            "kappa_q": round(kappa_q, 4),
            "ECR_q": round(ECR_q, 4),
            "H_final": round(H_final, 4),
            "SC_q": round(SC_q, 4) if SC_q is not None else None,
        }
        domain_medical = int(domain == "medical")

        # ── Gate 1: Confidence gate ──────────────────────────────────────
        # For models > 14B: SC is not computed (3× generate at 32B ≈ +2h per n=500).
        # Gate 1 reduces to H_final < tau_H_easy for those large models.
        #
        # For models ≤ 14B: Gate 1 requires *both* SC_q ≥ tau_SC_easy AND
        # H_final < tau_H_easy. SC_q = None when --compute-sc is absent (the
        # default in Phase 4 runs for efficiency), making Gate 1 permanently
        # inactive for 3B/8B in canonical evaluation. The scale-aware shortcut
        # (between Gates 1 and 2) handles confident-question routing instead.
        #
        # A production deployment using --compute-sc would activate Gate 1 for
        # all scales ≤ 14B and route high-SC, low-entropy questions to greedy_confident.
        if self.params_B > 14.0:
            gate1_fires = H_final < self.tau_H_easy
        else:
            gate1_fires = (
                SC_q is not None          # always False without --compute-sc
                and SC_q >= self.tau_SC_easy
                and H_final < self.tau_H_easy
            )
        if gate1_fires:
            routing_log["gate"] = 1
            return {
                "text": greedy_generate(self.model, self.tokenizer, prompt, max_new_tokens),
                "strategy": "greedy_confident",
                "domain": domain,
                **routing_log,
            }

        # ── Scale-aware shortcut (between Gate 1 and Gate 2) ─────────────
        # When model-level R² ≥ 0.55, ALTA is globally beneficial at this scale.
        # Bypass per-question Gate 2 gating for general-domain non-trivial questions,
        # replicating the robust behavior of the old CUREDRouter while retaining the
        # full 5-gate logic for medical domain and easy questions.
        if self.alta_globally_viable and not domain_medical and H_final > self.tau_H_easy:
            routing_log["gate"] = "2s"
            out = alta_generate(self.model, self.tokenizer, prompt, max_new_tokens)
            return {
                "text": out["text"],
                "strategy": "alta_global_viable",
                "domain": domain,
                **routing_log,
            }

        # ── Gate 2: Feasibility gate ─────────────────────────────────────
        alta_ok = (
            r2_q > self.tau_R2
            and kappa_q < self.tau_kappa
            and ECR_q > self.tau_ECR
        )
        routing_log["alta_feasible"] = int(alta_ok)
        if not alta_ok:
            return self._uncertainty_gate(
                question, prompt, domain, domain_medical, SC_q, H_final, max_new_tokens, routing_log
            )

        # ── Gate 3: Domain-safety gate ───────────────────────────────────
        # medical+large → ITI (CoVe snowballs on clinical MCQs at ≥8B)
        if domain_medical and self.params_B > 4.0 and self.iti_available:
            routing_log["gate"] = 3
            text = iti_generate(
                self.model,
                self.tokenizer,
                self.arch,
                self.top_heads,
                self.head_vectors,
                prompt,
                alpha=ITI_ALPHA,
                max_new_tokens=max_new_tokens,
            )
            return {"text": text, "strategy": "iti_medical_gate3", "domain": domain, **routing_log}

        # ── Gate 4: ALTA composite score ─────────────────────────────────
        s_alta = score_alta(
            r2_q, var_r2_q, kappa_q, H_final,
            tau_r2=self.tau_R2,
            beta1=self.beta1, beta2=self.beta2,
            beta3=self.beta3, beta4=self.beta4,
        )
        routing_log["S_ALTA"] = round(s_alta, 4)
        if s_alta > 0.5:
            routing_log["gate"] = 4
            out = alta_generate(self.model, self.tokenizer, prompt, max_new_tokens)
            return {"text": out["text"], "strategy": "alta_gate4", "domain": domain, **routing_log}

        # ── Gate 5: Uncertainty gate ─────────────────────────────────────
        return self._uncertainty_gate(
            question, prompt, domain, domain_medical, SC_q, H_final, max_new_tokens, routing_log
        )

    def _uncertainty_gate(
        self,
        question: str,
        prompt: str,
        domain: str,
        domain_medical: int,
        SC_q: float | None,
        H_final: float,
        max_new_tokens: int,
        routing_log: dict[str, Any],
    ) -> dict[str, Any]:
        s_cove = score_cove(SC_q, H_final, domain_medical, self.params_B,
                            tau_SC=self.tau_SC_hard, tau_H=self.tau_H_hard)
        routing_log["S_CoVe"] = round(s_cove, 4)
        routing_log["gate"] = 5
        # CoVe only for medical: on general/adversarial QA (TruthfulQA) it consistently
        # degrades performance at all scales by reinforcing the model's initial misconception.
        # Empirical finding: CoVe -10pp on TruthfulQA, +3–6pp on MedHallu.
        if s_cove > 0.5 and domain_medical:
            text = cove_generate(self.model, self.tokenizer, question, max_new_tokens)
            return {"text": text, "strategy": "cove_gate5_medical", "domain": domain, **routing_log}
        text = greedy_generate(self.model, self.tokenizer, prompt, max_new_tokens)
        return {"text": text, "strategy": "greedy_gate5", "domain": domain, **routing_log}


# ---------------------------------------------------------------------------
# Phase 4: CUREDRouter — legacy d2H-based router (--router old)
#   Kept for backwards compatibility and comparison experiments.
# ---------------------------------------------------------------------------


class CUREDRouter:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        arch: dict[str, Any],
        mean_r2: float,
        d2h_threshold: float,
        top_heads: np.ndarray | None,
        head_vectors: np.ndarray | None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.arch = arch
        self.mean_r2 = float(mean_r2)
        self.d2h_threshold = float(d2h_threshold)
        self.top_heads = top_heads
        self.head_vectors = head_vectors
        self.alta_viable = self.mean_r2 >= ALTA_R2_CUTOFF
        self.iti_available = self.top_heads is not None and self.head_vectors is not None

    def route(self, question: str, max_new_tokens: int = 80, scoring: str = "cosine") -> dict[str, Any]:
        domain = detect_domain(question)
        prompt = format_prompt(self.tokenizer, question)

        if scoring in ("letter", "yesno"):
            if self.alta_viable:
                out = alta_generate(self.model, self.tokenizer, prompt, max_new_tokens=max_new_tokens)
                return {
                    "text": out["text"],
                    "strategy": f"alta_{domain}_structured",
                    "domain": domain,
                    "r2": round(self.mean_r2, 4),
                    "alta_gate": round(float(out["mean_gate"]), 4),
                }

            text = greedy_generate(self.model, self.tokenizer, prompt, max_new_tokens=max_new_tokens)
            return {
                "text": text,
                "strategy": f"greedy_{domain}_structured",
                "domain": domain,
                "r2": round(self.mean_r2, 4),
            }

        if domain == "general":
            if self.alta_viable:
                out = alta_generate(self.model, self.tokenizer, prompt, max_new_tokens=max_new_tokens)
                return {
                    "text": out["text"],
                    "strategy": "alta_general",
                    "domain": domain,
                    "r2": round(self.mean_r2, 4),
                    "alta_gate": round(float(out["mean_gate"]), 4),
                }

            text = greedy_generate(self.model, self.tokenizer, prompt, max_new_tokens=max_new_tokens)
            return {
                "text": text,
                "strategy": "greedy_general",
                "domain": domain,
                "r2": round(self.mean_r2, 4),
            }

        feats = compute_d2h_features(self.model, self.tokenizer, prompt)
        d2h = float(feats["d2H"])

        if d2h <= self.d2h_threshold:
            text = cove_generate(self.model, self.tokenizer, question, max_new_tokens=max_new_tokens)
            return {
                "text": text,
                "strategy": "cove_medical",
                "domain": domain,
                "d2H": round(d2h, 4),
            }

        if self.iti_available:
            text = iti_generate(
                self.model,
                self.tokenizer,
                self.arch,
                self.top_heads,
                self.head_vectors,
                prompt,
                alpha=ITI_ALPHA,
                max_new_tokens=max_new_tokens,
            )
            return {
                "text": text,
                "strategy": "iti_medical",
                "domain": domain,
                "d2H": round(d2h, 4),
            }

        text = greedy_generate(self.model, self.tokenizer, prompt, max_new_tokens=max_new_tokens)
        return {
            "text": text,
            "strategy": "greedy_medical_no_iti",
            "domain": domain,
            "d2H": round(d2h, 4),
        }


# ---------------------------------------------------------------------------
# Phase 4: Benchmark loaders (TruthfulQA, MedHallu, custom CSV)
# ---------------------------------------------------------------------------


def _first_positive_choice(choices: list[str], labels: list[Any]) -> str:
    for choice, label in zip(choices, labels):
        try:
            if int(label) == 1:
                return str(choice)
        except Exception:
            if str(label).strip().lower() in {"1", "true", "yes"}:
                return str(choice)
    return ""


def load_truthfulqa(n: int, scoring: str = "cosine") -> list[dict[str, Any]]:
    if scoring == "mc":
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        out = []
        for row in ds.select(range(min(int(n), len(ds)))):
            mc1 = row.get("mc1_targets", {}) or {}
            mc2 = row.get("mc2_targets", {}) or {}
            mc1_choices = [str(c) for c in mc1.get("choices", [])]
            mc1_labels = list(mc1.get("labels", []))
            mc2_choices = [str(c) for c in mc2.get("choices", [])]
            mc2_labels = list(mc2.get("labels", []))
            out.append(
                {
                    "question": str(row["question"]),
                    "reference": _first_positive_choice(mc1_choices, mc1_labels),
                    "domain": "general",
                    "dataset": "truthfulqa_mc",
                    "mc1_choices": mc1_choices,
                    "mc1_labels": mc1_labels,
                    "mc2_choices": mc2_choices,
                    "mc2_labels": mc2_labels,
                }
            )
        return out

    ds = load_dataset("truthful_qa", "generation", split="validation")
    out = []
    for row in ds.select(range(min(int(n), len(ds)))):
        out.append(
            {
                "question": str(row["question"]),
                "reference": str(row["best_answer"]),
                "domain": "general",
                "dataset": "truthfulqa",
            }
        )
    return out


def load_medhallu_generation(n: int) -> list[dict[str, Any]]:
    ds, ds_id, ds_subset, ds_split = try_load_medhallu_dataset()
    p(f"  MedHallu source: {ds_id} subset={ds_subset} split={ds_split}")

    out = []
    for row in ds:
        if len(out) >= int(n):
            break
        q = medhallu_question(row)
        gt = medhallu_ground_truth(row)
        if q and gt:
            out.append({"question": q, "reference": gt, "domain": "medical", "dataset": "medhallu"})
    return out


def load_custom_csv(
    path: str,
    question_col: str,
    answer_col: str,
    n: int,
    allow_missing_reference: bool,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(samples) >= int(n):
                break
            q = str(row.get(question_col, "") or "").strip()
            a = str(row.get(answer_col, "") or "").strip() if answer_col else ""
            if not q:
                continue
            if (not a) and (not allow_missing_reference):
                continue
            samples.append(
                {
                    "question": q,
                    "reference": a,
                    "domain": detect_domain(q),
                    "dataset": "custom_csv",
                }
            )
    return samples


# ---------------------------------------------------------------------------
# Phase 4: Evaluation loop (run_protocol, run_api_protocol, result serialisation)
# ---------------------------------------------------------------------------


def run_protocol(
    model: Any,
    tokenizer: Any,
    arch: dict[str, Any],
    router: Any,  # CUREDRouter or CUREDRouterV2
    scorer: SentenceTransformer,
    protocol: str,
    samples: list[dict[str, Any]],
    cosine_threshold: float,
    scoring: str,
    max_new_tokens: int,
    selfcheck_k: int,
    iti_probes: tuple[np.ndarray | None, np.ndarray | None],
    save_per_question: bool = False,
    load_in_4bit: bool = False,
    model_params_b: float = 0.0,
) -> dict[str, Any]:
    top_heads, head_vectors = iti_probes
    correct = 0
    repeated = 0
    scored = 0
    mc_scored = 0
    mc1_total = 0.0
    mc2_total = 0.0
    by_strategy: dict[str, int] = {}
    per_question: list[dict[str, Any]] = []

    t0 = time.time()

    # ── Batched pre-generation for greedy and cove (GPU-bound; safe to batch) ──
    # These protocols do not need per-step hidden states, so B>1 is safe.
    # Results are identical: same do_sample=False argmax decoding, same penalties.
    # ALTA, delta_dola, ITI are left sequential (need output_hidden_states=True per step).
    _bsz = _auto_batch_size(model_params_b, load_in_4bit)
    _pregenerated: list[str] | None = None

    if protocol in ("greedy",) and _bsz > 1 and len(samples) > 1:
        p(f"  [batch] pre-generating {len(samples)} answers (batch_size={_bsz})...")
        all_prompts = [format_prompt(tokenizer, s["question"]) for s in samples]
        _pregenerated = []
        for _b0 in range(0, len(all_prompts), _bsz):
            _pregenerated.extend(
                batch_greedy_generate(model, tokenizer, all_prompts[_b0:_b0 + _bsz], max_new_tokens)
            )
        p(f"  [batch] generation done, starting scoring loop...")

    for i, sample in enumerate(samples):
        q = sample["question"]
        ref = sample.get("reference", "")
        prompt = format_prompt(tokenizer, q)
        extra: dict[str, Any] = {}

        if protocol == "greedy":
            text = _pregenerated[i] if _pregenerated is not None else greedy_generate(
                model, tokenizer, prompt, max_new_tokens=max_new_tokens
            )
            strategy = "greedy"

        elif protocol == "alta":
            out = alta_generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            text = out["text"]
            strategy = "alta"
            extra = {
                "alta_gate": round(float(out["mean_gate"]), 4),
                "alta_first_entropy": round(float(out["first_entropy"]), 4),
            }

        elif protocol == "delta_dola":
            text = delta_dola_generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            strategy = "delta_dola"

        elif protocol == "cove":
            text = cove_generate(model, tokenizer, q, max_new_tokens=max_new_tokens)
            strategy = "cove"

        elif protocol == "iti":
            if top_heads is not None and head_vectors is not None:
                text = iti_generate(
                    model,
                    tokenizer,
                    arch,
                    top_heads,
                    head_vectors,
                    prompt,
                    alpha=ITI_ALPHA,
                    max_new_tokens=max_new_tokens,
                )
                strategy = "iti"
            else:
                text = greedy_generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
                strategy = "greedy_no_iti"

        elif protocol == "selfcheck":
            out = selfcheck_generate(
                model,
                tokenizer,
                prompt,
                scorer,
                max_new_tokens=max_new_tokens,
                k_samples=selfcheck_k,
            )
            text = out["text"]
            strategy = "selfcheck"
            extra = {"consistency": round(float(out["consistency"]), 4), "n_samples": int(out["n_samples"])}

        elif protocol == "cured":
            out = router.route(q, max_new_tokens=max_new_tokens, scoring=scoring)
            text = out["text"]
            strategy = out["strategy"]
            extra = {k: v for k, v in out.items() if k not in ("text", "strategy")}

        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        # When saving per-question data for non-cured protocols, compute routing
        # features (r2_q, kappa_q, ECR_q, H_final) via one extra forward pass.
        # Required for R²-stratified analysis in compute_final_stats.py.
        if save_per_question and protocol != "cured" and "r2_q" not in extra:
            try:
                _dev = get_model_device(model)
                _ids = tokenizer.encode(prompt, return_tensors="pt").to(_dev)
                with torch.no_grad():
                    _fwd = model(_ids, output_hidden_states=True, use_cache=False)
                _hs = _fwd.hidden_states[1:]
                _norm, _lm_head = get_final_norm_and_lm_head(model)
                _r2_q, _var_r2_q, _kappa_q = _compute_layer_features(_hs, _lm_head, _norm)
                _ECR_q, _H_final, _ = compute_ecr(_hs, _lm_head, _norm)
                extra.update({
                    "r2_q": round(float(_r2_q), 4),
                    "var_r2_q": round(float(_var_r2_q), 4),
                    "kappa_q": round(float(_kappa_q), 4),
                    "ECR_q": round(float(_ECR_q), 4),
                    "H_final": round(float(_H_final), 4),
                })
            except Exception:
                pass  # degrade gracefully; r2_q stays None

        by_strategy[strategy] = by_strategy.get(strategy, 0) + 1

        rep = has_repetition(text)
        if rep:
            repeated += 1

        has_ref = isinstance(ref, str) and bool(ref.strip())
        is_correct = False
        if scoring == "mc":
            mc1_choices = sample.get("mc1_choices")
            mc1_labels = sample.get("mc1_labels")
            mc2_choices = sample.get("mc2_choices")
            mc2_labels = sample.get("mc2_labels")
            if isinstance(mc1_choices, list) and isinstance(mc1_labels, list) and mc1_choices and mc1_labels:
                # Strategy-faithful MC protocol selection:
                # - 'alta' protocol always uses ALTA-aware scoring
                # - 'cured' uses ALTA-aware scoring only when router chose an ALTA strategy
                # - all others use baseline scoring
                _mc_proto = "greedy"
                if protocol == "alta":
                    _mc_proto = "alta"
                elif protocol == "cured" and router.alta_viable:
                    _alta_strategies = {"alta_general", "alta_medical_structured", "alta_general_structured"}
                    if strategy in _alta_strategies:
                        _mc_proto = "alta"
                mc_scores = mc_score_sample(
                    model=model,
                    tokenizer=tokenizer,
                    question=q,
                    choices=[str(c) for c in mc1_choices],
                    labels=list(mc1_labels),
                    choices_mc2=[str(c) for c in mc2_choices] if isinstance(mc2_choices, list) else None,
                    labels_mc2=list(mc2_labels) if isinstance(mc2_labels, list) else None,
                    mc_protocol=_mc_proto,
                )
                scored += 1
                mc_scored += 1
                mc1_val = float(mc_scores["mc1"])
                mc2_val = float(mc_scores["mc2"])
                mc1_total += mc1_val
                mc2_total += mc2_val
                is_correct = mc1_val >= 0.5
                if is_correct:
                    correct += 1
                extra["mc1"] = round(mc1_val, 4)
                extra["mc2"] = round(mc2_val, 4)
            elif (not rep) and has_ref:
                scored += 1
                is_correct = reference_match(
                    scorer=scorer,
                    sample=sample,
                    generated=text,
                    reference=ref,
                    threshold=cosine_threshold,
                    scoring=scoring,
                )
                if is_correct:
                    correct += 1
        elif (not rep) and has_ref:
            scored += 1
            is_correct = reference_match(
                scorer=scorer,
                sample=sample,
                generated=text,
                reference=ref,
                threshold=cosine_threshold,
                scoring=scoring,
            )
            if is_correct:
                correct += 1

        pq_entry: dict[str, Any] = {
            "i": i + 1,
            "q_id": i,
            "question": q,
            "reference": ref,
            "domain": sample.get("domain", detect_domain(q)),
            "strategy": strategy,
            "correct": int(is_correct) if has_ref else None,
            "repeated": int(rep),
            "answer": text,
            **extra,
        }
        if save_per_question:
            # Feature fields from CUREDRouterV2 routing_log (None for non-cured protocols).
            pq_entry.update({
                "r2_q": extra.get("r2_q"),
                "var_r2_q": extra.get("var_r2_q"),
                "kappa_q": extra.get("kappa_q"),
                "ecr_q": extra.get("ECR_q"),
                "h_final": extra.get("H_final"),
                "sc_q": extra.get("SC_q"),
                "domain_medical": int(sample.get("domain", detect_domain(q)) == "medical"),
            })
        per_question.append(pq_entry)

        if (i + 1) % 10 == 0:
            acc = (correct / scored) if scored else 0.0
            rep_rate = repeated / (i + 1)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(samples) - (i + 1))
            p(
                f"  [{i+1}/{len(samples)}] acc={acc:.1%} rep={rep_rate:.1%} "
                f"strategies={by_strategy} eta={eta/60:.0f}min"
            )

    n_total = len(samples)
    accuracy = (correct / scored) if scored else None
    mc1 = (mc1_total / mc_scored) if mc_scored else None
    mc2 = (mc2_total / mc_scored) if mc_scored else None
    if scoring == "mc" and mc1 is not None:
        accuracy = mc1

    return {
        "n_total": n_total,
        "n_scored": scored,
        "accuracy": round(float(accuracy), 4) if accuracy is not None else None,
        "mc1": round(float(mc1), 4) if mc1 is not None else None,
        "mc2": round(float(mc2), 4) if mc2 is not None else None,
        "rep_rate": round(repeated / max(n_total, 1), 4),
        "routing": {k: round(v / max(n_total, 1), 4) for k, v in by_strategy.items()},
        "runtime_min": round((time.time() - t0) / 60.0, 2),
        "per_question": per_question,
    }


def run_api_protocol(
    api_mode: str,
    api_model: str,
    api_router: CUREDAPIRouter,
    scorer: SentenceTransformer,
    protocol: str,
    samples: list[dict[str, Any]],
    cosine_threshold: float,
    scoring: str,
    max_new_tokens: int,
    api_temperature: float,
) -> dict[str, Any]:
    correct = 0
    repeated = 0
    scored = 0
    by_strategy: dict[str, int] = {}
    per_question: list[dict[str, Any]] = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        q = sample["question"]
        ref = sample.get("reference", "")
        extra: dict[str, Any] = {}
        had_error = False

        try:
            if protocol == "greedy":
                text = api_generate(
                    api_mode=api_mode,
                    api_model=api_model,
                    prompt=q,
                    max_new_tokens=max_new_tokens,
                    temperature=api_temperature,
                )
                strategy = "api_greedy"
            elif protocol == "cove":
                text = api_cove_generate(
                    api_mode=api_mode,
                    api_model=api_model,
                    question=q,
                    max_new_tokens=max_new_tokens,
                    api_temperature=api_temperature,
                )
                strategy = "api_cove"
            elif protocol == "cured_api":
                out = api_router.route(q, max_new_tokens=max_new_tokens, scoring=scoring)
                text = out["text"]
                strategy = out["strategy"]
                extra = {k: v for k, v in out.items() if k not in ("text", "strategy")}
            else:
                raise ValueError(f"Unknown API protocol: {protocol}")
        except APIFatalError:
            # Stop the benchmark immediately for quota/auth fatals to avoid wasting calls.
            raise
        except Exception as exc:
            # Keep long benchmark jobs alive even when API backends transiently fail.
            fallback_ok = False
            primary_error = str(exc)

            if protocol in ("cove", "cured_api"):
                try:
                    text = api_generate(
                        api_mode=api_mode,
                        api_model=api_model,
                        prompt=q,
                        max_new_tokens=max_new_tokens,
                        temperature=api_temperature,
                    )
                    strategy = f"{protocol}_rate_limit_fallback_greedy"
                    extra["fallback_from_error"] = primary_error[:300]
                    fallback_ok = True
                except Exception as fallback_exc:
                    primary_error = f"{primary_error} | fallback_failed: {fallback_exc}"

            if not fallback_ok:
                text = ""
                strategy = f"{protocol}_error"
                extra["error"] = primary_error[:500]
                had_error = True

        by_strategy[strategy] = by_strategy.get(strategy, 0) + 1

        rep = has_repetition(text)
        if rep:
            repeated += 1

        has_ref = isinstance(ref, str) and bool(ref.strip())
        is_correct = False
        if (not had_error) and (not rep) and has_ref:
            scored += 1
            is_correct = reference_match(
                scorer=scorer,
                sample=sample,
                generated=text,
                reference=ref,
                threshold=cosine_threshold,
                scoring=scoring,
            )
            if is_correct:
                correct += 1

        per_question.append(
            {
                "i": i + 1,
                "question": q,
                "reference": ref,
                "domain": sample.get("domain", detect_domain(q)),
                "strategy": strategy,
                "correct": int(is_correct) if has_ref else None,
                "repeated": int(rep),
                "answer": text,
                **extra,
            }
        )

        if (i + 1) % 10 == 0:
            acc = (correct / scored) if scored else 0.0
            rep_rate = repeated / (i + 1)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(samples) - (i + 1))
            p(
                f"  [{i+1}/{len(samples)}] acc={acc:.1%} rep={rep_rate:.1%} "
                f"strategies={by_strategy} eta={eta/60:.0f}min"
            )

    n_total = len(samples)
    accuracy = (correct / scored) if scored else None
    return {
        "n_total": n_total,
        "n_scored": scored,
        "accuracy": round(float(accuracy), 4) if accuracy is not None else None,
        "rep_rate": round(repeated / max(n_total, 1), 4),
        "routing": {k: round(v / max(n_total, 1), 4) for k, v in by_strategy.items()},
        "runtime_min": round((time.time() - t0) / 60.0, 2),
        "per_question": per_question,
    }


def print_results_table(model_name: str, mean_r2: float, d2h_threshold: float, all_results: dict[str, Any]) -> None:
    benchmarks = list(all_results.keys())
    if not benchmarks:
        return

    protocols = list(next(iter(all_results.values())).keys())
    label_w = max(14, *(len(pn) for pn in protocols))
    col_w = 16

    p("\n" + "=" * 84)
    p(f"CURED RESULTS | model={model_name}")
    p(f"R2={mean_r2:.4f} (ALTA {'ON' if mean_r2 >= ALTA_R2_CUTOFF else 'OFF'}) | d2H_threshold={d2h_threshold:.4f}")
    p("=" * 84)

    header = f"{'protocol':<{label_w}}"
    for b in benchmarks:
        header += f"  {b:>{col_w}}"
    p(header)
    p("-" * len(header))

    for proto in protocols:
        row = f"{proto:<{label_w}}"
        for b in benchmarks:
            acc = all_results[b][proto]["accuracy"]
            if acc is None:
                cell = "n/a"
            else:
                cell = f"{acc:.1%}"
            row += f"  {cell:>{col_w}}"
        p(row)

    if "greedy" in protocols:
        p("\nDelta vs greedy:")
        for proto in protocols:
            if proto == "greedy":
                continue
            row = f"  {proto:<{label_w-2}}"
            for b in benchmarks:
                a = all_results[b][proto]["accuracy"]
                g = all_results[b]["greedy"]["accuracy"]
                if a is None or g is None:
                    cell = "n/a"
                else:
                    d = a - g
                    sign = "+" if d >= 0 else ""
                    cell = f"{sign}{d:.1%}"
                row += f"  {cell:>{col_w}}"
            p(row)

    if "cured" in protocols:
        p("\nCURED routing distribution:")
        for b in benchmarks:
            p(f"  {b}: {all_results[b]['cured']['routing']}")

    p("=" * 84)


def print_api_results_table(api_mode: str, api_model: str, all_results: dict[str, Any]) -> None:
    benchmarks = list(all_results.keys())
    if not benchmarks:
        return

    protocols = list(next(iter(all_results.values())).keys())
    label_w = max(14, *(len(pn) for pn in protocols))
    col_w = 16

    p("\n" + "=" * 84)
    p(f"CURED API RESULTS | mode={api_mode} model={api_model}")
    p("=" * 84)

    header = f"{'protocol':<{label_w}}"
    for b in benchmarks:
        header += f"  {b:>{col_w}}"
    p(header)
    p("-" * len(header))

    for proto in protocols:
        row = f"{proto:<{label_w}}"
        for b in benchmarks:
            acc = all_results[b][proto]["accuracy"]
            cell = "n/a" if acc is None else f"{acc:.1%}"
            row += f"  {cell:>{col_w}}"
        p(row)

    if "greedy" in protocols:
        p("\nDelta vs greedy:")
        for proto in protocols:
            if proto == "greedy":
                continue
            row = f"  {proto:<{label_w-2}}"
            for b in benchmarks:
                a = all_results[b][proto]["accuracy"]
                g = all_results[b]["greedy"]["accuracy"]
                if a is None or g is None:
                    cell = "n/a"
                else:
                    d = a - g
                    sign = "+" if d >= 0 else ""
                    cell = f"{sign}{d:.1%}"
                row += f"  {cell:>{col_w}}"
            p(row)

    if "cured_api" in protocols:
        p("\nCURED_API routing distribution:")
        for b in benchmarks:
            p(f"  {b}: {all_results[b]['cured_api']['routing']}")

    p("=" * 84)


# ---------------------------------------------------------------------------
# Phase 5: CLI entry point (parse_args, main)
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CURED standalone router and evaluator")

    # Local model
    parser.add_argument("--model", default="", help="HuggingFace model id or local path")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use bitsandbytes 4-bit loading")

    # API mode
    parser.add_argument(
        "--api-mode",
        default="none",
        choices=["none", "groq", "gemini", "cloudflare", "openrouter", "foundry"],
        help="Use online inference backend instead of local model",
    )
    parser.add_argument(
        "--api-model",
        default="",
        help="Comma-separated API model ids when --api-mode is set",
    )
    parser.add_argument(
        "--api-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for API generation",
    )
    parser.add_argument(
        "--openrouter-api-keys",
        default="",
        help="Optional comma/semicolon-separated OpenRouter backup keys for this run",
    )
    parser.add_argument(
        "--cloudflare-api-credentials",
        default="",
        help="Optional Cloudflare token/account pairs: token@account,token2@account2",
    )
    parser.add_argument(
        "--cloudflare-openrouter-fallback-model",
        default="",
        help="Optional OpenRouter model id used only if all Cloudflare credentials fail",
    )

    # Run mode
    parser.add_argument(
        "--protocols",
        default=DEFAULT_LOCAL_PROTOCOLS,
        help=(
            "Comma-separated protocols. "
            f"Local options: {','.join(ALL_PROTOCOLS)} | "
            f"API options: {','.join(API_PROTOCOLS)}"
        ),
    )
    parser.add_argument(
        "--benchmark",
        default="both",
        choices=["truthfulqa", "medhallu", "both", "custom"],
        help="Benchmark source",
    )
    parser.add_argument("--n", type=int, default=50, help="Questions per benchmark")

    # Custom data
    parser.add_argument("--custom-csv", default="", help="Path to custom CSV for --benchmark custom")
    parser.add_argument("--question-col", default="question", help="Question column in custom CSV")
    parser.add_argument("--answer-col", default="answer", help="Answer/reference column in custom CSV")
    parser.add_argument(
        "--allow-missing-reference",
        action="store_true",
        help="Allow custom CSV rows without references (accuracy becomes n/a for those rows)",
    )

    # Single-question quick test
    parser.add_argument("--question", default="", help="Route and answer one question, then exit")

    # Decoding/eval knobs
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--cosine-threshold", type=float, default=DEFAULT_COSINE_THRESHOLD)
    parser.add_argument(
        "--scoring",
        default="cosine",
        choices=["cosine", "letter", "yesno", "mc"],
        help="Reference scoring mode for evaluation.",
    )
    parser.add_argument("--selfcheck-k", type=int, default=4)

    # Calibration controls
    parser.add_argument("--r2-questions", type=int, default=DEFAULT_R2_QUESTIONS)
    parser.add_argument("--d2h-questions", type=int, default=DEFAULT_D2H_QUESTIONS)
    parser.add_argument("--skip-iti", action="store_true", help="Skip ITI probe training/loading")
    parser.add_argument("--iti-train-questions", type=int, default=120)
    parser.add_argument("--iti-top-k-heads", type=int, default=ITI_TOP_K_HEADS)
    parser.add_argument("--force-recalibrate", action="store_true")

    # Output/cache
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT), help="Calibration/probe cache root")
    parser.add_argument("--out", default="", help="Optional output JSON path")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable any dataset shuffling (deterministic question order)")

    # Per-question feature logging (required for calibrate_router.py and McNemar tests)
    parser.add_argument("--save-per-question", action="store_true",
                        help="Embed per-question feature vectors (r2_q, kappa_q, ecr_q, h_final, sc_q) in output JSON")

    # Router v2 controls
    parser.add_argument("--model-params-b", type=float, default=0.0,
                        help="Model size in billions (e.g. 3.0, 8.0, 14.0, 32.0) — used by CUREDRouterV2")
    parser.add_argument("--router", choices=["old", "new"], default="old",
                        help="old = original CUREDRouter; new = 5-gate CUREDRouterV2")
    parser.add_argument("--router-config", default="",
                        help="Path to router_thresholds.json (used with --router new)")
    parser.add_argument("--compute-sc", action="store_true",
                        help="Compute self-consistency (k=3 samples) for routing (models ≤14B only)")
    parser.add_argument(
        "--scorer-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="SentenceTransformer device: auto keeps scorer on CPU for 4-bit and models ≥12B params.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Apply seed for reproducibility — required for valid McNemar pairing across runs
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    configure_torch_backends_for_inference()

    if args.openrouter_api_keys.strip():
        os.environ["OPENROUTER_API_KEYS"] = args.openrouter_api_keys.strip()
    if args.cloudflare_api_credentials.strip():
        os.environ["CLOUDFLARE_API_CREDENTIALS"] = args.cloudflare_api_credentials.strip()
    if args.cloudflare_openrouter_fallback_model.strip():
        os.environ["CLOUDFLARE_OPENROUTER_FALLBACK_MODEL"] = args.cloudflare_openrouter_fallback_model.strip()

    if args.api_mode != "none" and args.protocols == DEFAULT_LOCAL_PROTOCOLS:
        args.protocols = DEFAULT_API_PROTOCOLS

    protocols = parse_csv_list(args.protocols)
    if not protocols:
        raise ValueError("No protocols provided. Use --protocols with at least one item.")

    cache_root = Path(args.cache_root).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    if args.api_mode != "none":
        unknown = [pname for pname in protocols if pname not in API_PROTOCOLS]
        if unknown:
            raise ValueError(f"Unknown API protocols: {unknown}. Valid: {API_PROTOCOLS}")

        if str(args.scoring) == "mc":
            raise ValueError("--scoring mc is only supported in local model mode (--api-mode none).")

        api_models = parse_csv_list(args.api_model)
        if not api_models:
            raise ValueError("--api-model is required when --api-mode is set.")

        p(f"Running in API mode: backend={args.api_mode} models={api_models}")

        if args.question.strip():
            question = args.question.strip()
            p(f"\nQuestion: {question}")
            for api_model in api_models:
                router = CUREDAPIRouter(args.api_mode, api_model, api_temperature=args.api_temperature)
                out = router.route(question, max_new_tokens=args.max_new_tokens, scoring=str(args.scoring))
                p(f"\nModel:    {api_model}")
                p(f"Answer:   {out['text']}")
                p(f"Strategy: {out['strategy']} | domain={out.get('domain')}")
            return

        sdev = resolve_scorer_device(str(args.scorer_device), False, 0.0)
        p(f"Loading sentence scorer ({sdev})...")
        scorer = SentenceTransformer(DEFAULT_SCORER, device=sdev)

        benchmark_data: dict[str, list[dict[str, Any]]] = {}
        if args.benchmark in ("both", "truthfulqa"):
            benchmark_data["truthfulqa"] = load_truthfulqa(args.n, scoring=str(args.scoring))
            p(f"Loaded TruthfulQA: n={len(benchmark_data['truthfulqa'])}")
        if args.benchmark in ("both", "medhallu"):
            benchmark_data["medhallu"] = load_medhallu_generation(args.n)
            p(f"Loaded MedHallu: n={len(benchmark_data['medhallu'])}")
        if args.benchmark == "custom":
            if not args.custom_csv:
                raise ValueError("--benchmark custom requires --custom-csv")
            benchmark_data["custom"] = load_custom_csv(
                path=args.custom_csv,
                question_col=args.question_col,
                answer_col=args.answer_col,
                n=args.n,
                allow_missing_reference=args.allow_missing_reference,
            )
            p(f"Loaded custom CSV: n={len(benchmark_data['custom'])}")

        by_model_results: dict[str, dict[str, Any]] = {}
        by_model_per_question: dict[str, dict[str, Any]] = {}

        for api_model in api_models:
            p(f"\n{'#'*84}\nAPI model: {api_model}")
            api_router = CUREDAPIRouter(args.api_mode, api_model, api_temperature=args.api_temperature)

            all_results: dict[str, dict[str, Any]] = {b: {} for b in benchmark_data}
            for bench, samples in benchmark_data.items():
                p(f"\n{'='*62}\nBenchmark: {bench} (n={len(samples)})")
                for protocol in protocols:
                    p(f"\n  Protocol: {protocol}")
                    result = run_api_protocol(
                        api_mode=args.api_mode,
                        api_model=api_model,
                        api_router=api_router,
                        scorer=scorer,
                        protocol=protocol,
                        samples=samples,
                        cosine_threshold=float(args.cosine_threshold),
                        scoring=str(args.scoring),
                        max_new_tokens=int(args.max_new_tokens),
                        api_temperature=float(args.api_temperature),
                    )
                    all_results[bench][protocol] = result
                    acc = result["accuracy"]
                    acc_str = "n/a" if acc is None else f"{acc:.1%}"
                    p(
                        f"  -> {protocol}: acc={acc_str} rep={result['rep_rate']:.1%} "
                        f"runtime={result['runtime_min']:.1f}min"
                    )

            print_api_results_table(args.api_mode, api_model, all_results)

            by_model_results[api_model] = {}
            by_model_per_question[api_model] = {}
            for bench, proto_map in all_results.items():
                by_model_results[api_model][bench] = {}
                by_model_per_question[api_model][bench] = {}
                for protocol, result in proto_map.items():
                    compact = dict(result)
                    compact.pop("per_question", None)
                    by_model_results[api_model][bench][protocol] = compact
                    by_model_per_question[api_model][bench][protocol] = result.get("per_question", [])

        payload = {
            "api_mode": args.api_mode,
            "api_models": api_models,
            "protocols": protocols,
            "benchmark": args.benchmark,
            "n_target": int(args.n),
            "max_new_tokens": int(args.max_new_tokens),
            "api_temperature": float(args.api_temperature),
            "cosine_threshold": float(args.cosine_threshold),
            "scoring": str(args.scoring),
            "results": by_model_results,
        }

        ts = int(time.time())
        out_tag = safe_model_name(f"api_{args.api_mode}")
        summary_out = Path(args.out) if args.out else (cache_root / f"results_{out_tag}_{ts}.json")
        log_out = cache_root / f"per_question_{out_tag}_{ts}.json"

        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log_out.write_text(json.dumps(by_model_per_question, indent=2), encoding="utf-8")

        p(f"\nSaved summary: {summary_out}")
        p(f"Saved per-question log: {log_out}")
        return

    if not args.model.strip():
        raise ValueError("--model is required when --api-mode is none.")

    unknown = [pname for pname in protocols if pname not in ALL_PROTOCOLS]
    if unknown:
        raise ValueError(f"Unknown protocols: {unknown}. Valid: {ALL_PROTOCOLS}")

    model_cache = cache_root / safe_model_name(args.model)
    model_cache.mkdir(parents=True, exist_ok=True)
    p(f"Cache directory: {model_cache}")

    model, tokenizer = load_model_and_tokenizer(args.model, args.load_in_4bit)
    arch = get_arch(model)
    p(f"Architecture: {arch}")

    sdev = resolve_scorer_device(str(args.scorer_device), bool(args.load_in_4bit), float(args.model_params_b))
    p(f"Loading sentence scorer ({sdev})...")
    scorer = SentenceTransformer(DEFAULT_SCORER, device=sdev)

    cal_path = model_cache / "calibration.json"
    if cal_path.exists() and not args.force_recalibrate:
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        mean_r2 = float(cal["mean_r2"])
        d2h_threshold = float(cal["d2h_threshold"])
        p(f"Loaded cached calibration: R2={mean_r2:.4f} d2H={d2h_threshold:.4f}")
    else:
        mean_r2 = measure_r2(model, tokenizer, n_questions=args.r2_questions)
        d2h_threshold = calibrate_d2h(model, tokenizer, n_questions=args.d2h_questions)
        cal = {
            "model": args.model,
            "mean_r2": mean_r2,
            "d2h_threshold": d2h_threshold,
            "timestamp": int(time.time()),
        }
        cal_path.write_text(json.dumps(cal, indent=2), encoding="utf-8")
        p(f"Saved calibration -> {cal_path}")

    top_heads = None
    head_vectors = None
    need_iti = ("iti" in protocols) or ("cured" in protocols and not args.skip_iti)
    if need_iti and not args.skip_iti:
        top_path = model_cache / "iti_top_heads.npy"
        vec_path = model_cache / "iti_head_vectors.npy"
        if top_path.exists() and vec_path.exists() and not args.force_recalibrate:
            top_heads = np.load(top_path)
            head_vectors = np.load(vec_path)
            p(f"Loaded cached ITI probes: {len(top_heads)} heads")
        else:
            top_heads, head_vectors = train_iti_probes(
                model,
                tokenizer,
                arch,
                model_cache,
                max_questions=args.iti_train_questions,
                top_k_heads=args.iti_top_k_heads,
            )
    elif args.skip_iti:
        p("ITI disabled by --skip-iti")

    router = CUREDRouter(
        model=model,
        tokenizer=tokenizer,
        arch=arch,
        mean_r2=mean_r2,
        d2h_threshold=d2h_threshold,
        top_heads=top_heads,
        head_vectors=head_vectors,
    )

    p(
        "Router ready: "
        f"R2={router.mean_r2:.3f} ALTA={'ON' if router.alta_viable else 'OFF'} "
        f"ITI={'ON' if router.iti_available else 'OFF'} d2H={router.d2h_threshold:.4f}"
    )

    # Select active router for the "cured" protocol
    if args.router == "new":
        thresholds: dict[str, Any] = {}
        if args.router_config and Path(args.router_config).exists():
            thresholds = json.loads(Path(args.router_config).read_text(encoding="utf-8"))
        _cured_router: Any = CUREDRouterV2(
            model=model,
            tokenizer=tokenizer,
            arch=arch,
            model_params_B=args.model_params_b,
            top_heads=top_heads,
            head_vectors=head_vectors,
            thresholds=thresholds,
            compute_sc=args.compute_sc,
        )
        p(f"Using CUREDRouterV2 (params_B={args.model_params_b}, compute_sc={args.compute_sc})")
    else:
        _cured_router = router

    if args.question.strip():
        question = args.question.strip()
        out = router.route(question, max_new_tokens=args.max_new_tokens, scoring=str(args.scoring))
        p(f"\nQuestion: {question}")
        p(f"Answer:   {out['text']}")
        p(f"Strategy: {out['strategy']} | domain={out.get('domain')}")
        if out.get("d2H") is not None:
            p(f"d2H={out['d2H']} threshold={router.d2h_threshold:.4f}")
        if out.get("alta_gate") is not None:
            p(f"ALTA_gate={out['alta_gate']}")
        return

    benchmark_data: dict[str, list[dict[str, Any]]] = {}
    if args.benchmark in ("both", "truthfulqa"):
        benchmark_data["truthfulqa"] = load_truthfulqa(args.n, scoring=str(args.scoring))
        p(f"Loaded TruthfulQA: n={len(benchmark_data['truthfulqa'])}")
    if args.benchmark in ("both", "medhallu"):
        benchmark_data["medhallu"] = load_medhallu_generation(args.n)
        p(f"Loaded MedHallu: n={len(benchmark_data['medhallu'])}")
    if args.benchmark == "custom":
        if not args.custom_csv:
            raise ValueError("--benchmark custom requires --custom-csv")
        benchmark_data["custom"] = load_custom_csv(
            path=args.custom_csv,
            question_col=args.question_col,
            answer_col=args.answer_col,
            n=args.n,
            allow_missing_reference=args.allow_missing_reference,
        )
        p(f"Loaded custom CSV: n={len(benchmark_data['custom'])}")

    all_results: dict[str, dict[str, Any]] = {b: {} for b in benchmark_data}
    for bench, samples in benchmark_data.items():
        p(f"\n{'='*62}\nBenchmark: {bench} (n={len(samples)})")
        for protocol in protocols:
            p(f"\n  Protocol: {protocol}")
            result = run_protocol(
                model=model,
                tokenizer=tokenizer,
                arch=arch,
                router=_cured_router,
                scorer=scorer,
                protocol=protocol,
                samples=samples,
                cosine_threshold=float(args.cosine_threshold),
                scoring=str(args.scoring),
                max_new_tokens=int(args.max_new_tokens),
                selfcheck_k=int(args.selfcheck_k),
                iti_probes=(top_heads, head_vectors),
                save_per_question=bool(args.save_per_question),
                load_in_4bit=bool(args.load_in_4bit),
                model_params_b=float(args.model_params_b),
            )
            all_results[bench][protocol] = result
            acc = result["accuracy"]
            acc_str = "n/a" if acc is None else f"{acc:.1%}"
            p(
                f"  -> {protocol}: acc={acc_str} rep={result['rep_rate']:.1%} "
                f"runtime={result['runtime_min']:.1f}min"
            )

    print_results_table(args.model, mean_r2, d2h_threshold, all_results)

    payload = {
        "model": args.model,
        "load_in_4bit": bool(args.load_in_4bit),
        "protocols": protocols,
        "benchmark": args.benchmark,
        "n_target": int(args.n),
        "seed": int(args.seed),
        "router_version": str(args.router),
        "model_params_b": float(args.model_params_b),
        "max_new_tokens": int(args.max_new_tokens),
        "cosine_threshold": float(args.cosine_threshold),
        "scoring": str(args.scoring),
        "calibration": {
            "mean_r2": mean_r2,
            "alta_viable": bool(router.alta_viable),
            "d2h_threshold": d2h_threshold,
            "iti_available": bool(router.iti_available),
        },
        "results": {},
    }

    per_question_log: dict[str, dict[str, Any]] = {}
    for bench, proto_map in all_results.items():
        payload["results"][bench] = {}
        per_question_log[bench] = {}
        for protocol, result in proto_map.items():
            compact = dict(result)
            pq_data = compact.pop("per_question", [])
            if args.save_per_question:
                compact["per_question"] = pq_data
            payload["results"][bench][protocol] = compact
            per_question_log[bench][protocol] = pq_data

    ts = int(time.time())
    summary_out = Path(args.out) if args.out else (model_cache / f"results_{ts}.json")
    log_out = model_cache / f"per_question_{ts}.json"

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_out.write_text(json.dumps(per_question_log, indent=2), encoding="utf-8")

    p(f"\nSaved summary: {summary_out}")
    p(f"Saved per-question log: {log_out}")


if __name__ == "__main__":
    main()
