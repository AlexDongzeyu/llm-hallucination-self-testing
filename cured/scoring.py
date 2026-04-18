"""
cured.scoring — Answer-scoring utilities.

Scoring functions map a model output string (or log-probability) to a
correctness signal for a specific answer format.

Scoring methods
---------------
cosine_match    semantic cosine similarity against reference answer
letter_match    multiple-choice letter extraction (A/B/C/D)
yesno_match     binary yes/no extraction
mc_score_sample TruthfulQA MC1/MC2 log-probability scoring
reference_match ensemble of cosine + keyword heuristics
"""

# ---------------------------------------------------------------------------
# NOTE: This module does NOT duplicate any logic.  It re-imports the
# canonical implementations from cured.py, which remains the single source
# of truth for the full monolithic script.  When cured.py is eventually
# split into per-module source files, the implementations will migrate here.
# ---------------------------------------------------------------------------

from __future__ import annotations

import sys
import importlib

def _import_from_script(name: str):
    """Lazily import a name from the top-level cured.py CLI script."""
    import importlib.util, os
    script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cured.py")
    spec = importlib.util.spec_from_file_location("_cured_script", script)
    mod = importlib.util.module_from_spec(spec)
    if "_cured_script" not in sys.modules:
        sys.modules["_cured_script"] = mod
        spec.loader.exec_module(mod)
    return getattr(sys.modules["_cured_script"], name)


def cosine_match(response: str, reference: str, scorer=None, threshold: float = 0.65) -> bool:
    """Return True if cosine similarity between response and reference exceeds threshold.

    Args:
        response:  Model-generated answer string.
        reference: Ground-truth reference answer.
        scorer:    SentenceTransformer instance (loaded lazily if None).
        threshold: Minimum cosine similarity to count as correct (default 0.65).

    Returns:
        bool: True if the response is semantically close enough to the reference.
    """
    return _import_from_script("cosine_match")(response, reference, scorer, threshold)


def letter_match(response: str, expected: str) -> bool:
    """Return True if the extracted option letter matches expected.

    Args:
        response: Raw model output (e.g. "The answer is B.").
        expected: Expected letter string, e.g. "B".

    Returns:
        bool: True if the extracted letter equals expected (case-insensitive).
    """
    return _import_from_script("letter_match")(response, expected)


def yesno_match(response: str, expected: str) -> bool:
    """Return True if the extracted yes/no label matches expected.

    Args:
        response: Raw model output.
        expected: "yes" or "no" (case-insensitive).

    Returns:
        bool: True if the extracted label matches.
    """
    return _import_from_script("yesno_match")(response, expected)


def mc_score_sample(
    model,
    tokenizer,
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

    Args:
        model:        HuggingFace CausalLM (already loaded, on GPU).
        tokenizer:    Matching tokenizer.
        question:     Question text used as the prompt prefix.
        choices:      List of answer strings for MC1.
        labels:       Binary label list (1=correct, 0=incorrect) for MC1 choices.
        choices_mc2:  Answer strings for MC2 (falls back to choices if None).
        labels_mc2:   Binary labels for MC2 choices (falls back to labels if None).
        mc_protocol:  "greedy" uses standard log-probs; "alta" uses ALTA-weighted
                      log-probs.

    Returns:
        dict with keys:
            "mc1" (float): 1.0 if the highest-log-prob choice is correct, else 0.0.
            "mc2" (float): fraction of probability mass assigned to correct choices.
    """
    return _import_from_script("mc_score_sample")(
        model, tokenizer, question, choices, labels, choices_mc2, labels_mc2, mc_protocol
    )


def reference_match(response: str, reference: str, scorer=None) -> bool:
    """Ensemble scorer: cosine similarity + keyword heuristics.

    Args:
        response:  Model-generated answer string.
        reference: Ground-truth reference answer.
        scorer:    SentenceTransformer instance (loaded lazily if None).

    Returns:
        bool: True if either cosine or keyword heuristic signals correctness.
    """
    return _import_from_script("reference_match")(response, reference, scorer)
