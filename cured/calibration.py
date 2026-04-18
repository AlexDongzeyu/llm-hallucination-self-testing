"""
cured.calibration — Model-level and question-level feature calibration.

Calibration routines measure per-model properties that control the CURED
routing decision.  Results are cached in ~/.cured_cache/<model_name>/.

Key functions
-------------
measure_r2            Calibrate mean late-layer R² on TruthfulQA (n=15 default).
compute_ecr           Compute Entropy Compression Ratio H_final / H_peak.
_compute_layer_features Single-pass (r2, var_r2, kappa) from late-layer logit matrix.
compute_per_question_r2 Thin wrapper returning (r2_mean, r2_var).
compute_curvature     Thin wrapper returning κ (quadratic gain fraction).
compute_self_consistency Sample k responses, return SC score + modal answer.
compute_semantic_entropy Cluster k sampled responses, return H(cluster distribution).
train_iti_probes      Fit logistic probes for ITI head selection.
calibrate_d2h         Calibrate Δ2H threshold on a held-out question set.

R² and κ definitions
---------------------
R² measures how linearly the top-k token logits grow across transformer layers.
High R² (≥ 0.55) → model's residual stream is well-behaved → ALTA is effective.

κ = (R²_quad − R²_lin) / (1 − R²_lin + ε) measures the *curvature* of that
growth.  High κ → logit trajectory bends significantly → ECR entropy gate fires.

ECR definition
--------------
ECR = H_final / H_peak
where H_layer = −∑ p_i log p_i over the softmax of layer logits.

Low ECR → the model compresses entropy toward the final layer (high confidence).
ECR < τ_ECR → Gate 2 fires → route to ALTA.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# NOTE: This module re-imports from cured.py (the canonical single-file script)
# until the implementations are migrated here as a later refactor step.
# ---------------------------------------------------------------------------

import sys
import os
import importlib.util


def _cured():
    """Load (or return cached) the top-level cured.py module."""
    key = "_cured_script"
    if key not in sys.modules:
        script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cured.py")
        spec = importlib.util.spec_from_file_location(key, script)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def measure_r2(model, tokenizer, n_questions: int = 15) -> float:
    """Calibrate mean late-layer R² on TruthfulQA.

    Loads n_questions from the TruthfulQA validation split, runs a forward
    pass with hidden states, fits a linear model to the top-50 logit
    trajectories, and returns the mean R².

    Args:
        model:       HuggingFace CausalLM loaded on GPU.
        tokenizer:   Matching tokenizer.
        n_questions: Number of calibration questions (default 15).

    Returns:
        float: Mean R² across calibration questions and top-50 tokens.
               Values ≥ 0.55 indicate ALTA is reliable for this model.
    """
    return _cured().measure_r2(model, tokenizer, n_questions)


def _compute_layer_features(
    hidden_states,
    lm_head,
    norm,
    layer_start_ratio: float = 0.7,
    top_k: int = 50,
) -> tuple[float, float, float]:
    """Single-pass computation of (mean_r2, var_r2, kappa) from late-layer logits.

    Extracts the (L, vocab) logit matrix once from the bottom ``layer_start_ratio``
    of transformer layers and reuses it for both R² and κ computation, avoiding
    redundant GPU→CPU transfers.

    Args:
        hidden_states:     Tuple of hidden state tensors, one per transformer layer.
                           Must have embedding layer removed (i.e. out.hidden_states[1:]).
        lm_head:           Language model head (linear projection to vocabulary).
        norm:              Final layer-norm module.
        layer_start_ratio: Fraction of layers to skip from the start (default 0.7,
                           meaning only the top 30% of layers are used).
        top_k:             Number of highest-logit tokens to compute R²/κ over.

    Returns:
        tuple(r2_mean, r2_var, kappa):
            r2_mean (float): Mean R² across top_k tokens.
            r2_var  (float): Variance of R² across top_k tokens.
            kappa   (float): Mean quadratic-gain fraction κ across top_k tokens.
    """
    return _cured()._compute_layer_features(
        hidden_states, lm_head, norm, layer_start_ratio, top_k
    )


def compute_per_question_r2(
    hidden_states,
    lm_head,
    norm,
    layer_start_ratio: float = 0.7,
    top_k: int = 50,
) -> tuple[float, float]:
    """Return (r2_mean, r2_var) — thin wrapper around _compute_layer_features.

    Args:
        hidden_states:     See _compute_layer_features.
        lm_head:           See _compute_layer_features.
        norm:              See _compute_layer_features.
        layer_start_ratio: See _compute_layer_features.
        top_k:             See _compute_layer_features.

    Returns:
        tuple(r2_mean, r2_var).
    """
    return _cured().compute_per_question_r2(hidden_states, lm_head, norm, layer_start_ratio, top_k)


def compute_curvature(
    hidden_states,
    lm_head,
    norm,
    layer_start_ratio: float = 0.7,
    top_k: int = 50,
) -> float:
    """Return κ (quadratic gain fraction) — thin wrapper around _compute_layer_features.

    Args:
        hidden_states:     See _compute_layer_features.
        lm_head:           See _compute_layer_features.
        norm:              See _compute_layer_features.
        layer_start_ratio: See _compute_layer_features.
        top_k:             See _compute_layer_features.

    Returns:
        float: κ in [0, 1]; higher values indicate more curvature in the logit trajectory.
    """
    return _cured().compute_curvature(hidden_states, lm_head, norm, layer_start_ratio, top_k)


def compute_ecr(hidden_states, lm_head, norm) -> tuple[float, float, float]:
    """Compute the Entropy Compression Ratio (ECR) across all transformer layers.

    ECR = H_final / H_peak, where H_layer = −∑ p_i log p_i over the softmax
    of layer-projected logits.

    Low ECR indicates that the model has compressed its prediction distribution
    by the final layer, which is correlated with high confidence.  The CURED
    Gate 2 uses ECR < τ_ECR to decide whether ALTA is beneficial.

    Args:
        hidden_states: Tuple of hidden state tensors (embedding layer removed,
                       i.e. out.hidden_states[1:]).  hidden_states[0] = layer 1,
                       hidden_states[-1] = final transformer layer.
        lm_head:       Language model head (linear projection to vocabulary).
        norm:          Final layer-norm module.

    Returns:
        tuple(ECR, H_final, H_peak):
            ECR     (float): H_final / H_peak ratio.
            H_final (float): Entropy at the final transformer layer.
            H_peak  (float): Maximum entropy across all layers.
    """
    return _cured().compute_ecr(hidden_states, lm_head, norm)


def compute_self_consistency(
    model,
    tokenizer,
    prompt: str,
    k: int = 3,
    temperature: float = 0.7,
    max_new_tokens: int = 40,
) -> tuple[float, str]:
    """Sample k responses and return (SC_score, modal_answer).

    Args:
        model:          HuggingFace CausalLM.
        tokenizer:      Matching tokenizer.
        prompt:         Full formatted prompt string.
        k:              Number of stochastic samples (default 3).
        temperature:    Sampling temperature (default 0.7).
        max_new_tokens: Maximum new tokens per sample.

    Returns:
        tuple(sc_score, modal_answer):
            sc_score     (float): Fraction of k samples agreeing with modal answer.
            modal_answer (str):   Most frequently generated response.
    """
    return _cured().compute_self_consistency(model, tokenizer, prompt, k, temperature, max_new_tokens)


def compute_semantic_entropy(
    model,
    tokenizer,
    prompt: str,
    scorer,
    k: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 40,
) -> float:
    """Compute semantic entropy over k sampled responses via clustering.

    Generates k stochastic responses, clusters them by cosine similarity,
    and returns H = −∑ p_c log p_c over the cluster distribution.

    Args:
        model:          HuggingFace CausalLM.
        tokenizer:      Matching tokenizer.
        prompt:         Full formatted prompt string.
        scorer:         SentenceTransformer instance for embedding.
        k:              Number of stochastic samples (default 5).
        temperature:    Sampling temperature (default 0.7).
        max_new_tokens: Maximum new tokens per sample.

    Returns:
        float: Semantic entropy H in nats.  Higher → model is more uncertain.
    """
    return _cured().compute_semantic_entropy(model, tokenizer, prompt, scorer, k, temperature, max_new_tokens)


def train_iti_probes(model, tokenizer, n_calibration: int = 256):
    """Fit logistic probes on attention head activations for ITI head selection.

    Args:
        model:         HuggingFace CausalLM.
        tokenizer:     Matching tokenizer.
        n_calibration: Number of TruthfulQA questions to use for probe training.

    Returns:
        tuple(head_scores, head_vectors, top_heads): Arrays saved to data/.
    """
    return _cured().train_iti_probes(model, tokenizer, n_calibration)


def calibrate_d2h(model, tokenizer, n_questions: int = 50) -> float:
    """Calibrate the Δ2H threshold on a held-out question set.

    Args:
        model:       HuggingFace CausalLM.
        tokenizer:   Matching tokenizer.
        n_questions: Number of questions for threshold estimation (default 50).

    Returns:
        float: Calibrated d2H threshold value.
    """
    return _cured().calibrate_d2h(model, tokenizer, n_questions)
