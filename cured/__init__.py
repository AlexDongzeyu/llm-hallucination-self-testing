"""
cured — CURED Decoding Router package.

Public API surfaces for the four submodules:
  cured.router      — CUREDRouterV2, CUREDRouter, CUREDAPIRouter
  cured.protocols   — greedy_generate, alta_generate, cove_generate, iti_generate,
                      selfcheck_generate, delta_dola_generate
  cured.scoring     — cosine_match, letter_match, yesno_match, mc_score_sample
  cured.calibration — measure_r2, compute_ecr, train_iti_probes, calibrate_d2h

All symbols are also importable directly from this top-level package for
backwards-compatible usage:

    from cured import CUREDRouterV2, cosine_match, measure_r2
"""

from cured.router import CUREDRouter, CUREDRouterV2, CUREDAPIRouter
from cured.protocols import (
    greedy_generate,
    batch_greedy_generate,
    alta_generate,
    cove_generate,
    api_cove_generate,
    iti_generate,
    selfcheck_generate,
    delta_dola_generate,
)
from cured.scoring import (
    cosine_match,
    letter_match,
    yesno_match,
    mc_score_sample,
    reference_match,
)
from cured.calibration import (
    measure_r2,
    compute_ecr,
    compute_per_question_r2,
    compute_curvature,
    _compute_layer_features,
    compute_self_consistency,
    compute_semantic_entropy,
    train_iti_probes,
    calibrate_d2h,
)

__all__ = [
    # Router
    "CUREDRouter",
    "CUREDRouterV2",
    "CUREDAPIRouter",
    # Protocols
    "greedy_generate",
    "batch_greedy_generate",
    "alta_generate",
    "cove_generate",
    "api_cove_generate",
    "iti_generate",
    "selfcheck_generate",
    "delta_dola_generate",
    # Scoring
    "cosine_match",
    "letter_match",
    "yesno_match",
    "mc_score_sample",
    "reference_match",
    # Calibration
    "measure_r2",
    "compute_ecr",
    "compute_per_question_r2",
    "compute_curvature",
    "_compute_layer_features",
    "compute_self_consistency",
    "compute_semantic_entropy",
    "train_iti_probes",
    "calibrate_d2h",
]
