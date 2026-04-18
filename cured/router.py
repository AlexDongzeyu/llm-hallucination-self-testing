"""
cured.router — CURED principled 5-gate routing framework.

The router decides, on a per-question basis, which decoding protocol to invoke.
Three router classes are provided:

CUREDRouterV2 (primary)
    5-gate principled router using per-question trajectory features.

    Gate flow (executed top-to-bottom, first match wins):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Gate 1 — Easy:   H_final < τ_H_easy (+ SC for ≤14B if computed) → greedy_confident │
    │ Scale:   profile_mean_r2 ≥ 0.55, non-medical, H_final > τ_H_easy → ALTA shortcut   │
    │ Gate 2 — Feat.:  R²_q > τ_R2, κ_q < τ_kappa, ECR_q > τ_ECR  →  continue to 3–5    │
    │ Gate 3 — ITI:    domain_medical and ITI available  →  ITI       │
    │ Gate 4 — Score:  composite S_ALTA > 0.5  →  ALTA                 │
    │ Gate 5 — CoVe:   domain_medical and SC > 0.5  →  CoVe          │
    │          else:   greedy                                         │
    └─────────────────────────────────────────────────────────────────┘

    Default thresholds (from configs/router_thresholds.json):
        tau_kappa = 0.70   (calibrated from 8B profile; mean kappa ≈ 0.597)
        tau_ECR   = 0.04   (calibrated from 8B profile; mean ECR ≈ 0.031–0.076)
        tau_R2    = 0.65
        tau_H_easy = 0.5
        tau_H_hard = 3.0
        profile_mean_r2 = 0.582

CUREDRouter (legacy v1)
    Older d2H-based router.  Kept for backwards compatibility.

CUREDAPIRouter
    API-backed router for cloud inference (Cloudflare, OpenRouter, Groq, Gemini).
"""

from __future__ import annotations
import sys, os, importlib.util


def _cured():
    key = "_cured_script"
    if key not in sys.modules:
        script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cured.py")
        spec = importlib.util.spec_from_file_location(key, script)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


class CUREDRouterV2:
    """5-gate principled router for local LLM inference.

    Args:
        model:         HuggingFace CausalLM (loaded, on GPU).
        tokenizer:     Matching tokenizer.
        thresholds:    Dict of threshold overrides.  Missing keys use defaults
                       from configs/router_thresholds.json or hard-coded fallbacks.
        compute_sc:    Whether to compute Self-Consistency score at Gate 5 (default True).

    Attributes:
        alta_globally_viable (bool): True when model's mean R² ≥ 0.55.
        profile_mean_r2      (float): Calibrated mean R² loaded from thresholds.
    """

    def __new__(cls, model, tokenizer, thresholds=None, compute_sc: bool = True):
        # Delegate construction to the canonical class in cured.py
        real_cls = _cured().CUREDRouterV2
        obj = real_cls.__new__(real_cls)
        real_cls.__init__(obj, model, tokenizer, thresholds, compute_sc)
        return obj

    def route(self, prompt: str, question: str, domain: str = "general",
              max_new_tokens: int = 80) -> dict:
        """Route a single question through the 5-gate decision tree.

        Args:
            prompt:         Full formatted prompt string.
            question:       Raw question text (used for domain detection and CoVe).
            domain:         "medical" or "general" (auto-detected if not provided).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            dict with keys:
                "text"     (str):  Generated answer.
                "strategy" (str):  Name of the selected protocol/gate.
                "domain"   (str):  Detected or provided domain.
                "gate"     (str):  Gate number that fired (e.g. "2").
        """
        raise NotImplementedError("Use the delegated instance returned by __new__")


class CUREDRouter:
    """Legacy d2H-based router (v1).

    Uses the Δ²H (second-difference of entropy) signal to distinguish easy
    from hard questions.  Superseded by CUREDRouterV2 but kept for comparisons.

    Args:
        model:     HuggingFace CausalLM.
        tokenizer: Matching tokenizer.
        d2h_threshold: Calibrated Δ²H threshold (default from calibrate_d2h()).
    """

    def __new__(cls, model, tokenizer, d2h_threshold: float = 0.0):
        real_cls = _cured().CUREDRouter
        obj = real_cls.__new__(real_cls)
        real_cls.__init__(obj, model, tokenizer, d2h_threshold)
        return obj


class CUREDAPIRouter:
    """API-backed CURED router for cloud inference.

    Routes questions to the best available API-based strategy (greedy, CoVe,
    or ensemble) depending on provider capabilities and rate limits.

    Args:
        api_config: Dict with provider credentials (see README for format).
    """

    def __new__(cls, api_config: dict):
        real_cls = _cured().CUREDAPIRouter
        obj = real_cls.__new__(real_cls)
        real_cls.__init__(obj, api_config)
        return obj
