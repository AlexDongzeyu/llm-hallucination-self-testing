"""
cured.protocols — Decoding protocol implementations.

Each protocol is a self-contained generation strategy that accepts a loaded
model/tokenizer and returns a text response.

Protocols
---------
greedy_generate        Standard greedy/top-1 decoding.
batch_greedy_generate  Batched greedy decoding for throughput.
alta_generate          ALTA: Anisotropic Logit Trajectory Amplification.
                       Boosts logits in the direction of maximum late-layer
                       R² improvement.  Gate: R² ≥ τ_R2 and ECR < τ_ECR.
cove_generate          Chain-of-Verification (CoVe): generates verification
                       questions and re-answers; restricted to medical domain.
api_cove_generate      API-backed variant of CoVe.
iti_generate           Inference-Time Intervention: steers attention heads
                       toward truthful activation directions.
selfcheck_generate     SelfCheckGPT-style consistency check.
delta_dola_generate    Δ-DoLa: contrastive decoding between early and late layers.

Usage
-----
All local protocols share the signature:
    fn(model, tokenizer, prompt, max_new_tokens=80) -> dict | str

The returned dict always contains at least {"text": str}.
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


def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> dict:
    """Standard greedy (temperature=0) decoding.

    Args:
        model:          HuggingFace CausalLM.
        tokenizer:      Matching tokenizer.
        prompt:         Full formatted prompt string.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        dict with key "text" (str).
    """
    return _cured().greedy_generate(model, tokenizer, prompt, max_new_tokens)


def batch_greedy_generate(model, tokenizer, prompts: list[str], max_new_tokens: int = 80) -> list[dict]:
    """Batched greedy decoding for multiple prompts.

    Args:
        model:          HuggingFace CausalLM.
        tokenizer:      Matching tokenizer.
        prompts:        List of formatted prompt strings.
        max_new_tokens: Maximum tokens to generate per prompt.

    Returns:
        List of dicts, each with key "text" (str).
    """
    return _cured().batch_greedy_generate(model, tokenizer, prompts, max_new_tokens)


def alta_generate(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> dict:
    """ALTA: Anisotropic Logit Trajectory Amplification.

    Amplifies logits in the direction that maximally increases R² across
    transformer layers, effectively suppressing the model's residual uncertainty.

    Gate condition: R² ≥ τ_R2 (≈ 0.50).  Below this, ALTA is unreliable and
    falls back to greedy.

    Args:
        model:          HuggingFace CausalLM.
        tokenizer:      Matching tokenizer.
        prompt:         Full formatted prompt string.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        dict with key "text" (str) and optionally "r2" (float).
    """
    return _cured().alta_generate(model, tokenizer, prompt, max_new_tokens)


def cove_generate(model, tokenizer, question: str, max_new_tokens: int = 80) -> str:
    """Chain-of-Verification (CoVe) decoding.

    Generates an initial answer, derives verification questions, re-answers
    each, and synthesises a revised final answer.

    DOMAIN RESTRICTION: CoVe is enabled only for the medical domain
    (detected via MEDICAL_KEYWORDS / MED_PATTERN).  On general QA it performs
    no better than greedy but costs ~3× the compute.

    Args:
        model:          HuggingFace CausalLM.
        tokenizer:      Matching tokenizer.
        question:       Raw question text (not formatted as a prompt).
        max_new_tokens: Maximum tokens per generation step.

    Returns:
        str: Final revised answer text.
    """
    return _cured().cove_generate(model, tokenizer, question, max_new_tokens)


def api_cove_generate(question: str, api_fn, max_new_tokens: int = 80) -> str:
    """API-backed Chain-of-Verification.

    Args:
        question:       Raw question text.
        api_fn:         Callable(prompt) -> str that calls an external LLM API.
        max_new_tokens: Maximum tokens per generation step.

    Returns:
        str: Final revised answer text.
    """
    return _cured().api_cove_generate(question, api_fn, max_new_tokens)


def iti_generate(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> dict:
    """Inference-Time Intervention (ITI) decoding.

    Steers the top-K attention heads toward the truthful activation direction
    learned during probe calibration (train_iti_probes).

    Requires ITI probes to be pre-trained and saved to data/.

    Args:
        model:          HuggingFace CausalLM.
        tokenizer:      Matching tokenizer.
        prompt:         Full formatted prompt string.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        dict with key "text" (str).
    """
    return _cured().iti_generate(model, tokenizer, prompt, max_new_tokens)


def selfcheck_generate(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> dict:
    """SelfCheckGPT-style consistency checking.

    Generates multiple stochastic responses and uses their mutual consistency
    as a hallucination signal, returning the most consistent answer.

    Args:
        model:          HuggingFace CausalLM.
        tokenizer:      Matching tokenizer.
        prompt:         Full formatted prompt string.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        dict with keys "text" (str) and "selfcheck_score" (float).
    """
    return _cured().selfcheck_generate(model, tokenizer, prompt, max_new_tokens)


def delta_dola_generate(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> dict:
    """Δ-DoLa: contrastive decoding between early and late transformer layers.

    Subtracts early-layer logits from late-layer logits to amplify factual
    tokens that emerge in deeper layers.

    Args:
        model:          HuggingFace CausalLM.
        tokenizer:      Matching tokenizer.
        prompt:         Full formatted prompt string.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        dict with key "text" (str).
    """
    return _cured().delta_dola_generate(model, tokenizer, prompt, max_new_tokens)
