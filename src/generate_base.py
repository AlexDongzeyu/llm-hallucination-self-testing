"""
generate.py -- Step 4: Custom generation loop with curvature-entropy gating.

At each token step:
  1. Extract full logit trajectory across all 28 transformer layers
  2. Compute curvature (trajectory instability via quadratic fit)
  3. Compute entropy (vocabulary uncertainty via Shannon entropy)
  4. If BOTH exceed thresholds -> apply DoLA contrastive decoding
  5. Otherwise -> decode normally from final layer

Three fixes from Step 3 are baked in (see get_layer_logits docstring).
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Model loading — Llama-3.2-3B (BoN Phase 2) ───────────────────────────────

MODEL_NAME = "meta-llama/Llama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

print(f"Model loaded | Device: {next(model.parameters()).device} | "
      f"Layers: {model.config.num_hidden_layers} | Vocab: {model.config.vocab_size}")
# ── 4A: Layer logit extraction (KV-cache aware) ─────────────────────────────

def get_layer_logits_cached(input_ids: torch.Tensor,
                            past_key_values=None):
    """
    Forward pass with KV caching.
    First call : pass full input_ids, past_key_values=None
    Subsequent : pass only the NEW token tensor, pass cached past_key_values

    Returns:
      layer_logits    [num_layers, vocab_size]  — identical shape to before
      past_key_values — updated cache for next step
    """
    # Debug: confirm KV cache is being passed (only verbose on first two calls)
    if input_ids.shape[1] > 1 or past_key_values is None:
        pass  # first call with full prompt
    # else: single-token calls with active cache — KV is working
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True
        )

    hidden_states = outputs.hidden_states[1:]  # skip embedding layer
    norm = model.model.norm
    lm_head = model.lm_head

    all_layer_logits = []
    for layer_hidden in hidden_states:
        last_token = layer_hidden[:, -1, :]                              # [1, hidden_dim]
        normed     = norm(last_token)                                    # LayerNorm fix
        logits     = lm_head(normed)                                     # [1, vocab_size]
        logits_np  = logits.squeeze(0).detach().cpu().float().numpy()   # detach fix
        all_layer_logits.append(logits_np)

    return np.array(all_layer_logits), outputs.past_key_values


# ── 4B: Feature functions ────────────────────────────────────────────────────

def compute_curvature(trajectory: np.ndarray) -> float:
    """
    Quadratic coefficient of the logit trajectory across layers.
    Matches how curvature was computed in eval.py (verified in Step 2A).

    High |curvature| = trajectory bends sharply = model changing its mind
    """
    layers = np.arange(len(trajectory))
    coefficients = np.polyfit(layers, trajectory, deg=2)
    return float(coefficients[0])  # 'a' in ax^2 + bx + c


def compute_entropy(final_layer_logits: np.ndarray) -> float:
    """
    Shannon entropy of the final layer's probability distribution.

    High entropy = probability spread across many tokens = uncertain
    Low entropy  = probability peaked on one token = confident
    """
    shifted = final_layer_logits - np.max(final_layer_logits)  # numerical stability
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0))))


# ── 4B.1: Repetition penalty ─────────────────────────────────────────────────

def apply_repetition_penalty(logits: np.ndarray,
                              generated_ids: list,
                              penalty: float = 1.3) -> np.ndarray:
    """
    Penalize tokens that already appeared in the generated sequence.
    Divides their logit by `penalty` if positive, multiplies if negative.
    This is the standard Keskar et al. repetition penalty.

    penalty=1.0 = no effect, penalty>1.0 = discourage repetition.
    Only applied to generated tokens (not prompt tokens).
    """
    adjusted = logits.copy()
    for token_id in set(generated_ids):  # set() avoids double-counting
        if adjusted[token_id] > 0:
            adjusted[token_id] /= penalty
        else:
            adjusted[token_id] *= penalty
    return adjusted


# ── 4C: Gated generation loop (KV-cached — ~15x faster) ─────────────────────

def gated_generate(
    prompt: str,
    max_new_tokens: int = 80,
    curve_threshold: float = 0.05,
    entropy_threshold: float = 3.0,
    early_layer_idx: int = 12,
    contrast_alpha: float = 0.3,
    repetition_penalty: float = 1.3,
    gate_mode: str = "joint"
) -> dict:
    """
    KV-cached generation loop. Same logic as before, ~15x faster.
    First step: full forward pass over the prompt.
    Each subsequent step: only the ONE new token is passed; the rest comes from cache.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # First full forward pass — builds the KV cache for the prompt
    layer_logits, past_key_values = get_layer_logits_cached(input_ids, None)

    generated_token_ids = []
    token_metadata = []

    for step in range(max_new_tokens):

        final_logits = layer_logits[-1]
        early_logits = layer_logits[early_layer_idx]

        # Curvature and entropy — identical logic to before
        top_token_id   = int(np.argmax(final_logits))
        token_trajectory = layer_logits[:, top_token_id]

        if gate_mode == "late_curve":
            curvature = compute_curvature(token_trajectory[-8:])
        else:
            curvature = compute_curvature(token_trajectory)

        entropy = compute_entropy(final_logits)

        # Gate — identical logic to before
        if gate_mode == "entropy_only" or gate_mode == "sled_entropy":
            gate_fires = entropy > entropy_threshold
        elif gate_mode == "late_curve" or gate_mode == "sled_gated":
            gate_fires = (abs(curvature) > curve_threshold) and (entropy > entropy_threshold)
        else:
            gate_fires = (abs(curvature) > curve_threshold) and (entropy > entropy_threshold)

        # Intervention — identical logic to before
        if gate_fires:
            if "sled" in gate_mode:
                k = 8
                early_avg = np.mean(layer_logits[:k], axis=0)
                adjusted_logits = final_logits + contrast_alpha * (early_avg - final_logits)
            else:
                adjusted_logits = final_logits - contrast_alpha * early_logits
        else:
            adjusted_logits = final_logits

        # Repetition penalty
        adjusted_logits = apply_repetition_penalty(
            adjusted_logits, generated_token_ids, repetition_penalty
        )

        next_token_id = int(np.argmax(adjusted_logits))
        generated_token_ids.append(next_token_id)

        token_metadata.append({
            "step":       step,
            "token":      tokenizer.decode([next_token_id]),
            "curvature":  round(curvature, 4),
            "entropy":    round(entropy, 4),
            "gate_fired": gate_fires,
        })

        if next_token_id == tokenizer.eos_token_id:
            break

        # KV cache: next step only processes the ONE new token
        next_tensor = torch.tensor([[next_token_id]]).to(model.device)
        layer_logits, past_key_values = get_layer_logits_cached(
            next_tensor, past_key_values
        )

    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    gate_fire_rate = sum(m["gate_fired"] for m in token_metadata) / max(len(token_metadata), 1)

    return {
        "text":           generated_text,
        "metadata":       token_metadata,
        "gate_fire_rate": round(gate_fire_rate, 4)
    }


# ── 4D: Sanity check — run with 3B to verify KV cache output ────────────────

if __name__ == "__main__":

    print("Sanity check: KV-cached greedy generation on Llama-3.2-3B")
    result = gated_generate(
        "The capital of Canada is",
        max_new_tokens=20,
        curve_threshold=999.0,   # gate never fires — pure greedy
        entropy_threshold=999.0,
        gate_mode="joint"
    )
    print(f"Generated: {result['text']}")
    print(f"Gate fire rate: {result['gate_fire_rate']:.1%}  (should be 0.0%)")
    # Expected: " Ottawa, Ontario..." or similar factual completion
    # Gate fire rate must be 0.0% — proves gate is inactive when thresholds are 999
