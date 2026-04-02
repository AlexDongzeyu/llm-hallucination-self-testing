"""
diagnose_jsd.py
Run on each model (after swapping generate.py MODEL_NAME) to measure
Jensen-Shannon Divergence between each intermediate layer and the final layer.

High JSD early layers → model changes prediction significantly across depth
Low JSD everywhere  → layers are redundant, early signal is weak (explains SLED failure)
"""

import numpy as np
import torch
from generate import model, tokenizer, get_layer_logits_cached


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def compute_jsd(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """Jensen-Shannon Divergence between two logit distributions."""
    p = softmax(p_logits)
    q = softmax(q_logits)
    m = np.clip(0.5 * (p + q), 1e-10, 1.0)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return float(
        0.5 * np.sum(p * np.log(p / m)) +
        0.5 * np.sum(q * np.log(q / m))
    )


TEST_PROMPTS = [
    "The capital of France is",
    "The chemical symbol for gold is",
    "The author of Hamlet is",
    "The speed of light is approximately",
    "The largest planet in the solar system is",
]


if __name__ == "__main__":
    model_label = getattr(model.config, "_name_or_path", "unknown")
    n_layers    = model.config.num_hidden_layers
    print(f"\nDiagnosing JSD: {model_label} | layers={n_layers}")
    print("=" * 60)

    all_jsds = []   # list of (prompt, layer_idx, jsd)

    for prompt in TEST_PROMPTS:
        input_ids           = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        layer_logits, _     = get_layer_logits_cached(input_ids, None)
        final_logits        = layer_logits[-1]

        prompt_jsds = []
        for layer_idx in range(len(layer_logits) - 1):
            jsd = compute_jsd(layer_logits[layer_idx], final_logits)
            all_jsds.append((prompt[:30], layer_idx, jsd))
            prompt_jsds.append(jsd)

        early_mean = float(np.mean([j for _, li, j in all_jsds if li < 8 and _ == prompt[:30]]))
        print(f"  {prompt[:40]:<42} | mean_JSD={np.mean(prompt_jsds):.4f}")

    jsds_by_layer = [(li, jsd) for (_, li, jsd) in all_jsds]
    all_jsd_vals  = [j for (_, j) in jsds_by_layer]
    early_vals    = [j for (li, j) in jsds_by_layer if li < 8]
    mid_vals      = [j for (li, j) in jsds_by_layer if 8 <= li < 20]
    late_vals     = [j for (li, j) in jsds_by_layer if li >= 20]

    print()
    print("=" * 60)
    print("JSD SUMMARY")
    print("=" * 60)
    print(f"  Mean JSD (all layers vs final)   : {np.mean(all_jsd_vals):.4f}")
    print(f"  Mean JSD (early layers  <8)      : {np.mean(early_vals):.4f}" if early_vals else "  No early layers")
    print(f"  Mean JSD (middle layers 8-20)    : {np.mean(mid_vals):.4f}"   if mid_vals  else "  No mid layers")
    print(f"  Mean JSD (late layers   >=20)    : {np.mean(late_vals):.4f}"  if late_vals else "  No late layers")
    print()
    print("Interpretation:")
    print("  High early JSD → early layers diverge from final → SLED has meaningful signal")
    print("  Low  early JSD → layers agree   → SLED correction is weak no matter the gate")
