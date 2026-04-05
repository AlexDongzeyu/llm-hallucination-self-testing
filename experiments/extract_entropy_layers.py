"""
extract_entropy_layers.py
Extracts actual per-layer entropy from Llama-3.2-3B-Instruct.
Output used by regenerate_figures.py to make Fig 1 accurate.

Runtime: ~15 min on 30 questions.
Usage: python experiments/extract_entropy_layers.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generate_instruct import model, tokenizer, format_instruct_prompt

N_QUESTIONS = 30


def compute_entropy(logits_np: np.ndarray) -> float:
    logits_np = logits_np - logits_np.max()
    probs = np.exp(logits_np) / np.sum(np.exp(logits_np))
    probs = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def get_per_layer_entropy(prompt: str) -> list:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hidden_states = out.hidden_states[1:]  # skip embedding layer
    norm = model.model.norm
    lm_head = model.lm_head
    entropies = []
    for h in hidden_states:
        hs_last = h[:, -1, :]
        logits = lm_head(norm(hs_last)).squeeze(0).detach().cpu().float().numpy()
        entropies.append(compute_entropy(logits))
    return entropies


if __name__ == "__main__":
    print("Loading TruthfulQA...", flush=True)
    ds = load_dataset("truthful_qa", "generation", split="validation")

    all_curves = []
    print(f"Computing per-layer entropy for {N_QUESTIONS} questions...", flush=True)

    for i, sample in enumerate(ds.select(range(N_QUESTIONS))):
        prompt = format_instruct_prompt(sample["question"])
        curve = get_per_layer_entropy(prompt)
        all_curves.append(curve)
        dH = curve[-1] - curve[0]
        print(
            f"  [{i+1}/{N_QUESTIONS}] layers={len(curve)} "
            f"H1={curve[0]:.2f} H_last={curve[-1]:.2f} dH={dH:+.2f}",
            flush=True,
        )

    n_layers = len(all_curves[0])
    layer_means = [float(np.mean([c[l] for c in all_curves])) for l in range(n_layers)]
    layer_stds  = [float(np.std( [c[l] for c in all_curves])) for l in range(n_layers)]
    layer_mins  = [float(np.min( [c[l] for c in all_curves])) for l in range(n_layers)]
    layer_maxs  = [float(np.max( [c[l] for c in all_curves])) for l in range(n_layers)]

    dH_values = [c[-1] - c[0] for c in all_curves]
    dH_neg_count = sum(1 for d in dH_values if d < 0)

    print(f"\n=== ENTROPY SUMMARY ===")
    print(f"H layer 1 (mean): {layer_means[0]:.3f}")
    print(f"H layer {n_layers} (mean): {layer_means[-1]:.3f}")
    print(f"dH mean: {np.mean(dH_values):.3f}")
    print(f"dH < 0: {dH_neg_count}/{N_QUESTIONS} ({dH_neg_count/N_QUESTIONS:.0%})")

    payload = {
        "n_questions": N_QUESTIONS,
        "n_layers": n_layers,
        "layer_means": layer_means,
        "layer_stds": layer_stds,
        "layer_mins": layer_mins,
        "layer_maxs": layer_maxs,
        "all_curves": all_curves,
        "dH_mean": float(np.mean(dH_values)),
        "dH_negative_pct": dH_neg_count / N_QUESTIONS,
    }

    out = ROOT / "results" / "entropy_by_layer.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved to {out}", flush=True)
