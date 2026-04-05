"""Compute late-layer logit linearity (R^2) for the current instruct model.

DeLTa assumes late-layer trajectories are close to linear. This script
measures that assumption directly on TruthfulQA prompts.

Output: results/logit_linearity_3b.json
"""

import argparse
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

from generate_instruct import format_instruct_prompt, model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Number of prompts")
    parser.add_argument(
        "--mid-layer",
        type=int,
        default=14,
        dest="mid_layer",
        help="1-based start layer for regression window",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        dest="top_k",
        help="Top-k final-layer tokens used for R^2 aggregation",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "logit_linearity_3b.json"),
    )
    return parser.parse_args()


def _r2_linear(x: np.ndarray, y: np.ndarray) -> float:
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 1e-12:
        return 0.0

    beta1 = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    beta0 = y_mean - beta1 * x_mean
    y_pred = beta0 + beta1 * x

    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_r2_for_prompt(prompt: str, mid_layer_1based: int, top_k: int) -> dict:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)

    hidden_states = out.hidden_states[1:]  # skip embedding layer
    total_layers = len(hidden_states)

    # Convert to 0-based inclusive slice start.
    start_idx = max(0, min(total_layers - 1, mid_layer_1based - 1))
    reg_hidden = hidden_states[start_idx:]

    norm = model.model.norm
    lm_head = model.lm_head

    reg_logits = []
    for h in reg_hidden:
        logits = lm_head(norm(h[:, -1, :])).squeeze(0).detach().cpu().float().numpy()
        reg_logits.append(logits)

    reg_logits = np.array(reg_logits)  # [n_reg_layers, vocab]
    n_reg_layers = int(reg_logits.shape[0])

    if n_reg_layers < 2:
        return {
            "mean_r2": 0.0,
            "median_r2": 0.0,
            "std_r2": 0.0,
            "n_layers_used": n_reg_layers,
            "layer_start": start_idx + 1,
            "layer_end": total_layers,
        }

    x = np.arange(n_reg_layers, dtype=np.float32)
    x = (x - float(np.mean(x))) / max(float(np.std(x)), 1e-8)

    k = max(1, min(int(top_k), int(reg_logits.shape[1])))
    top_tokens = np.argsort(reg_logits[-1])[-k:]

    r2_values = []
    for tok in top_tokens:
        y = reg_logits[:, int(tok)]
        r2_values.append(_r2_linear(x, y))

    return {
        "mean_r2": float(np.mean(r2_values)),
        "median_r2": float(np.median(r2_values)),
        "std_r2": float(np.std(r2_values)),
        "n_layers_used": n_reg_layers,
        "layer_start": start_idx + 1,
        "layer_end": total_layers,
    }


if __name__ == "__main__":
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading TruthfulQA...", flush=True)
    ds = load_dataset("truthful_qa", "generation", split="validation")

    print(
        f"Computing late-layer linearity on n={args.n}, mid_layer={args.mid_layer}, top_k={args.top_k}",
        flush=True,
    )

    rows = []
    for i, sample in enumerate(ds.select(range(args.n))):
        prompt = format_instruct_prompt(sample["question"])
        r = compute_r2_for_prompt(prompt, args.mid_layer, args.top_k)
        rows.append(r)
        print(
            f"  [{i + 1}/{args.n}] mean_R2={r['mean_r2']:.3f} "
            f"layers={r['layer_start']}..{r['layer_end']}",
            flush=True,
        )

    payload = {
        "model": getattr(model.config, "_name_or_path", "unknown"),
        "n_questions": int(args.n),
        "mid_layer": int(args.mid_layer),
        "top_k": int(args.top_k),
        "mean_r2": float(np.mean([r["mean_r2"] for r in rows])) if rows else 0.0,
        "median_r2": float(np.median([r["median_r2"] for r in rows])) if rows else 0.0,
        "std_r2": float(np.std([r["mean_r2"] for r in rows])) if rows else 0.0,
        "per_question": rows,
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== 3B LOGIT LINEARITY ===", flush=True)
    print(f"Mean R2: {payload['mean_r2']:.3f}", flush=True)
    print(f"Median R2: {payload['median_r2']:.3f}", flush=True)
    print(f"Std R2: {payload['std_r2']:.3f}", flush=True)
    print(f"Saved to {out_path}", flush=True)
