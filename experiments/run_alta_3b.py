"""ALTA-style entropy-gated logit correction at 3B scale.

This script applies token-level entropy weighting to a DeLTa+DoLa-style
correction and evaluates on TruthfulQA generation (cosine threshold metric)
to mirror the local 3B setup.

Usage:
  python -u experiments/run_alta_3b.py --n 50 --threshold 0.65
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import util

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generate_instruct import (
    apply_repetition_penalty,
    compute_entropy,
    format_instruct_prompt,
    get_layer_logits_cached,
    model,
    scorer,
    tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--max-new-tokens", type=int, default=80, dest="max_new_tokens")
    parser.add_argument("--early-idx", type=int, default=7, dest="early_idx")
    parser.add_argument("--mid-idx", type=int, default=14, dest="mid_idx")
    parser.add_argument("--top-k", type=int, default=200, dest="top_k")
    parser.add_argument("--alpha-contrast", type=float, default=0.3, dest="alpha_contrast")
    parser.add_argument("--alpha-extrap", type=float, default=0.3, dest="alpha_extrap")
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "alta_3b_results.json"),
    )
    return parser.parse_args()


def alta_logits_3b(
    layer_logits: np.ndarray,
    early_idx: int = 7,
    mid_idx: int = 14,
    top_k: int = 200,
    alpha_contrast: float = 0.3,
    alpha_extrap: float = 0.3,
) -> tuple[np.ndarray, float, float]:
    """Return entropy-gated corrected logits and diagnostics."""
    n_layers = int(layer_logits.shape[0])
    z_final = layer_logits[-1].astype(np.float32, copy=False)

    # Token-level entropy gating.
    h_final = float(compute_entropy(z_final))
    entropy_weight = float(np.clip(h_final / 3.0, 0.0, 1.0))

    early = layer_logits[min(max(int(early_idx), 0), n_layers - 1)].astype(np.float32, copy=False)
    z_dola = z_final - early

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
            y_m = y.mean(axis=0)
            b1 = np.sum(x_n[:, None] * (y - y_m[None, :]), axis=0) / denom
            b0 = y_m - b1 * float(x_n.mean())
            x_virt = float(x_n[-1] + (x_n[1] - x_n[0]))
            z_delta[top_idx] = b0 + b1 * x_virt

    correction = alpha_contrast * z_dola + alpha_extrap * (z_delta - z_final)
    return z_final + entropy_weight * correction, entropy_weight, h_final


def alta_generate_3b(
    question: str,
    max_new_tokens: int = 80,
    early_idx: int = 7,
    mid_idx: int = 14,
    top_k: int = 200,
    alpha_contrast: float = 0.3,
    alpha_extrap: float = 0.3,
) -> dict:
    """Generate an answer with ALTA-style entropy-gated correction."""
    prompt = format_instruct_prompt(question)
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    layer_logits, past_kv = get_layer_logits_cached(ids, None)

    generated = []
    gate_weights = []
    first_entropy = None

    for _ in range(max_new_tokens):
        corrected, w, h = alta_logits_3b(
            layer_logits,
            early_idx=early_idx,
            mid_idx=mid_idx,
            top_k=top_k,
            alpha_contrast=alpha_contrast,
            alpha_extrap=alpha_extrap,
        )
        if first_entropy is None:
            first_entropy = h
        gate_weights.append(w)

        logits = apply_repetition_penalty(corrected, generated)
        next_id = int(np.argmax(logits))
        generated.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

        next_t = torch.tensor([[next_id]], device=model.device)
        layer_logits, past_kv = get_layer_logits_cached(next_t, past_kv)

    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "strategy": "alta_3b_entropy_gated",
        "mean_gate_weight": float(np.mean(gate_weights)) if gate_weights else 0.0,
        "first_token_entropy": float(first_entropy) if first_entropy is not None else 0.0,
    }


def has_repetition(text: str, window: int = 5) -> bool:
    words = text.lower().split()
    if len(words) < max(window * 2, 20):
        return False
    ngrams = [tuple(words[i : i + window]) for i in range(len(words) - window + 1)]
    return len(ngrams) != len(set(ngrams))


def score_cosine(generated: str, reference: str, threshold: float) -> bool:
    if not generated.strip():
        return False
    eg = scorer.encode(generated, convert_to_tensor=True, device="cpu")
    er = scorer.encode(reference, convert_to_tensor=True, device="cpu")
    return float(util.cos_sim(eg, er).item()) >= threshold


if __name__ == "__main__":
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading TruthfulQA...", flush=True)
    ds = load_dataset("truthful_qa", "generation", split="validation")

    correct = 0
    reps = 0
    gate_means = []
    first_entropies = []
    t0 = time.time()

    for i, sample in enumerate(ds.select(range(args.n))):
        out = alta_generate_3b(
            sample["question"],
            max_new_tokens=args.max_new_tokens,
            early_idx=args.early_idx,
            mid_idx=args.mid_idx,
            top_k=args.top_k,
            alpha_contrast=args.alpha_contrast,
            alpha_extrap=args.alpha_extrap,
        )

        text = out["text"]
        gate_means.append(out["mean_gate_weight"])
        first_entropies.append(out["first_token_entropy"])

        if has_repetition(text):
            reps += 1
        elif score_cosine(text, sample["best_answer"], args.threshold):
            correct += 1

        if (i + 1) % 10 == 0:
            print(
                f"  [{i + 1}/{args.n}] acc={correct/(i + 1):.1%} "
                f"rep={reps/(i + 1):.1%} gate={np.mean(gate_means):.3f}",
                flush=True,
            )

    accuracy = round(correct / max(args.n, 1), 4)
    rep_rate = round(reps / max(args.n, 1), 4)

    result = {
        "method": "alta_3b_entropy_gated",
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "n": int(args.n),
        "threshold": float(args.threshold),
        "accuracy": accuracy,
        "rep_rate": rep_rate,
        "runtime_min": round((time.time() - t0) / 60, 2),
        "mean_gate_weight": round(float(np.mean(gate_means)), 4) if gate_means else 0.0,
        "mean_first_token_entropy": round(float(np.mean(first_entropies)), 4)
        if first_entropies
        else 0.0,
        "settings": {
            "early_idx": int(args.early_idx),
            "mid_idx": int(args.mid_idx),
            "top_k": int(args.top_k),
            "alpha_contrast": float(args.alpha_contrast),
            "alpha_extrap": float(args.alpha_extrap),
            "max_new_tokens": int(args.max_new_tokens),
        },
    }

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== ALTA-3B RESULT ===", flush=True)
    print(f"Accuracy: {accuracy:.1%}", flush=True)
    print(f"Repetition: {rep_rate:.1%}", flush=True)
    print(f"Mean gate weight: {result['mean_gate_weight']:.3f}", flush=True)
    print(f"Saved to {out_path}", flush=True)
