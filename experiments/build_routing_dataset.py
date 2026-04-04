"""
build_routing_dataset.py
Builds the feature-outcome dataset needed to train the GADR router.

For each question in TruthfulQA (n=50) and MedHallu (n=50):
  1. Compute trajectory features
  2. Run candidate methods and record correctness
  3. Save to results/routing_dataset.csv
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generate_instruct import (
    cove_generate,
    compute_entropy,
    format_instruct_prompt,
    get_layer_logits_cached,
    greedy_generate,
    iti_generate,
    model,
    tokenizer,
)

scorer = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def _jsd_from_logits(a: np.ndarray, b: np.ndarray) -> float:
    p = np.exp(a - np.max(a))
    p /= p.sum()
    q = np.exp(b - np.max(b))
    q /= q.sum()

    m = np.clip(0.5 * (p + q), 1e-10, 1.0)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def compute_features(prompt_formatted: str) -> dict:
    """
    One forward pass -> depth-wise trajectory features.
    """
    ids = tokenizer.encode(prompt_formatted, return_tensors="pt").to(model.device)
    layer_logits, _ = get_layer_logits_cached(ids, None)
    n = len(layer_logits)

    i1 = max(0, n // 4)
    i2 = max(0, n // 2)
    i3 = max(0, 3 * n // 4)
    i4 = n - 1

    h1 = compute_entropy(layer_logits[i1])
    h2 = compute_entropy(layer_logits[i2])
    h3 = compute_entropy(layer_logits[i3])
    h4 = compute_entropy(layer_logits[i4])

    dh = h4 - h1
    d2h = h4 - 2 * h2 + h1
    dh_late = h4 - h3

    jsd_early = _jsd_from_logits(layer_logits[i1], layer_logits[i4])
    jsd_late = _jsd_from_logits(layer_logits[i3], layer_logits[i4])
    jsd_conv = jsd_early - jsd_late

    return {
        "H_final": h4,
        "dH": dh,
        "d2H": d2h,
        "dH_late": dh_late,
        "jsd_early": jsd_early,
        "jsd_late": jsd_late,
        "jsd_conv": jsd_conv,
    }


def token_entropy_spike(prompt_formatted: str, k: int = 15) -> dict:
    """
    Track entropy over first k generated tokens and return summary stats.
    """
    ids = tokenizer.encode(prompt_formatted, return_tensors="pt").to(model.device)
    layer_logits, past_kv = get_layer_logits_cached(ids, None)

    entropies = []
    for _ in range(k):
        h = compute_entropy(layer_logits[-1])
        entropies.append(h)

        next_id = int(np.argmax(layer_logits[-1]))
        if next_id == tokenizer.eos_token_id:
            break
        next_t = torch.tensor([[next_id]], device=model.device)
        layer_logits, past_kv = get_layer_logits_cached(next_t, past_kv)

    e = np.array(entropies, dtype=np.float32)
    mu = float(e.mean()) if len(e) else 0.0
    sigma = float(e.std()) + 1e-8
    z = (e - mu) / sigma if len(e) else np.array([0.0], dtype=np.float32)
    return {
        "mean_token_entropy": mu,
        "max_spike_z": float(z.max()) if len(z) > 1 else 0.0,
    }


def is_correct(text: str, ref: str, thresh: float = 0.65) -> int:
    if not text.strip():
        return 0
    eg = scorer.encode(text, convert_to_tensor=True, device="cpu")
    er = scorer.encode(ref, convert_to_tensor=True, device="cpu")
    return int(util.cos_sim(eg, er).item() >= thresh)


def has_rep(text: str, w: int = 5) -> bool:
    words = text.lower().split()
    if len(words) < max(w * 2, 20):
        return False
    ngrams = [tuple(words[i : i + w]) for i in range(len(words) - w + 1)]
    return len(ngrams) != len(set(ngrams))


def get_samples() -> list[tuple[str, str, str]]:
    """Returns (question, reference, domain) tuples."""
    samples = []

    tqa = load_dataset("truthful_qa", "generation", split="validation")
    for s in tqa.select(range(50)):
        samples.append((s["question"], s["best_answer"], "general"))

    mh = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial", split="train")
    for s in list(mh)[:50]:
        q = s.get("Question", "")
        r = s.get("Ground Truth", "")
        if q and r:
            samples.append((q, r, "medical"))

    return samples


if __name__ == "__main__":
    samples = get_samples()
    out_path = ROOT / "results" / "routing_dataset.csv"
    out_path.parent.mkdir(exist_ok=True)

    fieldnames = [
        "question",
        "domain",
        "reference",
        "H_final",
        "dH",
        "d2H",
        "dH_late",
        "jsd_early",
        "jsd_late",
        "jsd_conv",
        "mean_token_entropy",
        "max_spike_z",
        "greedy_ok",
        "cove_ok",
        "iti_ok",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (question, reference, domain) in enumerate(samples):
            t0 = time.time()
            prompt = format_instruct_prompt(question)

            depth_feats = compute_features(prompt)
            token_feats = token_entropy_spike(prompt, k=15)

            def eval_method(text: str) -> int:
                if has_rep(text):
                    return -1
                return is_correct(text, reference)

            g_text = greedy_generate(prompt, max_new_tokens=80)
            g_ok = eval_method(g_text)

            c_res = cove_generate(question, max_new_tokens=80)
            c_ok = eval_method(c_res["text"])

            iti_res = iti_generate(prompt, alpha=0.5, max_new_tokens=80)
            iti_ok = eval_method(iti_res["text"])

            row = {
                "question": question[:200],
                "domain": domain,
                "reference": reference[:200],
                **depth_feats,
                **token_feats,
                "greedy_ok": g_ok,
                "cove_ok": c_ok,
                "iti_ok": iti_ok,
            }
            writer.writerow(row)
            f.flush()

            elapsed = time.time() - t0
            eta = elapsed * (len(samples) - i - 1)
            print(
                f"[{i + 1}/{len(samples)}] domain={domain} | "
                f"greedy={g_ok} cove={c_ok} iti={iti_ok} | "
                f"dH={depth_feats['dH']:.3f} d2H={depth_feats['d2H']:.3f} | "
                f"eta={eta / 60:.0f}min",
                flush=True,
            )

    print(f"\nSaved to {out_path}")
