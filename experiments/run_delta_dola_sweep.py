"""
run_delta_dola_sweep.py
Grid search over alpha1 (DeLTa weight) and alpha2 (DoLa weight) on TruthfulQA.

Usage:
    python -u experiments/run_delta_dola_sweep.py
    python -u experiments/run_delta_dola_sweep.py --n 50 --threshold 0.65
"""

import argparse
import json
import sys
import time
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generate_instruct import format_instruct_prompt, delta_dola_generate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of samples")
    parser.add_argument("--threshold", type=float, default=0.65, help="cosine threshold")
    parser.add_argument(
        "--out",
        type=str,
        # Canonical TruthfulQA sweep artifact.
        default=str(ROOT / "results" / "truthfulqa_delta_dola_sweep.json"),
        help="output JSON path",
    )
    return parser.parse_args()


print("Loading TruthfulQA...", flush=True)
dataset = load_dataset("truthful_qa", "generation", split="validation")

print("Loading sentence scorer...", flush=True)
scorer = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def score_answer(generated: str, reference: str, threshold: float = 0.65) -> bool:
    if not generated.strip():
        return False
    eg = scorer.encode(generated, convert_to_tensor=True, device="cpu")
    er = scorer.encode(reference, convert_to_tensor=True, device="cpu")
    return util.cos_sim(eg, er).item() >= threshold


def has_repetition(text: str, window: int = 5) -> bool:
    words = text.lower().split()
    if len(words) < max(window * 2, 20):
        return False
    ngrams = [tuple(words[i : i + window]) for i in range(len(words) - window + 1)]
    return len(ngrams) != len(set(ngrams))


def run_one(alpha1: float, alpha2: float, n: int, threshold: float) -> dict:
    correct = 0
    reps = 0
    t0 = time.time()

    for i, sample in enumerate(dataset.select(range(n))):
        q = sample["question"]
        best = sample["best_answer"]

        result = delta_dola_generate(
            format_instruct_prompt(q),
            max_new_tokens=80,
            alpha1=alpha1,
            alpha2=alpha2,
        )
        text = result["text"]

        if has_repetition(text):
            reps += 1
        elif score_answer(text, best, threshold=threshold):
            correct += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(
                f"    [{i+1}/{n}] acc={correct/(i+1):.1%} rep={reps/(i+1):.1%} eta={eta/60:.0f}min",
                flush=True,
            )

    runtime_min = (time.time() - t0) / 60
    return {
        "alpha1": alpha1,
        "alpha2": alpha2,
        "accuracy": round(correct / max(n, 1), 4),
        "rep_rate": round(reps / max(n, 1), 4),
        "runtime_min": round(runtime_min, 1),
    }


if __name__ == "__main__":
    args = parse_args()

    # 3x3 grid from proposed shortlist
    alphas = [
        (0.0, 0.0),
        (0.2, 0.2),
        (0.3, 0.3),
        (0.2, 0.0),
        (0.0, 0.2),
        (0.3, 0.1),
        (0.1, 0.3),
        (0.4, 0.2),
        (0.2, 0.4),
    ]

    results = []
    total_t0 = time.time()

    for alpha1, alpha2 in alphas:
        print(f"\nRunning alpha1={alpha1}, alpha2={alpha2} ...", flush=True)
        res = run_one(alpha1, alpha2, n=args.n, threshold=args.threshold)
        results.append(res)
        print(
            f"  done: acc={res['accuracy']:.1%} rep={res['rep_rate']:.1%} runtime={res['runtime_min']:.1f}min",
            flush=True,
        )

    payload = {
        "baseline": {
            "greedy_truthfulqa_instruct": 0.70,
            "sled_best": 0.64,
            "iti_alpha_0_5": 0.72,
        },
        "n": args.n,
        "threshold": args.threshold,
        "results": results,
        "best": max(results, key=lambda r: r["accuracy"]) if results else None,
        "runtime_min": round((time.time() - total_t0) / 60, 2),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSweep complete.", flush=True)
    print(json.dumps(payload["best"], indent=2), flush=True)
    print(f"Saved to {out_path}", flush=True)
