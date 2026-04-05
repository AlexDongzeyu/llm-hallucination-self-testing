"""Run MedHallu ablations for ITI, SLED, and BoN.

Writes `results/medhallu_ablation_results.json` and does not overwrite
`results/medhallu_generation_results.json`.

Expected labels (used by regenerate_figures.py):
- iti_alpha0.5
- sled
- bon3_t0.3
"""

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

from generate_instruct import (
    bon_generate,
    format_instruct_prompt,
    iti_generate,
    sled_generate,
)

N = 50
THRESHOLD = 0.65

print("Loading MedHallu...", flush=True)
dataset = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial", split="train")
print(f"Loaded {len(dataset)} rows", flush=True)

print("Loading scorer...", flush=True)
scorer = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def has_repetition(text: str, window: int = 5) -> bool:
    words = text.lower().split()
    if len(words) < max(window * 2, 20):
        return False
    ngrams = [tuple(words[i:i+window]) for i in range(len(words) - window + 1)]
    return len(ngrams) != len(set(ngrams))


def score_cosine(generated: str, reference: str) -> bool:
    if not generated.strip():
        return False
    eg = scorer.encode(generated, convert_to_tensor=True, device="cpu")
    er = scorer.encode(reference, convert_to_tensor=True, device="cpu")
    return util.cos_sim(eg, er).item() >= THRESHOLD


def run_strategy(label, strategy_fn):
    correct = reps = skipped = seen = 0
    by_difficulty = {}
    t0 = time.time()

    for sample in dataset:
        if seen >= N:
            break

        question = ""
        for key in ["Question", "question", "query"]:
            if key in sample and sample[key]:
                question = sample[key].strip()
                break
        gt = str(sample.get("Ground Truth", "") or "").strip()
        difficulty = str(
            sample.get("Difficulty Level", "unknown") or "unknown"
        ).lower()

        if not question or not gt:
            skipped += 1
            continue

        out = strategy_fn(question)
        text = out["text"] if isinstance(out, dict) else str(out)
        seen += 1

        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = {"correct": 0, "total": 0}
        by_difficulty[difficulty]["total"] += 1

        if has_repetition(text):
            reps += 1
        elif score_cosine(text, gt):
            correct += 1
            by_difficulty[difficulty]["correct"] += 1

        if seen % 10 == 0:
            elapsed = (time.time() - t0) / 60
            print(
                f"  [{seen}/{N}] acc={correct/seen:.0%} "
                f"rep={reps/seen:.0%} ({elapsed:.0f}min)",
                flush=True,
            )

    diff_summary = {
        d: {
            "n": v["total"],
            "accuracy": round(v["correct"] / max(v["total"], 1), 4),
        }
        for d, v in by_difficulty.items()
    }
    return {
        "label": label,
        "n_used": seen,
        "n_skipped": skipped,
        "accuracy": round(correct / max(seen, 1), 4),
        "rep_rate": round(reps / max(seen, 1), 4),
        "by_difficulty": diff_summary,
        "runtime_min": round((time.time() - t0) / 60, 2),
    }


# ── Strategies ────────────────────────────────────────────────────────────────
# Label names should match what regenerate_figures.py expects:
#   "iti_alpha0.5", "sled", "bon3_t0.3"
strategies = [
    (
        "iti_alpha0.5",
        lambda q: iti_generate(
            format_instruct_prompt(q), alpha=0.5, max_new_tokens=80
        ),
    ),
    (
        "sled",
        lambda q: sled_generate(
            format_instruct_prompt(q),
            max_new_tokens=80,
            entropy_threshold=3.5,
            alpha=0.3,
        ),
    ),
    (
        "bon3_t0.3",
        lambda q: bon_generate(
            format_instruct_prompt(q),
            n=3,
            temperature=0.3,
            max_new_tokens=80,
        ),
    ),
]

results = []
for label, fn in strategies:
    print(f"\n=== Running {label} ===", flush=True)
    r = run_strategy(label, fn)
    results.append(r)
    delta = r["accuracy"] - 0.50
    print(
        f"  DONE: acc={r['accuracy']:.0%} ({delta:+.0%} vs greedy) "
        f"rep={r['rep_rate']:.0%} ({r['runtime_min']:.0f}min)",
        flush=True,
    )

print("\n=== ABLATION SUMMARY ===")
print("Greedy baseline: 50%  CoVe: 50%  CURED: 54%")
for r in results:
    print(f"  {r['label']}: {r['accuracy']:.0%}")

out_path = ROOT / "results" / "medhallu_ablation_results.json"
out_path.write_text(
    json.dumps({"results": results, "greedy_baseline": 0.50}, indent=2)
)
print(f"\nSaved to {out_path}")
