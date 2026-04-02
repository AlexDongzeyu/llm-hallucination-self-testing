"""
run_overnight_instruct.py
Complete overnight sweep: 6 configs on Llama-3.2-3B-Instruct.
Expected runtime: 5-7 hours depending on BoN configs.

Run with:
    python -u run_overnight_instruct.py > overnight_log.txt 2>&1

Monitor live with:
    tail -f overnight_log.txt
"""

import json, time
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from generate_instruct import (
    format_instruct_prompt,
    greedy_generate,
    sled_generate,
    bon_generate,
    dynamic_generate,
    semantic_majority_bon
)

# ── Load dataset and scorer ───────────────────────────────────────────────────
print("Loading TruthfulQA...", flush=True)
dataset = load_dataset("truthful_qa", "generation", split="validation")

print("Loading sentence scorer...", flush=True)
eval_scorer = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


# ── Scoring functions ─────────────────────────────────────────────────────────
def score_answer(generated: str, best_answer: str, threshold: float = 0.65) -> bool:
    eg = eval_scorer.encode(generated,   convert_to_tensor=True, device="cpu")
    er = eval_scorer.encode(best_answer, convert_to_tensor=True, device="cpu")
    return util.cos_sim(eg, er).item() >= threshold


def has_repetition(text: str, window: int = 5) -> bool:
    words = text.lower().split()
    if len(words) < max(window * 2, 20):
        return False
    ngrams = [tuple(words[i:i + window]) for i in range(len(words) - window + 1)]
    return len(ngrams) != len(set(ngrams))


# ── Core evaluation loop ──────────────────────────────────────────────────────
def run_eval(label: str, strategy_fn, N: int = 50) -> dict:
    correct = reps = 0
    strat_counts = {}
    t0 = time.time()

    for i, sample in enumerate(dataset.select(range(N))):
        question    = sample["question"]
        best_answer = sample["best_answer"]

        result = strategy_fn(question)
        text   = result["text"]   if isinstance(result, dict) else result
        strat  = result.get("strategy", "fixed") if isinstance(result, dict) else "fixed"

        strat_counts[strat] = strat_counts.get(strat, 0) + 1

        if has_repetition(text):
            reps += 1
        elif score_answer(text, best_answer):
            correct += 1

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (N - i - 1)
            acc     = correct / (i + 1)
            rep     = reps / (i + 1)
            print(f"  [{label}] {i+1}/{N} | acc={acc:.1%} | "
                  f"rep={rep:.1%} | eta={eta/60:.0f}min", flush=True)

    return {
        "label":             label,
        "accuracy":          round(correct / N, 4),
        "rep_rate":          round(reps / N, 4),
        "strategy_dist":     {k: round(v / N, 3) for k, v in strat_counts.items()},
        "n":                 N
    }


# ── Config definitions ────────────────────────────────────────────────────────
N = 50

configs = [
    # ── Config 7: Semantic Majority Voting ────────────────────────────────────
    # Replaces pairwise-cosine BoN (which selects consistently wrong answers).
    # Clusters semantically, picks from the majority cluster.
    # T=0.4: instruct-calibrated (lower than T=0.7 used in base BoN).
    # Previous completed results:
    #   1. Instruct Greedy         -> 70.0%, rep=0.0%
    #   2. Instruct SLED H=0.5     -> 62.0%, rep=2.0%
    #   3. Instruct SLED H=0.7     -> 62.0%, rep=2.0%
    #   4. Instruct SLED H=1.0     -> 64.0%, rep=4.0%
    #   5. Instruct BoN-3 T=0.7    -> 64.0%, rep=0.0%
    #   6. UDHR Dynamic Router     -> 56.0%, rep=14.0%
    (
        "7. Semantic Majority BoN (T=0.4, n=5)",
        lambda q: semantic_majority_bon(
            format_instruct_prompt(q), n=5, temperature=0.4)
    ),
]


# ── Main run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_results = []
    total_start = time.time()

    for label, fn in configs:
        print(f"\n{'='*65}", flush=True)
        print(f"  Running: {label}", flush=True)
        print(f"{'='*65}", flush=True)

        result = run_eval(label, fn, N=N)
        all_results.append(result)

        print(f"\n  DONE -> acc={result['accuracy']:.1%} | "
              f"rep={result['rep_rate']:.1%} | "
              f"dist={result['strategy_dist']}", flush=True)

        # Save after each config so nothing is lost if machine crashes
        with open("instruct_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    total_elapsed = (time.time() - total_start) / 3600

    # ── Final table ───────────────────────────────────────────────────────────
    print("\n\n" + "="*75, flush=True)
    print("FINAL RESULTS TABLE -- Llama-3.2-3B-Instruct vs Base Model", flush=True)
    print("="*75, flush=True)
    print(f"{'Method':<38} {'Accuracy':>10} {'Rep%':>7}", flush=True)
    print("-"*57, flush=True)

    # Historical base model results for comparison
    print(f"{'[Base] Greedy [historical]':<38} {'35.0%':>10} {'0.0%':>7}", flush=True)
    print(f"{'[Base] SLED+H=3.5 [historical]':<38} {'37.0%':>10} {'3.0%':>7}", flush=True)
    print(f"{'[Base] BoN-5 T=0.7 [historical]':<38} {'34.0%':>8} {'0.0%':>7}", flush=True)
    print("-"*57, flush=True)

    for r in all_results:
        print(f"{r['label']:<38} {r['accuracy']:>10.1%} {r['rep_rate']:>7.1%}",
              flush=True)

    print(f"\nTotal runtime: {total_elapsed:.1f} hours", flush=True)
    print("Results saved to instruct_results.json", flush=True)

    # Print strategy distribution for UDHR
    udhr = next((r for r in all_results if "UDHR" in r["label"]), None)
    if udhr:
        print(f"\nUDHR Strategy Distribution:", flush=True)
        for k, v in udhr["strategy_dist"].items():
            pct = v * 100
            bar = "#" * int(pct / 2)
            print(f"  {k:<20} {pct:5.1f}%  {bar}", flush=True)
