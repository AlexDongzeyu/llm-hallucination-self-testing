"""
eval_base.py -- Open-ended generation and BoN evaluation.

The MC evaluation (max_new_tokens=5) was the wrong testbed -- the gate
had almost no opportunity to fire on 1-2 token answers.

This version uses TruthfulQA generation split with 80-token answers,
semantic similarity scoring, and three comparison rows:
  1. Baseline (greedy, no intervention)
  2. DoLA no-gate (gate always fires = pure DoLA at early_layer=12)
  3. Ours: joint curvature-entropy gate at early_layer=12

The key question: does gated DoLA beat ungated DoLA?
That isolates the gate's contribution from the layer-choice finding.
"""

import json
import time
import sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generate_base import gated_generate


# ── Load dataset and scorer ───────────────────────────────────────────────────

print("Loading TruthfulQA generation split...")
dataset = load_dataset("truthful_qa", "generation", split="validation")
print(f"Loaded {len(dataset)} samples")

print("Loading sentence scorer (forced to CPU to save VRAM for LLM)...")
scorer = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_answer(generated: str, best_answer: str, threshold: float = 0.65) -> bool:
    """
    Semantic similarity between generated answer and reference.
    Threshold 0.65 is deliberately lenient -- we're measuring hallucination
    reduction, not exact recall.
    """
    emb_gen = scorer.encode(generated, convert_to_tensor=True, device="cpu")
    emb_ref = scorer.encode(best_answer, convert_to_tensor=True, device="cpu")
    return util.cos_sim(emb_gen, emb_ref).item() >= threshold


def has_repetition(text: str, window: int = 5, min_words: int = 20) -> bool:
    """
    Detect repeated n-gram loops in generated text.
    Only checks outputs with >= min_words to avoid false positives on short answers.
    Window=5 catches phrase-level repetition without flagging common word pairs.
    """
    words = text.lower().split()
    if len(words) < min_words:
        return False
    if len(words) < window * 2:
        return False
    ngrams = [tuple(words[i:i + window]) for i in range(len(words) - window + 1)]
    return len(ngrams) != len(set(ngrams))


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate_config(
    label: str,
    curve_thresh: float,
    entropy_thresh: float,
    early_layer: int,
    alpha: float,
    gate_mode: str = "joint",
    n_samples: int = 50
) -> dict:
    """
    Run one configuration on n_samples questions from TruthfulQA generation.
    Prints progress as it runs.
    """
    correct = 0
    repetitions = 0
    fire_rates = []
    config_start = time.time()

    for i, sample in enumerate(dataset.select(range(n_samples))):
        sample_start = time.time()
        prompt = f"Q: {sample['question']}\nA:"
        best_answer = sample["best_answer"]

        result = gated_generate(
            prompt,
            max_new_tokens=80,
            curve_threshold=curve_thresh,
            entropy_threshold=entropy_thresh,
            early_layer_idx=early_layer,
            contrast_alpha=alpha,
            repetition_penalty=1.3,
            gate_mode=gate_mode
        )

        if has_repetition(result["text"]):
            repetitions += 1
        elif score_answer(result["text"], best_answer):
            correct += 1

        fire_rates.append(result["gate_fire_rate"])
        elapsed = time.time() - sample_start

        if i % 5 == 0:
            eta = (elapsed * (n_samples - i - 1))
            print(f"  [{label}] {i+1}/{n_samples} -- "
                  f"acc: {correct / (i + 1):.1%} | "
                  f"fire: {np.mean(fire_rates):.1%} | "
                  f"sample: {elapsed:.1f}s | "
                  f"ETA: {eta/60:.1f}min", flush=True)

    return {
        "label":           label,
        "gate_mode":       gate_mode,
        "curve_thresh":    curve_thresh,
        "entropy_thresh":  entropy_thresh,
        "early_layer":     early_layer,
        # NOTE: early_layer_idx=12 in generate.py indexes into hidden_states[1:],
        # so it corresponds to transformer layer 13 (off by one from CSV numbering).
        # This is documented here for the mentor report.
        "alpha":           alpha,
        "accuracy":        round(correct / n_samples, 4),
        "gate_fire_rate":  round(float(np.mean(fire_rates)), 4),
        "repetition_rate": round(repetitions / n_samples, 4),
        "n_samples":       n_samples
    }


# ── Best-of-N evaluation ──────────────────────────────────────────────────────

from best_of_n import best_of_n


def evaluate_bon(label: str, n_bon: int, n_questions: int = 50,
                 temperature: float = 0.7) -> dict:
    correct, repetitions = 0, 0
    import time
    for i, sample in enumerate(dataset.select(range(n_questions))):
        t0 = time.time()
        prompt = f"Q: {sample['question']}\nA:"
        result = best_of_n(prompt, n=n_bon, temperature=temperature)
        if has_repetition(result["text"]):
            repetitions += 1
        elif score_answer(result["text"], sample["best_answer"]):
            correct += 1
        if i % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (n_questions - i - 1)
            print(f"  [{label}] {i+1}/{n_questions} "
                  f"acc={correct/(i+1):.1%} | "
                  f"rep={repetitions/(i+1):.1%} | "
                  f"sample={elapsed:.1f}s | ETA={eta/60:.1f}min", flush=True)
    return {
        "label":        label,
        "accuracy":     round(correct / n_questions, 4),
        "rep_rate":     round(repetitions / n_questions, 4),
        "n_bon":        n_bon,
        "n_questions":  n_questions,
    }


# ── Main: run BoN configs ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    all_results = []

    # T=0.3 is calibrated for base models (not instruct).
    # T=0.7 produced garbage single-sample outputs (22% vs 35% greedy).
    for label, n in [("BoN-1 T=0.3", 1),
                     ("BoN-3 T=0.3", 3),
                     ("BoN-5 T=0.3", 5)]:
        print(f"\n{'='*60}", flush=True)
        print(f"Running: {label}", flush=True)
        print(f"{'='*60}", flush=True)
        r = evaluate_bon(label, n, temperature=0.3)
        all_results.append(r)
        print(f"DONE: {r['label']} | acc={r['accuracy']:.1%} | rep={r['rep_rate']:.1%}")

    # Save
    out_path = ROOT / "results" / "bon_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/bon_results.json")

    # Print final table
    print("\n" + "="*60)
    print("BEST-OF-N RESULTS TABLE")
    print("="*60)
    print(f"{'Method':<25} {'Accuracy':>10} {'Rep%':>8}")
    print("-"*45)
    print(f"{'Greedy baseline':<25} {'35.0%':>10} {'0.0%':>8}  [historical]")
    print(f"{'SLED + H=3.5 (N=100)':<25} {'37.0%':>10} {'3.0%':>8}  [historical]")
    for r in all_results:
        print(f"{r['label']:<25} {r['accuracy']:>10.1%} {r['rep_rate']:>8.1%}")


