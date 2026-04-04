"""
eval_medhallu.py
Evaluate local Llama-3.2-3B-Instruct strategies on a medical hallucination dataset.

Default behavior tries several likely MedHallu dataset IDs on Hugging Face.
You can override with explicit args if your dataset ID differs.

Usage:
    python -u experiments/eval_medhallu.py
    python -u experiments/eval_medhallu.py --dataset your_org/your_medhallu --split test
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

from generate_instruct import (
    format_instruct_prompt,
    greedy_generate,
    cove_generate,
    dynamic_generate,
    gadr2_generate,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="", help="HF dataset id")
    parser.add_argument("--subset", type=str, default="", help="HF config/subset")
    parser.add_argument("--split", type=str, default="", help="split name")
    parser.add_argument("--n", type=int, default=50, help="number of samples")
    parser.add_argument("--threshold", type=float, default=0.65, help="cosine threshold")
    return parser.parse_args()


def try_load_dataset(dataset_id: str, subset: str, split: str):
    if subset:
        return load_dataset(dataset_id, subset, split=split)
    return load_dataset(dataset_id, split=split)


def load_medhallu_dataset(dataset_arg: str, subset_arg: str, split_arg: str):
    if dataset_arg:
        split = split_arg or "test"
        ds = try_load_dataset(dataset_arg, subset_arg, split)
        return ds, dataset_arg, subset_arg or None, split

    candidates = [
        ("medhallu", "", "test"),
        ("medhallu", "", "validation"),
        ("medhallu", "", "train"),
        ("openlifescienceai/medhallu", "", "test"),
        ("FreedomIntelligence/MedHallu", "", "test"),
        ("UTAustin-AIHealth/MedHallu", "pqa_artificial", "train"),
        ("UTAustin-AIHealth/MedHallu", "pqa_labeled", "train"),
        ("hirundo-io/medhallu", "default", "train"),
        ("Lizong/MedHallu", "pqa_artificial", "train"),
        ("Lizong/MedHallu", "pqa_labeled", "train"),
    ]

    errors = []
    for dataset_id, subset, split in candidates:
        try:
            ds = try_load_dataset(dataset_id, subset, split)
            return ds, dataset_id, subset or None, split
        except Exception as e:
            errors.append(f"{dataset_id}:{subset}:{split} -> {type(e).__name__}: {e}")

    raise RuntimeError(
        "Could not load a MedHallu dataset automatically. "
        "Pass --dataset/--subset/--split explicitly.\n"
        + "\n".join(errors)
    )


def extract_question(sample: dict) -> str:
    for key in ["question", "query", "prompt", "input", "instruction", "claim", "Question"]:
        if key in sample and isinstance(sample[key], str) and sample[key].strip():
            return sample[key].strip()
    return ""


def extract_reference(sample: dict) -> str:
    for key in ["best_answer", "answer", "ground_truth", "reference", "output", "response", "Ground Truth"]:
        if key in sample and isinstance(sample[key], str) and sample[key].strip():
            return sample[key].strip()
    return ""


def has_repetition(text: str, window: int = 5) -> bool:
    words = text.lower().split()
    if len(words) < max(window * 2, 20):
        return False
    ngrams = [tuple(words[i:i + window]) for i in range(len(words) - window + 1)]
    return len(ngrams) != len(set(ngrams))


def score_answer(scorer, generated: str, reference: str, threshold: float) -> bool:
    eg = scorer.encode(generated, convert_to_tensor=True, device="cpu")
    er = scorer.encode(reference, convert_to_tensor=True, device="cpu")
    return util.cos_sim(eg, er).item() >= threshold


def evaluate_strategy(dataset, scorer, label: str, strategy_fn, n: int, threshold: float) -> dict:
    correct = 0
    reps = 0
    seen = 0
    skipped = 0

    for sample in dataset:
        if seen >= n:
            break
        question = extract_question(sample)
        reference = extract_reference(sample)
        if not question or not reference:
            skipped += 1
            continue

        out = strategy_fn(question)
        text = out["text"] if isinstance(out, dict) else out

        seen += 1
        if has_repetition(text):
            reps += 1
        elif score_answer(scorer, text, reference, threshold=threshold):
            correct += 1

    if seen == 0:
        raise RuntimeError("No usable samples found. Check dataset field mappings.")

    return {
        "label": label,
        "n_used": seen,
        "n_skipped": skipped,
        "accuracy": round(correct / seen, 4),
        "rep_rate": round(reps / seen, 4),
    }


if __name__ == "__main__":
    args = parse_args()
    start = time.time()

    print("Loading MedHallu dataset...", flush=True)
    dataset, ds_id, ds_subset, ds_split = load_medhallu_dataset(
        args.dataset,
        args.subset,
        args.split,
    )
    print(
        f"Loaded dataset={ds_id} subset={ds_subset} split={ds_split} rows={len(dataset)}",
        flush=True,
    )

    print("Loading sentence scorer...", flush=True)
    scorer = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    strategies = [
        ("greedy", lambda q: greedy_generate(format_instruct_prompt(q), max_new_tokens=80)),
        ("cove", lambda q: cove_generate(q, max_new_tokens=80)),
        ("dynamic", lambda q: dynamic_generate(q, max_new_tokens=80)),
        ("gadr2", lambda q: gadr2_generate(q, max_new_tokens=80)),
    ]

    results = []
    for label, fn in strategies:
        print(f"Running {label}...", flush=True)
        r = evaluate_strategy(dataset, scorer, label, fn, n=args.n, threshold=args.threshold)
        results.append(r)
        print(
            f"  {label}: acc={r['accuracy']:.1%} rep={r['rep_rate']:.1%} "
            f"used={r['n_used']} skipped={r['n_skipped']}",
            flush=True,
        )

    payload = {
        "dataset": {
            "id": ds_id,
            "subset": ds_subset,
            "split": ds_split,
        },
        "threshold": args.threshold,
        "n_target": args.n,
        "results": results,
        "runtime_min": round((time.time() - start) / 60, 2),
    }

    out_path = ROOT / "results" / "medhallu_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nMedHallu evaluation complete.", flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved results to {out_path}", flush=True)
