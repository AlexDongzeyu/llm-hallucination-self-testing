"""MedHallu generation evaluation (primary metric).

Scoring: cosine similarity to Ground Truth.
Default strategies: greedy, cove, gadr2_cured, cove_rag, delta_dola.
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
    cove_generate,
    cove_rag_generate,
    delta_dola_generate,
    format_instruct_prompt,
    gadr2_generate,
    greedy_generate,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument(
        "--strategies",
        type=str,
        default="greedy,cove,gadr2_cured,cove_rag,delta_dola",
        help="comma-separated subset in execution order",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80, dest="max_new_tokens")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="resume from existing output file if present",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="ignore existing output file and start from scratch",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "medhallu_generation_results.json"),
    )
    return parser.parse_args()


def has_repetition(text: str, window: int = 5) -> bool:
    words = text.lower().split()
    if len(words) < max(window * 2, 20):
        return False
    ngrams = [tuple(words[i : i + window]) for i in range(len(words) - window + 1)]
    return len(ngrams) != len(set(ngrams))


def score_cosine(scorer, generated: str, reference: str, threshold: float) -> bool:
    """Return True if cosine similarity >= threshold (original generation metric)."""
    if not generated.strip():
        return False
    eg = scorer.encode(generated, convert_to_tensor=True, device="cpu")
    er = scorer.encode(reference, convert_to_tensor=True, device="cpu")
    return util.cos_sim(eg, er).item() >= threshold


def evaluate_generation_strategy(
    dataset,
    scorer,
    label: str,
    strategy_fn,
    n: int,
    threshold: float,
    progress_hook=None,
    progress_every: int = 5,
) -> dict:
    """
    Original generation-based evaluation.
    Generate a free-form answer, score against Ground Truth via cosine similarity.
    This is the primary CURED paper metric for MedHallu.
    """
    correct = 0
    reps = 0
    skipped = 0
    seen = 0
    by_difficulty = {}

    for sample in dataset:
        if seen >= n:
            break

        # Resolve commonly observed field names.
        question = ""
        for key in ["Question", "question", "query"]:
            if key in sample and sample[key]:
                question = str(sample[key]).strip()
                break

        gt = str(sample.get("Ground Truth", "") or "").strip()
        difficulty = str(sample.get("Difficulty Level", "unknown") or "unknown").strip().lower()

        if not question or not gt:
            skipped += 1
            continue

        out = strategy_fn(question)
        text = out["text"] if isinstance(out, dict) else str(out)

        repeated = has_repetition(text)
        matched = False if repeated else score_cosine(scorer, text, gt, threshold)

        seen += 1
        if repeated:
            reps += 1
        elif matched:
            correct += 1

        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = {"correct": 0, "total": 0}
        by_difficulty[difficulty]["total"] += 1
        if matched:
            by_difficulty[difficulty]["correct"] += 1

        if seen % 10 == 0:
            print(
                f"  [{seen}/{n}] acc={correct/seen:.1%} rep={reps/seen:.1%}",
                flush=True,
            )

        if progress_hook and (seen % max(progress_every, 1) == 0 or seen == n):
            progress_hook(
                {
                    "label": label,
                    "seen": seen,
                    "n_target": n,
                    "correct": correct,
                    "repeated": reps,
                    "skipped": skipped,
                    "accuracy": round(correct / max(seen, 1), 4),
                    "rep_rate": round(reps / max(seen, 1), 4),
                }
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
    }


if __name__ == "__main__":
    args = parse_args()
    start = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading MedHallu pqa_artificial...", flush=True)
    dataset = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial", split="train")
    print(f"Loaded {len(dataset)} rows", flush=True)

    print("Loading scorer...", flush=True)
    scorer = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # Canonical strategy builders.
    all_strategies = [
        (
            "greedy",
            lambda q: greedy_generate(format_instruct_prompt(q), max_new_tokens=args.max_new_tokens),
        ),
        ("cove", lambda q: cove_generate(q, max_new_tokens=args.max_new_tokens)),
        ("gadr2_cured", lambda q: gadr2_generate(q, max_new_tokens=args.max_new_tokens)),
        ("cove_rag", lambda q: cove_rag_generate(q, max_new_tokens=args.max_new_tokens)),
        (
            "delta_dola",
            lambda q: delta_dola_generate(
                format_instruct_prompt(q),
                max_new_tokens=args.max_new_tokens,
                alpha1=0.3,
                alpha2=0.3,
            ),
        ),
    ]

    selected = [s.strip() for s in args.strategies.split(",") if s.strip()]
    valid_labels = {label for label, _ in all_strategies}
    unknown = [s for s in selected if s not in valid_labels]
    if unknown:
        raise ValueError(f"Unknown strategy labels: {unknown}. Valid: {sorted(valid_labels)}")

    # Respect caller order while still using canonical builders.
    builder_by_label = {label: fn for label, fn in all_strategies}
    strategies = [(label, builder_by_label[label]) for label in selected]

    results = []
    completed_labels = set()
    if args.resume and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            existing_results = existing.get("results", [])
            # Keep first occurrence of each label to avoid duplicates across retries.
            for item in existing_results:
                label = str(item.get("label", "")).strip()
                if not label or label in completed_labels:
                    continue
                results.append(item)
                completed_labels.add(label)
            if completed_labels:
                print(
                    f"Resuming run with completed strategies: {', '.join(sorted(completed_labels))}",
                    flush=True,
                )
        except Exception as exc:
            print(f"Could not parse existing output for resume: {exc}", flush=True)

    for label, fn in strategies:
        if label in completed_labels:
            print(f"\n=== Skipping {label} (already complete) ===", flush=True)
            continue

        print(f"\n=== Running {label} ===", flush=True)
        t0 = time.time()
        current_progress = {
            "label": label,
            "seen": 0,
            "n_target": args.n,
            "correct": 0,
            "repeated": 0,
            "skipped": 0,
            "accuracy": 0.0,
            "rep_rate": 0.0,
        }

        def checkpoint_progress(progress: dict):
            checkpoint = {
                "dataset": "UTAustin-AIHealth/MedHallu pqa_artificial train",
                "metric": "cosine_similarity_to_ground_truth",
                "threshold": args.threshold,
                "n_target": args.n,
                "status": "in_progress",
                "current_strategy": progress,
                "completed_labels": sorted(completed_labels),
                "results": results,
                "runtime_min": round((time.time() - start) / 60, 2),
            }
            out_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

        checkpoint_progress(current_progress)
        r = evaluate_generation_strategy(
            dataset,
            scorer,
            label,
            fn,
            n=args.n,
            threshold=args.threshold,
            progress_hook=checkpoint_progress,
            progress_every=5,
        )
        r["runtime_min"] = round((time.time() - t0) / 60, 2)
        results.append(r)
        completed_labels.add(label)

        # Checkpoint after each strategy so long runs can resume safely.
        checkpoint = {
            "dataset": "UTAustin-AIHealth/MedHallu pqa_artificial train",
            "metric": "cosine_similarity_to_ground_truth",
            "threshold": args.threshold,
            "n_target": args.n,
            "status": "in_progress",
            "current_strategy": None,
            "completed_labels": sorted(completed_labels),
            "results": results,
            "runtime_min": round((time.time() - start) / 60, 2),
        }
        out_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

        print(
            f"  DONE: acc={r['accuracy']:.1%} rep={r['rep_rate']:.1%} "
            f"({r['runtime_min']:.0f}min)",
            flush=True,
        )

    print("\n=== RESULTS SUMMARY ===", flush=True)
    print(f"{'Strategy':<20} {'Accuracy':>10} {'Rep Rate':>10}", flush=True)
    print("-" * 42, flush=True)
    for r in results:
        print(
            f"{r['label']:<20} {r['accuracy']:>10.1%} {r['rep_rate']:>10.1%}",
            flush=True,
        )

    print("\n=== BY DIFFICULTY ===", flush=True)
    for r in results:
        print(f"\n{r['label']}:", flush=True)
        for diff in ["easy", "medium", "hard"]:
            if diff in r["by_difficulty"]:
                d = r["by_difficulty"][diff]
                print(f"  {diff}: {d['accuracy']:.1%} (n={d['n']})", flush=True)

    payload = {
        "dataset": "UTAustin-AIHealth/MedHallu pqa_artificial train",
        "metric": "cosine_similarity_to_ground_truth",
        "threshold": args.threshold,
        "n_target": args.n,
        "status": "complete",
        "current_strategy": None,
        "completed_labels": sorted(completed_labels),
        "results": results,
        "runtime_min": round((time.time() - start) / 60, 2),
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}", flush=True)
