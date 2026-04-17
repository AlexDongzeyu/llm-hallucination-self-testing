#!/usr/bin/env python3
"""
prep_benchmarks.py - Download MedQA and PubMedQA as CSV files for cured.py.

Run this once before large benchmark sweeps.

Output files:
  benchmarks/medqa_usmle_n200.csv
  benchmarks/pubmedqa_n200.csv
  benchmarks/medhallu_n200.csv

Usage:
  python prep_benchmarks.py
  python prep_benchmarks.py --n 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def p(msg: str) -> None:
    print(msg, flush=True)


def save_csv(rows: list[dict[str, str]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    p(f"  Saved {len(rows)} rows -> {path}")


def prep_medqa(n: int, out_dir: Path) -> None:
    """
    MedQA USMLE benchmark in open-ended format:
    - question: exam question text
    - answer: correct option text
    """
    p(f"\nDownloading MedQA USMLE (n={n})...")
    try:
        from datasets import load_dataset
    except ImportError:
        p("  ERROR: install datasets first")
        return

    candidates = [
        ("GBaker/MedQA-USMLE-4-options", None, "test"),
        ("GBaker/MedQA-USMLE-4-options", None, "train"),
        ("bigbio/med_qa", "med_qa_en_bigbio_qa", "test"),
        ("medmcqa", None, "validation"),
    ]

    ds = None
    for ds_id, subset, split in candidates:
        try:
            ds = load_dataset(ds_id, subset, split=split) if subset else load_dataset(ds_id, split=split)
            p(f"  Loaded: {ds_id} split={split} rows={len(ds)}")
            break
        except Exception as exc:
            p(f"  Tried {ds_id}: {type(exc).__name__}")

    if ds is None:
        p("  Could not load MedQA. Skipping.")
        return

    rows: list[dict[str, str]] = []
    for row in ds:
        if len(rows) >= n:
            break

        question = str(row.get("question", row.get("sent1", "")) or "").strip()
        answer = ""

        if "options" in row and "answer" in row:
            options = row["options"]
            ans_key = str(row["answer"]).strip()
            if isinstance(options, dict):
                answer = str(options.get(ans_key, ans_key)).strip()
            else:
                answer = ans_key
        elif "opa" in row and "cop" in row:
            opt_map = {0: "opa", 1: "opb", 2: "opc", 3: "opd"}
            cop = int(row.get("cop", 0))
            answer = str(row.get(opt_map.get(cop, "opa"), "")).strip()
        elif "choices" in row and "answer_idx" in row:
            choices = row["choices"]
            idx_raw = str(row["answer_idx"])
            idx = int(idx_raw) if idx_raw.isdigit() else 0
            if isinstance(choices, list) and idx < len(choices):
                answer = str(choices[idx]).strip()

        if question and answer:
            rows.append({"question": question, "answer": answer, "domain": "medical"})

    save_csv(rows, out_dir / f"medqa_usmle_n{len(rows)}.csv", ["question", "answer", "domain"])


def prep_pubmedqa(n: int, out_dir: Path) -> None:
    """
    PubMedQA benchmark in open-ended format:
    - question: prompt
    - answer: long_answer (preferred) or answer fallback
    """
    p(f"\nDownloading PubMedQA (n={n})...")
    try:
        from datasets import load_dataset
    except ImportError:
        p("  ERROR: install datasets first")
        return

    candidates = [
        ("qiaojin/PubMedQA", "pqa_artificial", "train"),
        ("qiaojin/PubMedQA", "pqa_labeled", "train"),
        ("pubmed_qa", "pqa_artificial", "train"),
    ]

    ds = None
    for ds_id, subset, split in candidates:
        try:
            ds = load_dataset(ds_id, subset, split=split) if subset else load_dataset(ds_id, split=split)
            p(f"  Loaded: {ds_id}/{subset} split={split} rows={len(ds)}")
            break
        except Exception as exc:
            p(f"  Tried {ds_id}/{subset}: {type(exc).__name__}")

    if ds is None:
        p("  Could not load PubMedQA. Skipping.")
        return

    rows: list[dict[str, str]] = []
    for row in ds:
        if len(rows) >= n:
            break
        q = str(row.get("question", "") or "").strip()
        a = str(row.get("long_answer", row.get("answer", "")) or "").strip()
        if q and a:
            rows.append({"question": q, "answer": a, "domain": "medical"})

    save_csv(rows, out_dir / f"pubmedqa_n{len(rows)}.csv", ["question", "answer", "domain"])


def prep_medhallu_local(n: int, out_dir: Path) -> None:
    """Export MedHallu to a local CSV for reproducible reruns."""
    p(f"\nExporting MedHallu local copy (n={n})...")
    try:
        from datasets import load_dataset
    except ImportError:
        p("  ERROR: install datasets first")
        return

    try:
        ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial", split="train")
        p(f"  Loaded MedHallu: {len(ds)} rows")
    except Exception as exc:
        p(f"  Could not load MedHallu: {exc}")
        return

    rows: list[dict[str, str]] = []
    for row in ds:
        if len(rows) >= n:
            break
        q = str(row.get("Question", "") or "").strip()
        a = str(row.get("Ground Truth", "") or "").strip()
        if q and a:
            rows.append({"question": q, "answer": a, "domain": "medical"})

    save_csv(rows, out_dir / f"medhallu_n{len(rows)}.csv", ["question", "answer", "domain"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark CSVs for cured.py")
    parser.add_argument("--n", type=int, default=200, help="Max rows per benchmark")
    parser.add_argument("--out-dir", default="benchmarks", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep_medqa(args.n, out_dir)
    prep_pubmedqa(args.n, out_dir)
    prep_medhallu_local(args.n, out_dir)

    p(f"\nAll done. CSV files are in {out_dir}/")


if __name__ == "__main__":
    main()
