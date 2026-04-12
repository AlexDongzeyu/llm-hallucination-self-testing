#!/usr/bin/env python3
"""
rebuild_medhallu_csv.py

Rebuilds benchmarks/medhallu_n200.csv from the real MedHallu dataset.
"""

from __future__ import annotations

import csv
import re
import sys

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")

N = 200
OUTPUT = "benchmarks/medhallu_n200.csv"


def first_sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    sent = parts[0].strip() if parts else text
    return sent if sent else text


def pick(sample: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        val = sample.get(key)
        if val is None:
            continue
        s = str(val).strip()
        if s:
            return s
    return ""


def main() -> None:
    print(f"Downloading MedHallu (target n={N})...")

    ds = None
    for subset in ("pqa_artificial", "pqa_labeled"):
        try:
            ds = load_dataset("UTAustin-AIHealth/MedHallu", subset, split="train")
            print(f"  Loaded UTAustin-AIHealth/MedHallu subset={subset}, n={len(ds)}")
            break
        except Exception as exc:
            print(f"  Failed subset={subset}: {exc}")

    if ds is None:
        sys.exit("Could not load MedHallu dataset.")

    print(f"  Fields: {list(ds[0].keys())}")

    rows = 0
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "domain"])
        writer.writeheader()

        for sample in ds:
            if rows >= N:
                break

            q = pick(sample, ("Question", "question", "query", "prompt"))
            gt = pick(sample, ("Ground Truth", "ground_truth", "answer", "reference", "best_answer"))
            if not q or not gt:
                continue

            ref = first_sentence(gt)
            if len(ref.split()) < 5:
                ref = gt[:220].strip()

            prompt = f"Question: {q}\nAnswer concisely in 1-2 sentences."
            writer.writerow({"question": prompt, "answer": ref, "domain": "medical"})
            rows += 1

            if rows % 50 == 0:
                print(f"  Written {rows}/{N}")

    print(f"Done. Wrote {rows} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
