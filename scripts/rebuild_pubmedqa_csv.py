#!/usr/bin/env python3
"""
rebuild_pubmedqa_csv.py

Rebuilds benchmarks/pubmedqa_n200.csv with yes/no/maybe references.
"""

from __future__ import annotations

import csv
import sys

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")

N = 200
OUTPUT = "benchmarks/pubmedqa_n200.csv"
VALID = {"yes", "no", "maybe"}


def main() -> None:
    print(f"Downloading PubMedQA (target n={N})...")

    ds = None
    for subset in ("pqa_labeled", "pqa_artificial"):
        try:
            ds = load_dataset("qiaojin/PubMedQA", subset, split="train")
            print(f"  Loaded qiaojin/PubMedQA subset={subset}, n={len(ds)}")
            break
        except Exception as exc:
            print(f"  Failed subset={subset}: {exc}")

    if ds is None:
        sys.exit("Could not load PubMedQA.")

    rows = 0
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "domain"])
        writer.writeheader()

        for sample in ds:
            if rows >= N:
                break

            q = str(sample.get("question", "")).strip()
            decision = str(sample.get("final_decision", "")).strip().lower()

            if not q or decision not in VALID:
                continue

            prompt = f"Question: {q}\nRespond with only one token: yes, no, or maybe."
            writer.writerow({"question": prompt, "answer": decision, "domain": "medical"})
            rows += 1

            if rows % 50 == 0:
                print(f"  Written {rows}/{N}")

    print(f"Done. Wrote {rows} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
