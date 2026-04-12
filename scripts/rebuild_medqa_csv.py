#!/usr/bin/env python3
"""
rebuild_medqa_csv.py

Rebuilds benchmarks/medqa_usmle_n200.csv with options embedded in the question
and a single-letter answer key for MC scoring.
"""

from __future__ import annotations

import csv
import sys
from typing import Any

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")

N = 200
OUTPUT = "benchmarks/medqa_usmle_n200.csv"
LETTERS = ["A", "B", "C", "D", "E"]


def _extract_options(sample: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}

    opts = sample.get("options")
    if isinstance(opts, dict):
        for k, v in opts.items():
            key = str(k).strip().upper()
            if key in LETTERS:
                txt = str(v).strip()
                if txt:
                    out[key] = txt
        return out

    if isinstance(opts, list):
        for item in opts:
            if isinstance(item, dict):
                key = str(item.get("key", "")).strip().upper()
                val = str(item.get("value", "")).strip()
                if key in LETTERS and val:
                    out[key] = val
        if out:
            return out

    # Fallback schema used by some exports.
    for k in LETTERS:
        cand = sample.get(k) or sample.get(k.lower()) or sample.get(f"op{k.lower()}")
        if cand:
            out[k] = str(cand).strip()

    return out


def _extract_answer_letter(sample: dict[str, Any]) -> str:
    for key in ("answer_idx", "answer", "correct_option", "label", "cop"):
        if key not in sample:
            continue
        val = sample[key]
        if isinstance(val, int):
            if 0 <= val < len(LETTERS):
                return LETTERS[val]
        sval = str(val).strip().upper()
        if len(sval) == 1 and sval in LETTERS:
            return sval
        if sval.isdigit():
            idx = int(sval)
            if 0 <= idx < len(LETTERS):
                return LETTERS[idx]
    return ""


def main() -> None:
    print(f"Downloading MedQA-USMLE (target n={N})...")

    ds = None
    for ds_id, split in (
        ("GBaker/MedQA-USMLE-4-options", "test"),
        ("GBaker/MedQA-USMLE-4-options", "train"),
    ):
        try:
            ds = load_dataset(ds_id, split=split)
            print(f"  Loaded {ds_id} split={split}, n={len(ds)}")
            break
        except Exception as exc:
            print(f"  Failed {ds_id} split={split}: {exc}")

    if ds is None:
        sys.exit("Could not load MedQA source dataset.")

    rows = 0
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "domain"])
        writer.writeheader()

        for sample in ds:
            if rows >= N:
                break

            q = str(sample.get("question", "")).strip()
            if not q:
                continue

            options = _extract_options(sample)
            answer_letter = _extract_answer_letter(sample)

            if answer_letter not in options:
                continue

            option_lines = [f"{k}. {options[k]}" for k in LETTERS if k in options]
            if len(option_lines) < 4:
                continue

            prompt = (
                f"Question: {q}\n"
                + "\n".join(option_lines)
                + "\nAnswer with only the letter A, B, C, D, or E."
            )

            writer.writerow({"question": prompt, "answer": answer_letter, "domain": "medical"})
            rows += 1

            if rows % 50 == 0:
                print(f"  Written {rows}/{N}")

    print(f"Done. Wrote {rows} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
