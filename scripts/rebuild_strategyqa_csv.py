#!/usr/bin/env python3
"""Build benchmarks/strategyqa_n500.csv - binary yes/no reasoning benchmark.
Used by DoLA, DeLTa, SLED for direct comparison."""

import csv
import sys

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")


print("Downloading StrategyQA...")
candidates = [
    ("wics/strategy-qa", None, "test"),
    ("metaeval/strategy-qa", None, "test"),
    ("kelvin-jiang/strategy-qa", None, "test"),
]

ds = None
for ds_id, subset, split in candidates:
    try:
        ds = load_dataset(ds_id, split=split) if not subset else load_dataset(ds_id, subset, split=split)
        print(f"  Loaded {ds_id} split={split} rows={len(ds)}")
        break
    except Exception as e:
        print(f"  Tried {ds_id}: {e}")

if ds is None:
    sys.exit("Could not load StrategyQA. Try: pip install datasets --upgrade")

rows = []
for row in ds:
    if len(rows) >= 500:
        break
    q = str(row.get("question", "")).strip()
    ans_raw = row.get("answer")

    if isinstance(ans_raw, bool):
        ans = "yes" if ans_raw else "no"
    elif isinstance(ans_raw, str):
        ans = "yes" if ans_raw.strip().lower() in ("yes", "true", "1") else "no"
    else:
        continue

    if q:
        prompt = f"Question: {q}\nAnswer with only one word: yes or no."
        rows.append({"question": prompt, "answer": ans, "domain": "general"})

out = "benchmarks/strategyqa_n500.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["question", "answer", "domain"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {out}")
