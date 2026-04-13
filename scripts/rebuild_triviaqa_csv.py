#!/usr/bin/env python3
"""Build benchmarks/triviaqa_n1000.csv - closed-book factual QA benchmark.
Used by DeLTa and SLED for direct comparison."""

import csv
import sys

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")


print("Downloading TriviaQA...")
try:
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    print(f"  Loaded trivia_qa rc.nocontext validation rows={len(ds)}")
except Exception as e:
    sys.exit(f"Could not load TriviaQA: {e}")

rows = []
for row in ds:
    if len(rows) >= 1000:
        break

    q = str(row.get("question", "")).strip()
    ans_data = row.get("answer", {})

    aliases = ans_data.get("aliases", []) if isinstance(ans_data, dict) else []
    value = ans_data.get("value", "") if isinstance(ans_data, dict) else str(ans_data)
    ans = aliases[0] if aliases else value
    ans = str(ans).strip()

    if q and ans:
        rows.append({"question": q, "answer": ans, "domain": "general"})

out = "benchmarks/triviaqa_n1000.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["question", "answer", "domain"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {out}")
