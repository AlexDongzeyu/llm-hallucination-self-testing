#!/usr/bin/env python3
"""Download and convert FACTOR benchmark to cured.py CSV format.

Source: https://github.com/AI21Labs/factor
  data/wiki_factor.csv  — 2994 examples from Wikipedia
  data/news_factor.csv  — 1036 examples from Reuters

CSV schema (FACTOR original):
  full_prefix, completion, contradiction_0, contradiction_1, contradiction_2

Output format (cured.py custom-csv):
  question, answer, domain
  where question is an MC prompt with A-D choices, answer is the letter (A-D).
  Options are shuffled per-row (seeded by row index) so the correct answer
  is not always at position A.

Usage:
    python scripts/prep_factor_benchmark.py

Writes:
    benchmarks/factor_news_n200.csv
    benchmarks/factor_wiki_n200.csv
"""
import csv
import io
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"
BENCHMARKS_DIR.mkdir(exist_ok=True)

GITHUB_BASE = "https://raw.githubusercontent.com/AI21Labs/factor/main/data"
LETTERS = ["A", "B", "C", "D"]

SUBSETS = {
    "news": ("news_factor.csv", 200),
    "wiki": ("wiki_factor.csv", 200),
}


def download_csv(url: str) -> list[dict]:
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    return list(reader)


def build_mc_question(prefix: str, correct: str, distractors: list[str], seed: int) -> tuple[str, str]:
    """Return (question_text, answer_letter) with shuffled options."""
    rng = random.Random(seed)
    options = [correct] + distractors[:3]
    rng.shuffle(options)
    correct_idx = options.index(correct)
    option_lines = "\n".join(f"{LETTERS[i]}. {opt.strip()}" for i, opt in enumerate(options))
    question = (
        "Complete the following text with the most factually accurate option:\n"
        f"{prefix.strip()}\n{option_lines}\n"
        "Answer with only the letter A, B, C, or D."
    )
    return question, LETTERS[correct_idx]


def prep_subset(subset: str, filename: str, n_target: int) -> None:
    url = f"{GITHUB_BASE}/{filename}"
    try:
        rows = download_csv(url)
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        print(f"  Manual download: wget {url} -O benchmarks/{filename}")
        return

    print(f"  Loaded {len(rows)} rows from {filename}")
    out_rows = []

    for i, row in enumerate(rows):
        if len(out_rows) >= n_target:
            break
        prefix = row.get("full_prefix", "").strip()
        completion = row.get("completion", "").strip()
        c0 = row.get("contradiction_0", "").strip()
        c1 = row.get("contradiction_1", "").strip()
        c2 = row.get("contradiction_2", "").strip()

        if not prefix or not completion or not c0:
            continue

        distractors = [d for d in [c0, c1, c2] if d]
        question, answer = build_mc_question(prefix, completion, distractors, seed=i)
        out_rows.append({"question": question, "answer": answer, "domain": "general"})

    out_path = BENCHMARKS_DIR / f"factor_{subset}_n{len(out_rows)}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "answer", "domain"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"  Wrote {len(out_rows)} rows -> {out_path}")


if __name__ == "__main__":
    for subset, (filename, n_target) in SUBSETS.items():
        print(f"\nPreparing FACTOR {subset}...")
        prep_subset(subset, filename, n_target)
    print("\nDone.")
