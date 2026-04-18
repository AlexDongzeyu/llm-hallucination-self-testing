#!/usr/bin/env python3
"""Parse and display all v2 result files + update all_results.md."""
import json, glob, os, re
from datetime import datetime

CANONICAL = "results/CANONICAL_v2"

def parse_file(path):
    d = json.load(open(path))
    model = d.get("model", "?")
    bench_global = d.get("benchmark", "?")
    scoring = d.get("scoring", "cosine")
    n_target = d.get("n_target")
    rows = []
    results = d.get("results", {})
    if not isinstance(results, dict):
        return rows
    for bk, bval in results.items():
        if not isinstance(bval, dict):
            continue
        for proto, pval in bval.items():
            if not isinstance(pval, dict):
                continue
            acc = pval.get("accuracy")
            n = pval.get("n_questions", pval.get("n", n_target))
            rt = pval.get("runtime_min")
            if rt is None and pval.get("runtime_s"):
                rt = round(pval["runtime_s"] / 60, 2)
            rep = pval.get("repetition_rate", 0)
            n_scored = pval.get("n_scored", n)
            rows.append({
                "file": "results/CANONICAL_v2/" + os.path.basename(path),
                "model": model, "benchmark": bk, "protocol": proto,
                "scoring": scoring, "n": n,
                "acc": f"{acc*100:.1f}%" if acc is not None else "?",
                "rep": f"{rep*100:.1f}%" if rep is not None else "0.0%",
                "n_scored": n_scored,
                "runtime_min": round(rt, 2) if rt else "?",
            })
    return rows

# --- Collect all v2 new main results ---
v2_files = sorted(glob.glob(os.path.join(CANONICAL, "main_cured_*_v2.json")))
print(f"\n{'='*70}")
print(f"New CURED v2 Main Results ({len(v2_files)} files)")
print(f"{'='*70}")
all_rows = []
for f in v2_files:
    rows = parse_file(f)
    all_rows.extend(rows)
    for r in rows:
        print(f"  {os.path.basename(r['file'])}")
        print(f"    model={r['model'].split('/')[-1]}  bench={r['benchmark']}  protocol={r['protocol']}")
        print(f"    acc={r['acc']}  n={r['n']}  n_scored={r['n_scored']}  runtime={r['runtime_min']}min  rep={r['rep']}")

# --- Also check ablation _v2 if any ---
abl_v2 = sorted(glob.glob(os.path.join(CANONICAL, "ablation_*_v2.json")))
if abl_v2:
    print(f"\n{'='*70}")
    print(f"Ablation v2 files ({len(abl_v2)})")
    print(f"{'='*70}")
    for f in abl_v2:
        rows = parse_file(f)
        for r in rows:
            print(f"  {r['file']} | {r['protocol']} | acc={r['acc']}")

# --- Build markdown table for all_results.md update ---
print(f"\n{'='*70}")
print("Markdown table rows for all_results.md:")
print(f"{'='*70}")
print("| file | model | benchmark | protocol | scoring | n | acc | rep | n_scored | runtime_min |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in all_rows:
    print(f"| {r['file']} | {r['model']} | {r['benchmark']} | {r['protocol']} | {r['scoring']} | {r['n']} | {r['acc']} | {r['rep']} | {r['n_scored']} | {r['runtime_min']} |")

print("\nDone.")
