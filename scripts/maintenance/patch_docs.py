#!/usr/bin/env python3
"""
patch_docs.py — updates README.md, all_results.md, QUICKSTART.md
with MC validity caveats and rerun guidance.
Run from repo root: python patch_docs.py
"""
from pathlib import Path

# ── README.md ────────────────────────────────────────────────────────────────
README = Path("README.md")
README_ADDITION = """
## ⚠ MC Scoring Validity Note

Results in `results/CANONICAL_v2/results_*_truthfulqa_full_mc.json` (pre-v2 suffix)
were produced before the protocol-aware MC scoring fix and **must not be cited**
for protocol comparison. All protocols scored identically because `mc_score_sample()`
evaluated raw model log-probs independent of the generation strategy.

Use only files with the `_v2` suffix for TruthfulQA MC comparisons.
Generation-scored results (`results_8b_both.json`, `results_3b_*`) are unaffected
and remain valid.
"""

if README.exists():
    txt = README.read_text(encoding="utf-8")
    if "MC Scoring Validity Note" not in txt:
        txt += README_ADDITION
        README.write_text(txt, encoding="utf-8")
        print("✓ README.md updated")
    else:
        print("✓ README.md already has MC caveat")
else:
    print("⚠ README.md not found")

# ── all_results.md ───────────────────────────────────────────────────────────
ALL_RESULTS = Path("all_results.md")
MC_CAVEAT = """
## ⚠ MC Scoring Status

The following files are **superseded pending v2 reruns** and must not be used
for protocol comparison in the paper:

- `results/CANONICAL_v2/results_8b_truthfulqa_full_mc.json` — all protocols identical (pre-fix)
- `results/CANONICAL_v2/results_3b_truthfulqa_full_mc.json` — all protocols identical (pre-fix)

**Cause**: `mc_score_sample()` evaluated baseline model log-probs regardless of
protocol. Fixed in cured.py by `_average_choice_log_prob_alta()` + strategy-faithful
`mc_protocol` threading. See `patch_cured.py` for full diff.

**Valid for immediate use** (unaffected by MC bug):
- All cosine-scored results (`results_8b_both.json`, `results_3b_medhallu_n100.json`, etc.)
- All letter/yesno-scored results (`results_8b_medqa_v3_fixed.json`, `results_8b_pubmedqa_v2.json`)
- Generation results in `results/archive/comprehensive_results_legacy_origin_alex.md`

**Rerun status**: Pending — see QUICKSTART.md for exact commands.
"""

if ALL_RESULTS.exists():
    txt = ALL_RESULTS.read_text(encoding="utf-8")
    if "MC Scoring Status" not in txt:
        # Insert after the first heading block
        lines = txt.split("\n")
        insert_at = 5
        lines.insert(insert_at, MC_CAVEAT)
        ALL_RESULTS.write_text("\n".join(lines), encoding="utf-8")
        print("✓ all_results.md updated with MC status block")
    else:
        print("✓ all_results.md already has MC status")
else:
    print("⚠ all_results.md not found")

# ── QUICKSTART.md ────────────────────────────────────────────────────────────
QS_PATH = Path("scripts/autodl/QUICKSTART.md")
QS_ADDITION = """
## Protocol-Aware MC Rerun (Required for Paper)

After merging the cured.py patch (`python patch_cured.py cured.py`):

### Run 8B and 3B MC reruns in parallel (A100 40GB — both fit)

```bash
# Window 1 — 8B TruthfulQA MC v2
# No --load-in-4bit: A100 has 40GB, bfloat16 gives cleaner results
python -u cured.py \\
  --model meta-llama/Llama-3.1-8B-Instruct \\
  --protocols greedy,alta,cove,cured \\
  --skip-iti \\
  --benchmark truthfulqa \\
  --n 817 --scoring mc --max-new-tokens 50 \\
  --force-recalibrate \\
  --out results/CANONICAL_v2/results_8b_truthfulqa_full_mc_v2.json \\
  > logs/8b_tqa_mc_v2.log 2>&1 &

# Window 2 — 3B TruthfulQA MC v2 (run simultaneously)
python -u cured.py \\
  --model meta-llama/Llama-3.2-3B-Instruct \\
  --protocols greedy,alta,delta_dola,cove,cured \\
  --skip-iti \\
  --benchmark truthfulqa \\
  --n 817 --scoring mc --max-new-tokens 50 \\
  --force-recalibrate \\
  --out results/CANONICAL_v2/results_3b_truthfulqa_full_mc_v2.json \\
  > logs/3b_tqa_mc_v2.log 2>&1 &
```

### After MC v2 completes — generation eval with larger n

```bash
# 8B both benchmarks, cosine, n=100, no quantization
python -u cured.py \\
  --model meta-llama/Llama-3.1-8B-Instruct \\
  --protocols greedy,alta,cove,cured \\
  --skip-iti \\
  --benchmark both \\
  --n 100 --scoring cosine --max-new-tokens 80 \\
  --force-recalibrate \\
  --out results/CANONICAL_v2/results_8b_both_n100.json \\
  > logs/8b_both_n100.log 2>&1
```

### Archive old (pre-fix) MC files only after v2 validation

```bash
mv results/CANONICAL_v2/results_8b_truthfulqa_full_mc.json results/archive/
mv results/CANONICAL_v2/results_3b_truthfulqa_full_mc.json results/archive/
```

### --force-recalibrate is required

The cached `calibration.json` has `alta_viable: false` because it was generated
when `ALTA_R2_CUTOFF=0.65`. With the patch setting it to `0.50`, and R²=0.5557,
ALTA becomes viable — but only after a forced recalibrate overwrites the cache.
"""

if QS_PATH.exists():
    txt = QS_PATH.read_text(encoding="utf-8")
    if "Protocol-Aware MC Rerun" not in txt:
        txt += QS_ADDITION
        QS_PATH.write_text(txt, encoding="utf-8")
        print("✓ QUICKSTART.md updated with rerun commands")
    else:
        print("✓ QUICKSTART.md already has rerun section")
else:
    print(f"  QUICKSTART.md not at {QS_PATH} — writing standalone rerun_guide.md")
    Path("rerun_guide.md").write_text("# Rerun Guide\n" + QS_ADDITION, encoding="utf-8")
    print("✓ rerun_guide.md written")

print("\nDoc patch complete.")
