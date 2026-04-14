# CURED: Curvature-Informed Routing and Entropy-based Decoding

Inference-time hallucination mitigation experiments for RLHF-tuned LLMs.

## Repository Layout

- cured.py: unified runner for local and API evaluation protocols.
- benchmarks/: benchmark CSV inputs.
- experiments/: one-off and paper-generation experiment scripts.
- scripts/: reproducible automation scripts (GPU setup, API jobs, CSV rebuilders).
- src/: reusable model-analysis and decoding modules.
- results/: persistent structured artifacts.
- logs/: runtime logs.

## Canonical Result Location

Use results/CANONICAL_v2 as the single source of truth for reportable final tables.

Expected final files:

- results/CANONICAL_v2/results_8b_truthfulqa_full_mc.json
- results/CANONICAL_v2/results_3b_truthfulqa_full_mc.json
- results/CANONICAL_v2/results_8b_medhallu_v2.json
- results/CANONICAL_v2/results_8b_pubmedqa_v2.json
- results/CANONICAL_v2/results_8b_medqa_v3_fixed.json
- results/CANONICAL_v2/results_3b_medhallu_n100.json
- results/CANONICAL_v2/results_openrouter_medqa_v2.json
- results/CANONICAL_v2/results_openrouter_pubmedqa_v2.json
- results/CANONICAL_v2/results_openrouter_medhallu_v2.json

## Borrowed GPU Quickstart

See scripts/autodl/QUICKSTART.md.

Minimal flow:

1) bash scripts/autodl/bootstrap_gpu_env.sh
2) bash scripts/autodl/run_final_suite.sh
3) bash scripts/autodl/organize_final_results.sh

## API Rerun Helper (Windows)

For the fixed MedQA API rerun:

scripts\\run_openrouter_job.cmd medqa

This writes results/results_openrouter_medqa_v2.json.

## Scoring Modes

- cosine: semantic similarity scoring (default for free-form QA).
- letter: multiple-choice letter scoring.
- yesno: binary yes/no/maybe scoring.
- mc: TruthfulQA multiple-choice likelihood scoring with MC1/MC2 summary fields.

## Notes

- Legacy or invalid artifacts should be moved to results/archive.
- Top-level result JSON files are considered legacy and should be normalized via scripts/autodl/organize_final_results.sh.
- For current status narrative, use all_results.md.

## ⚠ MC Scoring Validity Note

Results in `results/CANONICAL_v2/results_*_truthfulqa_full_mc.json` (pre-v2 suffix)
were produced before the protocol-aware MC scoring fix and **must not be cited**
for protocol comparison. All protocols scored identically because `mc_score_sample()`
evaluated raw model log-probs independent of the generation strategy.

Use only files with the `_v2` suffix for TruthfulQA MC comparisons.
Generation-scored results (`results_8b_both.json`, `results_3b_*`) are unaffected
and remain valid.
