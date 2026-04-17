# CURED: Curvature-Informed Routing and Entropy-based Decoding

Inference-time hallucination mitigation experiments for RLHF-tuned LLMs.

## Repository Layout

```
LLM_Hallucination/
│
├── cured.py                  ← MAIN SCRIPT: run all protocols/benchmarks here
├── calibrate_router.py       ← Phase 3: learn routing thresholds from ablations
├── compute_final_stats.py    ← Phase 5: McNemar tests + bootstrap CI
├── requirements.txt
├── README.md
├── all_results.md            ← aggregated results narrative
│
├── configs/
│   └── router_thresholds.json   ← router tau/beta values (updated by calibrate_router.py)
│
├── benchmarks/               ← frozen benchmark CSVs (TruthfulQA, MedHallu, etc.)
│
├── experiments/              ← experiment scripts (sweeps, figures, profiling)
│   ├── compute_logit_linearity.py   ← Phase 1: measure R², κ, ECR per model
│   └── ...
│
├── src/                      ← reusable library modules (generation, probing, routing)
│
├── scripts/
│   ├── prep_benchmarks.py    ← download/rebuild benchmark CSVs
│   ├── deploy_and_queue.py   ← sync code + queue pipeline on remote GPU
│   ├── check_status.py       ← check GPU / pipeline status
│   ├── autodl/               ← GPU server shell scripts (run_phase*.sh, bootstrap)
│   └── maintenance/          ← one-time patch and diagnostic tools (not part of main pipeline)
│
├── data/                     ← ITI probes, trajectory datasets
├── results/
│   └── CANONICAL_v2/         ← SINGLE SOURCE OF TRUTH for all reportable results
├── logs/                     ← runtime logs
└── plots/                    ← visualization outputs
```

**Run the full pipeline on a GPU server:**
```bash
python scripts/deploy_and_queue.py   # syncs code + queues Phase 1→5 automatically
```

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

## Unattended Auto Queue (Safe To Leave VS Code)

Use this when you want post-final-suite jobs to run automatically even after you disconnect:

1) Start the main suite:

	bash scripts/autodl/run_final_suite.sh

2) Start the queued follow-up runner in detached mode:

	nohup bash scripts/autodl/queue_after_triviaqa_v2.sh >/tmp/queue_after_triviaqa_v2_boot.out 2>&1 < /dev/null &

What this queue runs automatically after the suite exits:

- 8B TruthfulQA MC v2 rerun
- 3B TruthfulQA MC v2 rerun (parallel with 8B)
- 8B both n=100 v2 (after both MC jobs succeed)

Live queue log location:

- logs/queue_after_triviaqa_v2_*.log

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
