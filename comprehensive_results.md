# Comprehensive Results Snapshot

Last update: 2026-04-14 03:12 (+08)

This is the canonical status summary for the repository.

## Current Run Status

- status: NOT_DONE
- active run log: logs/final_suite_nohup_20260413_112100.log
- latest markers:
  - [11:21:00] JOB 1: TruthfulQA FULL 817 - MC1/MC2 scoring
  - [21:54:36] JOB 1 DONE
  - [21:54:36] JOB 2: TruthfulQA FULL 817 - 3B model

## Canonical Results Location

Use only `results/CANONICAL_v2/` for report tables.

Expected files:

- results/CANONICAL_v2/results_8b_truthfulqa_full_mc.json
- results/CANONICAL_v2/results_3b_truthfulqa_full_mc.json
- results/CANONICAL_v2/results_8b_medhallu_v2.json
- results/CANONICAL_v2/results_8b_pubmedqa_v2.json
- results/CANONICAL_v2/results_8b_medqa_v3_fixed.json
- results/CANONICAL_v2/results_3b_medhallu_n100.json
- results/CANONICAL_v2/results_8b_strategyqa_v1.json
- results/CANONICAL_v2/results_8b_triviaqa_v1.json

## Job 1 Synced Summary (8B TruthfulQA MC)

Artifact:

- results/CANONICAL_v2/results_8b_truthfulqa_full_mc.json

Metrics:

- greedy: acc 0.4027, mc1 0.4027, mc2 0.4770, runtime_min 47.73
- alta: acc 0.4027, mc1 0.4027, mc2 0.4770, runtime_min 49.83
- cove: acc 0.4027, mc1 0.4027, mc2 0.4770, runtime_min 212.01
- cured: acc 0.4027, mc1 0.4027, mc2 0.4770, runtime_min 57.37

## Why MC Accuracies Match

This is expected in current code.

- TruthfulQA `scoring=mc` uses candidate choice log-likelihood scoring.
- Accuracy comes from the MC scoring function, not protocol-generated text.
- Therefore protocol accuracies can match while routing/runtime differ.

## Previous Summary Locations (No Data Lost)

- Previous path used for this summary: `results/comprehensive_results.md`
- Detailed legacy summary restored at root: `raw_results.md`
- Legacy copy of old results-folder summary: `results/archive/comprehensive_results_legacy_origin_alex.md`
- Historical versions are also recoverable from git history.

## Document Layout (Simple)

- `comprehensive_results.md`: active one-page status (this file)
- `raw_results.md`: detailed legacy snapshot from origin/alex
- `results/archive/comprehensive_results_legacy_origin_alex.md`: archived old results-folder summary

## Cleanup Rules

- Keep this summary at repo root: `comprehensive_results.md`
- Keep canonical JSON outputs in `results/CANONICAL_v2/`
- Keep archived/superseded artifacts in `results/archive/`