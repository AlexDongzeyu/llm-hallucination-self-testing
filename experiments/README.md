# Experiments

This folder contains research scripts used to build CURED profiles, ablations, figures, and diagnostic analyses. The current result narrative is in [../RESULTS.md](../RESULTS.md); this file documents how each phase is produced.

## Phase Map

| Phase | Goal | Primary Output |
|---|---|---|
| 1 | Measure logit linearity and entropy profiles by scale. | `results/CANONICAL_v2/profile_*.json` |
| 2 | Run protocol ablations for greedy, ALTA, CoVe, and ITI. | `results/CANONICAL_v2/ablation_*_n200.json` |
| 3 | Calibrate router thresholds. | `configs/router_thresholds.json` |
| 4 | Run CURED v2 main evaluations. | `results/CANONICAL_v2/main_cured_*` |
| 5 | Compute statistics, R2 analyses, and paper figures. | `statistics_table.json`, `r2_*`, `paper/figures/` |

## Phase 1 - Logit Linearity Profiles

```bash
python experiments/compute_logit_linearity.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --n 50 \
  --out results/CANONICAL_v2/profile_8b.json
```

Canonical profile summary:

| Scale | Mean R2 | Mean kappa | Mean ECR | H_final | H_peak |
|---|---:|---:|---:|---:|---:|
| 3B | 0.501 | 0.455 | 0.076 | 0.837 | 11.00 |
| 8B | 0.582 | 0.597 | 0.066 | 0.669 | 10.14 |
| 14B | 0.444 | 0.360 | 0.031 | 0.306 | 9.73 |
| 32B | 0.473 | 0.322 | 0.051 | 0.529 | 10.32 |

## Phase 2 - Protocol Ablations

```bash
bash scripts/autodl/run_phase2_ablations.sh
```

TruthfulQA and MedHallu ablations use cosine scoring with `n=200`. See [../RESULTS.md](../RESULTS.md) for the current tables.

## Phase 3 - Router Calibration

```bash
python calibrate_router.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --out configs/router_thresholds.json
```

Current calibrated defaults:

| Threshold | Value | Purpose |
|---|---:|---|
| `tau_R2` | 0.65 | Strict per-question linearity gate. |
| `tau_kappa` | 0.70 | Curvature gate. |
| `tau_ECR` | 0.04 | Entropy compression gate. |
| `tau_H_easy` | 0.5 | Low-entropy confidence threshold. |
| `profile_mean_r2` | 0.582 | Global scale shortcut profile. |

## Phase 4 - Main Evaluation

```bash
bash scripts/autodl/run_all_experiments.sh
```

Single 8B TruthfulQA example:

```bash
python cured.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --load-in-4bit \
  --protocols cured \
  --router new \
  --router-config configs/router_thresholds.json \
  --benchmark truthfulqa \
  --scoring cosine \
  --n 500 --seed 42 \
  --save-per-question \
  --out results/CANONICAL_v2/main_cured_8b_truthfulqa_n500_v2.json
```

## Phase 5 - Statistics And Figures

```bash
python compute_final_stats.py \
  --results-dir results/CANONICAL_v2 \
  --output results/CANONICAL_v2/statistics_table.json

python experiments/compute_scale_r2_correlation.py
python experiments/generate_paper_figures.py
python scripts/build_all_results_md.py
```

## Diagnostic Runs

| Diagnostic | Script or Command | Output |
|---|---|---|
| Semantic entropy ablation | `run_semantic_entropy_ablation.py` | `semantic_entropy_gate_comparison.json` |
| FACTOR benchmark prep | `scripts/prep_factor_benchmark.py` | `benchmarks/factor_*_n200.csv` |
| FACTOR-Wiki fixed rerun | command in [../RESULTS.md](../RESULTS.md) | `results_8b_factor_wiki_n200_fixed.json` |
| 3B native profile ablation | `configs/router_thresholds_3b.json` | `main_cured_3b_truthfulqa_n500_v2_native_profile.json` |

## Script Reference

| Script | Phase | Description |
|---|---:|---|
| `compute_logit_linearity.py` | 1 | Per-model R2 profile measurement. |
| `compute_linearity_8b_groq.py` | 1 | API-based 8B R2 check. |
| `compute_scale_r2_correlation.py` | 5 | Scale-level R2 correlation summary. |
| `run_alta_3b.py` | 2 | Standalone 3B ALTA ablation. |
| `run_delta_dola_sweep.py` | 2 | Delta-DoLa hyperparameter sweep. |
| `run_delta_dola_complete_grid.py` | 2 | Full Delta-DoLa grid. |
| `run_medhallu_ablations.py` | 2 | MedHallu protocol ablations. |
| `run_medhallu_eval.py` | 2 | MedHallu evaluation helper. |
| `run_medhallu_generation.py` | 2 | MedHallu generation batches. |
| `build_routing_dataset.py` | 3 | Routing dataset builder. |
| `check_low_threshold.py` | 3 | Threshold diagnosis. |
| `run_semantic_entropy_ablation.py` | 4/5 | ECR vs semantic entropy routing comparison. |
| `generate_paper_figures.py` | 5 | Paper figure generation. |
| `regenerate_figures.py` | 5 | Rebuild figures from saved data. |
| `extract_entropy_layers.py` | 5 | Per-layer entropy extraction. |
| `latency_benchmark.py` | 5 | Protocol latency measurement. |
