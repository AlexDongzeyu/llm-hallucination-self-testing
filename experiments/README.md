# experiments/ — Research Experiments

This folder contains the per-phase research scripts that produced the results
in the paper.  They are **not** part of the reproducible benchmark pipeline
(which lives in `scripts/autodl/`) but document the research process.

---

## 5-Phase Pipeline

### Phase 1 — Logit Linearity Measurement

**Goal**: Measure per-model late-layer R² to determine whether ALTA is viable.

```bash
python experiments/compute_logit_linearity.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --n 50 --out results/CANONICAL_v2/profile_8b.json
```

Key output: `results/CANONICAL_v2/profile_{3b,8b,14b,32b}.json`

Each profile contains `mean_r2`, `alta_viable`, `d2h_threshold`, `iti_available`.

**Key finding**: R² ≥ 0.55 for all tested models → ALTA globally viable.

---

### Phase 2 — Protocol Ablations

**Goal**: Establish individual protocol baselines before routing.

```bash
bash scripts/autodl/run_phase2_ablations.sh
```

Runs greedy, ALTA, CoVe, ITI, and SelfCheck on TruthfulQA and MedHallu
(n=200, all model sizes).

Key outputs: `results/CANONICAL_v2/ablation_*_n200.json`

**Key finding**: ALTA dominates on TruthfulQA (8B: 59.3% vs greedy 43.2%).
CoVe underperforms on general QA (8B: 39.2%) → restrict to medical domain.

---

### Phase 3 — Router Threshold Calibration

**Goal**: Calibrate τ_κ, τ_ECR, τ_R2, τ_H_easy, τ_H_hard from profile data.

```bash
python calibrate_router.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --out configs/router_thresholds.json
```

**Critical fix (Issue #1)**: Original thresholds `tau_kappa=0.08, tau_ECR=0.10`
passed no questions through Gate 2 (mean kappa=0.597, mean ECR=0.031–0.076).
Fixed to `tau_kappa=0.70, tau_ECR=0.04`.

---

### Phase 4 — Main CURED v2 Evaluation

**Goal**: Full n=500 evaluation with the fixed router on all models and benchmarks.

```bash
# Full pipeline (A100/A800):
bash scripts/autodl/run_all_experiments.sh

# Single model:
python cured.py \
  --model meta-llama/Llama-3.1-8B-Instruct --load-in-4bit \
  --protocols cured --router new \
  --router-config configs/router_thresholds.json \
  --benchmark truthfulqa --n 500 --seed 42 --no-shuffle \
  --scoring cosine --save-per-question --skip-iti \
  --out results/CANONICAL_v2/main_cured_8b_truthfulqa_n500_v2.json
```

Key outputs: `results/CANONICAL_v2/main_cured_*_n500_v2.json`

---

### Phase 5 — Statistics and R²-Stratified Analysis

**Goal**: McNemar tests, confidence intervals, and R²-stratified ALTA analysis.

```bash
python compute_final_stats.py \
  --results-dir results/CANONICAL_v2 \
  --output results/CANONICAL_v2/statistics_table.json
```

Key outputs:
- `statistics_table.json` — McNemar p-values, CIs, power analysis
- `r2_stratified_analysis.json` — ALTA accuracy by R² quartile

---

## Experiment Scripts Reference

| Script | Phase | Description |
|---|---|---|
| `compute_logit_linearity.py` | 1 | Per-model R² profile measurement |
| `compute_linearity_8b_groq.py` | 1 | R² via Groq API (no local GPU) |
| `eval_base.py` | 1 | Base evaluation with early stopping |
| `eval_calibration_phase1.py` | 1 | Calibration phase 1 evaluation |
| `run_alta_3b.py` | 2 | ALTA 3B standalone ablation |
| `run_delta_dola_sweep.py` | 2 | Δ-DoLa hyperparameter sweep |
| `run_medhallu_ablations.py` | 2 | MedHallu protocol ablations |
| `run_medhallu_eval.py` | 2 | MedHallu generation eval |
| `run_medhallu_generation.py` | 2 | MedHallu generation (batch) |
| `build_routing_dataset.py` | 3 | Build routing training dataset |
| `check_low_threshold.py` | 3 | Diagnose low-threshold issues |
| `eval_grid.py` | 3 | Grid search over thresholds |
| `run_semantic_entropy_ablation.py` | 4/5 | ECR vs semantic-entropy comparison |
| `generate_paper_figures.py` | 5 | Generate all paper figures |
| `regenerate_figures.py` | 5 | Re-generate figures from saved data |
| `extract_entropy_layers.py` | 5 | Extract per-layer entropy profiles |
| `latency_benchmark.py` | 5 | Measure per-protocol latency |

---

## Pipeline Scripts (PowerShell, local development)

These were used for local Windows development runs before the A100 pipeline
was established:

- `pipeline_all_steps.ps1` — Orchestrate all phases locally
- `pipeline_remaining_steps.ps1` — Continue from a checkpoint
- `pipeline_common.ps1` — Shared helpers

These are not needed for GPU-cluster reproduction.
