# CURED — Canonical Results

> **Single source of truth.** All numbers below come from
> `results/CANONICAL_v2/` and are fully reproducible via the pipeline in
> `scripts/autodl/run_all_experiments.sh`.

## Phase 4 Main Results — CURED v2 Router (Fixed Thresholds)

Router config: `configs/router_thresholds.json`
(`tau_kappa=0.70`, `tau_ECR=0.04`, `profile_mean_r2=0.582`)

| Model | Benchmark | n | Acc (CURED v2) | Runtime |
|---|---|---|---|---|
| Llama-3.2-3B | TruthfulQA | 500 | **60.6%** | 54 min |
| Llama-3.2-3B | MedHallu | 500 | **49.9%** | 52 min |
| Llama-3.2-3B | StrategyQA | 500 | **62.4%** | 4 min |
| Llama-3.1-8B | TruthfulQA | 500 | **60.2%** | 66 min |
| Llama-3.1-8B | MedHallu | 500 | **50.2%** | 55 min |
| Llama-3.1-8B | StrategyQA | 500 | **72.2%** | 5 min |

## Router v1 vs v2 — 8B TruthfulQA

| Router version | File | n | Acc |
|---|---|---|---|
| v1 (old thresholds: tau_kappa=0.08, tau_ECR=0.10) | `main_cured_old_8b_truthfulqa_n500.json` | 500 | 62.9% |
| v2 (fixed thresholds: tau_kappa=0.70, tau_ECR=0.04) | `main_cured_8b_truthfulqa_n500_v2.json` | 500 | **60.2%** |

> Note: v2 shows different routing behaviour (more ALTA + scale-adaptive shortcut)
> vs v1 (was incorrectly passing near nobody through Gate 2).

## Greedy Baseline

| Model | Benchmark | n | Acc (Greedy) | File |
|---|---|---|---|---|
| Llama-3.2-3B | TruthfulQA | 817 | 50.1% | `main_greedy_3b_truthfulqa_n817.json` |
| Llama-3.1-8B | TruthfulQA | 817 | 49.6% | `main_greedy_8b_truthfulqa_n817.json` |
| Qwen-14B | TruthfulQA | 817 | 62.2% | `main_greedy_14b_truthfulqa_n817.json` |
| Qwen-32B | TruthfulQA | 817 | 58.8% | `main_greedy_32b_truthfulqa_n817.json` |

## Phase 4 — Prior Runs (Older Router, for reference)

| Model | Benchmark | Acc | File |
|---|---|---|---|
| Llama-3.2-3B | TruthfulQA | 51.6% | `main_cured_3b_truthfulqa_n500.json` |
| Llama-3.2-3B | MedHallu | 47.5% | `main_cured_3b_medhallu_n500.json` |
| Llama-3.1-8B | TruthfulQA | 49.8% | `main_cured_8b_truthfulqa_n500.json` |
| Llama-3.1-8B | MedHallu | 48.3% | `main_cured_8b_medhallu_n500.json` |
| Qwen-14B | TruthfulQA | 64.0% | `main_cured_14b_truthfulqa_n500.json` |
| Qwen-32B | TruthfulQA | 60.1% | `main_cured_32b_truthfulqa_n500.json` |

## Phase 2 Ablations — Protocol Comparison (8B, n=200)

| Protocol | TruthfulQA Acc | MedHallu Acc |
|---|---|---|
| Greedy | 48.0% | 45.5% |
| ALTA | 60.6% | 59.8% |
| CoVe | 39.2% | 42.4% |
| ITI | 57.5% | 61.1% |

> Source: `results/CANONICAL_v2/ablation_8b_*_n200.json`


## A800 sync — FACTOR, semantic entropy, stats!

- **`results_8b_factor_news_n200.json`** (custom / greedy): acc **59.0%**, n_scored=200, runtime=1.39 min
- **`results_8b_factor_news_n200.json`** (custom / alta): acc **69.0%**, n_scored=200, runtime=1.99 min
- **`results_8b_factor_news_n200.json`** (custom / cured): acc **61.5%**, n_scored=200, runtime=1.57 min
- **`results_8b_factor_wiki_n200.json`** (custom / greedy): acc **29.0%**, n_scored=200, runtime=1.7 min
- **`results_8b_factor_wiki_n200.json`** (custom / alta): acc **64.0%**, n_scored=200, runtime=2.25 min
- **`results_8b_factor_wiki_n200.json`** (custom / cured): acc **43.0%**, n_scored=200, runtime=1.89 min

### Semantic entropy ablation (MedHallu, n=50, k=5)!

- Greedy **34.0%** vs ECR-gate **34.0%** vs SE-gate **44.0%** (`semantic_entropy_gate_comparison.json`).

- Paired tests: `results/CANONICAL_v2/statistics_table.json`.
- R² stratified ALTA analysis: `results/CANONICAL_v2/r2_stratified_analysis.json`.

## Reproducing Results

```bash
# Full pipeline on A100/A800 (run from repo root):
bash scripts/autodl/run_all_experiments.sh

# Quick smoke test (Gate 2 verification, n=20):
python cured.py \
  --model meta-llama/Llama-3.1-8B-Instruct --load-in-4bit \
  --protocols cured --router new \
  --router-config configs/router_thresholds.json \
  --benchmark truthfulqa --n 20 --save-per-question \
  --out /tmp/smoke_test.json
```

## Threshold Configuration

See `configs/router_thresholds.json` for all router thresholds.
Critical calibrated values:

```json
{
  "tau_kappa":       0.70,
  "tau_ECR":         0.04,
  "tau_R2":          0.50,
  "profile_mean_r2": 0.582
}
```
