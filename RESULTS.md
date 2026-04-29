# CURED Results

> This is the human-readable canonical result document. The raw source files live in `results/CANONICAL_v2/`; the full generated ledger is [all_results.md](all_results.md).

## Result Sources

| Source | Role |
|---|---|
| `results/CANONICAL_v2/main_cured_*` | Phase 4 CURED main runs. |
| `results/CANONICAL_v2/main_greedy_*` | Greedy reference baselines. |
| `results/CANONICAL_v2/ablation_*` | Phase 2 protocol ablations. |
| `results/CANONICAL_v2/results_8b_factor_*` | FACTOR diagnostic runs. |
| `results/CANONICAL_v2/statistics_table.json` | McNemar exact binomial statistics. |
| `results/CANONICAL_v2/r2_scale_correlation.json` | Scale-level R2 correlation. |
| `all_results.md` | Auto-generated inventory from every JSON under `results/`. |

When this document and the generated ledger disagree, prefer this document and the named JSON files above.

## 1. Main CURED V2 Results

Router config: `configs/router_thresholds.json`.

| Benchmark | 3B | 8B | 14B | 32B |
|---|---:|---:|---:|---:|
| TruthfulQA, cosine, n=500 | 60.6% | 60.2% | 64.0% | 60.1% |
| MedHallu, cosine, n=500 | 49.9% | 50.2% | 53.4% | 54.2% |
| StrategyQA, yes/no, n=500 | 62.4% | 72.2% | 70.0% | 76.4% |

| Model | Benchmark | File | Routing Summary |
|---|---|---|---|
| 3B | TruthfulQA | `main_cured_3b_truthfulqa_n500_v2.json` | 70.8% ALTA shortcut, 25.6% greedy fallback. |
| 8B | TruthfulQA | `main_cured_8b_truthfulqa_n500_v2.json` | 66.8% ALTA shortcut, 30.2% greedy fallback. |
| 3B | MedHallu | `main_cured_3b_medhallu_n500_v2.json` | Mostly greedy fallback, 20.6% ALTA shortcut. |
| 8B | MedHallu | `main_cured_8b_medhallu_n500_v2.json` | Mostly greedy fallback, 20.8% ALTA shortcut. |
| 3B | StrategyQA | `main_cured_3b_strategyqa_n500_v2.json` | Mixed greedy confidence, ALTA shortcut, and fallback. |
| 8B | StrategyQA | `main_cured_8b_strategyqa_n500_v2.json` | 66.6% ALTA shortcut. |

## 2. Greedy Reference Baselines

| Model | Benchmark | n | Greedy | File |
|---|---|---:|---:|---|
| 3B | TruthfulQA | 817 | 50.1% | `main_greedy_3b_truthfulqa_n817.json` |
| 8B | TruthfulQA | 817 | 49.6% | `main_greedy_8b_truthfulqa_n817.json` |
| 14B | TruthfulQA | 817 | 62.2% | `main_greedy_14b_truthfulqa_n817.json` |
| 32B | TruthfulQA | 817 | 58.8% | `main_greedy_32b_truthfulqa_n817.json` |
| 3B | StrategyQA | 500 | 65.0% | `main_greedy_3b_strategyqa_n500.json` |

TruthfulQA statistics use the first matched 500 questions when pairing `n=817` greedy references with `n=500` CURED runs.

## 3. Phase 2 Protocol Ablations

TruthfulQA ablations use cosine scoring with `n=200`.

| Protocol | 3B | 8B | 14B | 32B |
|---|---:|---:|---:|---:|
| Greedy | 56.5% | 48.0% | 64.5% | 57.6% |
| ALTA | 59.0% | 60.6% | 57.9% | 58.0% |
| CoVe | 46.0% | 39.2% | 45.5% | 49.8% |
| ITI | 56.5% | 57.5% | 67.0% | 64.3% |

MedHallu ablations use cosine scoring with `n=200`.

| Protocol | 3B | 8B | 14B | 32B |
|---|---:|---:|---:|---:|
| Greedy | 55.0% | 45.5% | 54.0% | 53.0% |
| ALTA | 58.0% | 59.8% | 57.5% | 60.0% |
| CoVe | 47.7% | 42.4% | 60.0% | 53.0% |
| ITI | 53.0% | 61.1% | 63.0% | 61.3% |

## 4. FACTOR Diagnostics

| Benchmark | File | Greedy | ALTA | CURED | Routing |
|---|---|---:|---:|---:|---|
| FACTOR-News, n=200 | `results_8b_factor_news_n200.json` | 59.0% | 69.0% | 61.5% | 27.5% ALTA shortcut, 72.5% greedy fallback. |
| FACTOR-Wiki original, n=200 | `results_8b_factor_wiki_n200.json` | 29.0% | 64.0% | 43.0% | Diagnostic regression from domain misclassification. |
| FACTOR-Wiki fixed, n=200 | `results_8b_factor_wiki_n200_fixed.json` | 29.5% | 65.0% | 65.0% | 100% ALTA shortcut. |

The fixed FACTOR-Wiki run follows two changes:

1. `detect_domain()` now uses word-boundary keyword matching, avoiding false medical matches such as `general` containing `gene`.
2. `configs/router_thresholds_factor.json` sets only `tau_H_easy` to `0.0`.

## 5. Medical QA And Other 8B Custom Runs

These are separate local protocols and should not be mixed with external ALTA MedQA/PubMedQA numbers.

| Benchmark | File | Scoring | n | Greedy | ALTA | CoVe | CURED |
|---|---|---|---:|---:|---:|---:|---:|
| MedQA v3-fixed | `results_8b_medqa_v3_fixed.json` | letter | 100 | 55.0% | 57.0% | 35.0% | 57.0% |
| PubMedQA v2 | `results_8b_pubmedqa_v2.json` | yes/no | 100 | 55.0% | 53.0% | 57.0% | 53.0% |
| TriviaQA v1 | `results_8b_triviaqa_v1.json` | cosine | 1000 | 18.4% | 18.3% | 15.1% | 18.1% |

## 6. Statistical Tests

McNemar exact binomial tests come from `results/CANONICAL_v2/statistics_table.json`.

| Comparison | Greedy | CURED | Delta | p_exact | Discordant | Significant |
|---|---:|---:|---:|---:|---:|---|
| 3B TruthfulQA | 51.8% | 60.2% | +8.4 pp | <0.0001 | 74/500 | yes |
| 8B TruthfulQA | 48.2% | 60.0% | +11.8 pp | <0.0001 | 89/500 | yes |
| 14B TruthfulQA | 63.6% | 64.0% | +0.4 pp | 0.8600 | 32/500 | no |
| 32B TruthfulQA | 59.8% | 59.6% | -0.2 pp | 1.0000 | 1/500 | no |
| 3B StrategyQA | 65.0% | 62.4% | -2.6 pp | 0.0984 | 53/500 | no |

## 7. R2 Analyses

| Analysis | r | p | Interpretation |
|---|---:|---:|---|
| Scale-level R2 vs ALTA gain | 0.9859 | 0.0141 | Scale-level profile predicts whether ALTA is viable. |
| Per-question R2 vs per-question gain | 0.0393 | 0.5803 | Per-question R2 does not predict individual gains. |

Source files:

| File | Purpose |
|---|---|
| `r2_scale_correlation.json` | Scale-level and per-question summary. |
| `r2_stratified_analysis.json` | R2 quartiles and point-biserial analysis. |
| `profile_3b.json`, `profile_8b.json`, `profile_14b.json`, `profile_32b.json` | Mechanistic profiles by scale. |

## 8. Known Diagnostic Runs

| Diagnostic | File | Result |
|---|---|---|
| 3B native profile ablation | `main_cured_3b_truthfulqa_n500_v2_native_profile.json` | 49.4%, showing the 3B TruthfulQA gain depends on the 8B-calibrated global scale shortcut. |
| Old 8B router | `main_cured_old_8b_truthfulqa_n500.json` | Kept for history; not the v2 headline result. |
| Semantic entropy gate | `semantic_entropy_gate_comparison.json` | MedHallu n=50: greedy 34.0%, ECR gate 34.0%, semantic entropy gate 44.0%. |

## 9. Reproduce And Regenerate

Run the full GPU suite:

```bash
bash scripts/autodl/run_all_experiments.sh
```

Regenerate derived result documents:

```bash
python compute_final_stats.py \
  --results-dir results/CANONICAL_v2 \
  --output results/CANONICAL_v2/statistics_table.json

python experiments/compute_scale_r2_correlation.py
python scripts/build_all_results_md.py
```

Run the fixed FACTOR-Wiki diagnostic:

```bash
python cured.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --load-in-4bit \
  --protocols greedy,alta,cured \
  --router new \
  --router-config configs/router_thresholds_factor.json \
  --benchmark custom \
  --custom-csv benchmarks/factor_wiki_n200.csv \
  --scoring letter \
  --max-new-tokens 5 \
  --n 200 --seed 42 \
  --out results/CANONICAL_v2/results_8b_factor_wiki_n200_fixed.json
```
