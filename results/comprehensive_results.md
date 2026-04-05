# Comprehensive Results Snapshot

This report consolidates the latest completed outputs across TruthfulQA, MedHallu generation (primary), corrected MedHallu MC (ablation), instruct strategy runs, ITI sweeps, online model comparisons, and SelfCheck.

## 1) Finalized Artifacts Used

- results/truthfulqa_delta_dola_sweep.json
- results/medhallu_generation_results.json
- results/medhallu_results.json
- results/archive/medhallu_results_snapshot_n50.json
- results/instruct_results.json
- results/iti_results.json
- results/online_results.json
- results/selfcheck_results.json
- results/grid_search_results.json

## 2) TruthfulQA: DeLTa+DoLa Alpha Sweep

Source: results/truthfulqa_delta_dola_sweep.json

- n: 50
- threshold: 0.65
- total sweep runtime: 84.55 min

| alpha1 | alpha2 | accuracy | rep_rate | runtime_min |
| :--: | :--: | --: | --: | --: |
| 0.0 | 0.0 | 0.74 | 0.00 | 9.6 |
| 0.2 | 0.2 | 0.68 | 0.00 | 9.6 |
| 0.3 | 0.3 | 0.74 | 0.00 | 9.4 |
| 0.2 | 0.0 | 0.66 | 0.00 | 9.2 |
| 0.0 | 0.2 | 0.72 | 0.02 | 8.9 |
| 0.3 | 0.1 | 0.72 | 0.00 | 9.4 |
| 0.1 | 0.3 | 0.68 | 0.02 | 9.4 |
| 0.4 | 0.2 | 0.72 | 0.00 | 9.7 |
| 0.2 | 0.4 | 0.70 | 0.00 | 9.5 |

Best from file: alpha1=0.0, alpha2=0.0, accuracy=0.74.

## 3) MedHallu: Generation Framing (Primary, n=50)

Source: results/medhallu_generation_results.json

- dataset: UTAustin-AIHealth/MedHallu pqa_artificial train
- metric: cosine_similarity_to_ground_truth
- threshold: 0.65
- n_target: 50
- runtime_min: 112.47

### 3.1 Overall Generation Results

| strategy | n_used | n_skipped | accuracy | rep_rate | runtime_min |
| :-- | --: | --: | --: | --: | --: |
| greedy | 50 | 0 | 0.50 | 0.00 | 11.11 |
| cove | 50 | 0 | 0.50 | 0.02 | 38.40 |
| gadr2_cured | 50 | 0 | 0.54 | 0.02 | 22.29 |
| cove_rag | 50 | 0 | 0.50 | 0.00 | 40.18 |
| delta_dola | 50 | 0 | 0.52 | 0.00 | 11.58 |

### 3.2 By Difficulty (Generation)

| strategy | easy | medium | hard |
| :-- | --: | --: | --: |
| greedy | 0.5714 (n=14) | 0.5263 (n=19) | 0.4118 (n=17) |
| cove | 0.5714 (n=14) | 0.5263 (n=19) | 0.4118 (n=17) |
| gadr2_cured | 0.7143 (n=14) | 0.5789 (n=19) | 0.3529 (n=17) |
| cove_rag | 0.6429 (n=14) | 0.5789 (n=19) | 0.2941 (n=17) |
| delta_dola | 0.7143 (n=14) | 0.5263 (n=19) | 0.3529 (n=17) |

### 3.3 Delta vs Greedy (Generation)

- gadr2_cured: +0.04 absolute over greedy (0.54 vs 0.50)
- delta_dola: +0.02 absolute over greedy (0.52 vs 0.50)
- cove_rag: 0.00 absolute over greedy (0.50 vs 0.50)
- cove: 0.00 absolute over greedy (0.50 vs 0.50)

## 4) MedHallu: Corrected MC Framing (Ablation, n=50)

Source: results/medhallu_results.json

- dataset: UTAustin-AIHealth/MedHallu
- subset: pqa_artificial
- split: train
- chooser: multiple_choice_by_candidate_loglikelihood
- n_target: 50
- runtime_min: 17.99
- delta_dola config: alpha1=0.3, alpha2=0.3, early_layer_idx=7, mid_layer_idx=14, top_k=200

### 4.1 Overall MC Results

| strategy | n_used | n_skipped | accuracy | mean_margin | abstain_band_rate |
| :-- | --: | --: | --: | --: | --: |
| greedy_mc | 50 | 0 | 0.02 | -1.0944 | 0.02 |
| delta_dola_mc_a10.3_a20.3 | 50 | 0 | 0.06 | -1.1638 | 0.02 |

### 4.2 By Difficulty

| strategy | easy | medium | hard |
| :-- | --: | --: | --: |
| greedy_mc | 0.0000 (n=14) | 0.0526 (n=19) | 0.0000 (n=17) |
| delta_dola_mc_a10.3_a20.3 | 0.0000 (n=14) | 0.1053 (n=19) | 0.0588 (n=17) |

### 4.3 Delta vs Greedy

- Absolute accuracy gain: +0.04
- Relative gain over greedy: +200% (small absolute baseline)

## 5) Instruct Strategy Snapshot

Source: results/instruct_results.json

| strategy | accuracy | rep_rate | n |
| :-- | --: | --: | --: |
| Semantic Majority BoN (T=0.4, n=5) | 0.70 | 0.00 | 50 |
| CoVe (2 checks) | 0.60 | 0.04 | 50 |
| GADR-2 Learned Router | 0.74 | 0.00 | 50 |

## 6) ITI Sweep Snapshot

Source: results/iti_results.json

| label | threshold | accuracy | rep_rate | n |
| :-- | --: | --: | --: | --: |
| ITI alpha=0.5 | 0.55 | 0.80 | 0.02 | 50 |
| ITI alpha=0.5 | 0.65 | 0.72 | 0.02 | 50 |
| ITI alpha=1.0 | 0.55 | 0.80 | 0.02 | 50 |
| ITI alpha=1.0 | 0.65 | 0.70 | 0.02 | 50 |
| ITI alpha=2.0 | 0.55 | 0.78 | 0.04 | 50 |
| ITI alpha=2.0 | 0.65 | 0.70 | 0.04 | 50 |

## 7) Online Multi-Model CoVe vs Greedy

Source: results/online_results.json

| model | greedy_acc@0.65 | cove_acc@0.65 | cove_delta@0.65 | runtime_min |
| :-- | --: | --: | --: | --: |
| Llama-3.3-70B | 0.74 | 0.66 | -0.08 | 40.5 |
| Llama-4-Scout-17B | 0.68 | 0.56 | -0.12 | 11.8 |
| Qwen3-32B | 0.70 | 0.70 | 0.00 | 7.5 |
| GPT-OSS-120B | 0.62 | 0.56 | -0.06 | 11.2 |

## 8) SelfCheck Snapshot

Source: results/selfcheck_results.json

- qa_accuracy: 0.72
- qa_rep_rate: 0.00
- detector_accuracy: 0.74
- detector_precision: 0.5556
- detector_recall: 0.3571
- detector_f1: 0.4348
- mean_confidence: 0.8036
- runtime_min: 23.69

## 9) Grid Search Snapshot (Base SLED/Gating)

Source: results/grid_search_results.json

- baseline_accuracy: 0.44
- Best phase2 point in prior summary: early_layer=12, alpha=0.3, accuracy=0.4667

## 10) High-Level Conclusions

1. On TruthfulQA (n=50), the DeLTa+DoLa alpha sweep did not beat the strongest baseline in this run; best observed value was 0.74, tied by (0.0,0.0) and (0.3,0.3).
2. On MedHallu generation framing (primary, n=50), gadr2_cured was best at 0.54, ahead of greedy/cove/cove_rag (0.50) and delta_dola (0.52).
3. On corrected MedHallu MC framing (ablation, n=50), delta_dola improved over greedy (0.06 vs 0.02), but absolute MC chooser accuracy remained low.
4. On instruct TruthfulQA-style runs, GADR-2 and strong ITI settings remain the strongest performers in current artifact set.
5. Online cross-model CoVe generally trailed greedy at threshold 0.65 except neutral behavior on Qwen3-32B.

## 11) Reproducibility Commands

- MedHallu generation final (primary):
  - python -u experiments/run_medhallu_generation.py --n 50 --threshold 0.65 --out results/medhallu_generation_results.json --resume
- TruthfulQA alpha sweep:
  - python -u experiments/run_delta_dola_sweep.py --n 50 --threshold 0.65
- MedHallu corrected MC final (ablation):
  - python -u experiments/eval_medhallu.py --n 50 --alpha1 0.3 --alpha2 0.3 --out results/medhallu_results.json
