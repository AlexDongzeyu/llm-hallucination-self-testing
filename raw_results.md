# Inference-Time Hallucination Reduction - Full Synced Results

This document is synchronized to the current workspace artifacts in `results/` and latest verified rerun outputs from logs/terminal.

## 0. Completion Status

All target result artifacts exist:

- `results/bon_results.json`
- `results/calibration_results.json`
- `results/generation_results.json`
- `results/generation_results_n100_4configs.json`
- `results/grid_search_results.json`
- `results/instruct_results.json`
- `results/iti_results.json`
- `results/selfcheck_results.json`
- `results/online_results.json`
- `results/medhallu_results.json`
- `results/routing_dataset.csv`
- `results/router_model.joblib`

Latest orchestrated completion marker:

- `results/logs/pipeline_remaining_steps.done` = `COMPLETED 2026-04-03T04:47:58`

Latest standalone artifact refresh:

- `results/medhallu_results.json` last written `2026-04-03 22:25:10`

## 1. Calibration Results (`results/calibration_results.json`)

| Key | Model | ECE | Entropy-Accuracy Corr | Mean Entropy | Mean Top-1 Prob | Accuracy | n |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| `meta_llama_Llama_3_1_8B` | meta-llama/Llama-3.1-8B | 0.2711 | 0.1413 | 3.3092 | 0.3456 | 0.48 | 50 |
| `meta_llama_Llama_3_2_3B` | meta-llama/Llama-3.2-3B | 0.0821 | -0.4211 | 3.5120 | 0.2140 | 0.35 | 100 |
| `Qwen_Qwen2_5_3B` | Qwen/Qwen2.5-3B | 0.3121 | -0.0814 | 2.1040 | 0.4420 | 0.14 | 100 |

## 2. Base Generation Runs (`results/generation_results.json`)

| Label | Gate Mode | Curve Thresh | Entropy Thresh | Early Layer | Alpha | Accuracy | Gate Fire Rate | Repetition Rate | n |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| 8B Baseline (greedy) | joint | 999.0 | 999.0 | 12 | 0.0 | 0.48 | 0.0000 | 0.00 | 50 |
| 8B SLED + entropy (H=4.5) | sled_entropy | 0.0 | 4.5 | 12 | 0.3 | 0.44 | 0.0467 | 0.00 | 50 |
| 8B SLED + entropy (H=5.0) | sled_entropy | 0.0 | 5.0 | 12 | 0.3 | 0.46 | 0.0243 | 0.00 | 50 |

## 3. N=100 Four-Config Sweep (`results/generation_results_n100_4configs.json`)

| Label | Gate Mode | Curve Thresh | Entropy Thresh | Early Layer | Alpha | Accuracy | Gate Fire Rate | Repetition Rate | n |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| Baseline (greedy) | joint | 999.0 | 999.0 | 12 | 0.0 | 0.35 | 0.0000 | 0.00 | 100 |
| SLED + entropy gate (H=3.0) | sled_entropy | 0.0 | 3.0 | 12 | 0.3 | 0.34 | 0.2755 | 0.05 | 100 |
| SLED + entropy gate (H=3.5) | sled_entropy | 0.0 | 3.5 | 12 | 0.3 | 0.36 | 0.1866 | 0.04 | 100 |
| SLED + entropy gate (H=4.0) | sled_entropy | 0.0 | 4.0 | 12 | 0.3 | 0.34 | 0.1219 | 0.01 | 100 |

## 4. Grid Search (`results/grid_search_results.json`)

### 4.1 Baseline

- baseline_accuracy: 0.44

### 4.2 Phase 1 (20 runs)

All phase1 runs use: `alpha=0.3`, `early_layer=8`.

| # | curve_thresh | entropy_thresh | accuracy | gate_fire_rate | repetition_rate |
| --: | --: | --: | --: | --: | --: |
| 1 | 0.01 | 2.0 | 0.4333 | 0.0833 | 0.0 |
| 2 | 0.01 | 2.5 | 0.4333 | 0.0000 | 0.0 |
| 3 | 0.01 | 3.0 | 0.4333 | 0.0000 | 0.0 |
| 4 | 0.01 | 3.5 | 0.4333 | 0.0000 | 0.0 |
| 5 | 0.02 | 2.0 | 0.4333 | 0.0833 | 0.0 |
| 6 | 0.02 | 2.5 | 0.4333 | 0.0000 | 0.0 |
| 7 | 0.02 | 3.0 | 0.4333 | 0.0000 | 0.0 |
| 8 | 0.02 | 3.5 | 0.4333 | 0.0000 | 0.0 |
| 9 | 0.03 | 2.0 | 0.4333 | 0.0833 | 0.0 |
| 10 | 0.03 | 2.5 | 0.4333 | 0.0000 | 0.0 |
| 11 | 0.03 | 3.0 | 0.4333 | 0.0000 | 0.0 |
| 12 | 0.03 | 3.5 | 0.4333 | 0.0000 | 0.0 |
| 13 | 0.05 | 2.0 | 0.4333 | 0.0833 | 0.0 |
| 14 | 0.05 | 2.5 | 0.4333 | 0.0000 | 0.0 |
| 15 | 0.05 | 3.0 | 0.4333 | 0.0000 | 0.0 |
| 16 | 0.05 | 3.5 | 0.4333 | 0.0000 | 0.0 |
| 17 | 0.07 | 2.0 | 0.4333 | 0.0000 | 0.0 |
| 18 | 0.07 | 2.5 | 0.4333 | 0.0000 | 0.0 |
| 19 | 0.07 | 3.0 | 0.4333 | 0.0000 | 0.0 |
| 20 | 0.07 | 3.5 | 0.4333 | 0.0000 | 0.0 |

### 4.3 Phase 2 (12 runs)

All phase2 runs use: `curve_thresh=0.01`, `entropy_thresh=2.0`.

| # | early_layer | alpha | accuracy | gate_fire_rate | repetition_rate |
| --: | --: | --: | --: | --: | --: |
| 1 | 4 | 0.1 | 0.4333 | 0.0833 | 0.0 |
| 2 | 4 | 0.3 | 0.4000 | 0.0833 | 0.0 |
| 3 | 4 | 0.5 | 0.4000 | 0.0867 | 0.0 |
| 4 | 8 | 0.1 | 0.4333 | 0.0833 | 0.0 |
| 5 | 8 | 0.3 | 0.4333 | 0.0833 | 0.0 |
| 6 | 8 | 0.5 | 0.4000 | 0.0683 | 0.0 |
| 7 | 12 | 0.1 | 0.4333 | 0.0833 | 0.0 |
| 8 | 12 | 0.3 | 0.4667 | 0.1100 | 0.0 |
| 9 | 12 | 0.5 | 0.4333 | 0.1167 | 0.0 |
| 10 | 16 | 0.1 | 0.4333 | 0.0833 | 0.0 |
| 11 | 16 | 0.3 | 0.4333 | 0.0833 | 0.0 |
| 12 | 16 | 0.5 | 0.4000 | 0.0933 | 0.0 |

## 5. Best-of-N (`results/bon_results.json`)

| Label | n_bon | Temperature | Accuracy | Repetition Rate | n_questions |
| :-- | --: | --: | --: | --: | --: |
| BoN-1 T=0.3 | 1 | 0.3 | 0.30 | 0.00 | 50 |
| BoN-3 T=0.3 | 3 | 0.3 | 0.28 | 0.02 | 50 |
| BoN-5 T=0.3 | 5 | 0.3 | 0.28 | 0.00 | 50 |

## 6. Instruct Sweep (`results/instruct_results.json`)

| Label | Accuracy | Repetition Rate | Strategy Dist | n |
| :-- | --: | --: | :-- | --: |
| 7. Semantic Majority BoN (T=0.4, n=5) | 0.70 | 0.00 | semantic_majority_voting: 1.0 | 50 |
| 8. CoVe (2 checks) | 0.60 | 0.04 | cove: 1.0 | 50 |
| 9. GADR-2 Learned Router | 0.74 | 0.00 | gadr2_greedy_general: 1.0 | 50 |

## 7. ITI Sweep (`results/iti_results.json`)

| Label | Threshold | Accuracy | Repetition Rate | n |
| :-- | --: | --: | --: | --: |
| ITI alpha=0.5 | 0.55 | 0.80 | 0.02 | 50 |
| ITI alpha=0.5 | 0.65 | 0.72 | 0.02 | 50 |
| ITI alpha=1.0 | 0.55 | 0.80 | 0.02 | 50 |
| ITI alpha=1.0 | 0.65 | 0.70 | 0.02 | 50 |
| ITI alpha=2.0 | 0.55 | 0.78 | 0.04 | 50 |
| ITI alpha=2.0 | 0.65 | 0.70 | 0.04 | 50 |

## 8. SelfCheck Evaluation (`results/selfcheck_results.json`)

- dataset: truthful_qa:generation:validation
- n: 50
- k_samples: 4
- selfcheck_similarity_threshold: 0.7
- reference_threshold: 0.65
- qa_accuracy: 0.72
- qa_rep_rate: 0.0
- detector_accuracy: 0.74
- detector_precision: 0.5556
- detector_recall: 0.3571
- detector_f1: 0.4348
- mean_confidence: 0.8036
- runtime_min: 23.69
- counts: tp=5, fp=4, tn=32, fn=9

## 9. Online Comparison (`results/online_results.json`)

| Model | n | Greedy Acc@0.55 | Greedy Acc@0.65 | CoVe Acc@0.55 | CoVe Acc@0.65 | Delta@0.55 | Delta@0.65 | Greedy Rep | CoVe Rep | Runtime (min) |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| Llama-3.3-70B | 50 | 0.84 | 0.74 | 0.78 | 0.66 | -0.06 | -0.08 | 0.04 | 0.10 | 40.5 |
| Llama-4-Scout-17B | 50 | 0.80 | 0.68 | 0.76 | 0.56 | -0.04 | -0.12 | 0.02 | 0.10 | 11.8 |
| Qwen3-32B | 50 | 0.84 | 0.70 | 0.82 | 0.70 | -0.02 | 0.00 | 0.02 | 0.04 | 7.5 |
| GPT-OSS-120B | 50 | 0.80 | 0.62 | 0.76 | 0.56 | -0.04 | -0.06 | 0.02 | 0.02 | 11.2 |

## 10. MedHallu Evaluation (`results/medhallu_results.json`)

- dataset_id: UTAustin-AIHealth/MedHallu
- subset: pqa_artificial
- split: train
- threshold: 0.65
- n_target: 50
- runtime_min: 91.95

| Strategy | n_used | n_skipped | Accuracy | Repetition Rate |
| :-- | --: | --: | --: | --: |
| greedy | 50 | 0 | 0.46 | 0.00 |
| cove | 50 | 0 | 0.54 | 0.00 |
| dynamic | 50 | 0 | 0.48 | 0.08 |
| gadr2 | 50 | 0 | 0.52 | 0.02 |

## 11. Latest Verified Rerun Outputs (Log/Terminal-Backed)

### 11.1 Phase 1 rerun (`experiments/eval_calibration_phase1.py`)

From successful terminal run (latest verified completion):

- calibration proof:
- model: meta-llama/Llama-3.2-3B
- ece: 0.2518
- entropy_accuracy_correlation: 0.0058
- mean_entropy: 4.0395
- mean_top1_prob: 0.2793
- accuracy: 0.35
- n_samples: 100
- JSD summary:
- mean_jsd_all_layers_vs_final: 0.5329
- mean_jsd_early_lt8: 0.6652
- mean_jsd_mid_8_to_20: 0.5976
- mean_jsd_late_ge20: 0.2708

Note: this rerun snapshot differs from the older multi-model JSON in `results/calibration_results.json`.

### 11.2 Low-threshold gate diagnostic (`experiments/check_low_threshold.py`)

From successful rerun output:

- prompt: "The capital of Canada is"
- gate_fire_rate: 60.0%
- generated sample starts with: "Ottawa, Ontario. The capital of Canada is located in the province called Quebec..."

### 11.3 Orchestrated sequence completion (`experiments/pipeline_remaining_steps.ps1`)

From `results/logs/pipeline_remaining_steps_20260403_023407.log` and completion marker:

- `[2026-04-03T02:34:10] DONE  learn_router`
- `[2026-04-03T03:35:07] DONE  eval_instruct`
- `[2026-04-03T04:47:58] DONE  eval_medhallu`
- `results/logs/pipeline_remaining_steps.done`: `COMPLETED 2026-04-03T04:47:58`

Router training snapshot in that orchestrated run:

- routing rows loaded: 100
- class distribution: greedy=93, iti=4, cove=3
- Decision Tree CV acc: 0.920
- Logistic baseline CV acc: 0.930
- saved router: `results/router_model.joblib`

### 11.4 Latest focused reruns (terminal-verified)

`src/learn_router.py` rerun (Apr 3, 2026, latest):

- routing rows loaded: 100
- domain distribution: general=50, medical=50
- class distribution: greedy=50, iti=25, cove=25
- Decision Tree CV acc: 0.830 (chosen)
- Logistic baseline CV acc: 0.780

`experiments/eval_medhallu.py` rerun (Apr 3, 2026, latest):

- `results/medhallu_results.json` rewritten at `2026-04-03 22:25:10`
- strategy metrics and runtime are unchanged from Section 10 (`results/medhallu_results.json`).

## 12. High-Level Takeaways (Strictly from Current Artifacts)

1. Base-model SLED/entropy gating gave mixed gains; best shown in `grid_search_results.json` phase2 at early layer 12, alpha 0.3, accuracy 0.4667 vs baseline 0.44 (on that sweep setup).
2. BoN on the base setup did not outperform `BoN-1` in this run snapshot.
3. Instruct sweep: GADR-2 (0.74) outperformed Semantic Majority BoN (0.70) and CoVe (0.60) on this 50-sample run.
4. ITI sweep at low alphas (0.5-2.0) stayed relatively stable with best 0.80 at threshold 0.55.
5. Cross-model online CoVe generally underperformed greedy at threshold 0.65 in 3/4 models; Qwen3-32B was neutral (0.00 delta).
6. Latest MedHallu rerun kept `cove` as best at 0.54 accuracy; dynamic was 0.48 and gadr2 improved to 0.52 (with rep_rate 0.02).
