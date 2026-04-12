# Raw Results (Synced)

Last sync: 2026-04-12

This file is the concise source-of-truth summary for completed and verified artifacts.

## 1) Integrity and Status

- JSON files checked (recursive under `results/`): 45
- JSON parse failures: 0
- Verified newly completed local v2 artifact: `results_8b_medqa_v2.json`
- Local v2 outputs still missing: `results_8b_pubmedqa_v2.json`, `results_8b_medhallu_v2.json`
- OpenRouter v2 and Cloudflare v2 pairs remain hash-identical (fallback provenance verified)

Core artifacts:
- `results/results_openrouter_medqa_v2.json`
- `results/results_openrouter_pubmedqa_v2.json`
- `results/results_openrouter_medhallu_v2.json`
- `results/results_cloudflare_medqa_v2.json`
- `results/results_cloudflare_pubmedqa_v2.json`
- `results/results_cloudflare_medhallu_v2.json`
- `results_8b_both.json`
- `results_8b_medqa_v2.json`
- `results/entropy_by_layer.json`
- `results/logit_linearity_3b.json`
- `results/alta_3b_results.json`
- `results/truthfulqa_delta_dola_sweep.json`
- `results/medhallu_generation_results.json`
- `results/medhallu_ablation_results.json`
- `results/medhallu_results.json`
- `results/figures/*.png`

## 2) Current V2 Completion Snapshot

### API custom benchmark v2 (`n=100`, complete)

| dataset | protocol | accuracy | n_scored | n_total | rep_rate |
|---|---|---:|---:|---:|---:|
| MedQA | greedy | 44.00% | 100 | 100 | 0.00 |
| MedQA | cove | 14.00% | 100 | 100 | 0.00 |
| MedQA | cured_api | 21.21% | 99 | 100 | 0.01 |
| PubMedQA | greedy | 57.00% | 100 | 100 | 0.00 |
| PubMedQA | cove | 47.00% | 100 | 100 | 0.00 |
| PubMedQA | cured_api | 52.00% | 100 | 100 | 0.00 |
| MedHallu | greedy | 59.00% | 100 | 100 | 0.00 |
| MedHallu | cove | 54.84% | 93 | 100 | 0.07 |
| MedHallu | cured_api | 55.91% | 93 | 100 | 0.07 |

Cloudflare v2 files are intentional fallback copies of OpenRouter v2 outputs; pairwise SHA256 values are identical.

### Local 8B status

Completed local files:

- `results_8b_both.json`
- `results_8b_medqa_v2.json`

Missing local files:

- `results_8b_pubmedqa_v2.json`
- `results_8b_medhallu_v2.json`

`results_8b_medqa_v2.json` metrics (`n=100`, custom MedQA letter scoring):

| protocol | accuracy | n_scored | n_total | rep_rate |
|---|---:|---:|---:|---:|
| greedy | 29.00% | 100 | 100 | 0.00 |
| alta | 28.00% | 100 | 100 | 0.00 |
| cove | 13.00% | 100 | 100 | 0.00 |
| cured | 14.00% | 100 | 100 | 0.00 |

## 3) Entropy Extraction

Source: `results/entropy_by_layer.json`

- n_questions: 30
- n_layers: 28
- H1 mean: 0.0806
- H7 mean: 10.8342
- H_last mean: 0.8516
- L1->L28 mean dH: +0.7711
- L1->L28 dH < 0: 16.7%
- L7->L28 mean dH: -9.9826
- L7->L28 dH < 0: 100%
- late-layer CV (L21-L28): min 0.582, max 0.838, mean 0.710

## 4) 3B Late-Layer Logit Linearity

Source: `results/logit_linearity_3b.json`

- model: meta-llama/Llama-3.2-3B-Instruct
- n_questions: 30
- regression window: layers 14-28
- top_k tokens per prompt: 50
- mean R2: 0.5557
- median R2: 0.5770
- std of per-question mean R2: 0.0696

## 5) TruthfulQA DeLTa+DoLa Sweep

Source: `results/truthfulqa_delta_dola_sweep.json`

- rows: 25 (full 5x5)
- n_target: 50
- threshold: 0.65
- runtime_min: 155.13
- max accuracy: 74%
- 74% settings include (alpha1=0.0, alpha2=0.0) and (alpha1=0.3, alpha2=0.3)

## 6) ALTA-Style Entropy-Gated 3B Run

Source: `results/alta_3b_results.json`

- method: alta_3b_entropy_gated
- model: meta-llama/Llama-3.2-3B-Instruct
- n: 50
- threshold: 0.65
- accuracy: 72%
- rep_rate: 0%
- mean_gate_weight: 0.2099
- mean_first_token_entropy: 0.8402
- runtime_min: 9.78

## 7) MedHallu Generation (Primary)

Source: `results/medhallu_generation_results.json`

- dataset: UTAustin-AIHealth/MedHallu pqa_artificial train
- metric: cosine_similarity_to_ground_truth
- n_target: 50
- threshold: 0.65
- runtime_min: 112.47

| method | acc | rep | n |
|---|---:|---:|---:|
| greedy | 50% | 0% | 50 |
| cove | 50% | 2% | 50 |
| cove_rag | 50% | 0% | 50 |
| delta_dola | 52% | 0% | 50 |
| gadr2_cured | 54% | 2% | 50 |

## 8) MedHallu Ablations

Source: `results/medhallu_ablation_results.json`

- greedy_baseline: 50%

| method | acc | rep | n |
|---|---:|---:|---:|
| iti_alpha0.5 | 52% | 4% | 50 |
| sled | 52% | 0% | 50 |
| bon3_t0.3 | 48% | 0% | 50 |

## 9) MedHallu MC Chooser (Ablation)

Source: `results/medhallu_results.json`

- chooser: multiple_choice_by_candidate_loglikelihood
- n_target: 50
- runtime_min: 17.99

| method | acc | mean_margin | abstain_band_rate | n |
|---|---:|---:|---:|---:|
| greedy_mc | 2% | -1.0944 | 2% | 50 |
| delta_dola_mc_a10.3_a20.3 | 6% | -1.1638 | 2% | 50 |

## 10) Other Benchmark Snapshots

### Instruct (`results/instruct_results.json`)

| method | acc | rep | n |
|---|---:|---:|---:|
| Semantic Majority BoN (T=0.4, n=5) | 70% | 0% | 50 |
| CoVe (2 checks) | 60% | 4% | 50 |
| GADR-2 Learned Router | 74% | 0% | 50 |

### ITI (`results/iti_results.json`)

| label | threshold | acc | rep | n |
|---|---:|---:|---:|---:|
| ITI alpha=0.5 | 0.55 | 80% | 2% | 50 |
| ITI alpha=0.5 | 0.65 | 72% | 2% | 50 |
| ITI alpha=1.0 | 0.55 | 80% | 2% | 50 |
| ITI alpha=1.0 | 0.65 | 70% | 2% | 50 |
| ITI alpha=2.0 | 0.55 | 78% | 4% | 50 |
| ITI alpha=2.0 | 0.65 | 70% | 4% | 50 |

### SelfCheck (`results/selfcheck_results.json`)

- qa_accuracy: 72%
- qa_rep_rate: 0%
- detector_accuracy: 74%
- detector_f1: 0.4348
- runtime_min: 23.69

### Online CoVe vs Greedy (`results/online_results.json`)

| model | greedy@0.65 | cove@0.65 | delta@0.65 |
|---|---:|---:|---:|
| Llama-3.3-70B | 74% | 66% | -8% |
| Llama-4-Scout-17B | 68% | 56% | -12% |
| Qwen3-32B | 70% | 70% | 0% |
| GPT-OSS-120B | 62% | 56% | -6% |

### Best-of-N (`results/bon_results.json`)

| label | acc | rep | n |
|---|---:|---:|---:|
| BoN-1 T=0.3 | 30% | 0% | 50 |
| BoN-3 T=0.3 | 28% | 2% | 50 |
| BoN-5 T=0.3 | 28% | 0% | 50 |

### Calibration (`results/calibration_results.json`)

| model | ece | entropy-acc corr | mean_entropy | mean_top1_prob | acc | n |
|---|---:|---:|---:|---:|---:|---:|
| meta-llama/Llama-3.1-8B | 0.2711 | 0.1413 | 3.3092 | 0.3456 | 48% | 50 |
| meta-llama/Llama-3.2-3B | 0.0821 | -0.4211 | 3.5120 | 0.2140 | 35% | 100 |
| Qwen/Qwen2.5-3B | 0.3121 | -0.0814 | 2.1040 | 0.4420 | 14% | 100 |

## 11) Figures

Current files in `results/figures/`:
- `fig1_entropy_compression.png` (273,699 bytes)
- `fig2_method_comparison.png` (129,361 bytes)
- `fig3_delta_dola_sweep.png` (130,404 bytes)
- `fig4_routing.png` (128,222 bytes)
- `fig5_cross_model_cove.png` (81,360 bytes)

## 12) Important Notes

- SelfCheck is reported for TruthfulQA only in current artifacts.
- MedHallu does not currently include a SelfCheck strategy run in `results/medhallu_generation_results.json`.
- Observed late-layer diagnostics at 3B: `mean R2=0.5557`, late-layer CV range `0.582-0.838`.
- Metric guardrail for joint writing: Alex TruthfulQA cosine-threshold accuracy is not directly comparable to ALTA `%True x %Info` values from 8B studies.

## 13) Metric-Separated Manuscript Tables

Table A (Alex section, generation cosine@0.65):

| Method | Model | TruthfulQA (gen) | MedHallu (gen) |
|---|---|---:|---:|
| Greedy | 3B-Instruct | 70% | 50% |
| DeLTa+DoLa (alpha1=0.3, alpha2=0.3) | 3B-Instruct | 74% | 52% |
| ALTA-3B (entropy-gated) | 3B-Instruct | 72% | - |
| CURED | 3B-Instruct | 74% | 54% |

Note: DeLTa+DoLa at (0.3, 0.3) ties the greedy sweep point (0.0, 0.0) at 74%.

Table B (Ben section, multiple-choice scoring):

| Method | Model | TruthfulQA (MC) | MedQA (MC) | PubMedQA (MC) |
|---|---|---:|---:|---:|
| Greedy | 8B-Instruct | - | - | - |
| ALTA | 8B-Instruct | 65.1% | 73.8% | 77.4% |

Caption note: Tables A and B use different models and evaluation protocols and should not be directly compared.
