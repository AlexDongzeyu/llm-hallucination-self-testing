# Comprehensive Results Snapshot

Last sync: 2026-04-05

This file is a short summary. Full detail is in `../raw_results.md`.

## Topline Summary

- Entropy profile is non-monotonic: `H1=0.0806 -> H7=10.8342 -> H28=0.8516`.
- Both delta definitions are reported: `L1->L28 mean dH=+0.7711 (dH<0=16.7%)` and `L7->L28 mean dH=-9.9826 (dH<0=100%)`.
- 3B late-layer logit linearity (`L14->L28`, top-50) is moderate: `mean R2=0.5557`.
- ALTA-style entropy-gated 3B run (`n=50`): 72% accuracy, 0% repetition.
- TruthfulQA DeLTa+DoLa sweep (5x5): max accuracy is 74%, including (alpha1=0.0, alpha2=0.0) and (alpha1=0.3, alpha2=0.3).
- MedHallu generation (primary): best method is `gadr2_cured` at 54% (greedy 50%).
- MedHallu ablations: ITI alpha0.5 = 52%, SLED = 52%, BoN-3 = 48%.
- MedHallu MC chooser (ablation): delta_dola 6%, greedy 2%.

## Core Artifact Files

- `entropy_by_layer.json`
- `logit_linearity_3b.json`
- `alta_3b_results.json`
- `truthfulqa_delta_dola_sweep.json`
- `medhallu_generation_results.json`
- `medhallu_ablation_results.json`
- `medhallu_results.json`
- `figures/*.png`

## MedHallu Generation (n=50, threshold=0.65)

| method | acc | rep |
|---|---:|---:|
| greedy | 50% | 0% |
| cove | 50% | 2% |
| cove_rag | 50% | 0% |
| delta_dola | 52% | 0% |
| gadr2_cured | 54% | 2% |

## ALTA-3B (n=50, threshold=0.65)

| method | acc | rep | mean gate weight |
|---|---:|---:|---:|
| alta_3b_entropy_gated | 72% | 0% | 0.2099 |

## MedHallu Ablations (n=50)

| method | acc | rep |
|---|---:|---:|
| iti_alpha0.5 | 52% | 4% |
| sled | 52% | 0% |
| bon3_t0.3 | 48% | 0% |

## Figure Outputs

- `fig1_entropy_compression.png`
- `fig2_method_comparison.png`
- `fig3_delta_dola_sweep.png`
- `fig4_routing.png`
- `fig5_cross_model_cove.png`

## Metric Separation Note

- Table A (this repo) uses free-form generation scored by cosine threshold.
- Table B (Ben ALTA) uses multiple-choice log-probability scoring.
- These tables are not directly comparable across metric/model settings.

### Table A (Alex, Generation Metric)

| Method | Model | TruthfulQA (gen) | MedHallu (gen) |
|---|---|---:|---:|
| Greedy | 3B-Instruct | 70% | 50% |
| DeLTa+DoLa (a1=0.3, a2=0.3) | 3B-Instruct | 74% | 52% |
| ALTA-3B (entropy-gated) | 3B-Instruct | 72% | - |
| CURED | 3B-Instruct | 74% | 54% |

### Table B (Ben, MC Metric)

| Method | Model | TruthfulQA (MC) | MedQA (MC) | PubMedQA (MC) |
|---|---|---:|---:|---:|
| Greedy | 8B-Instruct | - | - | - |
| ALTA | 8B-Instruct | 65.1% | 73.8% | 77.4% |

## Repro Commands

```bash
python -u experiments/compute_logit_linearity.py --n 30 --mid-layer 14 --top-k 50
python -u experiments/run_alta_3b.py --n 50 --threshold 0.65
python -u experiments/run_medhallu_generation.py --n 50 --threshold 0.65
python -u experiments/run_delta_dola_complete_grid.py --n 50 --threshold 0.65
python -u experiments/run_medhallu_ablations.py
python -u experiments/eval_medhallu.py --n 50 --alpha1 0.3 --alpha2 0.3
python experiments/regenerate_figures.py
```
