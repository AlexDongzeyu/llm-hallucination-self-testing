# Comprehensive Results Snapshot

Last sync: 2026-04-05

This file is a short summary. Full detail is in `../raw_results.md`.

## Topline Summary

- TruthfulQA DeLTa+DoLa sweep (5x5): best 74% at (alpha1=0.3, alpha2=0.3); the greedy point is also 74%.
- MedHallu generation (primary): best method is `gadr2_cured` at 54% (greedy 50%).
- MedHallu ablations: ITI alpha0.5 = 52%, SLED = 52%, BoN-3 = 48%.
- MedHallu MC chooser (ablation): delta_dola 6%, greedy 2%.

## Core Artifact Files

- `entropy_by_layer.json`
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

## Repro Commands

```bash
python -u experiments/run_medhallu_generation.py --n 50 --threshold 0.65
python -u experiments/run_delta_dola_complete_grid.py --n 50 --threshold 0.65
python -u experiments/run_medhallu_ablations.py
python -u experiments/eval_medhallu.py --n 50 --alpha1 0.3 --alpha2 0.3
python experiments/regenerate_figures.py
```
