# CURED: Curvature-Informed Routing and Entropy-based Decoding

Inference-time hallucination mitigation experiments for RLHF-tuned LLMs.

## Current Snapshot (2026-04-05)

Model: Llama-3.2-3B-Instruct.

- Entropy extraction (`n=30`): `H1=0.0806`, `H_last=0.8516`, `mean dH=+0.7711`, `dH<0=16.7%`.
- TruthfulQA DeLTa+DoLa 5x5 sweep (`n=50`, threshold `0.65`): best `74%` at `(alpha1=0.3, alpha2=0.3)`, and the greedy point `(0.0, 0.0)` is also `74%`.
- MedHallu generation (`n=50`, threshold `0.65`): best strategy is `gadr2_cured` at `54%`.

### MedHallu Generation Results

| Method | Accuracy | Repetition |
|---|---:|---:|
| greedy | 50% | 0% |
| cove | 50% | 2% |
| cove_rag | 50% | 0% |
| delta_dola | 52% | 0% |
| **gadr2_cured** | **54%** | 2% |

### MedHallu Ablations

| Method | Accuracy | Repetition |
|---|---:|---:|
| iti_alpha0.5 | 52% | 4% |
| sled | 52% | 0% |
| bon3_t0.3 | 48% | 0% |

Note: SelfCheck is available for TruthfulQA (`results/selfcheck_results.json`) and is not run on MedHallu in the current pipeline.

## Canonical Files

- `results/entropy_by_layer.json`
- `results/truthfulqa_delta_dola_sweep.json`
- `results/medhallu_generation_results.json`
- `results/medhallu_ablation_results.json`
- `results/medhallu_results.json`
- `results/figures/*.png`
- `raw_results.md`

## Run Commands

```bash
# MedHallu generation (primary)
python -u experiments/run_medhallu_generation.py --n 50 --threshold 0.65

# TruthfulQA DeLTa+DoLa full grid
python -u experiments/run_delta_dola_complete_grid.py --n 50 --threshold 0.65

# MedHallu ablations (SLED/BoN/ITI)
python -u experiments/run_medhallu_ablations.py

# MedHallu MC chooser (ablation)
python -u experiments/eval_medhallu.py --n 50 --alpha1 0.3 --alpha2 0.3

# Regenerate all figures
python experiments/regenerate_figures.py
```
