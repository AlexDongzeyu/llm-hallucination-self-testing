# CURED: Curvature-Informed Routing and Entropy-based Decoding

Inference-time hallucination mitigation experiments for RLHF-tuned LLMs.

## Current Snapshot (2026-04-05)

Model: Llama-3.2-3B-Instruct.

- Entropy shape (`n=30`): `H1=0.0806 -> H7=10.8342 -> H28=0.8516`; `L1->L28 mean dH=+0.7711 (dH<0=16.7%)`, `L7->L28 mean dH=-9.9826 (dH<0=100%)`.
- 3B late-layer diagnostics (`L14->L28`, top-50 logits, `n=30`): `mean R2=0.5557`, `median R2=0.5770`, `std=0.0696`; late-layer CV (L21-L28) `0.582-0.838`.
- ALTA-style 3B entropy-gated correction (`n=50`, threshold `0.65`): `72%` accuracy, `0%` repetition, mean gate weight `0.2099`.
- TruthfulQA DeLTa+DoLa 5x5 sweep (`n=50`, threshold `0.65`): best `74%` at `(alpha1=0.3, alpha2=0.3)`, and the greedy point `(0.0, 0.0)` is also `74%`.
- MedHallu generation (`n=50`, threshold `0.65`): best strategy is `gadr2_cured` at `54%`.

## Metric Note

- Do not directly compare Alex TruthfulQA cosine-threshold accuracy with Ben ALTA `%True x %Info`; they are different metrics on different models.

## R2 Framing

- This repo measures late-layer trajectory linearity at 3B as `mean R2=0.5557` (layers 14-28).
- DeLTa (He et al., 2025) reports higher late-layer linearity for 7B+ models (paper figures, commonly reported in the `~0.75-0.85` range).
- This provides a scale-linked context for trajectory-based correction behavior across model sizes.

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
- `results/logit_linearity_3b.json`
- `results/alta_3b_results.json`
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

# 3B late-layer logit linearity diagnostic
python -u experiments/compute_logit_linearity.py --n 30 --mid-layer 14 --top-k 50

# ALTA-style entropy-gated 3B run
python -u experiments/run_alta_3b.py --n 50 --threshold 0.65

# TruthfulQA DeLTa+DoLa full grid
python -u experiments/run_delta_dola_complete_grid.py --n 50 --threshold 0.65

# MedHallu ablations (SLED/BoN/ITI)
python -u experiments/run_medhallu_ablations.py

# MedHallu MC chooser (ablation)
python -u experiments/eval_medhallu.py --n 50 --alpha1 0.3 --alpha2 0.3

# Regenerate all figures
python experiments/regenerate_figures.py
```
