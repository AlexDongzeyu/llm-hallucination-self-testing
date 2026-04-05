# CURED: Curvature-Informed Routing and Entropy-based Decoding

Systematic study of inference-time hallucination mitigation for RLHF-tuned LLMs.

**Core finding:** RLHF training compresses layer-wise entropy (dH < 0 for 100% of
questions, mean delta H = -9.86), rendering all logit-space decoding interventions
(DoLa, DeLTa, SLED) ineffective on adversarial QA. CURED routes by domain +
entropy curvature, achieving +4% on MedHallu generation (54% vs 50% greedy)
without degrading TruthfulQA.

## Results Summary

| Method | TruthfulQA | MedHallu (gen) |
|--------|-----------|----------------|
| Greedy (baseline) | 70% | 50% |
| SLED | 64% | - |
| BoN-5 | 64% | - |
| CoVe | 60% | 50% |
| CoVe+RAG | - | 50% |
| ITI alpha=0.5 | 72% | - |
| DeLTa+DoLa | 74% (=greedy) | 52% |
| SelfCheck | 72% | - |
| **CURED (ours)** | **74%** | **54%** |

Model: Llama-3.2-3B-Instruct. n=50 per benchmark, threshold=0.65.

## Project Layout

- `src/generate_instruct.py` - all generation strategies + CURED router
- `experiments/` - runnable evaluation scripts
- `results/` - canonical output files
- `results/figures/` - paper figures

## Key Files

- `results/routing_dataset.csv` - 100-row feature+outcome dataset
- `results/truthfulqa_delta_dola_sweep.json` - DeLTa+DoLa alpha sweep
- `results/medhallu_generation_results.json` - generation eval (primary metric)
- `results/medhallu_results.json` - MC likelihood eval (ablation)
- `results/comprehensive_results.md` - consolidated results snapshot

## Run Commands

```bash
# MedHallu generation eval (primary) - ~4 hours
python -u experiments/run_medhallu_generation.py --n 50 --threshold 0.65

# TruthfulQA DeLTa+DoLa sweep - ~85 min
python -u experiments/run_delta_dola_sweep.py --n 50 --threshold 0.65

# MedHallu MC likelihood eval (ablation) - ~18 min
python -u experiments/eval_medhallu.py --n 50 --alpha1 0.3 --alpha2 0.3

# Generate paper figures - ~30 sec
python experiments/generate_paper_figures.py
```
