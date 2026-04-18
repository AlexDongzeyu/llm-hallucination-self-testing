# CURED: Complete Unified Routing and Evaluation for Decoding

> **TL;DR** — CURED is a principled 5-gate router that selects greedy,
> ALTA, CoVe, or ITI decoding **per question** using three lightweight
> trajectory features (R², κ, ECR) extracted from a single forward pass,
> reducing hallucinations without the compute overhead of always-on techniques.

---

## Key Results

| Model | Benchmark | Greedy | CURED v2 | Δ |
|---|---|---|---|---|
| Llama-3.2-3B | TruthfulQA | 50.1% | **60.6%** | +10.5pp |
| Llama-3.1-8B | TruthfulQA | 49.6% | **60.2%** | +10.6pp |
| Llama-3.1-8B | StrategyQA | — | **72.2%** | — |
| Qwen-14B | TruthfulQA | 62.2% | **64.0%** | +1.8pp |

Full result table: [RESULTS.md](RESULTS.md)

---

## Repository Structure

```
cured-decoding-router/
│
├── cured/                     ← Python package (importable API)
│   ├── __init__.py
│   ├── router.py              ← CUREDRouterV2, CUREDRouter, CUREDAPIRouter
│   ├── protocols.py           ← greedy, ALTA, CoVe, ITI, SelfCheck, DoLa
│   ├── scoring.py             ← cosine, letter, yesno, mc_score_sample
│   └── calibration.py        ← measure_r2, compute_ecr, train_iti_probes
│
├── cured.py                   ← CLI entry point (imports from cured/)
├── calibrate_router.py        ← standalone calibration script
├── compute_final_stats.py     ← statistical analysis + R²-stratified analysis
│
├── experiments/               ← research experiments
│   ├── README.md              ← Phase 1–5 pipeline documentation
│   ├── compute_logit_linearity.py
│   ├── run_alta_3b.py
│   ├── generate_paper_figures.py
│   └── run_semantic_entropy_ablation.py
│
├── scripts/
│   ├── autodl/                ← GPU shell scripts (A100/A800)
│   │   ├── run_all_experiments.sh  ← MAIN PIPELINE ENTRY POINT
│   │   └── ...
│   ├── prep_benchmarks.py     ← download/format benchmark CSVs
│   ├── build_all_results_md.py ← regenerate all_results.md
│   └── maintenance/           ← one-time tools (not part of pipeline)
│
├── benchmarks/                ← frozen benchmark CSVs
├── configs/
│   └── router_thresholds.json ← all router thresholds (tau_kappa, tau_ECR, …)
│
├── results/
│   ├── CANONICAL_v2/          ← SINGLE SOURCE OF TRUTH for all results
│   ├── figures/               ← paper figures (fig1–fig5 PNG)
│   └── archive/               ← non-canonical / debug runs
│
├── data/                      ← ITI probes, routing dataset
├── paper/                     ← PDF and figure sources
├── src/legacy/                ← early prototype code (pre-architecture)
│
├── README.md
├── RESULTS.md                 ← canonical result table
├── PAPER.md                   ← BibTeX + citation
├── requirements.txt
└── LICENSE                    ← MIT
```

---

## Installation

```bash
git clone https://github.com/your-org/cured-decoding-router.git
cd cured-decoding-router
pip install -r requirements.txt
```

GPU requirements: ≥ 24 GB VRAM for 8B models (4-bit), ≥ 40 GB for 32B models.

---

## Quickstart

```bash
# Run CURED router on TruthfulQA (n=100, 8B model)
python cured.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --load-in-4bit \
  --protocols cured \
  --router new \
  --router-config configs/router_thresholds.json \
  --benchmark truthfulqa \
  --n 100 --seed 42 \
  --save-per-question \
  --out results/my_run.json

# Compare protocols side-by-side
python cured.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --load-in-4bit \
  --protocols greedy,alta,cured \
  --benchmark medhallu \
  --n 50 --skip-iti \
  --out results/comparison.json
```

---

## Python API

```python
from cured import CUREDRouterV2
from cured.calibration import measure_r2
from cured.scoring import cosine_match

# Load model (standard HuggingFace)
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Calibrate and build router
r2 = measure_r2(model, tokenizer, n_questions=15)
router = CUREDRouterV2(model, tokenizer)

# Route a question
result = router.route(
    prompt="Answer concisely: What is the capital of France?",
    question="What is the capital of France?",
)
print(result["text"])       # → "Paris"
print(result["strategy"])   # → "greedy_confident" / "alta_global_viable" / …
```

---

## 5-Phase Experiment Pipeline

| Phase | Description | Key script |
|---|---|---|
| **Phase 1** | Measure logit linearity (R²) per model | `experiments/compute_logit_linearity.py` |
| **Phase 2** | Protocol ablations (greedy/ALTA/CoVe/ITI) | `scripts/autodl/run_phase2_ablations.sh` |
| **Phase 3** | Calibrate router thresholds | `calibrate_router.py` |
| **Phase 4** | Main CURED v2 evaluation (n=500) | `scripts/autodl/run_all_experiments.sh` |
| **Phase 5** | Statistics + R²-stratified analysis | `compute_final_stats.py` |

Full pipeline (A100/A800):
```bash
bash scripts/autodl/run_all_experiments.sh
```

---

## Router Architecture

```
Question → extract (R², κ, ECR, H_final, SC, domain)
                │
    ┌───────────▼───────────────┐
    │ Gate 1: H_final ≤ τ_H_easy│ → greedy_confident
    └───────────────────────────┘
                │ (not fired)
    ┌───────────▼────────────────────────┐
    │ Scale: model R² ≥ 0.55 + not med  │ → alta_global_viable
    └────────────────────────────────────┘
                │
    ┌───────────▼─────────────────────────────────┐
    │ Gate 2: kappa ≥ τ_κ=0.70 and ECR ≤ τ_E=0.04│ → alta_gate2
    └─────────────────────────────────────────────┘
                │
    ┌───────────▼──────────────────────┐
    │ Gate 3: medical and ITI available│ → iti_medical_gate3
    └──────────────────────────────────┘
                │
    ┌───────────▼──────────────────────────────┐
    │ Gate 4: H_final ≥ τ_H_hard and R² ≥ τ_R2│ → alta_gate4
    └──────────────────────────────────────────┘
                │
    ┌───────────▼──────────────────────┐
    │ Gate 5: medical and SC > 0.5     │ → cove_gate5_medical
    │         else                     │ → greedy_gate5
    └──────────────────────────────────┘
```

---

## Threshold Configuration

All thresholds in `configs/router_thresholds.json`:

```json
{
  "tau_kappa":        0.70,
  "tau_ECR":          0.04,
  "tau_R2":           0.50,
  "tau_H_easy":       1.0,
  "tau_H_hard":       3.5,
  "tau_SC_easy":      0.8,
  "tau_SC_hard":      0.5,
  "profile_mean_r2":  0.582
}
```

---

## ⚠ MC Scoring Validity Note

TruthfulQA MC1/MC2 scores require `--scoring mc` and the full MC answer set.
Default cosine scoring with `--scoring cosine` is the recommended mode for
generation evaluation and is used in all canonical results above.

---

## Citation

```bibtex
@misc{cured2026,
  title   = {{CURED}: Complete Unified Routing and Evaluation for Decoding},
  author  = {Author, A. and Author, B.},
  year    = {2026},
  url     = {https://github.com/your-org/cured-decoding-router},
  note    = {Preprint}
}
```

See [PAPER.md](PAPER.md) for full citation info and related work.
