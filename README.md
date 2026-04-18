# CURED: Curvature-Informed Routing and Entropy-Based Decoding

> A principled 5-gate router that selects greedy, ALTA, CoVe, or ITI decoding **per question** using three lightweight trajectory features (R², κ, ECR), reducing hallucinations without the overhead of always-on techniques.

## Key Results (CUREDRouterV2, fixed thresholds)

| Model | Benchmark | Greedy | CURED v2 | Δ |
|---|---|---|---|---|
| Llama-3.2-3B | TruthfulQA | 50.1% | **60.6%** | +10.5 pp |
| Llama-3.1-8B | TruthfulQA | 49.6% | **60.2%** | +10.6 pp |
| Llama-3.2-3B | MedHallu (n=500) | — | **49.9%** | — |
| Llama-3.1-8B | MedHallu (n=500) | — | **50.2%** | — |
| Llama-3.2-3B | StrategyQA | 65.0% | **62.4%** | −2.6 pp |
| Llama-3.1-8B | StrategyQA | 72.2% | **72.2%** | 0.0 pp |
| Qwen2.5-14B-Instruct | TruthfulQA | 62.2% | **64.0%** | +1.8 pp |
| Qwen2.5-32B-Instruct | TruthfulQA | 58.8% | **60.1%** | +1.3 pp |

TruthfulQA greedy columns use the full **n=817** MC1-style reference runs; CURED v2 uses **n=500** with fixed seed (see `results/CANONICAL_v2/`). MedHallu rows report CURED main only (no separate greedy n=500 in the canonical bundle; ablation slices at n=200 are in [RESULTS.md](RESULTS.md)).

---

## Repository structure

```
cured-decoding-router/
├── cured/                        # Python package
├── cured.py                      # CLI entry point
├── calibrate_router.py
├── compute_final_stats.py
├── experiments/
│   ├── README.md
│   ├── compute_logit_linearity.py
│   ├── run_alta_3b.py
│   ├── run_delta_dola_sweep.py
│   ├── run_semantic_entropy_ablation.py
│   ├── generate_paper_figures.py
│   ├── build_routing_dataset.py
│   └── …
├── scripts/
│   ├── autodl/                   # GPU pipeline (run_all_experiments.sh, …)
│   ├── prep_benchmarks.py
│   ├── prep_factor_benchmark.py
│   ├── rebuild_*_csv.py
│   ├── build_all_results_md.py
│   └── run_all_local.ps1
├── benchmarks/
├── configs/                      # router_thresholds.json (+ router_thresholds_3b.json)
├── results/CANONICAL_v2/
├── paper/figures/                # fig1–fig5 (see also results/figures/)
├── data/
├── README.md
├── RESULTS.md
├── PAPER.md
├── requirements.txt
└── LICENSE
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

## 5-Phase experiment pipeline

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

## Router architecture

```
Question → extract (R², κ, ECR, H_final, SC, domain)
                │
    ┌────────────────────────────────────────────────────────────┐
    │ Gate 1: H_final < τ_H_easy and (SC rules for ≤14B)         │ → greedy_confident
    │ Note: inactive for ≤14B in Phase 4 (--compute-sc omitted) │
    └────────────────────────────────────────────────────────────┘
                │ (not fired)
    ┌───────────▼────────────────────────────────────────┐
    │ Scale: profile_mean_r2 ≥ 0.55, not med, H_final > τ_H_easy │ → alta_global_viable
    └────────────────────────────────────────────────────┘
                │
    ┌───────────▼──────────────────────────────────────────────────┐
    │ Gate 2: R²_q > τ_R2, κ_q < τ_kappa, ECR_q > τ_ECR          │ → continue (ALTA path)
    └──────────────────────────────────────────────────────────────┘
                │
    ┌───────────▼──────────────────────┐
    │ Gate 3: medical and ITI available│ → iti_medical_gate3
    └──────────────────────────────────┘
                │
    ┌───────────▼──────────────────────────────┐
    │ Gate 4: composite S_ALTA > 0.5           │ → alta_gate4
    └──────────────────────────────────────────┘
                │
    ┌───────────▼──────────────────────┐
    │ Gate 5: medical and SC > 0.5     │ → cove_gate5_medical
    │         else                     │ → greedy_gate5
    └──────────────────────────────────┘
```

---

## Threshold configuration

All thresholds in `configs/router_thresholds.json`:

```json
{
  "tau_kappa":        0.70,
  "tau_ECR":          0.04,
  "tau_R2":           0.65,
  "tau_H_easy":       0.5,
  "tau_H_hard":       3.0,
  "tau_SC_easy":      0.90,
  "tau_SC_hard":      0.60,
  "profile_mean_r2":  0.582
}
```

> **Implementation notes:**
> - `tau_R2 = 0.65` means the strict per-question Gate 2 predicate `R²_q > tau_R2` fires on a small fraction of questions at 3B/8B; the scale shortcut routes most general-domain ALTA traffic.
> - `profile_mean_r2 = 0.582` comes from 8B profiling and is reused for all scales in Phase 4. Use `configs/router_thresholds_3b.json` for the native 3B profile ablation (see [RESULTS.md](RESULTS.md)).
> - Gate 1 requires `SC_q` for models ≤14B; Phase 4 omits `--compute-sc`, so Gate 1 is **inactive** for canonical 3B/8B. Models >14B use `H_final < tau_H_easy` only.

---

## MC scoring note

TruthfulQA MC1/MC2 scores require `--scoring mc` and the full MC answer set. Default cosine scoring with `--scoring cosine` is the recommended mode for generation evaluation and is used in all canonical results above.

---

## Citation

```bibtex
@misc{cured2026,
  title   = {{CURED}: Curvature-Informed Routing and Entropy-Based Decoding},
  author  = {Author, A. and Author, B.},
  year    = {2026},
  url     = {https://github.com/your-org/cured-decoding-router},
  note    = {Preprint}
}
```
