# CURED: Curvature-Informed Routing and Entropy-Based Decoding

> CURED is a test-time routing framework for hallucination reduction. It selects greedy, ALTA, CoVe, or ITI per question using lightweight trajectory features: R2, kappa, ECR, entropy, self-consistency when enabled, and domain routing.

## Source Of Truth

| Document | Purpose |
|---|---|
| [RESULTS.md](RESULTS.md) | Human-readable canonical result narrative and caveats. |
| [all_results.md](all_results.md) | Auto-generated ledger from `results/**/*.json`. |
| [experiments/README.md](experiments/README.md) | Phase-by-phase experiment guide and script index. |
| [scripts/autodl/QUICKSTART.md](scripts/autodl/QUICKSTART.md) | GPU runner quickstart. |
| `results/CANONICAL_v2/` | Canonical JSON outputs used by the docs. |

The public repository is [AlexDongzeyu/llm-hallucination-self-testing](https://github.com/AlexDongzeyu/llm-hallucination-self-testing).

## Headline Results

Canonical CURED v2 main runs use fixed thresholds and `n=500` unless noted.

| Benchmark | 3B | 8B | 14B | 32B |
|---|---:|---:|---:|---:|
| TruthfulQA, CURED v2 | 60.6% | 60.2% | 64.0% | 60.1% |
| MedHallu, CURED v2 | 49.9% | 50.2% | 53.4% | 54.2% |
| StrategyQA, CURED v2 | 62.4% | 72.2% | 70.0% | 76.4% |
| TruthfulQA greedy baseline, n=817 | 50.1% | 49.6% | 62.2% | 58.8% |

TruthfulQA greedy baselines are full `n=817` reference runs. CURED v2 rows are fixed-seed `n=500` generation runs from `results/CANONICAL_v2/`.

## 8B Diagnostic Results

| Benchmark | Scoring | Greedy | ALTA | CURED | Note |
|---|---|---:|---:|---:|---|
| FACTOR-News, n=200 | letter | 59.0% | 69.0% | 61.5% | Existing canonical run. |
| FACTOR-Wiki fixed, n=200 | letter | 29.5% | 65.0% | 65.0% | Domain word-boundary fix; 100% `alta_global_viable`. |
| MedQA v3-fixed, n=100 | letter | 55.0% | 57.0% | 57.0% | Free-form local split; not comparable to ALTA MedQA protocol. |
| PubMedQA v2, n=100 | yes/no | 55.0% | 53.0% | 53.0% | Local yes/no protocol. |
| TriviaQA v1, n=1000 | cosine | 18.4% | 18.3% | 18.1% | Retrieval-free trivia baseline. |

## Statistical Summary

| Comparison | Greedy | CURED | Delta | p_exact | Significant |
|---|---:|---:|---:|---:|---|
| 3B TruthfulQA | 51.8% | 60.2% | +8.4 pp | <0.0001 | yes |
| 8B TruthfulQA | 48.2% | 60.0% | +11.8 pp | <0.0001 | yes |
| 14B TruthfulQA | 63.6% | 64.0% | +0.4 pp | 0.8600 | no |
| 32B TruthfulQA | 59.8% | 59.6% | -0.2 pp | 1.0000 | no |
| 3B StrategyQA | 65.0% | 62.4% | -2.6 pp | 0.0984 | no |

Scale-level R2 predicts ALTA regime viability: Pearson `r=0.9859`, `p=0.0141`. Per-question R2 does not predict per-question gain: `r=0.0393`, `p=0.5803`.

## Repository Layout

```text
llm-hallucination-self-testing/
├── cured/                         # Python package API
├── cured.py                       # Main CLI entry point
├── calibrate_router.py            # Router threshold calibration
├── compute_final_stats.py         # McNemar and R2 stratified statistics
├── configs/                       # Router threshold JSON files
├── benchmarks/                    # Local benchmark CSVs
├── experiments/                   # Research scripts and figure builders
├── scripts/
│   ├── autodl/                    # GPU runner scripts
│   └── build_all_results_md.py    # all_results.md generator
├── results/CANONICAL_v2/          # Canonical result JSON files
├── paper/figures/                 # Paper figure assets
├── README.md
├── RESULTS.md
├── all_results.md
├── PAPER.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/AlexDongzeyu/llm-hallucination-self-testing.git
cd llm-hallucination-self-testing
pip install -r requirements.txt
```

GPU guidance: 8B 4-bit runs fit on the local 8 GB GPU used for the FACTOR rerun, but full canonical sweeps are intended for A100/A800-class machines. Larger 14B/32B runs need substantially more VRAM.

## Quickstart

Run CURED on TruthfulQA:

```bash
python cured.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --load-in-4bit \
  --protocols cured \
  --router new \
  --router-config configs/router_thresholds.json \
  --benchmark truthfulqa \
  --scoring cosine \
  --n 100 --seed 42 \
  --save-per-question \
  --out results/my_run.json
```

Reproduce the fixed FACTOR-Wiki diagnostic:

```bash
python cured.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --load-in-4bit \
  --protocols greedy,alta,cured \
  --router new \
  --router-config configs/router_thresholds_factor.json \
  --benchmark custom \
  --custom-csv benchmarks/factor_wiki_n200.csv \
  --scoring letter \
  --max-new-tokens 5 \
  --n 200 --seed 42 \
  --out results/CANONICAL_v2/results_8b_factor_wiki_n200_fixed.json
```

## Router Summary

```text
Question
  -> features: R2, kappa, ECR, H_final, SC, domain
  -> Gate 1: low entropy confidence path
  -> Scale shortcut: profile_mean_r2 >= 0.55 and non-medical -> ALTA
  -> Gate 2: per-question linearity predicate
  -> Gate 3: medical + ITI available
  -> Gate 4: composite ALTA score
  -> Gate 5: medical CoVe fallback or greedy fallback
```

Canonical thresholds live in `configs/router_thresholds.json`. FACTOR-Wiki uses `configs/router_thresholds_factor.json`, which only changes `tau_H_easy` to `0.0`.

## Documentation Workflow

```bash
python compute_final_stats.py \
  --results-dir results/CANONICAL_v2 \
  --output results/CANONICAL_v2/statistics_table.json

python experiments/compute_scale_r2_correlation.py
python scripts/build_all_results_md.py
```

After result JSONs change, regenerate the statistics and `all_results.md`, then update `RESULTS.md` if the human-facing narrative changed.

## Citation

```bibtex
@misc{cured2026,
  title  = {{CURED}: Curvature-Informed Routing and Entropy-Based Decoding},
  author = {Dong, Alex},
  year   = {2026},
  url    = {https://github.com/AlexDongzeyu/llm-hallucination-self-testing},
  note   = {Research code and experimental report}
}
```
