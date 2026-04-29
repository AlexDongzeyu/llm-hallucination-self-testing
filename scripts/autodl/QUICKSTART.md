# GPU Quickstart

Use this guide for borrowed GPU machines. The canonical result narrative is [../../RESULTS.md](../../RESULTS.md), and the raw outputs are in `results/CANONICAL_v2/`.

## 1. Clone And Bootstrap

```bash
git clone https://github.com/AlexDongzeyu/llm-hallucination-self-testing.git
cd llm-hallucination-self-testing
bash scripts/autodl/bootstrap_gpu_env.sh
```

## 2. Run The Canonical Suite

```bash
bash scripts/autodl/run_all_experiments.sh
```

Primary outputs:

| Output Pattern | Purpose |
|---|---|
| `results/CANONICAL_v2/profile_*.json` | Mechanistic profiles. |
| `results/CANONICAL_v2/ablation_*_n200.json` | Phase 2 protocol ablations. |
| `results/CANONICAL_v2/main_cured_*` | Phase 4 CURED main runs. |
| `results/CANONICAL_v2/statistics_table.json` | Paired McNemar statistics. |
| `results/CANONICAL_v2/r2_stratified_analysis.json` | Per-question R2 stratification. |

## 3. Regenerate Documentation Artifacts

```bash
python compute_final_stats.py \
  --results-dir results/CANONICAL_v2 \
  --output results/CANONICAL_v2/statistics_table.json

python experiments/compute_scale_r2_correlation.py
python scripts/build_all_results_md.py
```

## 4. Run The Fixed FACTOR-Wiki Diagnostic

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

Expected local result:

| Protocol | Accuracy |
|---|---:|
| Greedy | 29.5% |
| ALTA | 65.0% |
| CURED | 65.0% |

## 5. Optional Local Subsets

```bash
bash scripts/autodl/run_final_suite.sh
bash scripts/autodl/run_local_v2_suite.sh
```

Useful environment flags:

| Variable | Example | Effect |
|---|---|---|
| `FORCE_RERUN` | `FORCE_RERUN=1` | Rerun outputs that already exist. |
| `LOAD_IN_4BIT` | `LOAD_IN_4BIT=0` | Use higher precision when VRAM allows. |
| `MODEL_ID` | `MODEL_ID=meta-llama/Llama-3.1-8B-Instruct` | Override the default model. |

Example:

```bash
FORCE_RERUN=1 MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  bash scripts/autodl/run_local_v2_suite.sh
```

## 6. Cleanup And Archive

```bash
bash scripts/autodl/organize_final_results.sh
```

Legacy or failed API outputs should be moved under `results/archive/`. Do not use archived files in headline tables unless a diagnostic explicitly references them.
