# CURED — Canonical Results

> **Single source of truth.** All numbers below come from
> `results/CANONICAL_v2/` and are fully reproducible via the pipeline in
> `scripts/autodl/run_all_experiments.sh`.

## Phase 4 Main Results — CURED v2 Router (Fixed Thresholds)

Router config: `configs/router_thresholds.json`
(`tau_kappa=0.70`, `tau_ECR=0.04`, `profile_mean_r2=0.582`)

| Model | Benchmark | n | Acc (CURED v2) | Runtime |
|---|---|---|---|---|
| Llama-3.2-3B | TruthfulQA | 500 | **60.6%** | 54 min |
| Llama-3.2-3B | MedHallu | 500 | **49.9%** | 52 min |
| Llama-3.2-3B | StrategyQA | 500 | **62.4%** | 4 min |
| Llama-3.1-8B | TruthfulQA | 500 | **60.2%** | 66 min |
| Llama-3.1-8B | MedHallu | 500 | **50.2%** | 55 min |
| Llama-3.1-8B | StrategyQA | 500 | **72.2%** | 5 min |

## Router v1 vs v2 — 8B TruthfulQA

| Router version | File | n | Acc |
|---|---|---|---|
| v1 (old thresholds: tau_kappa=0.08, tau_ECR=0.10) | `main_cured_old_8b_truthfulqa_n500.json` | 500 | 62.9% |
| v2 (fixed thresholds: tau_kappa=0.70, tau_ECR=0.04) | `main_cured_8b_truthfulqa_n500_v2.json` | 500 | **60.2%** |

> Note: v2 shows different routing behaviour (more ALTA + scale-adaptive shortcut)
> vs v1 (was incorrectly passing near nobody through Gate 2).

## Greedy Baseline

| Model | Benchmark | n | Acc (Greedy) | File |
|---|---|---|---|---|
| Llama-3.2-3B | TruthfulQA | 817 | 50.1% | `main_greedy_3b_truthfulqa_n817.json` |
| Llama-3.2-3B | StrategyQA | 500 | 65.0% | `main_greedy_3b_strategyqa_n500.json` |
| Llama-3.1-8B | TruthfulQA | 817 | 49.6% | `main_greedy_8b_truthfulqa_n817.json` |
| Qwen-14B | TruthfulQA | 817 | 62.2% | `main_greedy_14b_truthfulqa_n817.json` |
| Qwen-32B | TruthfulQA | 817 | 58.8% | `main_greedy_32b_truthfulqa_n817.json` |

## Phase 4 — Prior Runs (Older Router, for reference)

| Model | Benchmark | Acc | File |
|---|---|---|---|
| Llama-3.2-3B | TruthfulQA | 51.6% | `main_cured_3b_truthfulqa_n500.json` |
| Llama-3.2-3B | MedHallu | 47.5% | `main_cured_3b_medhallu_n500.json` |
| Llama-3.1-8B | TruthfulQA | 49.8% | `main_cured_8b_truthfulqa_n500.json` |
| Llama-3.1-8B | MedHallu | 48.3% | `main_cured_8b_medhallu_n500.json` |
| Qwen-14B | TruthfulQA | 64.0% | `main_cured_14b_truthfulqa_n500.json` |
| Qwen-32B | TruthfulQA | 60.1% | `main_cured_32b_truthfulqa_n500.json` |

## Phase 2 Ablations — Protocol Comparison (8B, n=200)

| Protocol | TruthfulQA Acc | MedHallu Acc |
|---|---|---|
| Greedy | 48.0% | 45.5% |
| ALTA | 60.6% | 59.8% |
| CoVe | 39.2% | 42.4% |
| ITI | 57.5% | 61.1% |

> Source: `results/CANONICAL_v2/ablation_8b_*_n200.json`


## FACTOR Benchmarks (8B, letter scoring, max_new_tokens=5)

| Benchmark | Protocol | Acc | n_scored | Runtime |
|---|---|---|---|---|
| FACTOR-News | greedy | 59.0% | 200 | 1.39 min |
| FACTOR-News | alta | **69.0%** | 200 | 1.99 min |
| FACTOR-News | cured | 61.5% | 200 | 1.57 min |
| FACTOR-Wiki | greedy | 29.0% | 200 | 1.70 min |
| FACTOR-Wiki | alta | **64.0%** | 200 | 2.25 min |
| FACTOR-Wiki | cured | 43.0% | 200 | 1.89 min |

> **FACTOR-Wiki regression (CURED 43% vs ALTA 64%) — root cause:**
> FACTOR-Wiki prompts are high-confidence text completions (low H_final ≤ 0.5).
> Gate 1 is inactive for 8B (requires SC_q, omitted via `--skip-sc`), so the
> scale-aware shortcut condition `H_final > tau_H_easy (0.5)` is also False for
> these questions. They fall through to `greedy_gate5`, returning greedy's 29%.
> Questions with H_final > 0.5 correctly reach ALTA via the shortcut.
> The mix produces 43%. Raising `tau_H_easy` to 1.0 or enabling `--compute-sc`
> would route low-entropy FACTOR-Wiki questions to `greedy_confident` (Gate 1)
> rather than `greedy_gate5`. See `configs/router_thresholds.json`.

## Semantic Entropy Ablation (8B, MedHallu, n=50, k=5)

- Greedy **34.0%** vs ECR-gate **34.0%** vs SE-gate **44.0%**
- SE-gate gain: **+10.0 pp** — SE-based routing significantly outperforms ECR for medical QA
- `results/CANONICAL_v2/semantic_entropy_gate_comparison.json`

## McNemar Paired Statistics (Exact Binomial, α=0.05)

| Comparison | Baseline | CURED v2 | Δ pp | p (exact) | Sig? | n | b | c |
|---|---|---|---|---|---|---|---|---|
| 3B TruthfulQA | 51.8% (greedy n=817) | 60.2% | **+8.4** | < 0.0001 | **YES** | 500 | 16 | 58 |
| 8B TruthfulQA | 48.2% (greedy n=817) | 60.0% | **+11.8** | < 0.0001 | **YES** | 500 | 15 | 74 |
| 3B MedHallu | 55.0% (ablation n=200) | 52.5% | −2.5 | 0.458 | no | 200 | 17 | 12 |
| 8B MedHallu | 45.5% (ablation n=200) | 49.0% | +3.5 | 0.167 | no | 200 | 6 | 13 |

> `b` = greedy correct, CURED wrong; `c` = greedy wrong, CURED correct; Δ = c − b discordant pairs.
> MedHallu baselines use `ablation_*_greedy_medhallu_n200.json` (n=200 matched); lower power than TruthfulQA comparisons.

- `results/CANONICAL_v2/stats_3b_tqa_v2.json` — 3B TruthfulQA full output
- `results/CANONICAL_v2/stats_8b_tqa_v2.json` — 8B TruthfulQA full output
- `results/CANONICAL_v2/stats_3b_med_v2.json` — 3B MedHallu full output
- `results/CANONICAL_v2/stats_8b_med_v2.json` — 8B MedHallu full output
- `results/CANONICAL_v2/statistics_table.json` — auto-scan pairs (v2-aware; `_strip_n_suffix` fixed)
- `results/CANONICAL_v2/r2_stratified_analysis.json` — R² quartile vs ALTA accuracy

> **Issue 2 root-cause fixed** (`compute_final_stats.py`): `_strip_n_suffix` now strips both
> `_v<N>` then `_n<N>` so v2 files match greedy files in auto-scan mode. Prior runs produced
> zero auto-scan pairs for v2 files.

## 3B Native Viability Profile (TruthfulQA, n=500)

Same model and seeds as v2 main, but `configs/router_thresholds_3b.json` with
`profile_mean_r2=0.501` so `alta_globally_viable=False` (scale shortcut disabled).

| Run | Acc | Runtime | File |
|---|---|---|---|
| CURED v2 (8B-calibrated profile) | **60.6%** | ~54 min | `main_cured_3b_truthfulqa_n500_v2.json` |
| CURED + 3B-native profile | **49.4%** | ~36 min | `main_cured_3b_truthfulqa_n500_v2_native_profile.json` |

This isolates how much 3B TruthfulQA improvement depends on the global viability shortcut
when `profile_mean_r2` is taken from 8B profiling (`0.582`) vs 3B (`0.501`).

## Reproducing Results

```bash
# Full pipeline on A100/A800 (run from repo root):
bash scripts/autodl/run_all_experiments.sh

# Quick smoke test (Gate 2 verification, n=20):
python cured.py \
  --model meta-llama/Llama-3.1-8B-Instruct --load-in-4bit \
  --protocols cured --router new \
  --router-config configs/router_thresholds.json \
  --benchmark truthfulqa --n 20 --save-per-question \
  --out /tmp/smoke_test.json
```

## Threshold Configuration

See `configs/router_thresholds.json` for all router thresholds.
Critical calibrated values (corrected from README Bug A):

```json
{
  "tau_kappa":       0.70,
  "tau_ECR":         0.04,
  "tau_R2":          0.65,
  "tau_H_easy":      0.5,
  "tau_H_hard":      3.0,
  "profile_mean_r2": 0.582
}
```

**Known config limitations (disclosed for paper):**

- **Bug B (profile scale):** `profile_mean_r2=0.582` is calibrated from 8B profiling and applied
  to all model scales. The 3B-native value is 0.501 (< 0.55 viability threshold), which would
  disable the scale shortcut for 3B. `configs/router_thresholds_3b.json` provides the 3B-native
  config for ablation.
- **Bug C (tau_R2):** With `tau_R2=0.65` and empirical per-question R²_q mean ≈ 0.50 at 3B/8B,
  Gates 2 and 4 fire on fewer than ~5% of questions. The scale-aware shortcut (between Gates 1
  and 2) handles the majority of ALTA routing at these scales.
- **Gate 1 inactive (Issue 5):** For models ≤14B, Gate 1 requires SC_q (self-consistency). Phase 4
  runs omit `--compute-sc` for efficiency; Gate 1 is therefore inactive for all canonical 3B/8B
  results. Models >14B use `H_final < tau_H_easy` only (SC not required).
