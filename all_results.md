# All Results (Canonical)

Last updated: 2026-04-14 08:35 CST

This is the single canonical results document.
It consolidates:

1. Forked Alex legacy results (historical)
2. Current before-organization results inventory
3. Current after-organization canonical results inventory and metrics

## Live Run Status

- Final suite status: running
- Active stage: Job 8 (TriviaQA, cove)
- Latest progress: 280/1000
- ETA from live log: ~144 min
- Output pending: results/CANONICAL_v2/results_8b_triviaqa_v1.json

### Runtime tuning currently active

- GPU persistence mode: on
- App clocks: SM 1410 MHz, MEM 1215 MHz
- Process priorities: runner NI=-10, suite shell NI=-5
- Latest GPU snapshot: 67% GPU util, 63% memory util, ~206 W, P0

## Historical Legacy Results (Forked Alex)

Source: results/archive/comprehensive_results_legacy_origin_alex.md

### Topline summary

- Entropy profile: H1=0.0806 -> H7=10.8342 -> H28=0.8516
- L1->L28 mean dH=+0.7711 (dH<0=16.7%)
- L7->L28 mean dH=-9.9826 (dH<0=100%)
- 3B late-layer linearity mean R2=0.5557
- ALTA-style 3B (n=50): 72% acc, 0% rep
- TruthfulQA DeLTa+DoLa sweep max acc=74%
- MedHallu generation best: gadr2_cured 54% (greedy 50%)

### Legacy medhallu generation (n=50)

| method | acc | rep |
|---|---:|---:|
| greedy | 50% | 0% |
| cove | 50% | 2% |
| cove_rag | 50% | 0% |
| delta_dola | 52% | 0% |
| gadr2_cured | 54% | 2% |

### Legacy ablations (n=50)

| method | acc | rep |
|---|---:|---:|
| iti_alpha0.5 | 52% | 4% |
| sled | 52% | 0% |
| bon3_t0.3 | 48% | 0% |

## Before Organization Data

Files in results/*.json:

- results/alta_3b_results.json
- results/bon_results.json
- results/calibration_results.json
- results/entropy_by_layer.json
- results/generation_results.json
- results/generation_results_n100_4configs.json
- results/grid_search_results.json
- results/instruct_results.json
- results/iti_results.json
- results/logit_linearity_3b.json
- results/medhallu_ablation_results.json
- results/medhallu_generation_results.json
- results/medhallu_results.json
- results/online_results.json
- results/results_cloudflare_medhallu_v2.json
- results/results_cloudflare_medqa_fixed.json
- results/results_cloudflare_medqa_v2.json
- results/results_cloudflare_pubmedqa_v2.json
- results/results_openrouter_both.json
- results/results_openrouter_medhallu_v2.json
- results/results_openrouter_medqa_v2.json
- results/results_openrouter_pubmedqa_v2.json
- results/selfcheck_results.json
- results/truthfulqa_delta_dola_sweep.json

## After Organization Data

Files in results/CANONICAL_v2:

- results/CANONICAL_v2/results_3b_medhallu_n100.json
- results/CANONICAL_v2/results_3b_truthfulqa_full_mc.json
- results/CANONICAL_v2/results_8b_both.json
- results/CANONICAL_v2/results_8b_medhallu_v2.json
- results/CANONICAL_v2/results_8b_medqa_v2.json
- results/CANONICAL_v2/results_8b_medqa_v3_fixed.json
- results/CANONICAL_v2/results_8b_pubmedqa_v2.json
- results/CANONICAL_v2/results_8b_strategyqa_v1.json
- results/CANONICAL_v2/results_8b_truthfulqa_full_mc.json
- pending write: results/CANONICAL_v2/results_8b_triviaqa_v1.json

### After-organization metrics snapshot

| file | benchmark | scoring | protocol | acc | rep | runtime_min | n_scored |
|---|---|---|---|---:|---:|---:|---:|
| results_3b_medhallu_n100.json | medhallu | cosine | greedy | 0.5500 | 0.0000 | 6.32 | 100 |
| results_3b_medhallu_n100.json | medhallu | cosine | cove | 0.5600 | 0.0000 | 22.17 | 100 |
| results_3b_medhallu_n100.json | medhallu | cosine | cured | 0.5800 | 0.0000 | 12.04 | 100 |
| results_3b_truthfulqa_full_mc.json | truthfulqa | mc | greedy | 0.3684 | 0.0000 | 35.23 | 817 |
| results_3b_truthfulqa_full_mc.json | truthfulqa | mc | alta | 0.3684 | 0.0012 | 37.27 | 817 |
| results_3b_truthfulqa_full_mc.json | truthfulqa | mc | delta_dola | 0.3684 | 0.0012 | 37.58 | 817 |
| results_3b_truthfulqa_full_mc.json | truthfulqa | mc | cove | 0.3635 | 0.0000 | 155.30 | 817 |
| results_3b_truthfulqa_full_mc.json | truthfulqa | mc | cured | 0.3635 | 0.0000 | 41.67 | 817 |
| results_8b_both.json | truthfulqa |  | greedy | 0.7755 | 0.0200 | 62.97 | 49 |
| results_8b_both.json | truthfulqa |  | alta | 0.7959 | 0.0200 | 62.10 | 49 |
| results_8b_both.json | truthfulqa |  | cove | 0.7800 | 0.0000 | 257.26 | 50 |
| results_8b_both.json | truthfulqa |  | cured | 0.7755 | 0.0200 | 64.45 | 49 |
| results_8b_both.json | medhallu |  | greedy | 0.5306 | 0.0200 | 67.28 | 49 |
| results_8b_both.json | medhallu |  | alta | 0.4694 | 0.0200 | 67.43 | 49 |
| results_8b_both.json | medhallu |  | cove | 0.5400 | 0.0000 | 271.77 | 50 |
| results_8b_both.json | medhallu |  | cured | 0.5510 | 0.0200 | 167.31 | 49 |
| results_8b_medhallu_v2.json | custom | cosine | greedy | 0.5600 | 0.0000 | 6.08 | 100 |
| results_8b_medhallu_v2.json | custom | cosine | alta | 0.5400 | 0.0000 | 6.67 | 100 |
| results_8b_medhallu_v2.json | custom | cosine | cove | 0.5354 | 0.0100 | 27.21 | 99 |
| results_8b_medhallu_v2.json | custom | cosine | cured | 0.5300 | 0.0000 | 17.85 | 100 |
| results_8b_medqa_v2.json | custom | letter | greedy | 0.2900 | 0.0000 | 42.53 | 100 |
| results_8b_medqa_v2.json | custom | letter | alta | 0.2800 | 0.0000 | 45.87 | 100 |
| results_8b_medqa_v2.json | custom | letter | cove | 0.1300 | 0.0000 | 393.13 | 100 |
| results_8b_medqa_v2.json | custom | letter | cured | 0.1400 | 0.0000 | 468.18 | 100 |
| results_8b_medqa_v3_fixed.json | custom | letter | greedy | 0.5500 | 0.0000 | 2.54 | 100 |
| results_8b_medqa_v3_fixed.json | custom | letter | alta | 0.5700 | 0.0000 | 3.12 | 100 |
| results_8b_medqa_v3_fixed.json | custom | letter | cove | 0.3500 | 0.0000 | 20.68 | 100 |
| results_8b_medqa_v3_fixed.json | custom | letter | cured | 0.5700 | 0.0000 | 3.14 | 100 |
| results_8b_pubmedqa_v2.json | custom | yesno | greedy | 0.5500 | 0.0000 | 0.3400 | 100 |
| results_8b_pubmedqa_v2.json | custom | yesno | alta | 0.5300 | 0.0000 | 0.3300 | 100 |
| results_8b_pubmedqa_v2.json | custom | yesno | cove | 0.5700 | 0.0000 | 14.55 | 100 |
| results_8b_pubmedqa_v2.json | custom | yesno | cured | 0.5300 | 0.0000 | 0.3200 | 100 |
| results_8b_strategyqa_v1.json | custom | yesno | greedy | 0.7220 | 0.0000 | 1.54 | 500 |
| results_8b_strategyqa_v1.json | custom | yesno | alta | 0.7240 | 0.0000 | 1.58 | 500 |
| results_8b_strategyqa_v1.json | custom | yesno | cove | 0.6260 | 0.0000 | 66.92 | 500 |
| results_8b_strategyqa_v1.json | custom | yesno | cured | 0.7240 | 0.0000 | 1.58 | 500 |
| results_8b_truthfulqa_full_mc.json | truthfulqa | mc | greedy | 0.4027 | 0.0000 | 47.73 | 817 |
| results_8b_truthfulqa_full_mc.json | truthfulqa | mc | alta | 0.4027 | 0.0000 | 49.83 | 817 |
| results_8b_truthfulqa_full_mc.json | truthfulqa | mc | cove | 0.4027 | 0.0012 | 212.01 | 817 |
| results_8b_truthfulqa_full_mc.json | truthfulqa | mc | cured | 0.4027 | 0.0000 | 57.37 | 817 |

## Provenance

- historical legacy source: results/archive/comprehensive_results_legacy_origin_alex.md
- compatibility source: raw_results.md
- compatibility source: comprehensive_results.md
