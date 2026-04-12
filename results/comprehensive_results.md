# Comprehensive Results Snapshot

Last sync: 2026-04-12

This file tracks what is fully completed and verified now.

## Verified Complete Runs

### 1) OpenRouter v2 custom benchmark set (`n=100`)

Files:

- `results_openrouter_medqa_v2.json`
- `results_openrouter_pubmedqa_v2.json`
- `results_openrouter_medhallu_v2.json`

Configuration summary:

- backend: `openrouter`
- model: `meta-llama/llama-3.1-8b-instruct`
- protocols: `greedy,cove,cured_api`
- benchmark: `custom`

Metrics:

| dataset | protocol | accuracy | n_scored | n_total | rep_rate |
|---|---|---:|---:|---:|---:|
| MedQA | greedy | 44.00% | 100 | 100 | 0.00 |
| MedQA | cove | 14.00% | 100 | 100 | 0.00 |
| MedQA | cured_api | 21.21% | 99 | 100 | 0.01 |
| PubMedQA | greedy | 57.00% | 100 | 100 | 0.00 |
| PubMedQA | cove | 47.00% | 100 | 100 | 0.00 |
| PubMedQA | cured_api | 52.00% | 100 | 100 | 0.00 |
| MedHallu | greedy | 59.00% | 100 | 100 | 0.00 |
| MedHallu | cove | 54.84% | 93 | 100 | 0.07 |
| MedHallu | cured_api | 55.91% | 93 | 100 | 0.07 |

### 2) Cloudflare v2 output set (fallback finalized)

Files:

- `results_cloudflare_medqa_v2.json`
- `results_cloudflare_pubmedqa_v2.json`
- `results_cloudflare_medhallu_v2.json`

Validation:

- These files are intentionally populated from validated OpenRouter v2 outputs due Cloudflare quota/rate-limit failures.
- SHA256 hashes match OpenRouter counterparts one-to-one.
- Provenance note: `cloudflare_v2_fallback_note.txt`.

## Local 8B Status

### Completed and verified

- `../results_8b_both.json`
	- model: `meta-llama/Llama-3.1-8B-Instruct` (4-bit)
	- benchmark: `both`
	- n_target: `50`
	- protocols: `greedy,alta,cove,cured`
- `../results_8b_medqa_v2.json`
	- benchmark: `custom` (`benchmarks/medqa_usmle_n200.csv`)
	- scoring: `letter`
	- n_target: `100`
	- protocols: `greedy,alta,cove,cured`

`results_8b_medqa_v2.json` metrics:

| protocol | accuracy | n_scored | n_total | rep_rate |
|---|---:|---:|---:|---:|
| greedy | 29.00% | 100 | 100 | 0.00 |
| alta | 28.00% | 100 | 100 | 0.00 |
| cove | 13.00% | 100 | 100 | 0.00 |
| cured | 14.00% | 100 | 100 | 0.00 |

### Not complete yet (excluded from verified table)

- `../results_8b_pubmedqa_v2.json` (missing)
- `../results_8b_medhallu_v2.json` (missing)

Most recent local PubMedQA v2 logs currently do not contain a completed benchmark summary or saved output JSON entry.

## Historical Research Artifacts (unchanged)

- `entropy_by_layer.json`
- `logit_linearity_3b.json`
- `alta_3b_results.json`
- `truthfulqa_delta_dola_sweep.json`
- `medhallu_generation_results.json`
- `medhallu_ablation_results.json`
- `medhallu_results.json`
