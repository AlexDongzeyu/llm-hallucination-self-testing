# OpenRouter Online Run Status

Last updated: 2026-04-12

## Verified Outputs (safe to use)

### Primary v2 custom benchmark outputs (`n=100`)

- results/results_openrouter_medqa_v2.json
  - model: meta-llama/llama-3.1-8b-instruct
  - benchmark: custom (medqa_usmle_n200.csv)
  - greedy 44.0%, cove 14.0%, cured_api 21.21%

- results/results_openrouter_pubmedqa_v2.json
  - model: meta-llama/llama-3.1-8b-instruct
  - benchmark: custom (pubmedqa_n200.csv)
  - greedy 57.0%, cove 47.0%, cured_api 52.0%

- results/results_openrouter_medhallu_v2.json
  - model: meta-llama/llama-3.1-8b-instruct
  - benchmark: custom (medhallu_generation_n200.csv)
  - greedy 59.0%, cove 54.84%, cured_api 55.91%

### Legacy but valid OpenRouter output

- results/results_openrouter_both.json
  - complete and valid historical run (`both`, `n=50`)
  - keep for archival comparison only; do not mix directly with v2 custom benchmark reporting

## Completion State

- Complete and verified now:
  - OpenRouter v2 custom: MedQA, PubMedQA, MedHallu
  - OpenRouter legacy: both (`n=50`)
- Not complete yet:
  - local v2 custom output trio (`results_8b_medqa_v2.json`, `results_8b_pubmedqa_v2.json`, `results_8b_medhallu_v2.json`)

## Cloudflare v2 Relation

- `results_cloudflare_*_v2.json` files are finalized fallback copies of the validated OpenRouter v2 outputs.
- Pairwise SHA256 hashes are identical for MedQA, PubMedQA, and MedHallu v2 result files.
- Provenance is documented in `results/cloudflare_v2_fallback_note.txt`.

## Excluded Invalid Artifacts

The archived 401-failure artifacts remain excluded from all reporting:

- results/archive/invalid_openrouter_401_20260410/results_openrouter_both.json
- results/archive/invalid_openrouter_401_20260410/results_openrouter_medqa.json
- results/archive/invalid_openrouter_401_20260410/results_openrouter_pubmedqa.json
- logs/archive/invalid_openrouter_401_20260410/
