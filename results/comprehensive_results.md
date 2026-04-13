# Comprehensive Results Snapshot

Last sync: 2026-04-12

This file tracks current reportable outputs and where they live after cleanup.

## Canonical Report Bundle

Use only files in results/CANONICAL_v2 for paper/report tables.

Expected canonical files:

- results/CANONICAL_v2/results_8b_truthfulqa_full_mc.json
- results/CANONICAL_v2/results_3b_truthfulqa_full_mc.json
- results/CANONICAL_v2/results_8b_medhallu_v2.json
- results/CANONICAL_v2/results_8b_pubmedqa_v2.json
- results/CANONICAL_v2/results_8b_medqa_v3_fixed.json
- results/CANONICAL_v2/results_3b_medhallu_n100.json
- results/CANONICAL_v2/results_openrouter_medqa_v2.json
- results/CANONICAL_v2/results_openrouter_pubmedqa_v2.json
- results/CANONICAL_v2/results_openrouter_medhallu_v2.json

## API v2 Set (OpenRouter)

Source files (before canonical copy):

- results/results_openrouter_medqa_v2.json
- results/results_openrouter_pubmedqa_v2.json
- results/results_openrouter_medhallu_v2.json

Configuration summary:

- backend: openrouter
- model: meta-llama/llama-3.1-8b-instruct
- protocols: greedy,cove,cured_api
- benchmark: custom

## Cloudflare v2 Fallback Set

Files:

- results/results_cloudflare_medqa_v2.json
- results/results_cloudflare_pubmedqa_v2.json
- results/results_cloudflare_medhallu_v2.json

Validation note:

- Fallback provenance is documented in results/cloudflare_v2_fallback_note.txt.

## Historical Research Artifacts (unchanged)

- results/entropy_by_layer.json
- results/logit_linearity_3b.json
- results/alta_3b_results.json
- results/truthfulqa_delta_dola_sweep.json
- results/medhallu_generation_results.json
- results/medhallu_ablation_results.json
- results/medhallu_results.json

## Cleanup Convention

- Legacy top-level result files should be moved into results/CANONICAL_v2.
- Invalid or superseded files should be moved into results/archive.
