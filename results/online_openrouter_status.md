# OpenRouter Online Run Status

Last updated: 2026-04-10

## Verified Outputs (safe to use)

- results/results_openrouter_both.json
  - model: meta-llama/llama-3.1-8b-instruct
  - benchmark: both
  - n_target: 50
  - truthfulqa: greedy 64.0%, cove 62.0%, cured_api 68.0%
  - medhallu: greedy 10.0%, cove 10.0%, cured_api 10.0%

- results/results_openrouter_medqa.json
  - model: meta-llama/llama-3.1-8b-instruct
  - benchmark: custom (benchmarks/medqa_usmle_n200.csv)
  - n_target: 100
  - greedy 4.0%, cove 3.0%, cured_api 2.0%

- results/results_openrouter_pubmedqa.json
  - model: meta-llama/llama-3.1-8b-instruct
  - benchmark: custom (benchmarks/pubmedqa_n200.csv)
  - n_target: 100
  - greedy 16.0%, cove 15.0%, cured_api 12.0%

## Completion State

- OpenRouter online process is complete for:
  - both (n=50)
  - MedQA custom (n=100)
  - PubMedQA custom (n=100)

## Invalid Outputs (excluded from reporting)

The following runs were invalid because all API calls failed with OpenRouter HTTP 401 ("User not found").
They are archived and should not be used in summaries, plots, or papers.

- results/archive/invalid_openrouter_401_20260410/results_openrouter_both.json
- results/archive/invalid_openrouter_401_20260410/results_openrouter_medqa.json
- results/archive/invalid_openrouter_401_20260410/results_openrouter_pubmedqa.json
- logs/archive/invalid_openrouter_401_20260410/
