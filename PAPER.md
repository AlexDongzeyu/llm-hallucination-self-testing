# Paper Assets And Result Pointers

## Canonical Inputs

| Asset | Location |
|---|---|
| Human-readable results | [RESULTS.md](RESULTS.md) |
| Auto-generated result ledger | [all_results.md](all_results.md) |
| Canonical JSON outputs | `results/CANONICAL_v2/` |
| Figure assets | `paper/figures/` |
| Figure generator | `experiments/generate_paper_figures.py` |

## Paper Result Claims

Use the following result set for manuscript tables and poster figures:

| Claim | Source |
|---|---|
| 3B TruthfulQA CURED gain: +8.4 pp paired, p < 0.0001 | `statistics_table.json` |
| 8B TruthfulQA CURED gain: +11.8 pp paired, p < 0.0001 | `statistics_table.json` |
| 8B FACTOR-Wiki fixed result: CURED 65.0%, ALTA 65.0%, greedy 29.5% | `results_8b_factor_wiki_n200_fixed.json` |
| Scale-level R2 correlation: r=0.9859, p=0.0141 | `r2_scale_correlation.json` |
| Per-question R2 correlation: r=0.0393, p=0.5803 | `r2_scale_correlation.json` |

## Protocol Boundary Notes

| Boundary | Required Wording |
|---|---|
| TruthfulQA | Label cosine generation runs separately from MC scoring runs. |
| MedQA/PubMedQA | Do not compare local free-form or yes/no runs with external MC-letter protocols. |
| FACTOR-Wiki | Present original 43.0% CURED as a diagnostic regression and fixed 65.0% as the corrected domain-routing result. |
| 3B native profile | Present as an ablation, not as the headline CURED v2 result. |

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
