# Paper assets

- **Figures:** `paper/figures/` — copies of fig1–fig4 used in the manuscript (sources regenerated via `experiments/generate_paper_figures.py`).
- **PDF:** add `paper/cured_paper.pdf` when the camera-ready build is fixed.

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

## Related work (pointers)

- **Adaptive decoding / test-time compute:** contrast with fixed multi-sample or verifier-heavy pipelines; CURED uses a single forward pass for trajectory features (R², κ, ECR) plus routed second-stage decoding only when the selected protocol requires it.
- **Uncertainty and entropy:** ECR and entropy gates relate to confidence-based generation control; semantic-entropy ablation outputs live under `results/CANONICAL_v2/`.
- **Faithfulness / verification (CoVe, ITI):** CoVe and ITI are used as domain- and capability-conditioned branches rather than always-on costs.

Full numbers and McNemar tests: [RESULTS.md](RESULTS.md).
