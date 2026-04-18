# src/legacy/

Early prototype scripts written before the CURED architecture was finalised.
These files **predate** the current `cured.py` monolith and are kept for
historical reference only.

> **Do not import these from production code.**  All production functionality
> has been re-implemented and improved in `cured.py` and the `cured/` package.

| Script | Prototype functionality |
|---|---|
| `generate_base.py` | First greedy generation loop |
| `generate_instruct.py` | Instruct-model prompt formatting prototype |
| `load_model.py` | Early model loading helper |
| `iti_probe.py` | ITI probe training prototype (superseded by `train_iti_probes`) |
| `learn_router.py` | Exploratory router learning |
| `entropy_check.py` | Layer-wise entropy inspection |
| `entropy_gap.py` | Entropy gap heuristic experiments |
| `extract_logits.py` | Early logit extraction loop |
| `extraction.py` | Feature extraction utilities |
| `best_of_n.py` | Best-of-N sampling baseline |
| `best_of_n_entropy_gated.py` | Entropy-gated BoN variant |
| `calibration_proof.py` | Manual R² calibration proof-of-concept |
| `diagnose_jsd.py` | JSD diagnosis utility |
| `audit.py` | Dataset audit tools |
| `recover_calib_results.py` | Recover calibration results from partial runs |
| `rescore_threshold_test.py` | Threshold sensitivity analysis |
| `verify_features.py` | Feature sanity checks |
