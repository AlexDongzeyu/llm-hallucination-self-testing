# Archive Folder

Historical artifacts only. Do not use this folder as the current source of truth.

## Files

- `medhallu_detector_legacy_results.json`: legacy detector-style MedHallu output (comparison only).
- `medhallu_results_snapshot_n50.json`: snapshot copy of MedHallu MC n=50 output.

Canonical files:
- `results/medhallu_results.json` (MC ablation)
- `results/medhallu_generation_results.json` (primary generation metric)
- `all_results.md` (current consolidated summary)

Additional policy:
- Legacy API v1 outputs (for example results_openrouter_medqa.json without v2 tag)
	should live in this archive folder.
