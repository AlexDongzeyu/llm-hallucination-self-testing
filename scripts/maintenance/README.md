# Maintenance Scripts

These are one-time or rarely-used tools. They are **not** part of the main CURED pipeline.

| Script | Purpose |
|--------|---------|
| `patch_cured.py` | Applied historical text patches to cured.py (MC scoring fix, ALTA cutoff). Run once, no longer needed. |
| `patch_docs.py` | Injected MC-validity caveats into README and all_results.md. Run once. |
| `validate_patch.py` | AST smoke-test that checks cured.py has expected symbols/flags. Useful after large edits. |
| `run_stats.py` | Early ad-hoc z-tests on hardcoded n=50 results. Superseded by `compute_final_stats.py`. |
