# scripts/maintenance/

One-time and server-management scripts used during the experiment campaign.
These are **not** part of the reproducible experiment pipeline.

| Script | Purpose |
|---|---|
| `fix_and_restart.py` | Patch and restart a crashed local pipeline |
| `fix_and_restart_a800.py` | Fix Unicode chars in remote pipeline script and restart |
| `patch_server_wait_and_restart.py` | Wait for remote server and restart pipeline |
| `check_and_patch_a800.py` | Diagnose + patch A800 pipeline thresholds in-place |
| `push_profiler_and_restart_pipeline.py` | Sync profiler changes and restart remote job |
| `ensure_pipeline_watcher.py` | Confirm watchdog process is alive on remote |
| `restart_pipeline_robust.py` | Robust restart with exponential backoff |
| `redeploy_clean_pipeline.py` | Full clean-slate redeploy to A800 |
| `add_scoring_to_cured.py` | One-time patch to add scoring flags |
| `validate_patch.py` | Verify a patched cured.py compiles and passes smoke test |
| `patch_cured.py` | Apply one-time feature patch to cured.py |
| `patch_docs.py` | Update result tables in docs |
| `run_stats.py` | Run per-experiment statistics summary |
