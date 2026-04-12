# 2026-04-12 Cleanup Archive

This folder stores non-essential diagnostic logs moved out of the top-level `logs/` directory during repository cleanup.

Moved categories:
- Failed local PubMedQA v2 attempts (`local_pubmedqa_v2_*.log`)
- One-off prep/precheck logs (`prep_benchmarks.log`, `precheck_status.txt`)
- Dataset rebuild helper logs (`rebuild_*.log`)
- Scoring patch verification logs (`verify_scoring_patch_*.log`)
- Cloudflare probe smoke logs (`cloudflare_smoke_probe_*.log`)

Reason:
- Keep primary run logs in `logs/` easy to scan.
- Preserve debugging artifacts for traceability without cluttering active run outputs.
