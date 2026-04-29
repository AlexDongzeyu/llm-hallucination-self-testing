# Log Archive: 2026-04-12 Cleanup

This folder stores non-essential diagnostic logs moved out of the active `logs/` directory.

## Contents

| Category | Examples |
|---|---|
| Failed local runs | `local_pubmedqa_v2_*.log` |
| Prep and precheck logs | `prep_benchmarks.log`, `precheck_status.txt` |
| Dataset rebuild logs | `rebuild_*.log` |
| Scoring patch checks | `verify_scoring_patch_*.log` |
| Provider smoke probes | `cloudflare_smoke_probe_*.log` |

## Policy

| Location | Meaning |
|---|---|
| `logs/` | Active or primary run logs. |
| `logs/archive/` | Historical diagnostics retained for traceability. |
| `results/CANONICAL_v2/` | Canonical result artifacts; prefer these over logs for result claims. |

Use this archive only for debugging provenance, not for headline result numbers.
