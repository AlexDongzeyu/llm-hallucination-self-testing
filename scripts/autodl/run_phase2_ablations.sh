#!/usr/bin/env bash
# =============================================================================
# run_phase2_ablations.sh — Ablation grid: 4 models × 4 protocols × 2 benchmarks
# Expected runtime: ~10 hours on A800 80GB
#
# CRITICAL: --skip-iti is used ONLY for greedy/alta/cove (Loop 1).
# ITI runs in a SEPARATE loop (Loop 2) without --skip-iti so probes are trained.
# Mixing --skip-iti with --protocols iti silently falls back to greedy.
#
# Prerequisites:
#   - All 6 cured.py code changes implemented (run smoke test first)
#   - Phase 1 profiling complete (profile_*.json exist)
#   - ITI smoke test verified: python cured.py --model <MODEL> --protocols iti
#       --benchmark truthfulqa --n 5 (check for "iti" in routing)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON_BIN:-$ROOT/llm-env/bin/python}"
CANONICAL_DIR="results/CANONICAL_v2"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs "$CANONICAL_DIR"

if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] Python not found: $PYTHON" >&2
    exit 1
fi

log() { echo "[$(date +'%H:%M:%S')] $*"; }

declare -A MODEL_PARAMS
MODEL_PARAMS["meta-llama/Llama-3.2-3B-Instruct"]="3.0:3b"
MODEL_PARAMS["meta-llama/Llama-3.1-8B-Instruct"]="8.0:8b"
MODEL_PARAMS["Qwen/Qwen2.5-14B-Instruct"]="14.0:14b"
MODEL_PARAMS["Qwen/Qwen2.5-32B-Instruct"]="32.0:32b"

# ── Loop 1: greedy / alta / cove (no ITI probes needed) ───────────────────────
log "=== LOOP 1: greedy / alta / cove ==="
for MODEL in "${!MODEL_PARAMS[@]}"; do
    IFS=':' read -r PARAMS SIZE <<< "${MODEL_PARAMS[$MODEL]}"
    LOAD4BIT=""; [ "$SIZE" = "32b" ] && LOAD4BIT="--load-in-4bit"

    for PROTOCOL in greedy alta cove; do
        for BENCH in truthfulqa medhallu; do
            OUT="$CANONICAL_DIR/ablation_${SIZE}_${PROTOCOL}_${BENCH}_n200.json"
            log "  $SIZE | $PROTOCOL | $BENCH → $OUT"
            "$PYTHON" -u cured.py \
                --model "$MODEL" $LOAD4BIT \
                --model-params-b "$PARAMS" \
                --protocols "$PROTOCOL" \
                --benchmark "$BENCH" \
                --n 200 \
                --seed 42 \
                --no-shuffle \
                --scoring cosine \
                --save-per-question \
                --skip-iti \
                --out "$OUT" \
                2>&1 | tee "logs/ablation_${SIZE}_${PROTOCOL}_${BENCH}_${TS}.log"
        done
    done
done
log "Loop 1 complete."

# ── Loop 2: ITI (probe training required — DO NOT add --skip-iti) ─────────────
log "=== LOOP 2: ITI (requires probe training) ==="
log "Verifying ITI probe training works before full run..."
FIRST_MODEL="meta-llama/Llama-3.2-3B-Instruct"
"$PYTHON" -u cured.py \
    --model "$FIRST_MODEL" \
    --model-params-b "3.0" \
    --protocols iti \
    --benchmark truthfulqa \
    --n 5 \
    --seed 42 --no-shuffle \
    --out /tmp/iti_smoke_test.json \
    2>&1 | tee "logs/iti_smoke_${TS}.log"

# Check the smoke test routed to "iti" not "greedy_no_iti"
if python -c "
import json, sys
d = json.load(open('/tmp/iti_smoke_test.json'))
routing = d.get('results', {}).get('truthfulqa', {}).get('iti', {}).get('routing', {})
if 'greedy_no_iti' in routing:
    print('FAIL: ITI probes not trained, fell back to greedy')
    sys.exit(1)
print('OK: ITI routing confirmed')
"; then
    for MODEL in "${!MODEL_PARAMS[@]}"; do
        IFS=':' read -r PARAMS SIZE <<< "${MODEL_PARAMS[$MODEL]}"
        LOAD4BIT=""; [ "$SIZE" = "32b" ] && LOAD4BIT="--load-in-4bit"

        for BENCH in truthfulqa medhallu; do
            OUT="$CANONICAL_DIR/ablation_${SIZE}_iti_${BENCH}_n200.json"
            log "  $SIZE | iti | $BENCH → $OUT"
            "$PYTHON" -u cured.py \
                --model "$MODEL" $LOAD4BIT \
                --model-params-b "$PARAMS" \
                --protocols iti \
                --benchmark "$BENCH" \
                --n 200 \
                --seed 42 \
                --no-shuffle \
                --scoring cosine \
                --save-per-question \
                --out "$OUT" \
                2>&1 | tee "logs/ablation_${SIZE}_iti_${BENCH}_${TS}.log"
        done
    done
    log "Loop 2 complete."
else
    log "WARNING: ITI smoke test failed. Skipping ITI ablations."
    log "Check that iti_top_heads.npy / iti_head_vectors.npy are training correctly."
fi

log "Phase 2 ablations complete."
log "Next: run  python calibrate_router.py  to extract thresholds from ablation data."
