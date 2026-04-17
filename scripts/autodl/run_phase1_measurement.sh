#!/usr/bin/env bash
# =============================================================================
# run_phase1_measurement.sh — Model profiling: R², κ, ECR for 3B/8B/14B/32B
# Expected runtime: ~3-5 hours on A800 80GB
# Run BEFORE Phase 2 ablations to determine tau_R2 and tau_ECR priors.
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
    echo "[HINT] Set PYTHON_BIN or run scripts/autodl/bootstrap_gpu_env.sh first." >&2
    exit 1
fi

log() { echo "[$(date +'%H:%M:%S')] $*"; }

for MODEL in \
    "meta-llama/Llama-3.2-3B-Instruct" \
    "meta-llama/Llama-3.1-8B-Instruct" \
    "Qwen/Qwen2.5-14B-Instruct" \
    "Qwen/Qwen2.5-32B-Instruct"; do

    SIZE=$(echo "$MODEL" | grep -oP '\d+[Bb]' | head -1 | tr '[:upper:]' '[:lower:]')
    LOAD4BIT=""
    [ "$SIZE" = "32b" ] && LOAD4BIT="--load-in-4bit"

    log "Profiling $MODEL (size=$SIZE)..."
    "$PYTHON" -u experiments/compute_logit_linearity.py \
        --model "$MODEL" $LOAD4BIT \
        --n 50 \
        --start-layer-ratio 0.7 \
        --end-layer-ratio 1.0 \
        --compute-curvature \
        --compute-ecr \
        --out "$CANONICAL_DIR/profile_${SIZE}.json" \
        2>&1 | tee "logs/profile_${SIZE}_${TS}.log"
    log "Done: $CANONICAL_DIR/profile_${SIZE}.json"
done

log "Phase 1 complete. Review profile_*.json and update configs/router_thresholds.json:"
log "  Look at mean_r2, mean_ecr, mean_kappa for each model size."
log "  Set tau_R2 between the smallest scale that passes and the largest that fails."
