#!/usr/bin/env bash
# =============================================================================
# run_phase4_main.sh — Main publication experiments
# Expected runtime: ~18-24 hours on A800 80GB (includes 32B with SC disabled)
#
# Prerequisites:
#   - Phase 2 ablations complete
#   - Phase 3 calibration complete (configs/router_thresholds.json updated)
#   - StrategyQA smoke test: python cured.py --model ... --benchmark custom
#       --custom-csv benchmarks/strategyqa_n500.csv --n 5 --protocols greedy
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON_BIN:-$ROOT/llm-env/bin/python}"
CANONICAL_DIR="results/CANONICAL_v2"
ROUTER_CONFIG="configs/router_thresholds.json"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs "$CANONICAL_DIR"

if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] Python not found: $PYTHON" >&2
    exit 1
fi
if [ ! -f "$ROUTER_CONFIG" ]; then
    echo "[ERROR] $ROUTER_CONFIG not found. Run calibrate_router.py first." >&2
    exit 1
fi

log() { echo "[$(date +'%H:%M:%S')] $*"; }

# ── Verify StrategyQA CSV loads before full run ───────────────────────────────
log "Smoke-testing StrategyQA CSV..."
"$PYTHON" -u cured.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" --load-in-4bit \
    --protocols greedy \
    --benchmark custom \
    --custom-csv benchmarks/strategyqa_n500.csv \
    --question-col question --answer-col answer \
    --n 5 --seed 42 --no-shuffle --scoring yesno --skip-iti \
    --out /tmp/strategyqa_smoke.json \
    2>&1 | tee "logs/strategyqa_smoke_${TS}.log"
log "StrategyQA smoke test passed."

# Model specs: "HF_ID:params_B:size_tag"
MODELS=(
    "meta-llama/Llama-3.2-3B-Instruct:3.0:3b"
    "meta-llama/Llama-3.1-8B-Instruct:8.0:8b"
    "Qwen/Qwen2.5-14B-Instruct:14.0:14b"
    "Qwen/Qwen2.5-32B-Instruct:32.0:32b"
)

# ── Main CURED runs: 4 models × 3 benchmarks × n=500 ─────────────────────────
log "=== MAIN CURED RUNS ==="
for MODEL_SPEC in "${MODELS[@]}"; do
    IFS=':' read -r M P S <<< "$MODEL_SPEC"
    LOAD4BIT=""; [ "$S" = "32b" ] && LOAD4BIT="--load-in-4bit"
    # SC disabled for 32B (compute time: ~2h per benchmark at n=500)
    COMPUTE_SC="--compute-sc"; [ "$S" = "32b" ] && COMPUTE_SC=""

    for BENCH in truthfulqa medhallu strategyqa; do
        SCORING="cosine"
        if [ "$BENCH" = "strategyqa" ]; then
            SCORING="yesno"
            BENCH_ARGS=(--benchmark custom --custom-csv "benchmarks/strategyqa_n500.csv" --question-col question --answer-col answer)
        else
            BENCH_ARGS=(--benchmark "$BENCH")
        fi

        OUT="$CANONICAL_DIR/main_cured_${S}_${BENCH}_n500.json"
        log "  $S | cured (new router) | $BENCH → $OUT"
        "$PYTHON" -u cured.py \
            --model "$M" $LOAD4BIT \
            --model-params-b "$P" \
            --protocols cured \
            --router new \
            --router-config "$ROUTER_CONFIG" \
            "${BENCH_ARGS[@]}" \
            --n 500 \
            --seed 42 --no-shuffle \
            --scoring "$SCORING" \
            $COMPUTE_SC \
            --save-per-question \
            --skip-iti \
            --out "$OUT" \
            2>&1 | tee "logs/main_cured_${S}_${BENCH}_${TS}.log"
    done
done
log "Main CURED runs complete."

# ── Greedy baselines: full-scale for McNemar pairing ─────────────────────────
log "=== GREEDY BASELINES ==="
for MODEL_SPEC in "${MODELS[@]}"; do
    IFS=':' read -r M P S <<< "$MODEL_SPEC"
    LOAD4BIT=""; [ "$S" = "32b" ] && LOAD4BIT="--load-in-4bit"

    OUT="$CANONICAL_DIR/main_greedy_${S}_truthfulqa_n817.json"
    log "  $S | greedy | truthfulqa n=817 → $OUT"
    "$PYTHON" -u cured.py \
        --model "$M" $LOAD4BIT \
        --model-params-b "$P" \
        --protocols greedy \
        --benchmark truthfulqa \
        --n 817 \
        --seed 42 --no-shuffle \
        --scoring cosine \
        --save-per-question --skip-iti \
        --out "$OUT" \
        2>&1 | tee "logs/main_greedy_${S}_truthfulqa_${TS}.log"
done
log "Greedy baselines complete."

# ── Old-router CURED (ablation comparison) ───────────────────────────────────
log "=== OLD ROUTER ABLATION (8B TruthfulQA) ==="
"$PYTHON" -u cured.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" --load-in-4bit \
    --model-params-b 8.0 \
    --protocols cured \
    --router old \
    --benchmark truthfulqa \
    --n 500 \
    --seed 42 --no-shuffle \
    --scoring cosine \
    --save-per-question --skip-iti \
    --out "$CANONICAL_DIR/main_cured_old_8b_truthfulqa_n500.json" \
    2>&1 | tee "logs/main_cured_old_8b_${TS}.log"

log "Phase 4 complete."
log "Next: python compute_final_stats.py --results-dir $CANONICAL_DIR"
