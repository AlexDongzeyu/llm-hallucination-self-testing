#!/usr/bin/env bash
# Run FACTOR (news + wiki) benchmark on 8B model with greedy/alta/cured protocols.
# Prerequisite: python scripts/prep_factor_benchmark.py
#
# Usage (on A100):
#   bash scripts/autodl/run_factor_suite.sh
#
# Expected runtime: ~45 min per subset (letter scoring, max-new-tokens=5).
#
# GPU: keep one ``cured.py`` at a time on a single GPU; do not run news+wiki in parallel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON_BIN:-$ROOT/llm-env/bin/python}"
CANONICAL_DIR="results/CANONICAL_v2"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p logs "$CANONICAL_DIR"

for SUBSET in news wiki; do
    CSV="benchmarks/factor_${SUBSET}_n200.csv"
    if [[ ! -f "$CSV" ]]; then
        echo "[$(date)] ERROR: $CSV not found. Run: python scripts/prep_factor_benchmark.py"
        exit 1
    fi

    OUT="$CANONICAL_DIR/results_8b_factor_${SUBSET}_n200.json"
    LOG="logs/8b_factor_${SUBSET}_${TS}.log"

    echo "[$(date)] Running FACTOR ${SUBSET} -> $OUT"
    "$PYTHON" -u cured.py \
        --model "$MODEL" \
        --load-in-4bit \
        --skip-iti \
        --protocols greedy,alta,cured \
        --router new \
        --router-config configs/router_thresholds.json \
        --benchmark custom \
        --custom-csv "$CSV" \
        --question-col question \
        --answer-col answer \
        --n 200 \
        --seed 42 \
        --no-shuffle \
        --scoring letter \
        --max-new-tokens 5 \
        --save-per-question \
        --out "$OUT" \
        > "$LOG" 2>&1
    echo "[$(date)] Done: factor_${SUBSET} -> $OUT"
done

echo "[$(date)] All FACTOR runs complete."
