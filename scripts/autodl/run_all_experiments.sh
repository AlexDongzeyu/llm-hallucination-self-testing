#!/usr/bin/env bash
# ============================================================================
# run_all_experiments.sh — Master run script for CURED Phase 4 + ablations
#
# Runs in this order (each step checks prerequisites before proceeding):
#   1. Smoke test: verify Gate 2 fires after tau_kappa/tau_ECR fix
#   2. Phase 4 8B TruthfulQA (canonical result with fixed router)
#   3. 8B ablations with --save-per-question (for R²-stratified analysis)
#   4. FACTOR benchmark (news + wiki, 8B)
#   5. Semantic entropy ablation (n=50, MedHallu, 8B)
#   6. Compute final statistics + R² stratified analysis
#
# Usage:
#   bash scripts/autodl/run_all_experiments.sh [--skip-smoke] [--skip-factor]
#
# Environment overrides:
#   PYTHON_BIN=/path/to/python  (default: $ROOT/llm-env/bin/python)
#   MODEL=meta-llama/Llama-3.1-8B-Instruct
#   CANONICAL_DIR=results/CANONICAL_v2
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON_BIN:-$ROOT/llm-env/bin/python}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
CANONICAL_DIR="${CANONICAL_DIR:-results/CANONICAL_v2}"
TS="$(date +%Y%m%d_%H%M%S)"
SKIP_SMOKE=0
SKIP_FACTOR=0

for arg in "$@"; do
  case "$arg" in
    --skip-smoke)  SKIP_SMOKE=1 ;;
    --skip-factor) SKIP_FACTOR=1 ;;
  esac
done

mkdir -p logs "$CANONICAL_DIR" benchmarks

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { log "ERROR: $*"; exit 1; }

# ── Prereqs ─────────────────────────────────────────────────────────────────
[[ -f configs/router_thresholds.json ]] || die "configs/router_thresholds.json missing"
[[ -f cured.py ]] || die "cured.py missing"

KAPPA=$(python -c "import json; d=json.load(open('configs/router_thresholds.json')); print(d['tau_kappa'])")
ECR=$(python -c "import json; d=json.load(open('configs/router_thresholds.json')); print(d['tau_ECR'])")
log "Verified thresholds: tau_kappa=$KAPPA tau_ECR=$ECR"
if python -c "exit(0 if float('$KAPPA') >= 0.5 else 1)"; then
  log "  tau_kappa looks correct (>=0.5)"
else
  die "tau_kappa=$KAPPA still looks wrong (should be ~0.70). Fix router_thresholds.json first."
fi

# ── Step 1: Smoke test ───────────────────────────────────────────────────────
if [[ $SKIP_SMOKE -eq 0 ]]; then
  log "=== STEP 1: Smoke test (Gate 2 verification, n=20) ==="
  SMOKE_OUT="/tmp/gate_test_${TS}.json"
  "$PYTHON" -u cured.py \
    --model "$MODEL" --load-in-4bit \
    --protocols cured --router new \
    --router-config configs/router_thresholds.json \
    --benchmark truthfulqa --n 20 \
    --save-per-question \
    --out "$SMOKE_OUT" \
    > "logs/smoke_test_${TS}.log" 2>&1 || die "Smoke test failed. Check logs/smoke_test_${TS}.log"

  python -c "
import json, sys
d = json.load(open('$SMOKE_OUT'))
# Find routing distribution anywhere in the JSON
def find_routing(obj):
    if isinstance(obj, dict):
        if 'routing' in obj:
            return obj['routing']
        for v in obj.values():
            r = find_routing(v)
            if r: return r
    return None
routing = find_routing(d)
print('Routing distribution:', routing)
all_greedy5 = routing and all(k == 'greedy_gate5' for k in routing)
if all_greedy5:
    print('WARNING: All questions still routed to greedy_gate5 — Gate 2 fix may not have taken effect!')
    sys.exit(1)
else:
    print('Gate 2 is firing correctly.')
"
  log "Smoke test passed."
else
  log "Skipping smoke test (--skip-smoke)"
fi

# ── Step 2: Phase 4 — 8B TruthfulQA canonical result ─────────────────────────
log "=== STEP 2: Phase 4 — 8B TruthfulQA (n=500, fixed router) ==="
PHASE4_OUT="$CANONICAL_DIR/main_cured_8b_truthfulqa_n500_v2.json"
log "  Running Phase 4 -> $PHASE4_OUT  (expected ~1.5 hrs on A100)"
"$PYTHON" -u cured.py \
  --model "$MODEL" --load-in-4bit \
  --protocols cured --router new \
  --router-config configs/router_thresholds.json \
  --benchmark truthfulqa --n 500 --seed 42 --no-shuffle \
  --scoring cosine --save-per-question --skip-iti \
  --out "$PHASE4_OUT" \
  > "logs/phase4_8b_truthfulqa_${TS}.log" 2>&1 || die "Phase 4 failed. Check logs/phase4_8b_truthfulqa_${TS}.log"
log "Phase 4 8B TruthfulQA done -> $PHASE4_OUT"

# ── Step 3: 8B Ablations with --save-per-question ─────────────────────────────
# NOTE: Files saved WITHOUT _v2 suffix so r2_stratified_alta_analysis() finds them.
# These overwrite old ablation files (intentional: old ones may lack r2_q in per_question).
log "=== STEP 3: 8B ablations with --save-per-question (n=200) ==="

ALTA_ABL="$CANONICAL_DIR/ablation_8b_alta_truthfulqa_n200.json"
log "  Running ALTA ablation -> $ALTA_ABL"
"$PYTHON" -u cured.py \
  --model "$MODEL" --load-in-4bit \
  --model-params-b 8.0 \
  --protocols alta --benchmark truthfulqa \
  --n 200 --seed 42 --no-shuffle --scoring cosine \
  --save-per-question --skip-iti \
  --out "$ALTA_ABL" \
  > "logs/ablation_8b_alta_${TS}.log" 2>&1 || die "ALTA ablation failed. Check logs/ablation_8b_alta_${TS}.log"
log "  ALTA ablation done."

GREEDY_ABL="$CANONICAL_DIR/ablation_8b_greedy_truthfulqa_n200.json"
log "  Running Greedy ablation -> $GREEDY_ABL"
"$PYTHON" -u cured.py \
  --model "$MODEL" --load-in-4bit \
  --model-params-b 8.0 \
  --protocols greedy --benchmark truthfulqa \
  --n 200 --seed 42 --no-shuffle --scoring cosine \
  --save-per-question --skip-iti \
  --out "$GREEDY_ABL" \
  > "logs/ablation_8b_greedy_${TS}.log" 2>&1 || die "Greedy ablation failed. Check logs/ablation_8b_greedy_${TS}.log"
log "  Greedy ablation done."

# ── Step 4: FACTOR benchmark ─────────────────────────────────────────────────
if [[ $SKIP_FACTOR -eq 0 ]]; then
  log "=== STEP 4: FACTOR benchmark ==="

  # Prep CSVs if not already present
  if [[ ! -f benchmarks/factor_news_n200.csv || ! -f benchmarks/factor_wiki_n200.csv ]]; then
    log "  Downloading FACTOR data..."
    "$PYTHON" scripts/prep_factor_benchmark.py || die "FACTOR prep failed."
  else
    log "  FACTOR CSVs already present, skipping prep."
  fi

  for SUBSET in news wiki; do
    CSV="benchmarks/factor_${SUBSET}_n200.csv"
    [[ -f "$CSV" ]] || { log "  WARN: $CSV missing, skipping ${SUBSET}"; continue; }
    OUT="$CANONICAL_DIR/results_8b_factor_${SUBSET}_n200.json"
    log "  Running FACTOR ${SUBSET} -> $OUT"
    "$PYTHON" -u cured.py \
      --model "$MODEL" --load-in-4bit --skip-iti \
      --protocols greedy,alta,cured \
      --router new --router-config configs/router_thresholds.json \
      --benchmark custom \
      --custom-csv "$CSV" \
      --question-col question --answer-col answer \
      --n 200 --seed 42 --no-shuffle \
      --scoring letter --max-new-tokens 5 \
      --save-per-question \
      --out "$OUT" \
      > "logs/8b_factor_${SUBSET}_${TS}.log" 2>&1 || { log "  WARN: FACTOR ${SUBSET} failed. Check logs/8b_factor_${SUBSET}_${TS}.log"; continue; }
    log "  Done: factor_${SUBSET}"
  done
else
  log "Skipping FACTOR (--skip-factor)"
fi

# ── Step 5: Semantic entropy ablation ────────────────────────────────────────
log "=== STEP 5: Semantic entropy ablation (MedHallu, n=50, k=5) ==="
SE_OUT="$CANONICAL_DIR/semantic_entropy_gate_comparison.json"
"$PYTHON" -u experiments/run_semantic_entropy_ablation.py \
  --model "$MODEL" --load-in-4bit \
  --benchmark medhallu \
  --n 50 --k 5 --seed 42 \
  --out "$SE_OUT" \
  > "logs/semantic_entropy_${TS}.log" 2>&1 || log "  WARN: Semantic entropy ablation failed. Check logs/semantic_entropy_${TS}.log"
log "  Semantic entropy ablation done -> $SE_OUT"

# ── Step 6: Final statistics ──────────────────────────────────────────────────
log "=== STEP 6: Computing final statistics + R² stratified analysis ==="
"$PYTHON" compute_final_stats.py \
  --results-dir "$CANONICAL_DIR" \
  --output "$CANONICAL_DIR/statistics_table.json" \
  > "logs/final_stats_${TS}.log" 2>&1 || log "  WARN: compute_final_stats failed. Check logs/final_stats_${TS}.log"
log "Statistics saved -> $CANONICAL_DIR/statistics_table.json"

log ""
log "=== ALL RUNS COMPLETE ==="
log "Key outputs:"
log "  Phase 4 result:       $PHASE4_OUT"
log "  ALTA ablation:        $ALTA_ABL"
log "  Greedy ablation:      $GREEDY_ABL"
log "  FACTOR news:          $CANONICAL_DIR/results_8b_factor_news_n200.json"
log "  FACTOR wiki:          $CANONICAL_DIR/results_8b_factor_wiki_n200.json"
log "  Semantic entropy:     $SE_OUT"
log "  Statistics table:     $CANONICAL_DIR/statistics_table.json"
log "  R² stratified:        $CANONICAL_DIR/r2_stratified_analysis.json"
