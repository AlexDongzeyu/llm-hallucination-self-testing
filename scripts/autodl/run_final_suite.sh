#!/usr/bin/env bash
# =============================================================================
# run_final_suite.sh - Complete final evaluation suite
# Expected runtime: ~14 hours on A100 80GB
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON_BIN:-$ROOT/llm-env/bin/python}"
MODEL_8B="meta-llama/Llama-3.1-8B-Instruct"
MODEL_3B="meta-llama/Llama-3.2-3B-Instruct"
CANONICAL_DIR="results/CANONICAL_v2"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs "$CANONICAL_DIR"

if [ ! -x "$PYTHON" ]; then
  echo "[ERROR] Python executable not found: $PYTHON" >&2
  echo "[HINT] Set PYTHON_BIN or run scripts/autodl/bootstrap_gpu_env.sh first." >&2
  exit 1
fi

log() { echo "[$(date +'%H:%M:%S')] $*"; }

# =============================================================================
# JOB 1: TruthfulQA FULL - n=817, MC scoring (THE MOST CRITICAL JOB)
# =============================================================================
log "JOB 1: TruthfulQA FULL 817 - MC1/MC2 scoring"
"$PYTHON" -u cured.py \
  --model "$MODEL_8B" --load-in-4bit --skip-iti \
  --protocols greedy,alta,cove,cured \
  --benchmark truthfulqa \
  --n 817 \
  --scoring mc \
  --max-new-tokens 50 \
  --out "$CANONICAL_DIR/results_8b_truthfulqa_full_mc.json" \
  > "logs/8b_truthfulqa_full_mc_${TS}.log" 2>&1
log "JOB 1 DONE"

# =============================================================================
# JOB 2: TruthfulQA FULL - 3B model for comparison
# =============================================================================
log "JOB 2: TruthfulQA FULL 817 - 3B model"
"$PYTHON" -u cured.py \
  --model "$MODEL_3B" --skip-iti \
  --protocols greedy,alta,delta_dola,cove,cured \
  --benchmark truthfulqa \
  --n 817 \
  --scoring mc \
  --max-new-tokens 50 \
  --out "$CANONICAL_DIR/results_3b_truthfulqa_full_mc.json" \
  > "logs/3b_truthfulqa_full_mc_${TS}.log" 2>&1
log "JOB 2 DONE"

# =============================================================================
# JOB 3: MedHallu v2 - 8B, n=100, cosine scoring
# =============================================================================
log "JOB 3: MedHallu 8B n=100"
"$PYTHON" -u cured.py \
  --model "$MODEL_8B" --load-in-4bit --skip-iti \
  --protocols greedy,alta,cove,cured \
  --benchmark custom \
  --custom-csv benchmarks/medhallu_n200.csv \
  --n 100 --scoring cosine --max-new-tokens 80 \
  --out "$CANONICAL_DIR/results_8b_medhallu_v2.json" \
  > "logs/8b_medhallu_v2_${TS}.log" 2>&1
log "JOB 3 DONE"

# =============================================================================
# JOB 4: PubMedQA v2 - 8B, n=100, yesno scoring
# =============================================================================
log "JOB 4: PubMedQA 8B n=100"
"$PYTHON" -u cured.py \
  --model "$MODEL_8B" --load-in-4bit --skip-iti \
  --protocols greedy,alta,cove,cured \
  --benchmark custom \
  --custom-csv benchmarks/pubmedqa_n200.csv \
  --n 100 --scoring yesno --max-new-tokens 10 \
  --out "$CANONICAL_DIR/results_8b_pubmedqa_v2.json" \
  > "logs/8b_pubmedqa_v2_${TS}.log" 2>&1
log "JOB 4 DONE"

# =============================================================================
# JOB 5: MedQA v3 fixed - 8B, n=100, letter scoring
# =============================================================================
log "JOB 5: MedQA 8B n=100 fixed CURED"
"$PYTHON" -u cured.py \
  --model "$MODEL_8B" --load-in-4bit --skip-iti \
  --protocols greedy,alta,cove,cured \
  --benchmark custom \
  --custom-csv benchmarks/medqa_usmle_n200.csv \
  --n 100 --scoring letter --max-new-tokens 40 \
  --out "$CANONICAL_DIR/results_8b_medqa_v3_fixed.json" \
  > "logs/8b_medqa_v3_fixed_${TS}.log" 2>&1
log "JOB 5 DONE"

# =============================================================================
# JOB 6: MedHallu built-in - 3B, n=100
# =============================================================================
log "JOB 6: MedHallu 3B built-in n=100"
"$PYTHON" -u cured.py \
  --model "$MODEL_3B" --skip-iti \
  --protocols greedy,cove,cured \
  --benchmark medhallu \
  --n 100 \
  --scoring cosine --max-new-tokens 80 \
  --out "$CANONICAL_DIR/results_3b_medhallu_n100.json" \
  > "logs/3b_medhallu_n100_${TS}.log" 2>&1
log "JOB 6 DONE"

log "ALL JOBS COMPLETE. Check results/ directory."
