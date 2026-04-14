#!/usr/bin/env bash
set -euo pipefail

# Run this on the GPU host after run_final_suite.sh has started.
# It waits for the suite to finish, then runs the MC v2 reruns and both_n100 v2.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PY="llm-env/bin/python"
LOCK_FILE="/tmp/queue_after_triviaqa_v2.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[queue] already running: $(date -Is)"
  exit 0
fi

TS="$(date +%Y%m%d_%H%M%S)"
QLOG="logs/queue_after_triviaqa_v2_${TS}.log"

log() {
  printf '[queue] %s\n' "$*" | tee -a "$QLOG"
}

log "start $(date -Is)"

while pgrep -f 'bash scripts/autodl/run_final_suite.sh' >/dev/null; do
  log "waiting final suite $(date -Is)"
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits >> "$QLOG" 2>&1 || true
  sleep 300
done

log "final suite done $(date -Is)"
mkdir -p results/archive
[ -f results/CANONICAL_v2/results_8b_truthfulqa_full_mc.json ] && mv -f results/CANONICAL_v2/results_8b_truthfulqa_full_mc.json results/archive/
[ -f results/CANONICAL_v2/results_3b_truthfulqa_full_mc.json ] && mv -f results/CANONICAL_v2/results_3b_truthfulqa_full_mc.json results/archive/

RUNTS="$(date +%Y%m%d_%H%M%S)"
LOG8="logs/8b_tqa_mc_v2_${RUNTS}.log"
LOG3="logs/3b_tqa_mc_v2_${RUNTS}.log"

log "launch MC v2 $(date -Is)"
"$PY" -u cured.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --protocols greedy,alta,cove,cured \
  --skip-iti \
  --benchmark truthfulqa \
  --n 817 \
  --scoring mc \
  --max-new-tokens 50 \
  --force-recalibrate \
  --out results/CANONICAL_v2/results_8b_truthfulqa_full_mc_v2.json \
  > "$LOG8" 2>&1 & PID8=$!

"$PY" -u cured.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --protocols greedy,alta,delta_dola,cove,cured \
  --skip-iti \
  --benchmark truthfulqa \
  --n 817 \
  --scoring mc \
  --max-new-tokens 50 \
  --force-recalibrate \
  --out results/CANONICAL_v2/results_3b_truthfulqa_full_mc_v2.json \
  > "$LOG3" 2>&1 & PID3=$!

log "pids $PID8 $PID3"

wait "$PID8"; RC8=$?
wait "$PID3"; RC3=$?
log "MC done rc8=$RC8 rc3=$RC3"

if [ "$RC8" -eq 0 ] && [ "$RC3" -eq 0 ]; then
  LOGB="logs/8b_both_n100_v2_${RUNTS}.log"
  log "launch both_n100_v2 $(date -Is)"
  "$PY" -u cured.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --protocols greedy,alta,cove,cured \
    --skip-iti \
    --benchmark both \
    --n 100 \
    --scoring cosine \
    --max-new-tokens 80 \
    --force-recalibrate \
    --out results/CANONICAL_v2/results_8b_both_n100_v2.json \
    > "$LOGB" 2>&1
  log "both_n100_v2 done rc=$?"
else
  log "skip both_n100_v2 due to MC failure"
fi

log "end $(date -Is)"
