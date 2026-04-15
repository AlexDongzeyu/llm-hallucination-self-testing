#!/usr/bin/env bash
set -euo pipefail

cd ~/llm-hallucination-self-testing
mkdir -p logs results/CANONICAL_v2 results/calibration

LOCK_FILE=/tmp/queue_scale_plan.lock
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[queue] Another queue_scale_plan is already running."
  exit 0
fi

TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="logs/queue_scale_plan_${TS}.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[queue] start $(date -Is)"

PY=python3
if [ -x llm-env/bin/python ]; then
  PY=llm-env/bin/python
fi

echo "[queue] python=$PY"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-180}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export TOKENIZERS_PARALLELISM="false"
if "$PY" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('hf_transfer') else 1)"; then
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
else
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi
echo "[queue] HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER"

echo "[queue] step0 grep check"
grep -n "force.recalibrate\|ALTA_R2_CUTOFF\|r2_cutoff\|measure_r2\|calibrat" cured.py | head -40 || true
grep -n "calibrate-only\|load-calibration" cured.py | head -20 || true
grep -n "factor_news\|factor_wiki\|factor" cured.py | head -40 || true

BACKUP_FILE="cured.py.scale_queue_backup"
cp cured.py "$BACKUP_FILE"

restore_cured() {
  if [ -f "$BACKUP_FILE" ]; then
    cp "$BACKUP_FILE" cured.py
    rm -f "$BACKUP_FILE"
    echo "[queue] restored cured.py"
  fi
}
trap restore_cured EXIT

set_cutoff() {
  local cutoff="$1"
  "$PY" - "$cutoff" <<'PY'
import pathlib
import re
import sys

cutoff = sys.argv[1]
p = pathlib.Path("cured.py")
s = p.read_text(encoding="utf-8")
ns, n = re.subn(r"(?m)^ALTA_R2_CUTOFF\s*=\s*[0-9.]+\s*$", f"ALTA_R2_CUTOFF = {cutoff}", s, count=1)
if n != 1:
    raise SystemExit("failed to patch ALTA_R2_CUTOFF")
p.write_text(ns, encoding="utf-8")
print(f"patched ALTA_R2_CUTOFF={cutoff}")
PY
}

if [ -f results/CANONICAL_v2/results_3b_both_n100_v2.json ]; then
  echo "[queue] skip priority1; results_3b_both_n100_v2.json already exists"
else
  echo "[queue] run priority1 3B both n=100"
  "$PY" -u cured.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --protocols greedy,alta,delta_dola,cove,cured \
    --skip-iti \
    --benchmark both \
    --n 100 \
    --scoring cosine \
    --max-new-tokens 80 \
    --force-recalibrate \
    --out results/CANONICAL_v2/results_3b_both_n100_v2.json \
    > logs/3b_both_n100_v2.log 2>&1
  echo "[queue] priority1 done rc=$?"
fi

if grep -q "factor_news" cured.py; then
  echo "[queue] factor benchmark appears supported; running data pull + factor_news"
  "$PY" -c "from datasets import load_dataset; ds=load_dataset('EleutherAI/factor','news',split='test'); ds.to_csv('data/factor_news.csv', index=False); ds2=load_dataset('EleutherAI/factor','wiki',split='test'); ds2.to_csv('data/factor_wiki.csv', index=False); print('FACTOR downloaded', len(ds), len(ds2))" > logs/factor_download.log 2>&1 || true
  "$PY" -u cured.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --protocols greedy,cured \
    --benchmark factor_news \
    --n 200 \
    --scoring letter \
    --out results/CANONICAL_v2/results_3b_factor_news.json \
    > logs/factor_news.log 2>&1 || true
else
  echo "[queue] skip priority2; factor benchmark not supported in cured.py"
fi

run_scale() {
  local tag="$1"
  local model="$2"
  local out_json="results/CANONICAL_v2/results_${tag}_both_n100.json"
  local calib_json="results/calibration/calib_${tag}.json"

  if [ -f "$out_json" ]; then
    echo "[queue] skip ${tag}; output already exists"
    return 0
  fi

  echo "[queue] calibrate ${tag}"
  "$PY" scripts/autodl/recalibrate_scale.py \
    --model "$model" \
    --out "$calib_json" \
    --n-prompts 30 \
    --batch-size 64 \
    --load-in-4bit \
    > "logs/calib_${tag}.log" 2>&1

  local cutoff
  cutoff=$("$PY" -c "import json; print(json.load(open('$calib_json'))['r2_cutoff_used'])")
  echo "[queue] ${tag} cutoff=${cutoff}"
  set_cutoff "$cutoff"

  echo "[queue] run ${tag} both n=100"
  "$PY" -u cured.py \
    --model "$model" \
    --load-in-4bit \
    --protocols greedy,alta,cove,cured \
    --skip-iti \
    --benchmark both \
    --n 100 \
    --scoring cosine \
    --max-new-tokens 80 \
    --force-recalibrate \
    --out "$out_json" \
    > "logs/${tag}_both_n100.log" 2>&1

  echo "[queue] done ${tag} rc=$?"
}

run_scale qwen14b "Qwen/Qwen2.5-14B-Instruct"
run_scale qwen32b "Qwen/Qwen2.5-32B-Instruct"

echo "[queue] all requested tasks complete $(date -Is)"
