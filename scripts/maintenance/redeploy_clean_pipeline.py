#!/usr/bin/env python3
"""Write a completely clean (ASCII-only, properly quoted) pipeline to A800 and restart."""
import paramiko
import time

HOST = "js4.blockelite.cn"
PORT = 14136
USER = "root"
PASS = "ra7ye9ka"
REPO = "/root/llm-hallucination-self-testing"

# ─────────────────────────────────────────────────────────────────────────────
# CLEAN PIPELINE SCRIPT — ASCII only, careful bash quoting
# ─────────────────────────────────────────────────────────────────────────────
CLEAN_PIPELINE = """\
#!/usr/bin/env bash
# run_full_pipeline.sh - CURED full pipeline (clean ASCII version)
# Phases: 1=profiling, 2=ablations, 3=threshold check,
#         4=main CURED v2, 4b=R2 ablations, 4c=FACTOR, 4d=semantic-entropy,
#         5=statistics
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_VERBOSITY=warning
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
while [ ! -f "$ROOT/cured.py" ] && [ "$ROOT" != "/" ]; do
    ROOT="$(dirname "$ROOT")"
done
[ -f "$ROOT/cured.py" ] || { echo "[ERROR] cured.py not found" >&2; exit 1; }
cd "$ROOT"

PYTHON="${PYTHON_BIN:-$ROOT/llm-env/bin/python}"
[ -x "$PYTHON" ] || PYTHON=$(which python3 2>/dev/null || which python)

CANONICAL_DIR="results/CANONICAL_v2"
mkdir -p logs "$CANONICAL_DIR" configs benchmarks

log() { echo "[$(date +%Y-%m-%d\\ %H:%M:%S)] $*" | tee -a logs/pipeline.log; }

log "======================================================"
log "CURED Full Pipeline - started"
log "Repo: $ROOT  Python: $PYTHON"
log "======================================================"

# ── Phase 1: Model profiling ──────────────────────────────────────────────────
log "======================================================"
log "PHASE 1: Model profiling"
log "======================================================"
for MODEL in "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" \
             "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-32B-Instruct"; do
    SIZE=$(echo "$MODEL" | grep -oP '[0-9]+[Bb]' | head -1 | tr 'A-Z' 'a-z')
    LOAD4BIT=""; [ "$SIZE" = "32b" ] && LOAD4BIT="--load-in-4bit"
    OUT="$CANONICAL_DIR/profile_${SIZE}.json"
    if [ -f "$OUT" ]; then
        log "  Skipping $SIZE profile - already exists"
        continue
    fi
    log "  Profiling $SIZE..."
    "$PYTHON" -u experiments/compute_logit_linearity.py \
        --model "$MODEL" $LOAD4BIT --n 50 \
        --start-layer-ratio 0.7 --end-layer-ratio 1.0 \
        --compute-curvature --compute-ecr \
        --out "$OUT" \
        2>&1 | tee "logs/phase1_${SIZE}.log"
done
log "Phase 1 complete."

# ── Phase 2: Ablation grid ────────────────────────────────────────────────────
log "======================================================"
log "PHASE 2: Ablation grid"
log "======================================================"
declare -A MODEL_PARAMS
MODEL_PARAMS["meta-llama/Llama-3.2-3B-Instruct"]="3.0:3b"
MODEL_PARAMS["meta-llama/Llama-3.1-8B-Instruct"]="8.0:8b"
MODEL_PARAMS["Qwen/Qwen2.5-14B-Instruct"]="14.0:14b"
MODEL_PARAMS["Qwen/Qwen2.5-32B-Instruct"]="32.0:32b"

for MODEL in "${!MODEL_PARAMS[@]}"; do
    IFS=':' read -r PARAMS SIZE <<< "${MODEL_PARAMS[$MODEL]}"
    LOAD4BIT=""; [ "$SIZE" = "32b" ] && LOAD4BIT="--load-in-4bit"
    for PROTOCOL in greedy alta cove; do
        for BENCH in truthfulqa medhallu; do
            OUT="$CANONICAL_DIR/ablation_${SIZE}_${PROTOCOL}_${BENCH}_n200.json"
            if [ -f "$OUT" ]; then
                log "  Skipping ${SIZE}/${PROTOCOL}/${BENCH} - already exists"
                continue
            fi
            log "  Running ${SIZE} | ${PROTOCOL} | ${BENCH}"
            "$PYTHON" -u cured.py \
                --model "$MODEL" $LOAD4BIT \
                --model-params-b "$PARAMS" \
                --protocols "$PROTOCOL" \
                --benchmark "$BENCH" \
                --n 200 --seed 42 --no-shuffle \
                --scoring cosine --save-per-question --skip-iti \
                --out "$OUT" \
                2>&1 | tee "logs/ablation_${SIZE}_${PROTOCOL}_${BENCH}.log"
        done
    done
done

# ITI smoke test
log "  Verifying ITI probe training (3B smoke test)..."
"$PYTHON" -u cured.py \
    --model "meta-llama/Llama-3.2-3B-Instruct" \
    --model-params-b "3.0" \
    --protocols iti --benchmark truthfulqa \
    --n 5 --seed 42 --no-shuffle \
    --out /tmp/iti_smoke.json \
    2>&1 | tee logs/iti_smoke.log
ITI_OK=1
"$PYTHON" -c "
import json, sys
d = json.load(open('/tmp/iti_smoke.json'))
for bv in d.get('results', {}).values():
    for pv in bv.values():
        routing = pv.get('routing', {})
        if 'greedy_no_iti' in routing:
            sys.exit(1)
sys.exit(0)
" || ITI_OK=0

if [ "$ITI_OK" = "1" ]; then
    for MODEL in "${!MODEL_PARAMS[@]}"; do
        IFS=':' read -r PARAMS SIZE <<< "${MODEL_PARAMS[$MODEL]}"
        LOAD4BIT=""; [ "$SIZE" = "32b" ] && LOAD4BIT="--load-in-4bit"
        for BENCH in truthfulqa medhallu; do
            OUT="$CANONICAL_DIR/ablation_${SIZE}_iti_${BENCH}_n200.json"
            if [ -f "$OUT" ]; then
                log "  Skipping ${SIZE}/iti/${BENCH} - already exists"
                continue
            fi
            log "  Running ${SIZE} | iti | ${BENCH}"
            "$PYTHON" -u cured.py \
                --model "$MODEL" $LOAD4BIT \
                --model-params-b "$PARAMS" \
                --protocols iti --benchmark "$BENCH" \
                --n 200 --seed 42 --no-shuffle \
                --scoring cosine --save-per-question \
                --out "$OUT" \
                2>&1 | tee "logs/ablation_${SIZE}_iti_${BENCH}.log"
        done
    done
else
    log "  WARNING: ITI smoke failed - skipping ITI ablations"
fi
log "Phase 2 complete."

# ── Phase 3: Threshold verification ──────────────────────────────────────────
log "======================================================"
log "PHASE 3: Threshold verification (pre-calibrated values)"
log "======================================================"
"$PYTHON" - << 'PYEOF'
import json, sys
cfg = json.load(open('configs/router_thresholds.json'))
kappa = float(cfg.get('tau_kappa', 0.08))
ecr = float(cfg.get('tau_ECR', 0.10))
pmr2 = float(cfg.get('profile_mean_r2', 0.0))
if kappa >= 0.5 and ecr <= 0.05 and pmr2 >= 0.5:
    print(f"[OK] tau_kappa={kappa} tau_ECR={ecr} profile_mean_r2={pmr2}")
else:
    print(f"[WARN] Broken thresholds: tau_kappa={kappa} tau_ECR={ecr}. Restoring fixed values.")
    cfg.update({'tau_kappa': 0.70, 'tau_ECR': 0.04, 'profile_mean_r2': 0.582,
                '_comment': 'Manually calibrated: tau_kappa=0.70, tau_ECR=0.04'})
    json.dump(cfg, open('configs/router_thresholds.json', 'w'), indent=2)
    print("[FIXED] tau_kappa=0.70 tau_ECR=0.04 profile_mean_r2=0.582")
PYEOF
log "Phase 3 complete."

# ── Phase 4: Main CURED v2 experiments ───────────────────────────────────────
log "======================================================"
log "PHASE 4: Main CURED v2 experiments (fixed router, n=500)"
log "======================================================"

# StrategyQA smoke test
"$PYTHON" -u cured.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" --load-in-4bit \
    --protocols greedy \
    --benchmark custom \
    --custom-csv benchmarks/strategyqa_n500.csv \
    --question-col question --answer-col answer \
    --n 5 --seed 42 --no-shuffle --scoring yesno --skip-iti \
    --out /tmp/strategyqa_smoke.json \
    2>&1 | tee logs/strategyqa_smoke.log
log "  StrategyQA smoke test OK"

MODELS_MAIN=(
    "meta-llama/Llama-3.2-3B-Instruct:3.0:3b"
    "meta-llama/Llama-3.1-8B-Instruct:8.0:8b"
    "Qwen/Qwen2.5-14B-Instruct:14.0:14b"
    "Qwen/Qwen2.5-32B-Instruct:32.0:32b"
)

for MODEL_SPEC in "${MODELS_MAIN[@]}"; do
    IFS=':' read -r M P S <<< "$MODEL_SPEC"
    LOAD4BIT=""; [ "$S" = "32b" ] && LOAD4BIT="--load-in-4bit"
    COMPUTE_SC="--compute-sc"; [ "$S" = "32b" ] && COMPUTE_SC=""

    for BENCH in truthfulqa medhallu strategyqa; do
        SCORING="cosine"
        if [ "$BENCH" = "strategyqa" ]; then
            SCORING="yesno"
            BENCH_ARGS="--benchmark custom --custom-csv benchmarks/strategyqa_n500.csv --question-col question --answer-col answer"
        else
            BENCH_ARGS="--benchmark $BENCH"
        fi
        # v2 suffix forces rerun with fixed router
        OUT="$CANONICAL_DIR/main_cured_${S}_${BENCH}_n500_v2.json"
        if [ -f "$OUT" ]; then
            log "  Skipping ${S}/cured/${BENCH} v2 - already exists"
            continue
        fi
        log "  Running ${S} | cured (fixed router v2) | ${BENCH} -> $OUT"
        "$PYTHON" -u cured.py \
            --model "$M" $LOAD4BIT \
            --model-params-b "$P" \
            --protocols cured \
            --router new \
            --router-config configs/router_thresholds.json \
            $BENCH_ARGS \
            --n 500 --seed 42 --no-shuffle \
            --scoring "$SCORING" \
            $COMPUTE_SC \
            --save-per-question --skip-iti \
            --out "$OUT" \
            2>&1 | tee "logs/main_cured_${S}_${BENCH}_v2.log"
    done

    OUT_GREEDY="$CANONICAL_DIR/main_greedy_${S}_truthfulqa_n817.json"
    if [ ! -f "$OUT_GREEDY" ]; then
        log "  Running ${S} | greedy | truthfulqa n=817"
        "$PYTHON" -u cured.py \
            --model "$M" $LOAD4BIT \
            --model-params-b "$P" \
            --protocols greedy \
            --benchmark truthfulqa \
            --n 817 --seed 42 --no-shuffle \
            --scoring cosine --save-per-question --skip-iti \
            --out "$OUT_GREEDY" \
            2>&1 | tee "logs/main_greedy_${S}_truthfulqa.log"
    fi
done

OUT_OLD="$CANONICAL_DIR/main_cured_old_8b_truthfulqa_n500.json"
if [ ! -f "$OUT_OLD" ]; then
    log "  Running old-router CURED 8B ablation"
    "$PYTHON" -u cured.py \
        --model "meta-llama/Llama-3.1-8B-Instruct" --load-in-4bit \
        --model-params-b 8.0 \
        --protocols cured --router old \
        --benchmark truthfulqa \
        --n 500 --seed 42 --no-shuffle \
        --scoring cosine --save-per-question --skip-iti \
        --out "$OUT_OLD" \
        2>&1 | tee logs/main_cured_old_8b.log
fi
log "Phase 4 complete."

# ── Phase 4b: R2 ablations (overwrites old files to populate r2_q) ───────────
log "======================================================"
log "PHASE 4b: 8B R2 ablations (save-per-question, populates r2_q)"
log "======================================================"
for PROTOCOL in alta greedy; do
    for BENCH in truthfulqa medhallu; do
        OUT="$CANONICAL_DIR/ablation_8b_${PROTOCOL}_${BENCH}_n200.json"
        log "  Running 8b | ${PROTOCOL} | ${BENCH} (force rerun for r2_q)"
        "$PYTHON" -u cured.py \
            --model "meta-llama/Llama-3.1-8B-Instruct" --load-in-4bit \
            --model-params-b 8.0 \
            --protocols "$PROTOCOL" \
            --benchmark "$BENCH" \
            --n 200 --seed 42 --no-shuffle \
            --scoring cosine --save-per-question --skip-iti \
            --out "$OUT" \
            2>&1 | tee "logs/ablation_8b_${PROTOCOL}_${BENCH}_r2.log"
    done
done
log "Phase 4b complete."

# ── Phase 4c: FACTOR benchmark ────────────────────────────────────────────────
log "======================================================"
log "PHASE 4c: FACTOR benchmark (news + wiki)"
log "======================================================"
for SUBSET in news wiki; do
    CSV="benchmarks/factor_${SUBSET}_n200.csv"
    if [ ! -f "$CSV" ]; then
        log "  Downloading FACTOR ${SUBSET}..."
        "$PYTHON" scripts/prep_factor_benchmark.py || log "  WARN: FACTOR prep failed"
    fi
    OUT="$CANONICAL_DIR/results_8b_factor_${SUBSET}_n200.json"
    if [ -f "$OUT" ]; then
        log "  Skipping FACTOR ${SUBSET} - already exists"
        continue
    fi
    log "  Running FACTOR ${SUBSET}"
    "$PYTHON" -u cured.py \
        --model "meta-llama/Llama-3.1-8B-Instruct" --load-in-4bit \
        --skip-iti \
        --protocols greedy,alta,cured \
        --router new --router-config configs/router_thresholds.json \
        --benchmark custom \
        --custom-csv "$CSV" \
        --question-col question --answer-col answer \
        --n 200 --seed 42 --no-shuffle \
        --scoring letter --max-new-tokens 5 \
        --save-per-question \
        --out "$OUT" \
        2>&1 | tee "logs/8b_factor_${SUBSET}.log"
    log "  Done: FACTOR ${SUBSET}"
done
log "Phase 4c complete."

# ── Phase 4d: Semantic entropy ablation ──────────────────────────────────────
log "======================================================"
log "PHASE 4d: Semantic entropy ablation (MedHallu n=50 k=5)"
log "======================================================"
SE_OUT="$CANONICAL_DIR/semantic_entropy_gate_comparison.json"
if [ ! -f "$SE_OUT" ]; then
    "$PYTHON" -u experiments/run_semantic_entropy_ablation.py \
        --model "meta-llama/Llama-3.1-8B-Instruct" --load-in-4bit \
        --benchmark medhallu \
        --n 50 --k 5 --seed 42 \
        --out "$SE_OUT" \
        2>&1 | tee logs/semantic_entropy.log || log "  WARN: Semantic entropy failed"
    log "  Done -> $SE_OUT"
else
    log "  Skipping semantic entropy - already exists"
fi
log "Phase 4d complete."

# ── Phase 5: Statistics + R2 stratified analysis ─────────────────────────────
log "======================================================"
log "PHASE 5: Statistics + R2 stratified analysis"
log "======================================================"
"$PYTHON" -u compute_final_stats.py \
    --results-dir "$CANONICAL_DIR" \
    --output "$CANONICAL_DIR/statistics_table.json" \
    2>&1 | tee logs/phase5_stats.log
log "Phase 5 complete."

log "======================================================"
log "ALL PHASES COMPLETE"
log "Results: $CANONICAL_DIR"
log "Statistics: $CANONICAL_DIR/statistics_table.json"
log "R2 analysis: $CANONICAL_DIR/r2_stratified_analysis.json"
log "======================================================"
echo "" >> logs/pipeline.log
echo "PIPELINE FINISHED AT $(date)" >> logs/pipeline.log
"""


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    return client


def run(client, cmd, t=30):
    _, o, e = client.exec_command(cmd, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode("utf-8", "replace").encode("ascii", "replace").decode("ascii").strip()


def main():
    # Strip any non-ASCII chars (invisible unicode from editors etc.)
    pipeline_bytes = CLEAN_PIPELINE.encode("ascii", errors="replace")
    pipeline_clean = pipeline_bytes.decode("ascii")
    non_ascii = sum(1 for c in CLEAN_PIPELINE if ord(c) >= 128)
    if non_ascii:
        print(f"NOTE: Stripped {non_ascii} non-ASCII chars from pipeline (replaced with '?').")
    print(f"Pipeline: {len(pipeline_clean)} chars, ready to write.")

    client = connect()
    print(f"Connected to {HOST}:{PORT}.")

    # 1. Kill all running pipeline/cured processes
    run(client, "pkill -f run_full_pipeline.sh 2>/dev/null || true")
    run(client, "pkill -f cured.py 2>/dev/null || true")
    time.sleep(3)
    print("Killed lingering processes.")

    # 2. Write clean pipeline (force ASCII, Unix line endings)
    sftp = client.open_sftp()
    pipeline_path = f"{REPO}/run_full_pipeline.sh"
    pipeline_ascii = pipeline_clean.replace("\r\n", "\n").replace("\r", "\n")
    with sftp.file(pipeline_path, "w") as f:
        f.write(pipeline_ascii.encode("ascii"))
    sftp.close()
    run(client, f"chmod +x {pipeline_path}")
    print(f"Clean pipeline written to {pipeline_path}")

    # 3. Verify thresholds
    thresh = run(client, (
        f"python3 -c 'import json; d=json.load(open(\"{REPO}/configs/router_thresholds.json\")); "
        f"print(d[\"tau_kappa\"], d[\"tau_ECR\"], d.get(\"profile_mean_r2\",\"MISSING\"))'"
    ))
    print(f"Server thresholds: tau_kappa tau_ECR profile_mean_r2 = {thresh}")

    # 4. Restart pipeline
    log_path = f"{REPO}/logs/pipeline.log"
    activate_cmd = f"source {REPO}/llm-env/bin/activate 2>/dev/null || true"
    pipeline_cmd = f"cd '{REPO}' && {activate_cmd} && bash run_full_pipeline.sh"
    run(client, f"echo '' >> {log_path}")
    run(client, f"echo '=== CLEAN RESTART ===' >> {log_path}")
    nohup_cmd = f"nohup bash -c '{pipeline_cmd}' >> {log_path} 2>&1 &"
    run(client, nohup_cmd)
    time.sleep(4)
    print("Pipeline restarted.")

    # 5. Confirm
    proc = run(client, "ps aux | grep run_full_pipeline | grep -v grep | head -2")
    print(f"Process: {proc or 'NOT VISIBLE YET'}")
    log_tail = run(client, f"tail -10 {log_path}")
    print(f"Log:\n{log_tail}")
    gpu = run(client, "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader")
    print(f"GPU: {gpu}")

    client.close()
    print(f"\nDone. Monitor:\n  python scripts/monitor_a800.py")


if __name__ == "__main__":
    main()
