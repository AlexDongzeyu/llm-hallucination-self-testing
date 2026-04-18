#!/usr/bin/env python3
"""
deploy_and_queue.py — Sync code to A800, queue full CURED pipeline after current job.

1. Connects via SSH (paramiko)
2. Checks GPU / running jobs
3. Syncs changed source files via SFTP
4. Creates the full pipeline script on the server
5. Launches it in a tmux session so it survives disconnect
   - The script waits for the current Qwen job to finish first
   - Then runs: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

Usage:
    python scripts/deploy_and_queue.py
"""

import io
import os
import sys
import time
from pathlib import Path

import paramiko

# ── Server credentials ──────────────────────────────────────────────────────
HOST = "js4.blockelite.cn"
PORT = 14136
USER = "root"
PASS = "ra7ye9ka"

# ── Files to sync to server ─────────────────────────────────────────────────
LOCAL_ROOT = Path(__file__).resolve().parents[1]
# (local_path, remote_path_relative_to_repo_root)
FILES_TO_SYNC = [
    # Core scripts
    ("cured.py", "cured.py"),
    ("calibrate_router.py", "calibrate_router.py"),
    ("compute_final_stats.py", "compute_final_stats.py"),
    # Configs (with fixed tau_kappa=0.70, tau_ECR=0.04, profile_mean_r2=0.582)
    ("configs/router_thresholds.json", "configs/router_thresholds.json"),
    # Experiments
    ("experiments/compute_logit_linearity.py", "experiments/compute_logit_linearity.py"),
    ("experiments/run_semantic_entropy_ablation.py", "experiments/run_semantic_entropy_ablation.py"),
    # Benchmarks (strategyqa + FACTOR CSVs)
    ("benchmarks/strategyqa_n500.csv", "benchmarks/strategyqa_n500.csv"),
    ("benchmarks/factor_news_n200.csv", "benchmarks/factor_news_n200.csv"),
    ("benchmarks/factor_wiki_n200.csv", "benchmarks/factor_wiki_n200.csv"),
    # Scripts
    ("scripts/prep_factor_benchmark.py", "scripts/prep_factor_benchmark.py"),
    ("scripts/autodl/run_phase1_measurement.sh", "scripts/autodl/run_phase1_measurement.sh"),
    ("scripts/autodl/run_phase2_ablations.sh",   "scripts/autodl/run_phase2_ablations.sh"),
    ("scripts/autodl/run_phase4_main.sh",         "scripts/autodl/run_phase4_main.sh"),
    ("scripts/autodl/run_factor_suite.sh",        "scripts/autodl/run_factor_suite.sh"),
    ("scripts/autodl/run_all_experiments.sh",     "scripts/autodl/run_all_experiments.sh"),
]


# ── Master pipeline script ───────────────────────────────────────────────────
# NOTE: Phase 4 CURED outputs use _v2 suffix so they always rerun with the
# fixed router (tau_kappa=0.70, tau_ECR=0.04). Ablations overwrite old files
# to populate r2_q in per_question entries for R²-stratified analysis.
PIPELINE_SCRIPT = r"""#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh — Full CURED pipeline, auto-queued after current job
# Runs: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
# Started by deploy_and_queue.py inside a tmux session.
# =============================================================================
set -euo pipefail

# Hugging Face hub robustness (mirror + timeouts; safe to override HF_ENDPOINT before launch)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_VERBOSITY=warning
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DOWNLOAD_MAX_WORKERS=1

# PyTorch: reduce VRAM fragmentation on long multi-model runs (override if already set)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── find repo root ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
# Walk up to find cured.py
while [ ! -f "$ROOT/cured.py" ] && [ "$ROOT" != "/" ]; do
    ROOT="$(dirname "$ROOT")"
done
if [ ! -f "$ROOT/cured.py" ]; then
    echo "[ERROR] Cannot find repo root (cured.py not found)" >&2
    exit 1
fi
cd "$ROOT"

PYTHON="${PYTHON_BIN:-$ROOT/llm-env/bin/python}"
if [ ! -x "$PYTHON" ]; then
    # Try system python as fallback
    PYTHON=$(which python3 2>/dev/null || which python)
fi

CANONICAL_DIR="results/CANONICAL_v2"
mkdir -p logs "$CANONICAL_DIR" configs

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a logs/pipeline.log; }

log "======================================================"
log "CURED Full Pipeline — started"
log "Repo: $ROOT"
log "Python: $PYTHON"
log "======================================================"

# ── Wait for current GPU job to finish ───────────────────────────────────────
log "Waiting for current GPU job to finish..."
WAIT_COUNT=0
while nvidia-smi | grep -q "MiB /"; do
    # Check if GPU memory is nearly fully occupied (>10GB used = job still running)
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "${USED:-0}" -lt 5000 ]; then
        log "GPU memory low (${USED}MiB used) — job appears done."
        break
    fi
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $((WAIT_COUNT % 12)) -eq 0 ]; then
        log "Still waiting... GPU memory: ${USED}MiB used (check $WAIT_COUNT)"
    fi
    sleep 30
done
log "GPU is free. Starting pipeline."
sleep 5  # brief pause to let GPU memory fully release

# ── Phase 1: Model profiling ─────────────────────────────────────────────────
log "======================================================"
log "PHASE 1: Model profiling (R², κ, ECR)"
log "======================================================"

for MODEL in \
    "meta-llama/Llama-3.2-3B-Instruct" \
    "meta-llama/Llama-3.1-8B-Instruct" \
    "Qwen/Qwen2.5-14B-Instruct" \
    "Qwen/Qwen2.5-32B-Instruct"; do

    SIZE=$(echo "$MODEL" | grep -oP '\d+[Bb]' | head -1 | tr '[:upper:]' '[:lower:]')
    LOAD4BIT=""
    [ "$SIZE" = "32b" ] && LOAD4BIT="--load-in-4bit"

    OUT="$CANONICAL_DIR/profile_${SIZE}.json"
    if [ -f "$OUT" ]; then
        log "  Skipping $SIZE — $OUT already exists"
        continue
    fi

    log "  Profiling $SIZE..."
    "$PYTHON" -u experiments/compute_logit_linearity.py \
        --model "$MODEL" $LOAD4BIT \
        --n 50 \
        --start-layer-ratio 0.7 \
        --end-layer-ratio 1.0 \
        --compute-curvature \
        --compute-ecr \
        --out "$OUT" \
        2>&1 | tee "logs/phase1_${SIZE}.log"
    log "  Done: $OUT"
done
log "Phase 1 complete."

# ── Phase 2: Ablation grid ────────────────────────────────────────────────────
log "======================================================"
log "PHASE 2: Ablation grid (4 models × 4 protocols × 2 benchmarks)"
log "======================================================"

declare -A MODEL_PARAMS
MODEL_PARAMS["meta-llama/Llama-3.2-3B-Instruct"]="3.0:3b"
MODEL_PARAMS["meta-llama/Llama-3.1-8B-Instruct"]="8.0:8b"
MODEL_PARAMS["Qwen/Qwen2.5-14B-Instruct"]="14.0:14b"
MODEL_PARAMS["Qwen/Qwen2.5-32B-Instruct"]="32.0:32b"

# Loop 1: greedy / alta / cove
for MODEL in "${!MODEL_PARAMS[@]}"; do
    IFS=':' read -r PARAMS SIZE <<< "${MODEL_PARAMS[$MODEL]}"
    LOAD4BIT=""; [ "$SIZE" = "32b" ] && LOAD4BIT="--load-in-4bit"

    for PROTOCOL in greedy alta cove; do
        for BENCH in truthfulqa medhallu; do
            OUT="$CANONICAL_DIR/ablation_${SIZE}_${PROTOCOL}_${BENCH}_n200.json"
            if [ -f "$OUT" ]; then
                log "  Skipping $SIZE/$PROTOCOL/$BENCH — already exists"
                continue
            fi
            log "  $SIZE | $PROTOCOL | $BENCH → $OUT"
            "$PYTHON" -u cured.py \
                --model "$MODEL" $LOAD4BIT \
                --model-params-b "$PARAMS" \
                --protocols "$PROTOCOL" \
                --benchmark "$BENCH" \
                --n 200 \
                --seed 42 --no-shuffle \
                --scoring cosine \
                --save-per-question \
                --skip-iti \
                --out "$OUT" \
                2>&1 | tee "logs/ablation_${SIZE}_${PROTOCOL}_${BENCH}.log"
        done
    done
done

# Loop 2: ITI — probe training required
log "  Verifying ITI probe training (3B smoke test)..."
"$PYTHON" -u cured.py \
    --model "meta-llama/Llama-3.2-3B-Instruct" \
    --model-params-b "3.0" \
    --protocols iti \
    --benchmark truthfulqa \
    --n 5 \
    --seed 42 --no-shuffle \
    --out /tmp/iti_smoke.json \
    2>&1 | tee logs/iti_smoke.log

ITI_OK=0
"$PYTHON" -c "
import json, sys
d = json.load(open('/tmp/iti_smoke.json'))
routing = d.get('results', {}).get('truthfulqa', {}).get('iti', {}).get('routing', {})
if 'greedy_no_iti' in routing:
    sys.exit(1)
sys.exit(0)
" && ITI_OK=1

if [ $ITI_OK -eq 1 ]; then
    for MODEL in "${!MODEL_PARAMS[@]}"; do
        IFS=':' read -r PARAMS SIZE <<< "${MODEL_PARAMS[$MODEL]}"
        LOAD4BIT=""; [ "$SIZE" = "32b" ] && LOAD4BIT="--load-in-4bit"

        for BENCH in truthfulqa medhallu; do
            OUT="$CANONICAL_DIR/ablation_${SIZE}_iti_${BENCH}_n200.json"
            if [ -f "$OUT" ]; then
                log "  Skipping $SIZE/iti/$BENCH — already exists"
                continue
            fi
            log "  $SIZE | iti | $BENCH → $OUT"
            "$PYTHON" -u cured.py \
                --model "$MODEL" $LOAD4BIT \
                --model-params-b "$PARAMS" \
                --protocols iti \
                --benchmark "$BENCH" \
                --n 200 \
                --seed 42 --no-shuffle \
                --scoring cosine \
                --save-per-question \
                --out "$OUT" \
                2>&1 | tee "logs/ablation_${SIZE}_iti_${BENCH}.log"
        done
    done
else
    log "  WARNING: ITI smoke test failed — skipping ITI ablations"
fi
log "Phase 2 complete."

# ── Phase 3: Threshold calibration ───────────────────────────────────────────
log "======================================================"
log "PHASE 3: Router threshold calibration"
log "======================================================"
"$PYTHON" -u calibrate_router.py \
    --results-dir "$CANONICAL_DIR" \
    --pattern "ablation_*.json" \
    --out configs/router_thresholds.json \
    2>&1 | tee logs/phase3_calibrate.log
log "Phase 3 complete. Thresholds saved to configs/router_thresholds.json"

# ── Phase 4: Main publication experiments ─────────────────────────────────────
log "======================================================"
log "PHASE 4: Main experiments (n=500, new router, 4 scales × 3 benchmarks)"
log "======================================================"

# Verify StrategyQA CSV
log "  Smoke-testing StrategyQA..."
"$PYTHON" -u cured.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" --load-in-4bit \
    --protocols greedy \
    --benchmark custom \
    --custom-csv benchmarks/strategyqa_n500.csv \
    --question-col question --answer-col answer \
    --n 5 --seed 42 --no-shuffle --scoring yesno --skip-iti \
    --out /tmp/strategyqa_smoke.json \
    2>&1 | tee logs/strategyqa_smoke.log
log "  StrategyQA OK"

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
            BENCH_ARGS=(--benchmark custom --custom-csv "benchmarks/strategyqa_n500.csv" --question-col question --answer-col answer)
        else
            BENCH_ARGS=(--benchmark "$BENCH")
        fi
        OUT="$CANONICAL_DIR/main_cured_${S}_${BENCH}_n500.json"
        if [ -f "$OUT" ]; then
            log "  Skipping $S/cured/$BENCH — already exists"
            continue
        fi
        log "  $S | cured (new router) | $BENCH → $OUT"
        # Phase 4 CURED: use _v2 suffix so rerun always happens with fixed router
        OUT="$CANONICAL_DIR/main_cured_${S}_${BENCH}_n500_v2.json"
        if [ -f "$OUT" ]; then
            log "  Skipping $S/cured/$BENCH v2 — already exists"
            continue
        fi
        log "  $S | cured (fixed router v2) | $BENCH → $OUT"
        "$PYTHON" -u cured.py \
            --model "$M" $LOAD4BIT \
            --model-params-b "$P" \
            --protocols cured \
            --router new \
            --router-config configs/router_thresholds.json \
            "${BENCH_ARGS[@]}" \
            --n 500 \
            --seed 42 --no-shuffle \
            --scoring "$SCORING" \
            $COMPUTE_SC \
            --save-per-question \
            --skip-iti \
            --out "$OUT" \
            2>&1 | tee "logs/main_cured_${S}_${BENCH}_v2.log"
    done

    # Greedy baseline (full 817 TruthfulQA) — skip if already exists
    OUT_GREEDY="$CANONICAL_DIR/main_greedy_${S}_truthfulqa_n817.json"
    if [ ! -f "$OUT_GREEDY" ]; then
        log "  $S | greedy | truthfulqa n=817 → $OUT_GREEDY"
        "$PYTHON" -u cured.py \
            --model "$M" $LOAD4BIT \
            --model-params-b "$P" \
            --protocols greedy \
            --benchmark truthfulqa \
            --n 817 \
            --seed 42 --no-shuffle \
            --scoring cosine \
            --save-per-question --skip-iti \
            --out "$OUT_GREEDY" \
            2>&1 | tee "logs/main_greedy_${S}_truthfulqa.log"
    fi
done

# Old-router CURED ablation comparison (8B) — skip if already exists
OUT_OLD="$CANONICAL_DIR/main_cured_old_8b_truthfulqa_n500.json"
if [ ! -f "$OUT_OLD" ]; then
    log "  Old-router CURED ablation (8B)..."
    "$PYTHON" -u cured.py \
        --model "meta-llama/Llama-3.1-8B-Instruct" --load-in-4bit \
        --model-params-b 8.0 \
        --protocols cured --router old \
        --benchmark truthfulqa \
        --n 500 --seed 42 --no-shuffle \
        --scoring cosine \
        --save-per-question --skip-iti \
        --out "$OUT_OLD" \
        2>&1 | tee logs/main_cured_old_8b.log
fi
log "Phase 4 (fixed router v2) complete."

# ── Phase 4b: R² ablations — 8B with --save-per-question (populates r2_q) ────
log "======================================================"
log "PHASE 4b: 8B R²-stratified ablations (--save-per-question, overwrites old)"
log "======================================================"
# Always overwrite: old files have r2_q=None; new cured.py computes them.
for PROTOCOL in alta greedy; do
    for BENCH in truthfulqa medhallu; do
        OUT="$CANONICAL_DIR/ablation_8b_${PROTOCOL}_${BENCH}_n200.json"
        log "  8b | $PROTOCOL | $BENCH (force rerun for r2_q) → $OUT"
        "$PYTHON" -u cured.py \
            --model "meta-llama/Llama-3.1-8B-Instruct" --load-in-4bit \
            --model-params-b 8.0 \
            --protocols "$PROTOCOL" \
            --benchmark "$BENCH" \
            --n 200 --seed 42 --no-shuffle \
            --scoring cosine \
            --save-per-question --skip-iti \
            --out "$OUT" \
            2>&1 | tee "logs/ablation_8b_${PROTOCOL}_${BENCH}_r2.log"
    done
done
log "Phase 4b complete."

# ── Phase 4c: FACTOR benchmark ────────────────────────────────────────────────
log "======================================================"
log "PHASE 4c: FACTOR benchmark (news + wiki, 8B)"
log "======================================================"
for SUBSET in news wiki; do
    CSV="benchmarks/factor_${SUBSET}_n200.csv"
    if [ ! -f "$CSV" ]; then
        log "  Downloading FACTOR $SUBSET..."
        "$PYTHON" scripts/prep_factor_benchmark.py || log "  WARN: FACTOR prep failed"
    fi
    OUT="$CANONICAL_DIR/results_8b_factor_${SUBSET}_n200.json"
    if [ -f "$OUT" ]; then
        log "  Skipping FACTOR $SUBSET — already exists"
        continue
    fi
    log "  FACTOR $SUBSET → $OUT"
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
    log "  Done: FACTOR $SUBSET"
done
log "Phase 4c complete."

# ── Phase 4d: Semantic entropy ablation ───────────────────────────────────────
log "======================================================"
log "PHASE 4d: Semantic entropy ablation (MedHallu, n=50, k=5)"
log "======================================================"
SE_OUT="$CANONICAL_DIR/semantic_entropy_gate_comparison.json"
if [ ! -f "$SE_OUT" ]; then
    "$PYTHON" -u experiments/run_semantic_entropy_ablation.py \
        --model "meta-llama/Llama-3.1-8B-Instruct" --load-in-4bit \
        --benchmark medhallu \
        --n 50 --k 5 --seed 42 \
        --out "$SE_OUT" \
        2>&1 | tee logs/semantic_entropy.log || log "  WARN: Semantic entropy ablation failed"
    log "  Semantic entropy done → $SE_OUT"
else
    log "  Skipping semantic entropy — already exists"
fi
log "Phase 4d complete."

# ── Phase 5: Statistics + R²-stratified analysis ─────────────────────────────
log "======================================================"
log "PHASE 5: Statistical analysis + R²-stratified ALTA analysis"
log "======================================================"
"$PYTHON" -u compute_final_stats.py \
    --results-dir "$CANONICAL_DIR" \
    --output "$CANONICAL_DIR/statistics_table.json" \
    2>&1 | tee logs/phase5_stats.log
log "Phase 5 complete."

log "======================================================"
log "ALL PHASES COMPLETE"
log "Summary files in: $CANONICAL_DIR"
log "Statistics:        $CANONICAL_DIR/statistics_table.json"
log "R² stratified:     $CANONICAL_DIR/r2_stratified_analysis.json"
log "======================================================"

echo "" >> logs/pipeline.log
echo "PIPELINE FINISHED AT $(date)" >> logs/pipeline.log
"""


def connect(host: str, port: int, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=30)
    return client


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return exit_code, out, err


def sftp_put(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    # Ensure remote directory exists
    remote_dir = remote.rsplit("/", 1)[0] if "/" in remote else "."
    try:
        sftp.makedirs(remote_dir)
    except Exception:
        # makedirs may not exist on all versions; try mkdir -p via exec
        pass
    sftp.put(str(local), remote)


def makedirs_remote(client: paramiko.SSHClient, path: str) -> None:
    run(client, f"mkdir -p '{path}'")


def main(sync_only: bool = False) -> None:
    print(f"Connecting to {USER}@{HOST}:{PORT}...")
    client = connect(HOST, PORT, USER, PASS)
    print("Connected.")

    # ── 1. GPU / job status ──────────────────────────────────────────────────
    print("\n=== GPU Status ===")
    _, out, _ = run(client, "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available'")
    print(out)

    print("\n=== Current running Python processes ===")
    _, out, _ = run(client, "ps aux | grep python | grep -v grep | head -10 || echo 'none'")
    print(out.encode("ascii", errors="replace").decode("ascii"))

    # ── 2. Locate / create repo dir ──────────────────────────────────────────
    # Canonical AutoDL path used by watchers / prior runs (must take priority over ~/LLM_Hallucination).
    _, canon, _ = run(client, "test -f /root/llm-hallucination-self-testing/cured.py && echo CANON || echo NOCANON")
    if "CANON" in canon:
        repo_dir = "/root/llm-hallucination-self-testing"
        print(f"\nRepo found at {repo_dir} (canonical)")
    else:
        _, out, _ = run(client, "ls ~/LLM_Hallucination/cured.py 2>/dev/null && echo FOUND || echo MISSING")
        if "FOUND" in out:
            repo_dir = "~/LLM_Hallucination"
            print(f"\nRepo found at {repo_dir}")
        else:
            _, out2, _ = run(client, "find ~ -name cured.py -maxdepth 6 2>/dev/null | head -3")
            if out2.strip():
                repo_dir = out2.strip().split("\n")[0].replace("/cured.py", "")
                print(f"\nRepo found at {repo_dir}")
            else:
                print("\nRepo not found — creating ~/LLM_Hallucination (sync will populate).")
                repo_dir = "~/LLM_Hallucination"
                run(client, f"mkdir -p {repo_dir}")

    # Expand ~
    _, repo_abs, _ = run(client, f"echo {repo_dir}")
    repo_abs = repo_abs.strip()
    print(f"Repo absolute path: {repo_abs}")

    # ── 3. Sync files ────────────────────────────────────────────────────────
    print("\n=== Syncing code files ===")
    sftp = client.open_sftp()

    for local_rel, remote_rel in FILES_TO_SYNC:
        local_path = LOCAL_ROOT / local_rel
        remote_path = f"{repo_abs}/{remote_rel}"
        remote_dir = remote_path.rsplit("/", 1)[0]
        run(client, f"mkdir -p '{remote_dir}'")
        print(f"  {local_rel} -> {remote_path}")
        sftp.put(str(local_path), remote_path)

    sftp.close()
    print("Files synced.")

    # ── 4. Write master pipeline script ─────────────────────────────────────
    print("\n=== Writing pipeline script ===")
    pipeline_path = f"{repo_abs}/run_full_pipeline.sh"
    stdin, stdout, stderr = client.exec_command(f"cat > '{pipeline_path}'")
    stdin.write(PIPELINE_SCRIPT)
    stdin.channel.shutdown_write()
    stdout.channel.recv_exit_status()
    run(client, f"chmod +x '{pipeline_path}'")
    print(f"Pipeline script written to {pipeline_path}")

    if sync_only:
        print(
            "\n=== --sync-only: not restarting pipeline ===\n"
            "Updated files are on the server. The next cured.py job started by the\n"
            "existing orchestrator will load the new code; or restart the orchestrator manually.\n"
        )
        client.close()
        return

    # Avoid duplicate orchestrators if a prior nohup/watcher is still running
    run(client, "pkill -f run_full_pipeline.sh 2>/dev/null || true")
    time.sleep(2)

    # ── 5. Check/create persistent session (screen preferred, nohup fallback) ──
    log_path = f"{repo_abs}/logs/pipeline.log"
    run(client, f"mkdir -p '{repo_abs}/logs'")
    activate_cmd = f"source {repo_abs}/llm-env/bin/activate 2>/dev/null || true"
    pipeline_cmd = f"cd '{repo_abs}' && {activate_cmd} && bash run_full_pipeline.sh"
    session = "cured_pipeline"

    # Check what's available: screen > tmux > nohup
    _, screen_check, _ = run(client, "which screen 2>/dev/null && echo SCREEN_OK || echo NO_SCREEN")
    _, tmux_check, _   = run(client, "which tmux   2>/dev/null && echo TMUX_OK  || echo NO_TMUX")

    if "SCREEN_OK" in screen_check:
        print("\n=== Setting up screen session ===")
        # Kill existing session if present
        run(client, f"screen -X -S {session} quit 2>/dev/null || true")
        time.sleep(1)
        # Start detached screen session that logs to file
        screen_cmd = f"screen -dmS {session} bash -c '{pipeline_cmd} 2>&1 | tee {log_path}; echo PIPELINE_DONE >> {log_path}'"
        code, out, err = run(client, screen_cmd, timeout=10)
        print(f"  screen session '{session}' started (code={code})")
        time.sleep(2)
        _, out, _ = run(client, "screen -list 2>/dev/null || echo 'no screen'")
        print(f"  screen sessions:\n{out}")

    elif "TMUX_OK" in tmux_check:
        print("\n=== Setting up tmux session ===")
        run(client, f"tmux kill-session -t {session} 2>/dev/null || true")
        tmux_cmd = (f"tmux new-session -d -s {session} "
                    f"\"bash -c '{pipeline_cmd} 2>&1 | tee {log_path}'\"")
        run(client, tmux_cmd)
        print(f"  tmux session '{session}' started.")
        _, out, _ = run(client, "tmux list-sessions")
        print(f"  {out}")

    else:
        print("\n=== Using nohup (no screen/tmux) ===")
        nohup_cmd = f"nohup bash -c '{pipeline_cmd}' > {log_path} 2>&1 &"
        run(client, nohup_cmd)
        print("  Pipeline launched via nohup.")

    # ── 6. Verify it's running ───────────────────────────────────────────────
    time.sleep(4)
    _, out, _ = run(client, f"ps aux | grep run_full_pipeline | grep -v grep || echo 'not visible yet'")
    print(f"\n=== Pipeline process check ===\n{out.encode('ascii', errors='replace').decode('ascii')}")

    _, out, _ = run(client, f"tail -5 {log_path} 2>/dev/null || echo 'log not created yet'")
    print(f"\n=== Pipeline log (last 5 lines) ===\n{out.encode('ascii', errors='replace').decode('ascii')}")

    # ── 7. Print monitoring instructions ────────────────────────────────────
    monitor_cmd = "screen -r cured_pipeline" if "SCREEN_OK" in screen_check else "tmux attach -t cured_pipeline"
    print(f"""
==========================================================
PIPELINE QUEUED SUCCESSFULLY
==========================================================
Server : {HOST}:{PORT}
Repo   : {repo_abs}
Session: {session}

Monitor:
  ssh root@{HOST} -p {PORT}
  {monitor_cmd}
  # or just tail the log:
  tail -f {log_path}

Phases (auto-queued):
  1. Model profiling   (R-sq, kappa, ECR for 3B/8B/14B/32B)
  2. Ablation grid     (4x4x2 runs, n=200 each)
  3. Calibration       (decision tree -> router_thresholds.json)
  4. Main experiments  (n=500, new router, all scales x 3 benchmarks)
  5. Statistics        (McNemar + bootstrap CI -> statistics_table.json)

The pipeline WAITS for the current Qwen job to finish first.

Results will appear in:
  {repo_abs}/results/CANONICAL_v2/
  {repo_abs}/logs/
==========================================================
""")
    client.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Sync CURED repo to GPU server and optionally start the full pipeline.")
    ap.add_argument(
        "--sync-only",
        action="store_true",
        help="Only upload files and refresh run_full_pipeline.sh; do not kill or launch the orchestrator.",
    )
    a = ap.parse_args()
    main(sync_only=a.sync_only)
