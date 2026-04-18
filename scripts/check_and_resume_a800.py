#!/usr/bin/env python3
"""
Connect to A800, check what's done vs missing, and launch all remaining steps.
Run from local with: python scripts/check_and_resume_a800.py

GPU safety: the generated ``resume_pipeline.sh`` never runs two ``cured.py`` jobs at
once on the same machine (single A800). Optional overlap is **CPU/IO only** (FACTOR
CSV prep) while ablations hold the GPU.
"""
import paramiko, time, textwrap

HOST = "js4.blockelite.cn"; PORT = 14036; USER = "root"; PASS = "aiPh9chu"
REPO = "/root/llm-hallucination-self-testing"
PYTHON = f"{REPO}/llm-env/bin/python"
CANONICAL = f"{REPO}/results/CANONICAL_v2"

def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    return c

def run(c, cmd, t=60):
    _, o, e = c.exec_command(cmd, timeout=t)
    o.channel.recv_exit_status()
    out = o.read().decode("utf-8", "replace").strip()
    err = e.read().decode("utf-8", "replace").strip()
    return out, err

def exists(c, path):
    out, _ = run(c, f"test -f {path} && echo YES || echo NO")
    return out.strip() == "YES"

def check_r2q(c, path):
    """Check if ablation file has r2_q populated in per_question."""
    script = (
        "import json,sys; "
        f"d=json.load(open('{path}')); "
        "pq=d.get('per_question',[]); "
        "r2s=[x.get('r2_q') for x in pq[:5]]; "
        "print('n=%d' % len(pq), 'r2_sample=%s' % str(r2s))"
    )
    out, err = run(c, f"python3 -c \"{script}\" 2>&1")
    return out

print("Connecting to A800...")
c = connect()
print("Connected!")

# --- Check what's present ---
print("\n=== STATUS CHECK ===")
needed = {
    "8b_alta_ablation":   f"{CANONICAL}/ablation_8b_alta_truthfulqa_n200.json",
    "8b_greedy_ablation": f"{CANONICAL}/ablation_8b_greedy_truthfulqa_n200.json",
    "factor_news":        f"{CANONICAL}/results_8b_factor_news_n200.json",
    "factor_wiki":        f"{CANONICAL}/results_8b_factor_wiki_n200.json",
    "sem_entropy":        f"{CANONICAL}/semantic_entropy_gate_comparison.json",
    "stats_table":        f"{CANONICAL}/statistics_table.json",
    "r2_analysis":        f"{CANONICAL}/r2_stratified_analysis.json",
}
status = {}
for k, path in needed.items():
    status[k] = exists(c, path)
    print(f"  {'OK' if status[k] else 'XX'} {k}: {path}")

# Check if ablation files have r2_q
print("\n=== r2_q CHECK ===")
for k in ["8b_alta_ablation", "8b_greedy_ablation"]:
    if status[k]:
        result = check_r2q(c, needed[k])
        print(f"  {k}: {result}")
        # If r2_q values are None, mark as needing redo
        if "None" in result or "n=0" in result:
            print(f"    -> r2_q is None, will re-run with --save-per-question")
            status[k] = False
    else:
        print(f"  {k}: MISSING")

# Check if factor CSVs exist on server
print("\n=== FACTOR CSVs ===")
for subset in ["news", "wiki"]:
    csv_path = f"{REPO}/benchmarks/factor_{subset}_n200.csv"
    e = exists(c, csv_path)
    print(f"  {'OK' if e else 'XX'} factor_{subset}_n200.csv")

print("\n=== BUILDING RESUME SCRIPT ===")

need_factor_gpu = (not status["factor_news"]) or (not status["factor_wiki"])
news_csv = f"{REPO}/benchmarks/factor_news_n200.csv"
wiki_csv = f"{REPO}/benchmarks/factor_wiki_n200.csv"
factor_csvs_ok = exists(c, news_csv) and exists(c, wiki_csv)
# Download/build CSVs with CPU+network while GPU runs ALTA/greedy (never overlap two cured.py).
overlap_factor_prep = need_factor_gpu and (not factor_csvs_ok)

# Build targeted resume script
resume_lines = [
    "#!/usr/bin/env bash",
    "set -uo pipefail",
    f'REPO="{REPO}"',
    f'PYTHON="{PYTHON}"',
    f'CANONICAL="{CANONICAL}"',
    f'MODEL="meta-llama/Llama-3.1-8B-Instruct"',
    f'cd "$REPO"',
    'TS=$(date +%Y%m%d_%H%M%S)',
    'mkdir -p logs "$CANONICAL" benchmarks',
    'log() { echo "[$(date +\"%Y-%m-%d %H:%M:%S\")] $*"; }',
    '',
]

if overlap_factor_prep:
    resume_lines += [
        'log "=== FACTOR CSV prep (background; CPU/IO only - never run two cured.py at once) ==="',
        '"$PYTHON" scripts/prep_factor_benchmark.py > "logs/prep_factor_bg_${TS}.log" 2>&1 &',
        "FACTOR_PREP_PID=$!",
        "",
    ]

# 8B ablations if needed
if not status["8b_alta_ablation"]:
    resume_lines += [
        'log "=== 8B ALTA ablation (save-per-question) ==="',
        '"$PYTHON" -u cured.py \\',
        '  --model "$MODEL" --load-in-4bit --model-params-b 8.0 \\',
        '  --protocols alta --benchmark truthfulqa \\',
        '  --n 200 --seed 42 --no-shuffle --scoring cosine \\',
        '  --save-per-question --skip-iti \\',
        f'  --out "$CANONICAL/ablation_8b_alta_truthfulqa_n200.json" \\',
        '  > "logs/resume_8b_alta_${TS}.log" 2>&1',
        'log "  ALTA ablation done."',
        '',
    ]

if not status["8b_greedy_ablation"]:
    resume_lines += [
        'log "=== 8B Greedy ablation (save-per-question) ==="',
        '"$PYTHON" -u cured.py \\',
        '  --model "$MODEL" --load-in-4bit --model-params-b 8.0 \\',
        '  --protocols greedy --benchmark truthfulqa \\',
        '  --n 200 --seed 42 --no-shuffle --scoring cosine \\',
        '  --save-per-question --skip-iti \\',
        f'  --out "$CANONICAL/ablation_8b_greedy_truthfulqa_n200.json" \\',
        '  > "logs/resume_8b_greedy_${TS}.log" 2>&1',
        'log "  Greedy ablation done."',
        '',
    ]

# Factor (sequential GPU jobs only — one cured.py at a time)
if need_factor_gpu:
    if overlap_factor_prep:
        resume_lines += [
            'log "=== wait for FACTOR CSV prep (background) before GPU FACTOR runs ==="',
            "wait ${FACTOR_PREP_PID:-} || log \"WARN: factor prep failed\"",
            "",
        ]
    else:
        resume_lines += [
            'log "=== FACTOR CSVs present; skip prep ==="',
            "",
        ]
    for subset, key in [("news", "factor_news"), ("wiki", "factor_wiki")]:
        if not status[key]:
            resume_lines += [
                f'log "=== FACTOR {subset} ==="',
                '"$PYTHON" -u cured.py \\',
                '  --model "$MODEL" --load-in-4bit --skip-iti \\',
                '  --protocols greedy,alta,cured \\',
                '  --router new --router-config configs/router_thresholds.json \\',
                '  --benchmark custom \\',
                f'  --custom-csv benchmarks/factor_{subset}_n200.csv \\',
                '  --question-col question --answer-col answer \\',
                '  --n 200 --seed 42 --no-shuffle \\',
                '  --scoring letter --max-new-tokens 5 \\',
                '  --save-per-question \\',
                f'  --out "$CANONICAL/results_8b_factor_{subset}_n200.json" \\',
                f'  > "logs/resume_factor_{subset}_${{TS}}.log" 2>&1 || log "WARN: factor_{subset} failed"',
                f'log "  FACTOR {subset} done."',
                '',
            ]

# Semantic entropy
if not status["sem_entropy"]:
    resume_lines += [
        'log "=== Semantic entropy ablation ==="',
        '"$PYTHON" -u experiments/run_semantic_entropy_ablation.py \\',
        '  --model "$MODEL" --load-in-4bit \\',
        '  --benchmark medhallu --n 50 --k 5 --seed 42 \\',
        f'  --out "$CANONICAL/semantic_entropy_gate_comparison.json" \\',
        '  > "logs/resume_sem_entropy_${TS}.log" 2>&1 || log "WARN: sem entropy failed"',
        'log "  Semantic entropy done."',
        '',
    ]

# Final stats always re-run to pick up new v2 files
resume_lines += [
    'log "=== Final statistics + R2 analysis ==="',
    '"$PYTHON" compute_final_stats.py \\',
    '  --results-dir "$CANONICAL" \\',
    f'  --output "$CANONICAL/statistics_table.json" \\',
    '  > "logs/resume_stats_${TS}.log" 2>&1 || log "WARN: stats failed"',
    'log "Statistics done."',
    '',
    'log "=== ALL REMAINING STEPS COMPLETE ==="',
    'log "Results in: $CANONICAL"',
]

script_content = "\n".join(resume_lines)
print(script_content[:500], "\n...")

# Write script to server
sftp = c.open_sftp()
with sftp.file(f"{REPO}/resume_pipeline.sh", "w") as f:
    f.write(script_content)
sftp.close()
print(f"\nScript written to {REPO}/resume_pipeline.sh")

# Make executable and launch with nohup
run(c, f"chmod +x {REPO}/resume_pipeline.sh")
run(c, f"cd {REPO} && nohup bash resume_pipeline.sh > logs/resume_pipeline.log 2>&1 &")
time.sleep(3)

# Verify it started
out, _ = run(c, "ps aux | grep resume_pipeline | grep -v grep | head -3")
if out:
    print(f"\nPipeline STARTED:\n{out}")
else:
    # check if it finished instantly or failed
    out2, _ = run(c, f"tail -5 {REPO}/logs/resume_pipeline.log 2>/dev/null")
    print(f"\nProcess check (may have started and is loading model):\n{out2}")

# tail log
time.sleep(3)
out3, _ = run(c, f"tail -10 {REPO}/logs/resume_pipeline.log 2>/dev/null")
print(f"\nLog tail:\n{out3}")

c.close()
print("\nDone. Monitor with: python scripts/monitor_a800.py")
