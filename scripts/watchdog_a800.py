#!/usr/bin/env python3
"""
watchdog_a800.py — Monitors the A800 pipeline and auto-restarts on crash.

Checks every 5 minutes:
  - Is the pipeline process still alive?
  - Is cured.py making progress (GPU utilization > 0)?
  - If stuck/crashed → restart via redeploy_clean_pipeline.py

Runs indefinitely until pipeline log shows "PIPELINE FINISHED".
Usage: python scripts/watchdog_a800.py
"""
import subprocess
import sys
import time
from pathlib import Path
import paramiko

HOST = "js4.blockelite.cn"
PORT = 14136
USER = "root"
PASS = "ra7ye9ka"
REPO = "/root/llm-hallucination-self-testing"
CHECK_INTERVAL = 300   # 5 minutes
STUCK_THRESHOLD = 600  # 10 minutes with 0% GPU = stuck


def connect(timeout=30):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=timeout)
    return client


def run(client, cmd, t=30):
    _, o, e = client.exec_command(cmd, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode("utf-8", "replace").encode("ascii", "replace").decode("ascii").strip()


def get_status(client):
    gpu_raw = run(client, "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits")
    parts = gpu_raw.split(",")
    mem_mb = int(parts[0].strip()) if parts else 0
    util_pct = int(parts[1].strip()) if len(parts) > 1 else 0

    pipeline_running = bool(run(client, "ps aux | grep run_full_pipeline | grep -v grep | head -1"))
    cured_running = bool(run(client, "ps aux | grep cured.py | grep -v grep | head -1"))

    log_tail = run(client, f"tail -5 {REPO}/logs/pipeline.log 2>/dev/null || echo ''")
    finished = "PIPELINE FINISHED" in log_tail

    progress_line = run(client,
        f"grep -oP '\\[\\d+/\\d+\\].*' {REPO}/logs/pipeline.log 2>/dev/null | tail -1 || echo ''")
    last_result = run(client,
        f"ls -t {REPO}/results/CANONICAL_v2/main_cured_*_v2.json 2>/dev/null | head -1 || echo ''")
    n_results = run(client,
        f"ls {REPO}/results/CANONICAL_v2/main_cured_*_v2.json 2>/dev/null | wc -l || echo 0")

    return {
        "mem_mb": mem_mb, "util_pct": util_pct,
        "pipeline_running": pipeline_running, "cured_running": cured_running,
        "finished": finished, "log_tail": log_tail,
        "progress": progress_line, "last_result": last_result,
        "n_v2_results": n_results.strip(),
    }


def restart_pipeline():
    print("  Restarting pipeline via redeploy_clean_pipeline.py...")
    result = subprocess.run(
        [sys.executable, "scripts/redeploy_clean_pipeline.py"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        print("  Restart OK.")
        return True
    else:
        print(f"  Restart FAILED: {result.stderr[:300]}")
        return False


def main():
    root = Path(__file__).resolve().parents[1]
    import os; os.chdir(root)

    print(f"Watchdog started. Checking every {CHECK_INTERVAL}s.")
    print(f"Target: {REPO}/logs/pipeline.log")
    print("Will auto-restart on crash. Ctrl-C to stop.\n")

    zero_util_count = 0
    check_num = 0

    while True:
        check_num += 1
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            client = connect(timeout=20)
            s = get_status(client)
            client.close()

            print(f"\n[{ts}] Check #{check_num}")
            print(f"  GPU:      {s['mem_mb']} MiB | {s['util_pct']}%")
            print(f"  Pipeline: {'RUNNING' if s['pipeline_running'] else 'STOPPED'}")
            print(f"  cured.py: {'RUNNING' if s['cured_running'] else 'idle'}")
            print(f"  Progress: {s['progress'] or 'no progress line yet'}")
            print(f"  v2 results: {s['n_v2_results']}")
            if s["last_result"]:
                print(f"  Latest:   {s['last_result'].split('/')[-1]}")
            print(f"  Log tail: {s['log_tail'].splitlines()[-1] if s['log_tail'] else '(empty)'}")

            if s["finished"]:
                print("\n=== PIPELINE FINISHED! ===")
                print(f"  v2 results created: {s['n_v2_results']}")
                print(f"  Log: {REPO}/logs/pipeline.log")
                print("  Watchdog exiting.")
                break

            if not s["pipeline_running"] and not s["cured_running"]:
                print("  !! Pipeline AND cured.py are both stopped — RESTARTING")
                if restart_pipeline():
                    zero_util_count = 0
                continue

            if s["util_pct"] == 0 and s["mem_mb"] < 1000:
                zero_util_count += 1
                if zero_util_count >= 2:
                    print(f"  !! GPU idle for {zero_util_count * CHECK_INTERVAL}s — possible stall")
                    if not s["pipeline_running"]:
                        print("  !! Pipeline not running — RESTARTING")
                        if restart_pipeline():
                            zero_util_count = 0
                    else:
                        print("  Pipeline still alive but GPU idle (between models). Continuing.")
                        zero_util_count = max(0, zero_util_count - 1)
            else:
                zero_util_count = 0

        except Exception as e:
            print(f"\n[{ts}] Check #{check_num} FAILED: {type(e).__name__}: {str(e)[:80]}")
            print("  Server unreachable. Will retry next cycle.")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
