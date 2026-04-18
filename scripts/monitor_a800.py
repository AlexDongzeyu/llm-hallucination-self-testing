#!/usr/bin/env python3
"""Monitor A800 pipeline status.

Snapshot (default): GPU, running ``cured.py`` command, last lines of pipeline logs.

Live monitoring options:

1. **True live stream (recommended):** SSH into the box and follow the job log, e.g.
   ``ssh root@js4.blockelite.cn -p 14036`` then::

       tail -f /root/llm-hallucination-self-testing/logs/resume_8b_alta_*.log

   Use ``ls -lt logs/resume_*.log | head`` to pick the newest file (ALTA / greedy / FACTOR
   each write their own ``resume_*`` log).

2. **From this repo (polling):** ``python scripts/monitor_a800.py --watch 15`` refreshes
   a snapshot every 15 seconds until Ctrl+C (works on Windows without SSH installed).
"""
import argparse
import datetime
import paramiko
import time

HOST = "js4.blockelite.cn"
PORT = 14036
USER = "root"
PASS = "aiPh9chu"
REPO = "/root/llm-hallucination-self-testing"


def run(client, cmd, t=30):
    _, o, e = client.exec_command(cmd, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode("utf-8", "replace").encode("ascii", "replace").decode("ascii").strip()


def snapshot(client, log_lines: int) -> None:
    print("=== GPU ===")
    print(run(client, "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader"))

    print("\n=== Active cured.py command ===")
    args = run(client, "ps aux | grep cured.py | grep -v grep | head -1")
    print(args[:500] if args else "No cured.py running")

    print("\n=== Newest resume job log (last N lines) ===")
    newest = run(
        client,
        f"ls -1t {REPO}/logs/resume_*.log 2>/dev/null | grep -v resume_pipeline | head -1",
        t=15,
    )
    if newest:
        print(f"(file: {newest})")
        print(run(client, f"tail -{log_lines} {newest} 2>/dev/null", t=20))
    else:
        print("(no resume_*.log yet)")

    print("\n=== Pipeline wrapper log (last N lines) ===")
    log = run(
        client,
        f"tail -{log_lines} {REPO}/logs/resume_pipeline.log 2>/dev/null || tail -{log_lines} {REPO}/logs/pipeline.log 2>/dev/null",
        t=15,
    )
    print(log or "(empty / missing)")

    print("\n=== New CURED v2 results on server ===")
    print(run(client, f"ls {REPO}/results/CANONICAL_v2/main_cured_*_v2.json 2>/dev/null || echo 'NONE YET'"))

    print("\n=== Pipeline process running ===")
    proc = run(client, "ps aux | grep -E 'resume_pipeline|run_full_pipeline' | grep -v grep | head -3")
    print(proc or "NOT RUNNING")


def main():
    p = argparse.ArgumentParser(description="Monitor A800 experiment pipeline (SSH via paramiko).")
    p.add_argument(
        "--watch",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Re-print snapshot every SEC seconds until Ctrl+C (0 = run once).",
    )
    p.add_argument(
        "--log-lines",
        type=int,
        default=25,
        help="Lines of tail for each log section (default: 25).",
    )
    args = p.parse_args()

    interval = float(args.watch)
    while True:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'=' * 60}\n  {ts}  (local)\n{'=' * 60}")
        snapshot(client, args.log_lines)
        client.close()

        if interval <= 0:
            break
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
