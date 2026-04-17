"""Quick status check for the remote pipeline.

Usage: python scripts/check_status.py

Prints GPU state, running processes, latest pipeline log lines, and a count of
completed Phase 4 main-run JSON files (so you can eyeball progress without
opening an SSH session).
"""
from __future__ import annotations

import sys

import paramiko

HOST = "js4.blockelite.cn"
PORT = 14136
USER = "root"
PASS = "ra7ye9ka"
REPO = "/root/llm-hallucination-self-testing"
CANON = f"{REPO}/results/CANONICAL_v2"
LOG = f"{REPO}/logs/pipeline.log"


def _p(s: str) -> None:
    sys.stdout.write(s.encode("ascii", errors="replace").decode("ascii") + "\n")


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

    def run(cmd: str, t: int = 60) -> str:
        _, o, _ = c.exec_command(cmd, timeout=t)
        return o.read().decode("utf-8", errors="replace")

    _p("=== GPU ===")
    _p(run("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader").strip())

    _p("\n=== Orchestrator ===")
    _p(run("pgrep -af run_full_pipeline.sh | head -3").strip() or "(not running)")

    _p("\n=== Active cured.py ===")
    _p(run("pgrep -af 'llm-env/bin/python -u cured.py' | head -3").strip() or "(none)")

    _p("\n=== Latest progress lines ===")
    _p(run(f"tail -25 {LOG}").strip())

    _p("\n=== Phase 4 file counts ===")
    _p(run(f"ls {CANON}/main_cured_*.json 2>/dev/null | wc -l").strip() + " main_cured_*.json (target: 13 incl. old-router)")
    _p(run(f"ls {CANON}/main_greedy_*.json 2>/dev/null | wc -l").strip() + " main_greedy_*.json (target: 4)")
    _p(run(f"ls {CANON}/statistics_table.json 2>/dev/null | wc -l").strip() + " statistics_table.json (target: 1 at pipeline end)")

    c.close()


if __name__ == "__main__":
    main()
