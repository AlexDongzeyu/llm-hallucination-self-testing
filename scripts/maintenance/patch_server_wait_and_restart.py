"""Patch remote run_full_pipeline.sh to wait for near-zero VRAM, then restart.

This prevents the pipeline from trying to load 32B while another heavy job is using VRAM.
"""

import time

import paramiko


HOST = "js4.blockelite.cn"
PORT = 14136
USER = "root"
PASS = "ra7ye9ka"

REPO = "/root/llm-hallucination-self-testing"
PIPELINE = f"{REPO}/run_full_pipeline.sh"
LOG = f"{REPO}/logs/pipeline.log"


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 120) -> str:
    _, stdout, _ = client.exec_command(cmd, timeout=timeout)
    stdout.channel.recv_exit_status()
    return stdout.read().decode("utf-8", errors="replace").strip()


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

    # Stop any current orchestrator
    run(client, "pkill -f run_full_pipeline 2>/dev/null || true")

    # Tighten the VRAM-free heuristic: require <2000 MiB used (was 5000)
    run(client, f"perl -pi -e 's/< 5000\\)/< 2000\\)/g' {PIPELINE}")
    run(client, f"perl -pi -e 's/< 5000/< 2000/g' {PIPELINE}")

    # Restart orchestrator
    run(client, f"mkdir -p '{REPO}/logs'")
    activate = f"source {REPO}/llm-env/bin/activate 2>/dev/null || true"
    cmd = f"cd {REPO} && {activate} && bash run_full_pipeline.sh"
    run(client, f"nohup bash -c '{cmd}' >> {LOG} 2>&1 & echo STARTED")

    time.sleep(2)
    print(run(client, "pgrep -a -f run_full_pipeline || echo NONE").encode("ascii", "replace").decode("ascii"))
    print(run(client, f"tail -12 {LOG} 2>/dev/null || echo NO_LOG").encode("ascii", "replace").decode("ascii"))

    client.close()


if __name__ == '__main__':
    main()

