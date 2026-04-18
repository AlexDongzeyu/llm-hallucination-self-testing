"""Ensure the remote pipeline (run_full_pipeline.sh) will run automatically.

If run_full_pipeline is not currently running, start a server-side watcher:
  - polls GPU VRAM used
  - when VRAM < 2000 MiB, starts run_full_pipeline.sh via nohup

This avoids the previous failure mode where the pipeline tried to load 32B
while a 32B generation job was already occupying VRAM.
"""

from __future__ import annotations

import time

import paramiko


HOST = "js4.blockelite.cn"
PORT = 14136
USER = "root"
PASS = "ra7ye9ka"

REPO = "/root/llm-hallucination-self-testing"
LOG = f"{REPO}/logs/pipeline.log"


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 60) -> str:
    _, stdout, _ = client.exec_command(cmd, timeout=timeout)
    stdout.channel.recv_exit_status()
    return stdout.read().decode("utf-8", errors="replace").strip()

def fire_and_forget(client: paramiko.SSHClient, cmd: str) -> None:
    """Run a command but don't wait for output (useful for nohup/&)."""
    transport = client.get_transport()
    if not transport:
        raise RuntimeError("SSH transport not available")
    ch = transport.open_session()
    ch.exec_command(cmd)
    # Close immediately; command continues server-side.
    ch.close()


def safe_print(label: str, s: str) -> None:
    print(label)
    print(s.encode("ascii", errors="replace").decode("ascii"))


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30, banner_timeout=30, auth_timeout=30)
    t = client.get_transport()
    if t:
        t.set_keepalive(10)

    gpu = run(client, "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader")
    pipe = run(client, "ps aux | grep run_full_pipeline | grep -v grep || echo NONE")
    top = run(client, "ps -eo pid,pcpu,etime,cmd --sort=-pcpu | head -8")
    tail = run(client, f"tail -10 {LOG} 2>/dev/null || echo NO_LOG")

    safe_print("=== GPU ===", gpu)
    safe_print("\n=== run_full_pipeline ===", pipe)
    safe_print("\n=== Top processes ===", top)
    safe_print("\n=== pipeline.log tail ===", tail)

    if pipe.strip() == "NONE":
        run(client, f"mkdir -p '{REPO}/logs'")
        # Start watcher (nohup bash -c '...') fully quoted for remote shell
        watcher_cmd = (
            f"cd '{REPO}' && "
            f"nohup bash -lc \""
            f"echo '' >> '{LOG}'; "
            f"echo '[WATCHER] started '$(date) >> '{LOG}'; "
            f"while true; do "
            f"USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); "
            f"if [ \\\"${{USED:-999999}}\\\" -lt 2000 ]; then "
            f"echo '[WATCHER] GPU free '\"${{USED}}\"'MiB '$(date) >> '{LOG}'; break; "
            f"fi; "
            f"sleep 60; "
            f"done; "
            f"source '{REPO}/llm-env/bin/activate' 2>/dev/null || true; "
            f"bash run_full_pipeline.sh >> '{LOG}' 2>&1; "
            f"echo '[WATCHER] finished '$(date) >> '{LOG}';"
            f"\" >/dev/null 2>&1 &"
        )
        # Start watcher without waiting for remote output; the watcher is nohup'd.
        fire_and_forget(client, watcher_cmd)
        time.sleep(1)
        safe_print("\n=== watcher started (ps check) ===", run(client, "ps aux | grep WATCHER | grep -v grep || echo (watcher hidden in nohup bash -lc)", timeout=20))
        safe_print("\n=== pipeline.log tail (post) ===", run(client, f"tail -12 {LOG} 2>/dev/null || echo NO_LOG", timeout=20))

    client.close()


if __name__ == "__main__":
    main()

