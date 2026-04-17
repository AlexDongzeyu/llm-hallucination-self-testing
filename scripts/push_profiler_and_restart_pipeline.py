"""Push patched compute_logit_linearity.py to server and restart pipeline.

Avoids PowerShell quoting issues by running as a normal Python script.
"""

from pathlib import Path

import paramiko


HOST = "js4.blockelite.cn"
PORT = 14136
USER = "root"
PASS = "ra7ye9ka"

REPO = "/root/llm-hallucination-self-testing"


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30, banner_timeout=30, auth_timeout=30)
    t = client.get_transport()
    if t:
        t.set_keepalive(10)

    # Ensure directory exists
    client.exec_command(f"mkdir -p '{REPO}/experiments' '{REPO}/logs'")

    # Push file
    local = Path(__file__).resolve().parents[1] / "experiments" / "compute_logit_linearity.py"
    remote = f"{REPO}/experiments/compute_logit_linearity.py"
    sftp = client.open_sftp()
    sftp.put(str(local), remote)
    sftp.close()

    # Stop any orchestrators + start pipeline
    client.exec_command("pkill -f run_full_pipeline 2>/dev/null; true")
    cmd = f"cd '{REPO}'; source '{REPO}/llm-env/bin/activate' 2>/dev/null; true; bash run_full_pipeline.sh"
    client.exec_command(f"nohup bash -lc \"{cmd}\" >> '{REPO}/logs/pipeline.log' 2>&1 &")

    # Quick status
    def run(cmd: str) -> str:
        _, o, _ = client.exec_command(cmd, timeout=30)
        o.channel.recv_exit_status()
        return o.read().decode("utf-8", errors="replace").strip()

    print(run("nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader").encode("ascii", "replace").decode("ascii"))
    print(run("ps aux | grep run_full_pipeline | grep -v grep || echo NONE").encode("ascii", "replace").decode("ascii"))
    print(run(f"tail -12 '{REPO}/logs/pipeline.log' 2>/dev/null || echo NO_LOG").encode("ascii", "replace").decode("ascii"))

    client.close()


if __name__ == "__main__":
    main()

