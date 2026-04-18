"""Restart the remote pipeline with robust HF download settings.

Use this when the server-side pipeline dies during HuggingFace downloads.
It patches /root/llm-hallucination-self-testing/run_full_pipeline.sh to set:
  - HF_ENDPOINT (mirror)
  - longer timeouts
  - max workers = 1 (reduces chunked encoding / incomplete reads)

Then it restarts the pipeline via nohup.
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

    # Stop any orchestrators (safe even if none)
    run(client, "pkill -f run_full_pipeline 2>/dev/null || true")

    # Patch env vars at the top of the script (idempotent)
    patch = (
        "perl -0777 -pe '"
        "s/^#!\\\\/usr\\\\/bin\\\\/env bash\\\\n/"
        "#!\\\\/usr\\\\/bin\\\\/env bash\\\\n"
        "export HF_ENDPOINT=https:\\\\/\\\\/hf-mirror.com\\\\n"
        "export HUGGINGFACE_HUB_VERBOSITY=warning\\\\n"
        "export HF_HUB_DOWNLOAD_TIMEOUT=300\\\\n"
        "export HF_HUB_ETAG_TIMEOUT=60\\\\n"
        "export HF_HUB_DISABLE_PROGRESS_BARS=1\\\\n"
        "export HF_HUB_DOWNLOAD_MAX_WORKERS=1\\\\n"
        "/' -i "
        + PIPELINE
    )
    run(client, patch)

    # Ensure logs dir exists
    run(client, f"mkdir -p '{REPO}/logs'")

    # Restart pipeline
    activate = f"source {REPO}/llm-env/bin/activate 2>/dev/null || true"
    cmd = f"cd {REPO} && {activate} && bash run_full_pipeline.sh"
    run(client, f"nohup bash -c '{cmd}' >> {LOG} 2>&1 & echo STARTED")

    time.sleep(2)
    def safe_print(s: str) -> None:
        print(s.encode("ascii", errors="replace").decode("ascii"))

    safe_print(run(client, "pgrep -a -f run_full_pipeline || echo NONE"))
    safe_print(run(client, f"tail -12 {LOG} 2>/dev/null || echo NO_LOG")[:2000])

    client.close()


if __name__ == "__main__":
    main()

