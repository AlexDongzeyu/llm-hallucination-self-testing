#!/usr/bin/env python3
"""Fix Unicode characters in the A800 pipeline script and restart it."""
import paramiko
import time

HOST = "js4.blockelite.cn"
PORT = 14136
USER = "root"
PASS = "ra7ye9ka"
REPO = "/root/llm-hallucination-self-testing"


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    return client


def run(client, cmd, timeout=60):
    _, o, e = client.exec_command(cmd, timeout=timeout)
    o.channel.recv_exit_status()
    out = o.read().decode("utf-8", "replace").encode("ascii", "replace").decode("ascii").strip()
    return out


def fix_pipeline(content: str) -> str:
    """Replace non-ASCII characters with ASCII equivalents."""
    table = {
        "\u2192": "->",   # →
        "\u2190": "<-",   # ←
        "\u00d7": "x",    # ×
        "\u2264": "<=",   # ≤
        "\u2265": ">=",   # ≥
        "\u00b2": "2",    # ²
        "\u03ba": "kappa",# κ
        "\u2013": "-",    # –
        "\u2014": "--",   # —
        "\u2019": "'",    # '
        "\u201c": '"',    # "
        "\u201d": '"',    # "
        "\ufffd": "?",    # replacement char
        "\u2026": "...",  # …
        "\u00e9": "e",    # é
        "\u00e0": "a",    # à
    }
    for old, new in table.items():
        content = content.replace(old, new)
    # Final pass: strip remaining non-ASCII
    return "".join(c if ord(c) < 128 else "?" for c in content)


def main():
    print(f"Connecting to {USER}@{HOST}:{PORT}...")
    client = connect()
    print("Connected.\n")

    # 1. Read and fix pipeline script
    sftp = client.open_sftp()
    print("Reading pipeline script...")
    with sftp.file(f"{REPO}/run_full_pipeline.sh", "r") as f:
        content = f.read().decode("utf-8", "replace")

    non_ascii_before = sum(1 for c in content if ord(c) >= 128)
    print(f"Non-ASCII chars before fix: {non_ascii_before}")

    content_fixed = fix_pipeline(content)
    non_ascii_after = sum(1 for c in content_fixed if ord(c) >= 128)
    print(f"Non-ASCII chars after fix:  {non_ascii_after}")

    # Write fixed version
    with sftp.file(f"{REPO}/run_full_pipeline.sh", "w") as f:
        f.write(content_fixed.encode("ascii", "replace"))
    sftp.close()
    print("Pipeline script fixed and written back.")

    # 2. Kill any lingering processes
    run(client, "pkill -f run_full_pipeline.sh 2>/dev/null || true")
    run(client, "pkill -f cured.py 2>/dev/null || true")
    time.sleep(3)
    print("Killed lingering processes.")

    # 3. Verify thresholds still correct
    thresh = run(client, (
        f"python3 -c \"import json; d=json.load(open('{REPO}/configs/router_thresholds.json')); "
        "print('tau_kappa:', d['tau_kappa'], '| tau_ECR:', d['tau_ECR'], '| profile_mean_r2:', d.get('profile_mean_r2', 'MISSING'))\" 2>/dev/null"
    ))
    print(f"\nThresholds on server: {thresh}")

    # 4. Append to existing log (not overwrite) so we keep history
    log_path = f"{REPO}/logs/pipeline.log"
    activate_cmd = f"source {REPO}/llm-env/bin/activate 2>/dev/null || true"
    pipeline_cmd = f"cd '{REPO}' && {activate_cmd} && bash run_full_pipeline.sh"

    run(client, f"echo '' >> {log_path}")
    run(client, f"echo '=== PIPELINE RESTARTED AFTER UNICODE FIX ===' >> {log_path}")

    # 5. Restart with nohup
    nohup_cmd = f"nohup bash -c '{pipeline_cmd}' >> {log_path} 2>&1 &"
    run(client, nohup_cmd)
    time.sleep(4)
    print("Pipeline restarted via nohup.")

    # 6. Verify running
    proc = run(client, "ps aux | grep run_full_pipeline | grep -v grep | head -3")
    print(f"\nProcess check:\n{proc or 'NOT VISIBLE YET'}")

    log_tail = run(client, f"tail -15 {log_path}")
    print(f"\nLog tail:\n{log_tail}")

    gpu = run(client, "nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader")
    print(f"\nGPU: {gpu}")

    print(f"\n=== DONE ===")
    print(f"Monitor: tail -f {log_path}")
    client.close()


if __name__ == "__main__":
    main()
