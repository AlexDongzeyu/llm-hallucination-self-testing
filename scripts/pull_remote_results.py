#!/usr/bin/env python3
"""Pull completed CANONICAL_v2 result JSONs from the GPU server.

Downloads remote files from:
  /root/llm-hallucination-self-testing/results/CANONICAL_v2/
into local:
  results/CANONICAL_v2/

Only downloads files that are missing locally (or size differs).
"""

from __future__ import annotations

import os
from pathlib import Path

import paramiko

HOST = "js4.blockelite.cn"
PORT = 14136
USER = "root"
PASS = "ra7ye9ka"

REMOTE_DIR = "/root/llm-hallucination-self-testing/results/CANONICAL_v2"
LOCAL_DIR = Path(__file__).resolve().parents[1] / "results" / "CANONICAL_v2"


def _safe_print(s: str) -> None:
    print(s.encode("ascii", errors="replace").decode("ascii"))


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30, banner_timeout=30, auth_timeout=30)

    sftp = client.open_sftp()
    try:
        remote_names = sorted([n for n in sftp.listdir(REMOTE_DIR) if n.endswith(".json")])
    except FileNotFoundError:
        raise SystemExit(f"Remote directory not found: {REMOTE_DIR}")

    pulled = 0
    skipped = 0
    for name in remote_names:
        remote_path = f"{REMOTE_DIR}/{name}"
        local_path = LOCAL_DIR / name

        try:
            rstat = sftp.stat(remote_path)
        except FileNotFoundError:
            continue

        need = False
        if not local_path.exists():
            need = True
        else:
            try:
                if local_path.stat().st_size != rstat.st_size:
                    need = True
            except OSError:
                need = True

        if not need:
            skipped += 1
            continue

        _safe_print(f"Downloading {name} ({rstat.st_size} bytes)")
        sftp.get(remote_path, str(local_path))
        pulled += 1

    _safe_print(f"\nDONE. downloaded={pulled} skipped={skipped} total_remote_json={len(remote_names)}")

    sftp.close()
    client.close()


if __name__ == "__main__":
    main()

