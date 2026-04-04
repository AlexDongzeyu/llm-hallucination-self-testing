"""
eval_calibration_phase1.py
Workspace-compatible Phase 1 calibration runner.

Runs:
1) src/calibration_proof.py
2) src/diagnose_jsd.py

Usage:
    python -u experiments/eval_calibration_phase1.py
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PYTHON = ROOT / "llm-env" / "Scripts" / "python.exe"


def run_script(script_path: Path, label: str) -> None:
    print("\n" + "=" * 70, flush=True)
    print(f"Running: {label}", flush=True)
    print("=" * 70, flush=True)

    result = subprocess.run([str(PYTHON), "-u", str(script_path)], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


if __name__ == "__main__":
    if not PYTHON.exists():
        raise FileNotFoundError(f"Python interpreter not found: {PYTHON}")

    run_script(SRC / "calibration_proof.py", "Calibration Proof")
    run_script(SRC / "diagnose_jsd.py", "JSD Diagnostics")

    print("\nPhase 1 calibration run completed successfully.", flush=True)
