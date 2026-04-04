"""
pipeline_after_n100.py

Unattended pipeline runner:
1) Waits until eval_n100 is complete (based on completion marker in its log).
2) Runs the remaining experiment scripts sequentially.
3) Writes per-step logs plus a resumable state file.

Usage:
    python -u experiments/pipeline_after_n100.py

Optional flags:
    --skip-wait           Start immediately without waiting for n100 marker.
    --online auto|always|never
                          Control whether eval_online.py is included.
                          auto: include only if API keys are present.
    --continue-on-error   Keep going after a failed step (default).
    --stop-on-error       Stop on first failed step.
    --include-grid        Also run eval_grid.py at the end.
    --force-rerun         Ignore saved state and re-run completed steps.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
RESULTS_LOGS = ROOT / "results" / "logs"
RESULTS_LOGS.mkdir(parents=True, exist_ok=True)

DEFAULT_PYTHON = ROOT / "llm-env" / "Scripts" / "python.exe"
PYTHON = DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)

N100_LOG = RESULTS_LOGS / "eval_n100.log"
N100_DONE_MARKER = "RESULTS TABLE (N=100, 4 requested configs)"
STATE_PATH = RESULTS_LOGS / "pipeline_after_n100_state.json"


@dataclass(frozen=True)
class Step:
    name: str
    script: str
    args: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full experiment pipeline after n100.")
    parser.add_argument("--skip-wait", action="store_true", help="Do not wait for n100 completion marker.")
    parser.add_argument("--online", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--include-grid", action="store_true", help="Include eval_grid.py at the end.")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore completed steps in state file.")
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"completed": [], "failed": [], "last_update": None}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"completed": [], "failed": [], "last_update": None}


def save_state(state: dict) -> None:
    state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def should_include_online(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(os.getenv("GROQ_API_KEY") and os.getenv("GEMINI_API_KEY"))


def n100_completed() -> bool:
    if not N100_LOG.exists():
        return False
    try:
        text = N100_LOG.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return N100_DONE_MARKER in text


def wait_for_n100_completion(poll_seconds: int) -> None:
    print("Waiting for eval_n100 completion marker...", flush=True)
    while True:
        if n100_completed():
            print("Detected n100 completion marker. Continuing pipeline.", flush=True)
            return
        if N100_LOG.exists():
            age = time.time() - N100_LOG.stat().st_mtime
            print(f"  n100 not done yet. Last log update {age/60:.1f} min ago.", flush=True)
        else:
            print("  n100 log not found yet; waiting...", flush=True)
        time.sleep(max(5, poll_seconds))


def run_step(step: Step) -> int:
    script_path = EXPERIMENTS / step.script
    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}", flush=True)
        return 127

    log_path = RESULTS_LOGS / f"{script_path.stem}.log"
    cmd = [str(PYTHON), "-u", str(script_path), *step.args]

    print("\n" + "=" * 80, flush=True)
    print(f"Running step: {step.name}", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)
    print(f"Log file: {log_path}", flush=True)
    print("=" * 80, flush=True)

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {step.name}\n")
        log_file.write("Command: " + " ".join(cmd) + "\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)

        return_code = proc.wait()
        log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] END {step.name} rc={return_code}\n")
        log_file.flush()

    print(f"Step finished: {step.name} (rc={return_code})", flush=True)
    return return_code


def build_steps(include_online: bool, include_grid: bool) -> list[Step]:
    steps = [
        Step("Phase1 Calibration", "eval_calibration_phase1.py"),
        Step("Best-of-N Evaluation", "eval_base.py"),
        Step("Instruct Sweep", "eval_instruct.py"),
        Step("ITI Sweep", "eval_iti.py"),
        Step("SelfCheck Evaluation", "eval_selfcheck.py"),
        Step("MedHallu Evaluation", "eval_medhallu.py"),
        Step("Low Threshold Smoke Test", "check_low_threshold.py"),
    ]

    if include_online:
        steps.append(Step("Online Cross-Model Comparison", "eval_online.py"))

    if include_grid:
        steps.append(Step("Legacy Grid Evaluation", "eval_grid.py"))

    return steps


def main() -> int:
    args = parse_args()
    continue_on_error = args.continue_on_error and not args.stop_on_error
    include_online = should_include_online(args.online)

    print("pipeline_after_n100 starting", flush=True)
    print(f"Using python: {PYTHON}", flush=True)
    print(f"Workspace root: {ROOT}", flush=True)
    print(f"Online step enabled: {include_online} (mode={args.online})", flush=True)
    print(f"State file: {STATE_PATH}", flush=True)

    if not args.skip_wait:
        wait_for_n100_completion(args.poll_seconds)
    else:
        print("Skipping wait for n100 completion (requested).", flush=True)

    steps = build_steps(include_online=include_online, include_grid=args.include_grid)
    state = load_state()

    completed = set(state.get("completed", []))
    failed = set(state.get("failed", []))

    for step in steps:
        if not args.force_rerun and step.name in completed:
            print(f"Skipping completed step: {step.name}", flush=True)
            continue

        rc = run_step(step)
        if rc == 0:
            completed.add(step.name)
            failed.discard(step.name)
        else:
            failed.add(step.name)
            if not continue_on_error:
                state["completed"] = sorted(completed)
                state["failed"] = sorted(failed)
                save_state(state)
                print("Stopping on error as requested.", flush=True)
                return rc

        state["completed"] = sorted(completed)
        state["failed"] = sorted(failed)
        save_state(state)

    state["completed"] = sorted(completed)
    state["failed"] = sorted(failed)
    save_state(state)

    if failed:
        print(f"Pipeline finished with failures: {sorted(failed)}", flush=True)
        return 1

    print("Pipeline finished successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
