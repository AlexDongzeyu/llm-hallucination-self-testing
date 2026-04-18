#!/usr/bin/env python3
"""Check A800 pipeline status and patch Phase 3 to preserve fixed thresholds."""
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


def run(client, cmd, timeout=30):
    _, o, e = client.exec_command(cmd, timeout=timeout)
    o.channel.recv_exit_status()
    return o.read().decode("utf-8", "replace").encode("ascii", "replace").decode("ascii").strip()


def main():
    print(f"Connecting to {USER}@{HOST}:{PORT}...")
    client = connect()
    print("Connected.\n")

    # 1. GPU status
    gpu = run(client, "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null")
    print(f"=== GPU ===\n{gpu}\n")

    # 2. Running processes
    procs = run(client, "ps aux | grep -E 'python|cured|pipeline' | grep -v grep | head -10")
    print(f"=== Running processes ===\n{procs}\n")

    # 3. Pipeline log
    log = run(client, f"tail -30 {REPO}/logs/pipeline.log 2>/dev/null || echo 'LOG NOT CREATED YET'")
    print(f"=== Pipeline log (last 30 lines) ===\n{log}\n")

    # 4. Results present
    results = run(client, f"ls {REPO}/results/CANONICAL_v2/*.json 2>/dev/null | wc -l")
    print(f"=== Existing result files: {results} ===\n")

    # 5. Check if Phase 3 calibration is still in pipeline script
    phase3_check = run(client, f"grep -c 'PHASE 3' {REPO}/run_full_pipeline.sh 2>/dev/null || echo 0")
    print(f"Phase 3 blocks in pipeline: {phase3_check}")

    if phase3_check.strip() != "0":
        print("\n=== Patching pipeline: replacing Phase 3 calibration with threshold preservation ===")
        # Replace the Phase 3 block: instead of running calibrate_router.py,
        # log that we're using the pre-fixed thresholds (tau_kappa=0.70, tau_ECR=0.04)
        patch_script = r"""
# Backup fixed thresholds before any potential overwrite
cp {repo}/configs/router_thresholds.json {repo}/configs/router_thresholds.json.bak

# Verify tau_kappa is correct (should be 0.7 from our fix)
KAPPA=$(python3 -c "import json; print(json.load(open('{repo}/configs/router_thresholds.json'))['tau_kappa'])" 2>/dev/null || echo "0.08")
if python3 -c "exit(0 if float('$KAPPA') >= 0.5 else 1)" 2>/dev/null; then
    log "  Thresholds OK: tau_kappa=$KAPPA (pre-calibrated, skipping decision-tree calibration)"
else
    log "  WARNING: tau_kappa=$KAPPA looks wrong, restoring fixed thresholds"
    cat > {repo}/configs/router_thresholds.json << 'THRESHOLD_EOF'
{{
  "_comment": "Manually calibrated: tau_kappa=0.70 (mean kappa=0.597@8B), tau_ECR=0.04 (mean ECR=0.031-0.076)",
  "tau_R2": 0.65,
  "tau_kappa": 0.70,
  "tau_ECR": 0.04,
  "tau_H_easy": 0.5,
  "tau_H_hard": 3.0,
  "tau_SC_easy": 0.90,
  "tau_SC_hard": 0.60,
  "beta1": 3.0,
  "beta2": 0.5,
  "beta3": 5.0,
  "beta4": 2.0,
  "profile_mean_r2": 0.582
}}
THRESHOLD_EOF
fi
""".replace("{repo}", REPO)

        # Write the replacement snippet to a temp file on server
        stdin, stdout, stderr = client.exec_command(f"cat > /tmp/phase3_patch.sh")
        stdin.write(patch_script)
        stdin.channel.shutdown_write()
        stdout.channel.recv_exit_status()

        # Use Python-based sed replacement (more reliable than sed on different platforms)
        # Read the pipeline script
        sftp = client.open_sftp()
        with sftp.file(f"{REPO}/run_full_pipeline.sh", "r") as f:
            pipeline_content = f.read().decode("utf-8")

        # Find the Phase 3 block and replace it
        old_phase3_marker = '# ── Phase 3: Threshold calibration ──'
        old_phase4_marker = '# ── Phase 4: Main publication experiments ──'

        if old_phase3_marker in pipeline_content and old_phase4_marker in pipeline_content:
            # Find where Phase 3 ends (start of Phase 4)
            p3_start = pipeline_content.find(old_phase3_marker)
            p4_start = pipeline_content.find(old_phase4_marker)

            new_phase3 = '''# ── Phase 3: Threshold verification (calibration replaced with pre-fixed values) ──
log "======================================================"
log "PHASE 3: Threshold verification (using pre-calibrated values)"
log "======================================================"
KAPPA=$(python3 -c "import json; print(json.load(open('configs/router_thresholds.json'))['tau_kappa'])" 2>/dev/null || echo "0.08")
if python3 -c "exit(0 if float('$KAPPA') >= 0.5 else 1)" 2>/dev/null; then
    log "  Thresholds OK: tau_kappa=$KAPPA tau_ECR=$(python3 -c \\"import json; print(json.load(open('configs/router_thresholds.json'))['tau_ECR'])\\" 2>/dev/null)"
    log "  Using pre-fixed thresholds (tau_kappa=0.70, tau_ECR=0.04, profile_mean_r2=0.582)"
else
    log "  WARN: tau_kappa=$KAPPA seems wrong. Restoring manually calibrated thresholds."
    python3 -c "
import json
cfg = {
  '_comment': 'Manually calibrated: tau_kappa=0.70, tau_ECR=0.04',
  'tau_R2': 0.65, 'tau_kappa': 0.70, 'tau_ECR': 0.04,
  'tau_H_easy': 0.5, 'tau_H_hard': 3.0,
  'tau_SC_easy': 0.90, 'tau_SC_hard': 0.60,
  'beta1': 3.0, 'beta2': 0.5, 'beta3': 5.0, 'beta4': 2.0,
  'profile_mean_r2': 0.582
}
json.dump(cfg, open('configs/router_thresholds.json', 'w'), indent=2)
print('Thresholds restored.')
"
fi
log "Phase 3 complete."

'''
            new_content = pipeline_content[:p3_start] + new_phase3 + pipeline_content[p4_start:]
            with sftp.file(f"{REPO}/run_full_pipeline.sh", "w") as f:
                f.write(new_content.encode("utf-8"))
            sftp.close()
            print("  Phase 3 patched: calibrate_router.py replaced with threshold verification.")
        else:
            sftp.close()
            print(f"  Could not find Phase 3/4 markers in pipeline. Skipping patch.")
            print(f"  Markers found: Phase3={'found' if old_phase3_marker in pipeline_content else 'NOT FOUND'}")
            print(f"               Phase4={'found' if old_phase4_marker in pipeline_content else 'NOT FOUND'}")
    else:
        print("Phase 3 already patched or not present.")

    # 6. Check thresholds on server
    thresh = run(client, f"python3 -c \"import json; d=json.load(open('{REPO}/configs/router_thresholds.json')); print('tau_kappa:', d['tau_kappa'], 'tau_ECR:', d['tau_ECR'])\"")
    print(f"\n=== Server thresholds ===\n{thresh}\n")

    # 7. Verify pipeline is still running
    running = run(client, "ps aux | grep run_full_pipeline | grep -v grep | head -3")
    print(f"=== Pipeline still running ===\n{running or 'NOT RUNNING'}\n")

    print("Done. Monitor with:")
    print(f"  tail -f {REPO}/logs/pipeline.log")

    client.close()


if __name__ == "__main__":
    main()
