"""Fix the stuck 32B download (use hf-mirror.com) and restart cleanly."""
import paramiko, time

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('js4.blockelite.cn', port=14136, username='root', password='ra7ye9ka', timeout=20)

def run(cmd, timeout=30):
    _, o, e = client.exec_command(cmd, timeout=timeout)
    o.channel.recv_exit_status()
    return o.read().decode('utf-8', errors='replace').strip()

def p(text): print(text.encode('ascii', errors='replace').decode())

REPO = '/root/llm-hallucination-self-testing'

# 1. Kill all stuck 32B download processes
p('=== Killing stuck 32B download processes ===')
p(run('pkill -f "compute_logit_linearity.*32B" 2>/dev/null && echo killed || echo none'))
p(run('pkill -f "compute_logit_linearity.*Qwen2.5-32B" 2>/dev/null && echo killed || echo none'))
time.sleep(2)

# 2. Kill the pipeline orchestrators (we'll restart with the fix)
p('\n=== Killing pipeline orchestrators ===')
p(run('pkill -f run_full_pipeline 2>/dev/null && echo killed || echo none'))
time.sleep(2)

# 3. Verify everything is stopped
p('\n=== Remaining processes ===')
p(run('pgrep -a -f "run_full_pipeline\\|compute_logit" 2>/dev/null || echo all clear'))

# 4. Patch the pipeline script to add HF_ENDPOINT mirror before model downloads
p('\n=== Patching pipeline script with HF mirror ===')
patch_cmd = f"""
sed -i 's|#!/usr/bin/env bash|#!/usr/bin/env bash\\nexport HF_ENDPOINT=https://hf-mirror.com\\nexport HUGGINGFACE_HUB_VERBOSITY=warning|' {REPO}/run_full_pipeline.sh
"""
p(run(patch_cmd))

# Verify patch applied
p(run(f'head -5 {REPO}/run_full_pipeline.sh'))

# 5. Also install hf_xet for faster downloads
p('\n=== Installing hf_xet for faster HF downloads ===')
p(run(f'source {REPO}/llm-env/bin/activate && pip install hf_xet -q 2>&1 | tail -3', timeout=60))

# 6. Restart the pipeline fresh (it will skip completed phases)
p('\n=== Restarting pipeline with HF mirror ===')
log_path = f'{REPO}/logs/pipeline.log'
activate = f'source {REPO}/llm-env/bin/activate 2>/dev/null || true'
pipeline_cmd = f'cd {REPO} && {activate} && bash run_full_pipeline.sh'

# Use nohup
nohup_cmd = f'nohup bash -c \'{pipeline_cmd}\' >> {log_path} 2>&1 &'
p(run(nohup_cmd))
time.sleep(3)

p('\n=== New pipeline process ===')
p(run('pgrep -a -f run_full_pipeline 2>/dev/null || echo not started yet'))

p('\n=== Log tail (should show Phase 1 resuming) ===')
p(run(f'tail -8 {log_path}'))

client.close()
print('\nDONE')
