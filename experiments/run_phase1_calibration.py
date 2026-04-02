"""
run_phase1_overnight.py
Sequentially runs calibration_proof.py + diagnose_jsd.py on all 3 models.
Between each run it patches the model loading block in generate.py so the
next subprocess picks up the right model. Runs fully unattended overnight.

Run with:
    llm-env\\Scripts\\python.exe -u run_phase1_overnight.py
"""

import subprocess
import sys
import re
from pathlib import Path

PYTHON      = str(Path("llm-env/Scripts/python.exe").resolve())
GENERATE_PY = Path("generate.py")

# ── Model loading blocks ──────────────────────────────────────────────────────
# Each block replaces lines 14-32 in generate.py (everything up to "# ── 4A:")
# Anchors: starts at "import torch" (line 14), ends just before "# ── 4A:"

BLOCK_LLAMA3B = """\
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Model loading — Llama-3.2-3B ─────────────────────────────────────────────

MODEL_NAME = "meta-llama/Llama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

print(f"Model loaded | Device: {next(model.parameters()).device} | "
      f"Layers: {model.config.num_hidden_layers} | Vocab: {model.config.vocab_size}")

"""

BLOCK_QWEN = """\
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Model loading — Qwen2.5-3B ───────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-3B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

print(f"Model loaded | Device: {next(model.parameters()).device} | "
      f"Layers: {model.config.num_hidden_layers} | Vocab: {model.config.vocab_size}")

"""

BLOCK_LLAMA8B = """\
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ── Model loading — Llama-3.1-8B at 4-bit NF4 ────────────────────────────────

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

MODEL_NAME = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

print(f"Model loaded | Device: {next(model.parameters()).device} | "
      f"Layers: {model.config.num_hidden_layers} | Vocab: {model.config.vocab_size}")

"""

# ── Run config ────────────────────────────────────────────────────────────────
RUNS = [
    {"label": "Llama-3.2-3B",  "block": BLOCK_LLAMA3B, "n": 100},
    {"label": "Qwen-2.5-3B",   "block": BLOCK_QWEN,    "n": 100},
    {"label": "Llama-3.1-8B",  "block": BLOCK_LLAMA8B, "n": 50},
]


def patch_generate_py(new_block: str) -> None:
    """Replace the model loading section in generate.py (lines 14 up to '# ── 4A:')."""
    content = GENERATE_PY.read_text(encoding="utf-8")
    # Anchor: everything from 'import torch' through the blank line
    # before '# ── 4A:' is the loading block
    pattern = r"(import torch.*?)(?=\n# ── 4A:)"
    replacement = new_block.rstrip()
    new_content = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)
    if new_content == content:
        raise RuntimeError("Pattern not found in generate.py — cannot patch model block")
    GENERATE_PY.write_text(new_content, encoding="utf-8")
    print(f"  [orchestrator] generate.py patched OK")


def run_script(script: str, label: str, env: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Running {script} -- {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        [PYTHON, "-u", script],
        env=env,
        check=False
    )
    if result.returncode != 0:
        print(f"\n  [WARNING] {script} exited with code {result.returncode}")


if __name__ == "__main__":
    print("Phase 1 overnight calibration proof — all 3 models")
    print("="*60)

    for run in RUNS:
        label = run["label"]
        print(f"\n>>> Patching generate.py -> {label}")
        patch_generate_py(run["block"])

        # calibration_proof.py auto-detects n_samples from model size,
        # but we also pass it via environment so it can override
        import os
        env = os.environ.copy()
        env["CALIB_N_SAMPLES"] = str(run["n"])

        run_script("calibration_proof.py", label, env)
        run_script("diagnose_jsd.py",      label, env)

    print("\n" + "="*60)
    print("Phase 1 complete. Review PNGs and console output above.")
    print("="*60)
