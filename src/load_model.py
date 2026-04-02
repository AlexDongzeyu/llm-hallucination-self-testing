"""
load_model.py -- Step 3A: Load Llama-3.2-3B for inference.

Matches eval.py settings:
  - bfloat16 (not float16) for consistency with stored CSV values
  - device_map="auto" for GPU placement
  - model.eval() to disable dropout
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-3B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model in bfloat16 to match eval.py (which generated the CSV)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()  # disables dropout for deterministic outputs

print("Model loaded")
print("Device:", next(model.parameters()).device)
print("Number of layers:", model.config.num_hidden_layers)  # should be 28
print("Vocab size:", model.config.vocab_size)                # should be 128256
