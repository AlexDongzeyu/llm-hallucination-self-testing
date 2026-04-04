"""
extract_logits.py -- Step 3B/3C: Extract full vocab logits at every layer
and compute entropy from the distribution.

IMPORTANT: Applies LayerNorm before lm_head projection (matching eval.py).
Without this, logits would be incorrect and the cross-check would fail.
"""

import torch
import numpy as np
import pandas as pd
from load_model import model, tokenizer


# ── 3B: Extract full vocab logits at every layer ─────────────────────────────

def get_layer_logits(prompt: str) -> np.ndarray:
    """
    Run one forward pass on a prompt.
    Returns full vocab logit distribution at every layer
    for the LAST token position (what the model predicts next).

    Skips embedding layer (index 0) to match the user's convention.
    Applies LayerNorm before lm_head to match eval.py.

    Output shape: [num_transformer_layers, vocab_size]
                = [28, 128256]
    """
    # Tokenize and move to model device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True  # gives hidden state at every layer
        )

    # outputs.hidden_states is a tuple of length num_layers + 1
    # Index 0 = embedding layer output (skip)
    # Index 1..28 = transformer layer outputs
    hidden_states = outputs.hidden_states[1:]  # length 28

    lm_head = model.lm_head     # linear: hidden_dim -> vocab_size
    norm = model.model.norm      # final LayerNorm (required before lm_head)

    all_layer_logits = []

    for layer_hidden in hidden_states:
        # layer_hidden shape: [batch=1, seq_len, hidden_dim]
        last_token = layer_hidden[:, -1, :]             # [1, hidden_dim]

        # Apply LayerNorm before projecting (matches eval.py behavior)
        normed = norm(last_token)
        logits = lm_head(normed)                         # [1, vocab_size]

        logits_np = logits.squeeze(0).detach().cpu().float().numpy()  # [vocab_size]
        all_layer_logits.append(logits_np)

    return np.array(all_layer_logits)  # [28, 128256]


# ── Quick verification ────────────────────────────────────────────────────────
test_prompt = "The capital of Canada is"

layer_logits = get_layer_logits(test_prompt)

print("Output shape:", layer_logits.shape)  # expect (28, 128256)
print("Final layer logits range:",
      layer_logits[-1].min().round(3),
      "to",
      layer_logits[-1].max().round(3))

# Top predicted token at final layer should make sense
top_token_id = np.argmax(layer_logits[-1])
top_token = tokenizer.decode([top_token_id])
print("Top predicted next token:", repr(top_token))  # should be " Ottawa"


# ── 3C: Compute entropy from the full distribution ───────────────────────────

def compute_entropy(logits_1d: np.ndarray) -> float:
    """
    Shannon entropy of the probability distribution at one layer.

    Higher entropy = model spread across many tokens = uncertain
    Lower entropy  = model confident in one token

    Args:
        logits_1d: shape [vocab_size] -- raw logits, NOT probabilities

    Returns:
        Scalar entropy value
    """
    # Numerically stable softmax (subtract max to prevent overflow)
    shifted = logits_1d - np.max(logits_1d)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))

    # Shannon entropy: H = -sum(p * log(p)), clip to avoid log(0)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0)))

    return float(entropy)


# ── Entropy sanity checks ────────────────────────────────────────────────────
print()
print("=" * 50)
print("ENTROPY SANITY CHECKS")
print("=" * 50)

# Check 1: uniform distribution = maximum entropy
uniform_logits = np.zeros(128256)
uniform_entropy = compute_entropy(uniform_logits)
max_possible_entropy = np.log(128256)  # ln(vocab_size) ~ 11.76
print(f"Uniform entropy:       {uniform_entropy:.4f}")
print(f"Max possible entropy:  {max_possible_entropy:.4f}")
print(f"Match: {np.isclose(uniform_entropy, max_possible_entropy, atol=1e-3)}")
print()

# Check 2: peaked distribution = near-zero entropy
peaked_logits = np.zeros(128256)
peaked_logits[0] = 100.0
peaked_entropy = compute_entropy(peaked_logits)
print(f"Peaked entropy (should be ~0): {peaked_entropy:.6f}")
print()

# Check 3: actual model output entropy
final_logits = layer_logits[-1]  # shape: [128256]
real_entropy = compute_entropy(final_logits)
print(f"Entropy on '{test_prompt}': {real_entropy:.4f}")
print("(Expect somewhere between 2.0 and 6.0 for a normal prompt)")


# ── Cross-check against CSV ──────────────────────────────────────────────────
print()
print("=" * 50)
print("CROSS-CHECK AGAINST CSV")
print("=" * 50)

df = pd.read_csv("data/trajectories_dataset.csv")

# Find the stored final_logit for Ottawa on this prompt
stored_row = df[
    (df["prompt"].str.contains("capital of Canada")) &
    (df["candidate_token"].str.contains("Ottawa")) &
    (df["prompt_type"] == "original") &
    (df["layer"] == 28)
]

if len(stored_row) > 0:
    stored_final_logit = stored_row["final_logit"].values[0]

    # Get the live logit for Ottawa's token ID at the final layer
    ottawa_token_id = tokenizer.encode(" Ottawa", add_special_tokens=False)[0]
    live_final_logit = layer_logits[-1][ottawa_token_id]

    print(f"Stored final logit for Ottawa:  {stored_final_logit:.4f}")
    print(f"Live final logit for Ottawa:    {live_final_logit:.4f}")
    print(f"Match: {np.isclose(stored_final_logit, live_final_logit, atol=0.5)}")
    # Note: atol=0.5 because eval.py used bfloat16 which has limited precision.
    # The CSV was generated layer 28 = hidden_states[28] (with embedding at 0),
    # while this script uses hidden_states[1:][-1] = hidden_states[28].
    # Both refer to the same final transformer layer output.
else:
    print("Could not find matching row in CSV for cross-check.")
