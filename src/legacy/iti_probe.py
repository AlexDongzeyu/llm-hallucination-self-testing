"""
iti_probe.py — ITI Probe Training for Llama-3.2-3B-Instruct
Trains logistic regression probes on attention head activations to identify
truth-tracking attention heads. Uses TruthfulQA MC split questions 200-817
to avoid overlap with the generation split evaluation (questions 0-50).

Runtime: ~45-90 min depending on GPU speed.
Output: iti_head_scores.npy, iti_head_vectors.npy, iti_top_heads.npy
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()

# ── Architecture constants ────────────────────────────────────────────────────
N_LAYERS   = model.config.num_hidden_layers        # 28
N_HEADS    = model.config.num_attention_heads       # 24 query heads
N_KV_HEADS = model.config.num_key_value_heads      # 8 KV heads (GQA)
HEAD_DIM   = model.config.hidden_size // N_HEADS   # 128

print(f"Layers={N_LAYERS} | Q-heads={N_HEADS} | KV-heads={N_KV_HEADS} | head_dim={HEAD_DIM}")
if N_KV_HEADS != N_HEADS:
    print(f"  NOTE: GQA model (KV-heads={N_KV_HEADS} != Q-heads={N_HEADS})")
    print(f"  Attention OUTPUT is still [hidden_size={model.config.hidden_size}] — reshape OK")

# ── Data: use questions 200-817 (ZERO overlap with evaluation on 0-50) ────────
print("\nLoading TruthfulQA MC split (train on questions 200-817)...")
dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
train_dataset = dataset.select(range(200, len(dataset)))
print(f"Training samples: {len(train_dataset)} questions from idx 200-{len(dataset)-1}")


# ── Attention head activation extraction ─────────────────────────────────────
def get_attention_head_activations(prompt: str):
    """
    Extract attention output at the LAST token for every (layer, head).
    Returns shape: [num_layers, num_heads, head_dim]

    Hook target: layer.self_attn output[0] = [batch, seq, hidden_size]
    We take last token, reshape to [num_heads, head_dim].
    This works correctly even under GQA because the OUTPUT projection
    maps back to full hidden_size regardless of KV-head count.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    head_activations = []
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output[0]: [batch, seq, hidden_size]
            last_token = output[0][0, -1, :]        # [hidden_size]
            per_head   = last_token.view(N_HEADS, HEAD_DIM)
            head_activations.append(per_head.detach().cpu().float().numpy())
        return hook_fn

    for i, layer in enumerate(model.model.layers):
        h = layer.self_attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        model(input_ids=input_ids)

    for h in hooks:
        h.remove()

    return np.array(head_activations)  # [num_layers, num_heads, head_dim]


def format_mc_prompt(question: str, choice: str) -> str:
    """Format an MC answer-pair as an instruct prompt for activation extraction."""
    messages = [
        {"role": "system",  "content": "You are a helpful, honest assistant."},
        {"role": "user",    "content": question},
        {"role": "assistant", "content": choice}
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)


# ── Collect activations + labels ──────────────────────────────────────────────
print(f"\nExtracting activations...")
print(f"  Each question: all MC choices processed")
print(f"  Expected: {len(train_dataset)} questions x avg 4 choices ~= {len(train_dataset)*4} fwd passes\n")

all_activations = []   # [N_total, num_layers, num_heads, head_dim]
all_labels      = []   # 1 = truthful, 0 = hallucinated

for i, sample in enumerate(train_dataset):
    q      = sample["question"]
    mc     = sample["mc1_targets"]
    choices = mc["choices"]
    labels  = mc["labels"]

    for choice, label in zip(choices, labels):
        prompt = format_mc_prompt(q, choice)
        acts   = get_attention_head_activations(prompt)
        all_activations.append(acts)
        all_labels.append(label)

    if (i + 1) % 25 == 0:
        total_so_far = len(all_labels)
        print(f"  Q {i+1}/{len(train_dataset)} | total pairs so far: {total_so_far}", flush=True)

all_activations = np.array(all_activations)   # [N, layers, heads, head_dim]
all_labels      = np.array(all_labels)

print(f"\nCollected {len(all_labels)} total samples")
print(f"  Truthful (label=1): {all_labels.sum()}")
print(f"  Hallucinated (label=0): {(1 - all_labels).sum()}")
print(f"  Activation array shape: {all_activations.shape}")


# ── Train one probe per (layer, head) ─────────────────────────────────────────
print("\nTraining probes on all attention heads...")
head_scores  = np.zeros((N_LAYERS, N_HEADS))         # cross-val accuracy
head_vectors = np.zeros((N_LAYERS, N_HEADS, HEAD_DIM))  # steering directions

for layer in range(N_LAYERS):
    for head in range(N_HEADS):
        X = all_activations[:, layer, head, :]   # [N, head_dim]
        y = all_labels

        probe  = LogisticRegression(max_iter=1000, C=0.1)
        cv_acc = cross_val_score(probe, X, y, cv=5, n_jobs=-1).mean()
        head_scores[layer, head] = cv_acc

        probe.fit(X, y)
        direction = probe.coef_[0]
        head_vectors[layer, head] = direction / (np.linalg.norm(direction) + 1e-8)

    if (layer + 1) % 4 == 0:
        best_in_layer = head_scores[layer].max()
        best_head     = head_scores[layer].argmax()
        print(f"  Layer {layer+1:2d}/{N_LAYERS} | best head: #{best_head} acc={best_in_layer:.3f}",
              flush=True)


# ── Select top K heads by probe accuracy ─────────────────────────────────────
K = 20
flat_scores = head_scores.flatten()
top_k_flat  = np.argsort(flat_scores)[-K:]
top_k_heads = np.array([(idx // N_HEADS, idx % N_HEADS) for idx in top_k_flat])

print(f"\nTop {K} truthful attention heads:")
for layer, head in sorted(top_k_heads.tolist()):
    print(f"  Layer {layer:2d}, Head {head:2d} | probe acc = {head_scores[layer, head]:.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
np.save("iti_head_scores.npy",  head_scores)
np.save("iti_head_vectors.npy", head_vectors)
np.save("iti_top_heads.npy",    top_k_heads)

print("\nSaved:")
print("  iti_head_scores.npy  — probe accuracy per (layer, head)")
print("  iti_head_vectors.npy — steering directions per (layer, head)")
print("  iti_top_heads.npy    — top-20 [layer, head] index pairs")
print("\nProbe training complete. Next: run iti_generate evaluation.")
