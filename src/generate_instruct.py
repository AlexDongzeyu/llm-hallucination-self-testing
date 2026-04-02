"""
generate_instruct.py — Universal Dynamic Hallucination Reducer (UDHR)
Core generation module for Llama-3.2-3B-Instruct with:
  1. Correct chat template formatting (CRITICAL — fixes all base model failures)
  2. SLED + entropy gate
  3. Best-of-N self-consistency
  4. Dynamic routing (UDHR) — the novel contribution
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
print(f"Loaded | device={next(model.parameters()).device} "
      f"| layers={model.config.num_hidden_layers} "
      f"| vocab={model.config.vocab_size}")

# Head dimension — needed for ITI steering slice indexing
HEAD_DIM = model.config.hidden_size // model.config.num_attention_heads

scorer = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


# ── Chat template formatting ──────────────────────────────────────────────────
def format_instruct_prompt(question: str) -> str:
    """
    CRITICAL: instruct models require chat template.
    Without this, model behaves like base model — the root cause of all failures.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful, honest assistant. "
                       "Answer questions accurately and concisely."
        },
        {
            "role": "user",
            "content": question
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


# ── Core infrastructure ───────────────────────────────────────────────────────
def get_layer_logits_cached(input_ids, past_key_values=None):
    """KV-cached forward pass. Returns [num_layers, vocab] logits + updated cache."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True
        )
    hidden_states = outputs.hidden_states[1:]  # skip embedding layer
    norm    = model.model.norm
    lm_head = model.lm_head
    all_logits = []
    for h in hidden_states:
        last   = h[:, -1, :]
        logits = lm_head(norm(last)).squeeze(0).detach().cpu().float().numpy()
        all_logits.append(logits)
    return np.array(all_logits), outputs.past_key_values


def compute_entropy(logits: np.ndarray) -> float:
    """Shannon entropy of the output distribution. High = uncertain."""
    shifted = logits - np.max(logits)
    probs   = np.exp(shifted) / np.sum(np.exp(shifted))
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0))))


def apply_repetition_penalty(logits: np.ndarray,
                              generated_ids: list,
                              penalty: float = 1.3) -> np.ndarray:
    adjusted = logits.copy()
    for token_id in set(generated_ids):
        if adjusted[token_id] > 0:
            adjusted[token_id] /= penalty
        else:
            adjusted[token_id] *= penalty
    return adjusted


def compute_question_diagnostics(prompt_formatted: str):
    """
    One forward pass -> entropy + early-layer JSD.
    Used by UDHR router to decide which strategy to apply.
    Entropy: model confidence on first token
    JSD: how much early layers disagree with final layer
    """
    input_ids   = tokenizer.encode(prompt_formatted, return_tensors="pt").to(model.device)
    layer_logits, _ = get_layer_logits_cached(input_ids, None)
    final = layer_logits[-1]

    entropy = compute_entropy(final)

    # JSD between each of first 8 layers and final layer
    early_jsds = []
    for i in range(min(8, len(layer_logits) - 1)):
        p = np.exp(layer_logits[i] - np.max(layer_logits[i]));  p /= p.sum()
        q = np.exp(final - np.max(final));                       q /= q.sum()
        m = np.clip(0.5 * (p + q), 1e-10, 1.0)
        p = np.clip(p, 1e-10, 1.0);  q = np.clip(q, 1e-10, 1.0)
        jsd = (0.5 * np.sum(p * np.log(p / m)) +
               0.5 * np.sum(q * np.log(q / m)))
        early_jsds.append(float(jsd))

    return entropy, float(np.mean(early_jsds)), layer_logits


# ── Strategy 1: Greedy ────────────────────────────────────────────────────────
def greedy_generate(prompt_formatted: str, max_new_tokens: int = 80) -> str:
    """Pure greedy decoding -- baseline."""
    input_ids = tokenizer.encode(prompt_formatted, return_tensors="pt").to(model.device)
    layer_logits, past_kv = get_layer_logits_cached(input_ids, None)
    generated = []
    for _ in range(max_new_tokens):
        logits  = apply_repetition_penalty(layer_logits[-1], generated)
        next_id = int(np.argmax(logits))
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        next_t = torch.tensor([[next_id]]).to(model.device)
        layer_logits, past_kv = get_layer_logits_cached(next_t, past_kv)
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── Strategy 2: SLED + entropy gate ──────────────────────────────────────────
def sled_generate(prompt_formatted: str,
                  entropy_threshold: float = 3.5,
                  alpha: float = 0.3,
                  max_new_tokens: int = 80) -> dict:
    """
    SLED soft interpolation gated by entropy.
    When uncertain (high entropy), nudge logits toward early-layer average.
    Validated on base model: +2% at H=3.5.
    Expected to gain more on instruct model.
    """
    input_ids = tokenizer.encode(prompt_formatted, return_tensors="pt").to(model.device)
    layer_logits, past_kv = get_layer_logits_cached(input_ids, None)
    generated  = []
    gate_fired = 0

    for _ in range(max_new_tokens):
        final   = layer_logits[-1]
        entropy = compute_entropy(final)

        if entropy > entropy_threshold:
            early_avg = np.mean(layer_logits[:8], axis=0)
            final     = final + alpha * (early_avg - final)
            gate_fired += 1

        logits  = apply_repetition_penalty(final, generated)
        next_id = int(np.argmax(logits))
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        next_t = torch.tensor([[next_id]]).to(model.device)
        layer_logits, past_kv = get_layer_logits_cached(next_t, past_kv)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "text": text,
        "strategy": "sled",
        "gate_fire_rate": round(gate_fired / max(len(generated), 1), 4)
    }


# ── Strategy 3: Best-of-N ─────────────────────────────────────────────────────
def bon_generate(prompt_formatted: str,
                 n: int = 3,
                 temperature: float = 0.7,
                 max_new_tokens: int = 80) -> dict:
    """
    Sample N answers, return the most self-consistent one via cosine similarity.
    T=0.7 works on instruct models (unlike base models where T=0.7 -> garbage).
    """
    candidates = []
    for _ in range(n):
        input_ids = tokenizer.encode(prompt_formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        if len(text.strip()) > 5:
            candidates.append(text)

    if not candidates:
        return {"text": "", "strategy": "bon_fallback"}
    if len(candidates) == 1:
        return {"text": candidates[0], "strategy": "bon_single"}

    # Pairwise cosine -- most consistent answer wins
    embeddings = scorer.encode(candidates, convert_to_tensor=True, device="cpu")
    scores     = np.zeros(len(candidates))
    for i in range(len(candidates)):
        for j in range(len(candidates)):
            if i != j:
                scores[i] += float(util.cos_sim(embeddings[i], embeddings[j]).item())

    return {
        "text":     candidates[int(np.argmax(scores))],
        "strategy": "bon",
        "n_generated": len(candidates)
    }


# ── Strategy 3b: Semantic Majority Voting BoN ────────────────────────────────
def semantic_majority_bon(prompt_formatted: str,
                          n: int = 5,
                          temperature: float = 0.4,
                          max_new_tokens: int = 80) -> dict:
    """
    Semantic Majority Voting -- different from pairwise cosine BoN.

    Instead of picking the answer most similar to all others (which selects
    consistently wrong answers), we cluster by semantic meaning and pick
    the answer from the LARGEST cluster. This is genuine majority voting.

    Why T=0.4 not 0.7: instruct model needs low noise to stay coherent.
    Why n=5: need enough samples to form meaningful clusters.
    """
    candidates = []
    for _ in range(n):
        input_ids = tokenizer.encode(
            prompt_formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=True)
        if len(text.strip()) > 5:
            candidates.append(text)

    if not candidates:
        return {"text": "", "strategy": "smv_fallback"}
    if len(candidates) == 1:
        return {"text": candidates[0], "strategy": "smv_single"}

    # Embed all candidates
    embeddings = scorer.encode(
        candidates, convert_to_tensor=True, device="cpu")

    # Build similarity matrix
    sim_matrix = np.zeros((len(candidates), len(candidates)))
    for i in range(len(candidates)):
        for j in range(len(candidates)):
            sim_matrix[i][j] = float(
                util.cos_sim(embeddings[i], embeddings[j]).item())

    # Assign each candidate to a cluster (threshold=0.75 = same meaning)
    CLUSTER_THRESHOLD = 0.75
    cluster_id = [-1] * len(candidates)
    next_cluster = 0
    for i in range(len(candidates)):
        if cluster_id[i] == -1:
            cluster_id[i] = next_cluster
            for j in range(i + 1, len(candidates)):
                if sim_matrix[i][j] >= CLUSTER_THRESHOLD:
                    cluster_id[j] = next_cluster
            next_cluster += 1

    # Count cluster sizes, pick candidate from largest cluster
    from collections import Counter
    cluster_counts = Counter(cluster_id)
    majority_cluster = cluster_counts.most_common(1)[0][0]

    # From the majority cluster, pick the candidate with
    # highest average similarity to its cluster-mates
    majority_indices = [i for i, c in enumerate(cluster_id)
                        if c == majority_cluster]
    best_idx = majority_indices[0]
    if len(majority_indices) > 1:
        best_score = -1
        for i in majority_indices:
            score = np.mean([sim_matrix[i][j]
                             for j in majority_indices if j != i])
            if score > best_score:
                best_score = score
                best_idx = i

    return {
        "text": candidates[best_idx],
        "strategy": "semantic_majority_voting",
        "n_candidates": len(candidates),
        "n_clusters": next_cluster,
        "majority_cluster_size": cluster_counts.most_common(1)[0][1]
    }


# ── Strategy 4: UDHR — Universal Dynamic Hallucination Reducer ───────────────
def dynamic_generate(question: str, max_new_tokens: int = 80) -> dict:
    """
    The novel contribution: per-question routing using entropy + JSD.

    2x2 routing matrix (from experimental findings):
    +--------------------+----------------------+----------------------+
    |                    | Low JSD (<0.45)       | High JSD (>=0.45)    |
    +--------------------+----------------------+----------------------+
    | Low entropy (<3.5) | GREEDY               | SLED aggressive      |
    |                    | (confident+stable)   | (confident but layer |
    |                    |                      | signal says correct) |
    +--------------------+----------------------+----------------------+
    | High entropy (>=3.5)| BoN                 | SLED standard        |
    |                    | (uncertain, no layer | (uncertain + layer   |
    |                    | signal -> diversity) | signal -> correction)|
    +--------------------+----------------------+----------------------+

    Thresholds RECALIBRATED for instruct model (entropy_check.py measured):
      H=0.7  -> mean instruct entropy is 0.680; this gates ~40% of questions
      JSD=0.45 -> unchanged from Phase 1 (midpoint between 3B=0.44 and 8B=0.68)
    """
    prompt = format_instruct_prompt(question)

    # Diagnostic pass -- one forward call, ~same cost as first greedy step
    entropy, jsd, _ = compute_question_diagnostics(prompt)

    # Recalibrated for instruct model: mean entropy=0.680, so H=0.7 gates ~40%
    HIGH_ENTROPY = 0.7
    HIGH_JSD     = 0.45

    if entropy < HIGH_ENTROPY and jsd < HIGH_JSD:
        # Zone 1: Confident + stable -> greedy is optimal
        text     = greedy_generate(prompt, max_new_tokens)
        strategy = "greedy"

    elif entropy >= HIGH_ENTROPY and jsd >= HIGH_JSD:
        # Zone 2: Uncertain + layers diverge -> SLED correction
        result   = sled_generate(prompt, entropy_threshold=HIGH_ENTROPY,
                                 alpha=0.3, max_new_tokens=max_new_tokens)
        text     = result["text"]
        strategy = "sled"

    elif entropy >= HIGH_ENTROPY and jsd < HIGH_JSD:
        # Zone 3: Uncertain + no layer signal -> diversity selection
        result   = bon_generate(prompt, n=3, temperature=0.7,
                                max_new_tokens=max_new_tokens)
        text     = result["text"]
        strategy = "bon"

    else:
        # Zone 4: Confident + layers disagree -> aggressive SLED to align
        result   = sled_generate(prompt, entropy_threshold=0.5,
                                 alpha=0.4, max_new_tokens=max_new_tokens)
        text     = result["text"]
        strategy = "sled_aggressive"

    return {
        "text":     text,
        "strategy": strategy,
        "entropy":  round(entropy, 3),
        "jsd":      round(jsd, 3)
    }


# ── Strategy 5: ITI — Inference-Time Intervention ───────────────────────────
def iti_generate(prompt_formatted: str,
                 alpha: float = 15.0,
                 max_new_tokens: int = 80) -> dict:
    """
    ITI: Inference-Time Intervention (Li et al., 2023).
    Shifts top-K attention head activations along truth-correlated directions
    at every generation step. Requires pre-trained probes from iti_probe.py.

    Key architectural difference vs SLED/DoLA:
      SLED/DoLA: modify OUTPUT LOGITS (vocabulary distribution surface)
      ITI:       modify INTERMEDIATE ATTENTION ACTIVATIONS (internal belief)
    This is why ITI can fix false beliefs where SLED can only nudge outputs.

    alpha=15 from original paper. Higher = more truthful, potentially less fluent.
    """
    top_heads    = np.load("iti_top_heads.npy")    # [K, 2] int array
    head_vectors = np.load("iti_head_vectors.npy") # [layers, heads, head_dim]

    input_ids = tokenizer.encode(
        prompt_formatted, return_tensors="pt").to(model.device)

    generated = []

    for _ in range(max_new_tokens):
        hooks = []

        def make_iti_hook(layer_idx):
            def hook_fn(module, input, output):
                attn_out = output[0]  # [batch, seq, hidden_size]
                for (l, h) in top_heads:
                    if int(l) != layer_idx:
                        continue
                    direction = torch.tensor(
                        head_vectors[int(l), int(h)],
                        dtype=attn_out.dtype,
                        device=attn_out.device
                    )
                    head_start = int(h) * HEAD_DIM
                    head_end   = head_start + HEAD_DIM
                    # Steer last token position only
                    attn_out[0, -1, head_start:head_end] += alpha * direction
                return (attn_out,) + output[1:]
            return hook_fn

        for i, layer in enumerate(model.model.layers):
            h = layer.self_attn.register_forward_hook(make_iti_hook(i))
            hooks.append(h)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        for h in hooks:
            h.remove()

        logits  = outputs.logits[0, -1, :]
        next_id = int(torch.argmax(logits).item())
        generated.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([
            input_ids,
            torch.tensor([[next_id]]).to(model.device)
        ], dim=1)

    return {
        "text":     tokenizer.decode(generated, skip_special_tokens=True),
        "strategy": "iti",
        "alpha":    alpha
    }


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nSanity check on Llama-3.2-3B-Instruct")
    q = "What is the capital of France?"
    p = format_instruct_prompt(q)
    print(f"Formatted prompt (first 150 chars): {p[:150]}")

    result = dynamic_generate(q, max_new_tokens=40)
    print(f"\nQuestion: {q}")
    print(f"Answer:   {result['text']}")
    print(f"Strategy: {result['strategy']} "
          f"(entropy={result['entropy']}, jsd={result['jsd']})")
    print("\nExpected: 'Paris' routed to GREEDY (low entropy, confident answer)")
