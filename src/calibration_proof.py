"""
calibration_proof.py
Run on each model by swapping generate.py MODEL_NAME, then:
  python calibration_proof.py

Measures:
  - ECE (Expected Calibration Error): does confidence align with accuracy?
  - Pearson r: does first-token entropy predict errors?
  - Reliability diagram + entropy-vs-correctness scatter (saved as PNG)
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from generate_base import model, tokenizer, get_layer_logits_cached, compute_entropy, gated_generate


dataset = load_dataset("truthful_qa", "generation", split="validation")
scorer  = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def run_calibration_proof(n_samples: int = 100, model_name: str = "model"):
    """
    Collects three things per question:
      1. First-token entropy (model uncertainty before generating)
      2. Whether the greedy answer was correct (cosine >= 0.65)
      3. Top-1 probability (confidence proxy for ECE)

    Computes:
      - ECE  : do confidence bins match accuracy bins?
      - r    : Pearson correlation between entropy and correctness
               Negative = higher entropy → more errors (calibrated)
               Near zero = entropy doesn't predict errors (overconfident)
      - Reliability diagram + scatter plot saved as PNG
    """
    entropies    = []
    top1_probs   = []
    correct_flags = []

    for i, sample in enumerate(dataset.select(range(n_samples))):
        prompt      = f"Q: {sample['question']}\nA:"
        best_answer = sample["best_answer"]

        # First-token logits (no generation yet)
        input_ids              = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        layer_logits, _        = get_layer_logits_cached(input_ids, None)
        final_logits           = layer_logits[-1]

        entropy                = compute_entropy(final_logits)

        shifted                = final_logits - np.max(final_logits)
        probs                  = np.exp(shifted) / np.sum(np.exp(shifted))
        top1_prob              = float(np.max(probs))

        # Pure greedy generation — no intervention
        result = gated_generate(
            prompt, max_new_tokens=80,
            curve_threshold=999.0,
            entropy_threshold=999.0,
            gate_mode="joint"
        )

        # Correctness via cosine similarity
        emb_gen  = scorer.encode(result["text"],  convert_to_tensor=True, device="cpu")
        emb_ref  = scorer.encode(best_answer,     convert_to_tensor=True, device="cpu")
        is_correct = float(util.cos_sim(emb_gen, emb_ref).item() >= 0.65)

        entropies.append(entropy)
        top1_probs.append(top1_prob)
        correct_flags.append(is_correct)

        if i % 10 == 0:
            print(f"  {i:>3}/{n_samples} | acc: {np.mean(correct_flags):.1%} "
                  f"| mean_H: {np.mean(entropies):.3f} | top1_p: {np.mean(top1_probs):.3f}")

    entropies     = np.array(entropies)
    top1_probs    = np.array(top1_probs)
    correct_flags = np.array(correct_flags)

    # ── ECE ─────────────────────────────────────────────────────────────────
    n_bins     = 10
    bin_edges  = np.linspace(0, 1, n_bins + 1)
    ece        = 0.0
    bin_data   = []

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask   = (top1_probs >= lo) & (top1_probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf   = float(top1_probs[mask].mean())
        bin_acc    = float(correct_flags[mask].mean())
        bin_weight = float(mask.mean())
        ece       += bin_weight * abs(bin_acc - bin_conf)
        bin_data.append((bin_conf, bin_acc, int(mask.sum())))

    # ── Pearson r: entropy → correctness ────────────────────────────────────
    correlation = float(np.corrcoef(entropies, correct_flags)[0, 1])

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
    if bin_data:
        bconfs, baccs, bcounts = zip(*bin_data)
        ax.scatter(bconfs, baccs, s=[c * 2 for c in bcounts],
                   color='royalblue', alpha=0.8, zorder=3)
        ax.plot(bconfs, baccs, 'b-o', lw=2,
                label=f'{model_name}\nECE={ece:.3f}')
    ax.set_xlabel('Top-1 Probability (Confidence)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Right: entropy vs correctness
    ax = axes[1]
    wrong_mask = correct_flags == 0
    right_mask = correct_flags == 1
    ax.scatter(entropies[wrong_mask], [0] * wrong_mask.sum(),
               alpha=0.3, color='red',   label='Wrong', s=20)
    ax.scatter(entropies[right_mask], [1] * right_mask.sum(),
               alpha=0.3, color='green', label='Correct', s=20)

    # Decile bin means to show trend
    ent_bins = np.percentile(entropies, np.linspace(0, 100, 11))
    for b in range(len(ent_bins) - 1):
        mask = (entropies >= ent_bins[b]) & (entropies < ent_bins[b + 1])
        if mask.sum() > 2:
            mid = (ent_bins[b] + ent_bins[b + 1]) / 2
            ax.plot(mid, correct_flags[mask].mean(), 'ko', markersize=8)

    ax.set_xlabel('First-Token Entropy')
    ax.set_ylabel('Correct (1) / Wrong (0)')
    ax.set_title(f'Entropy vs Correctness\nPearson r = {correlation:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Calibration Analysis: {model_name}', fontsize=14)
    plt.tight_layout()

    safe_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    out_path  = f'calibration_{safe_name}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    return {
        "model":                       model_name,
        "ece":                         round(ece, 4),
        "entropy_accuracy_correlation": round(correlation, 4),
        "mean_entropy":                round(float(entropies.mean()), 4),
        "mean_top1_prob":              round(float(top1_probs.mean()), 4),
        "accuracy":                    round(float(correct_flags.mean()), 4),
        "n_samples":                   n_samples,
    }


if __name__ == "__main__":
    import os
    model_label = getattr(model.config, "_name_or_path", "unknown")

    # Respect orchestrator override, else auto-detect: 50 for 8B, 100 for 3B-scale
    if "CALIB_N_SAMPLES" in os.environ:
        n = int(os.environ["CALIB_N_SAMPLES"])
    else:
        is_8b = (model.config.num_hidden_layers >= 32
                 and model.config.vocab_size >= 128000
                 and "3.1" in model_label)
        n = 50 if is_8b else 100

    print(f"\nRunning calibration proof: {model_label} | n_samples={n}")
    print("=" * 60)
    results = run_calibration_proof(n_samples=n, model_name=model_label)

    print("\n" + "=" * 60)
    print("CALIBRATION PROOF RESULTS")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:<35}: {v}")
