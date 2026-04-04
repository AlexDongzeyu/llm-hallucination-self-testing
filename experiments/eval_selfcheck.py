"""
eval_selfcheck.py
SelfCheck-style hallucination detection on TruthfulQA using local
Llama-3.2-3B-Instruct generations.

Idea:
1) Generate one deterministic draft answer (greedy).
2) Generate K stochastic samples for the same question.
3) Use semantic agreement between draft and sampled answers as confidence.
4) Flag likely hallucination if confidence is below threshold.

Usage:
    python -u experiments/eval_selfcheck.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generate_instruct import format_instruct_prompt, greedy_generate, model, tokenizer


N = 50
K_SAMPLES = 4
SIM_THRESHOLD = 0.70
REF_THRESHOLD = 0.65

print("Loading TruthfulQA...", flush=True)
dataset = load_dataset("truthful_qa", "generation", split="validation")

print("Loading sentence scorer...", flush=True)
scorer = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def score_to_reference(generated: str, reference: str, threshold: float = REF_THRESHOLD) -> bool:
    eg = scorer.encode(generated, convert_to_tensor=True, device="cpu")
    er = scorer.encode(reference, convert_to_tensor=True, device="cpu")
    return util.cos_sim(eg, er).item() >= threshold


def has_repetition(text: str, window: int = 5) -> bool:
    words = text.lower().split()
    if len(words) < max(window * 2, 20):
        return False
    ngrams = [tuple(words[i:i + window]) for i in range(len(words) - window + 1)]
    return len(ngrams) != len(set(ngrams))


def sampled_answer(prompt_formatted: str,
                   max_new_tokens: int = 80,
                   temperature: float = 0.7,
                   top_p: float = 0.95) -> str:
    input_ids = tokenizer.encode(prompt_formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


def selfcheck_confidence(draft: str, samples: list[str]) -> float:
    clean = [s for s in samples if s and len(s.strip()) > 0]
    if not draft.strip() or not clean:
        return 0.0
    embs = scorer.encode([draft] + clean, convert_to_tensor=True, device="cpu")
    sims = [float(util.cos_sim(embs[0], embs[i]).item()) for i in range(1, len(embs))]
    return float(np.mean(sims)) if sims else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N, help="Number of evaluation samples")
    parser.add_argument("--k", type=int, default=K_SAMPLES, help="Number of stochastic samples")
    parser.add_argument("--sim-threshold", type=float, default=SIM_THRESHOLD)
    parser.add_argument("--ref-threshold", type=float, default=REF_THRESHOLD)
    args = parser.parse_args()

    N = args.n
    K_SAMPLES = args.k
    SIM_THRESHOLD = args.sim_threshold
    REF_THRESHOLD = args.ref_threshold

    t0 = time.time()

    tp = fp = tn = fn = 0
    total_correct = 0
    total_rep = 0
    confidences = []

    for i, sample in enumerate(dataset.select(range(N))):
        question = sample["question"]
        best = sample["best_answer"]

        prompt = format_instruct_prompt(question)
        draft = greedy_generate(prompt, max_new_tokens=80)

        samples = [sampled_answer(prompt) for _ in range(K_SAMPLES)]
        conf = selfcheck_confidence(draft, samples)
        confidences.append(conf)

        rep = has_repetition(draft)
        correct = (not rep) and score_to_reference(draft, best, threshold=REF_THRESHOLD)

        total_rep += int(rep)
        total_correct += int(correct)

        # Positive class = hallucination/incorrect answer.
        pred_hall = conf < SIM_THRESHOLD
        true_hall = not correct

        if pred_hall and true_hall:
            tp += 1
        elif pred_hall and not true_hall:
            fp += 1
        elif (not pred_hall) and (not true_hall):
            tn += 1
        else:
            fn += 1

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N - i - 1)
            print(
                f"[{i+1}/{N}] qa_acc={total_correct/(i+1):.1%} | "
                f"rep={total_rep/(i+1):.1%} | det_acc={(tp+tn)/(i+1):.1%} | "
                f"eta={eta/60:.0f}min",
                flush=True,
            )

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    det_acc = (tp + tn) / N

    summary = {
        "dataset": "truthful_qa:generation:validation",
        "n": N,
        "k_samples": K_SAMPLES,
        "selfcheck_similarity_threshold": SIM_THRESHOLD,
        "reference_threshold": REF_THRESHOLD,
        "qa_accuracy": round(total_correct / N, 4),
        "qa_rep_rate": round(total_rep / N, 4),
        "detector_accuracy": round(det_acc, 4),
        "detector_precision": round(precision, 4),
        "detector_recall": round(recall, 4),
        "detector_f1": round(f1, 4),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "runtime_min": round((time.time() - t0) / 60, 2),
        "counts": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
    }

    out_path = ROOT / "results" / "selfcheck_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSelfCheck evaluation complete.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved results to {out_path}", flush=True)
