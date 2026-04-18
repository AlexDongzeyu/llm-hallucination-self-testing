#!/usr/bin/env python3
"""Compare ECR-based vs semantic-entropy-based gate for deciding when to use CoVe.

Research question: does semantic entropy (output diversity) better predict when
CoVe is helpful compared to H_final (logit-space entropy proxy via ECR)?

Routing schemes compared on n questions:
  (a) ECR-based: route to CoVe if H_final > tau_H  (current Gate 5 logic)
  (b) SE-based:  route to CoVe if sem_entropy > tau_SE

Both CoVe routes are medical-only per the Issue #2 fix, so this ablation is most
meaningful on MedHallu. For TruthfulQA the baseline should be greedy for both.

Usage:
    python experiments/run_semantic_entropy_ablation.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --load-in-4bit \\
        --benchmark medhallu \\
        --n 50 --k 5 \\
        --out results/CANONICAL_v2/semantic_entropy_gate_comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from cured import (
    compute_semantic_entropy,
    cosine_match,
    cove_generate,
    detect_domain,
    format_prompt,
    get_model_device,
    greedy_generate,
    letter_match,
    load_medhallu_generation,
    load_model_and_tokenizer,
    load_truthfulqa,
)

DEFAULT_SCORER = "all-MiniLM-L6-v2"
DEFAULT_COSINE_THRESHOLD = 0.65


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semantic entropy vs ECR gate ablation")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument(
        "--benchmark", default="medhallu",
        choices=["medhallu", "truthfulqa"],
    )
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--k", type=int, default=5, help="Stochastic samples for semantic entropy")
    p.add_argument("--tau-H", type=float, default=3.0, help="H_final threshold for ECR-based route")
    p.add_argument("--tau-SE", type=float, default=0.5, help="Semantic entropy threshold")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scoring", default="cosine", choices=["cosine", "letter", "yesno"])
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument(
        "--out",
        default="results/CANONICAL_v2/semantic_entropy_gate_comparison.json",
    )
    return p.parse_args()


def score_response(scorer: SentenceTransformer, generated: str, reference: str, scoring: str) -> bool:
    if scoring == "letter":
        return letter_match(generated, reference)
    return cosine_match(scorer, generated, reference, threshold=DEFAULT_COSINE_THRESHOLD)


def load_questions(benchmark: str, n: int, seed: int) -> list[dict]:
    if benchmark == "truthfulqa":
        return load_truthfulqa(n, scoring="cosine")
    return load_medhallu_generation(n)


def compute_h_final(model: Any, tokenizer: Any, prompt: str) -> float:
    """Shannon entropy of the next-token distribution at the last input position."""
    dev = get_model_device(model)
    ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(ids, output_hidden_states=False)
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    return float(-torch.sum(probs * torch.log(probs + 1e-10)).item())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, load_in_4bit=args.load_in_4bit)

    print(f"Loading sentence scorer: {DEFAULT_SCORER}")
    scorer = SentenceTransformer(DEFAULT_SCORER)

    print(f"Loading benchmark: {args.benchmark} (n={args.n})")
    questions = load_questions(args.benchmark, args.n, args.seed)

    records = []
    for i, item in enumerate(questions):
        question = item["question"]
        reference = item.get("reference", "")
        domain = item.get("domain") or detect_domain(question)
        domain_medical = int(domain == "medical")
        prompt = format_prompt(tokenizer, question)

        print(f"  [{i+1}/{len(questions)}] Computing...", end=" ", flush=True)

        # Greedy baseline (generated once, reused for non-CoVe routes)
        greedy_text = greedy_generate(model, tokenizer, prompt, args.max_new_tokens)
        greedy_correct = int(score_response(scorer, greedy_text, reference, args.scoring))

        # H_final (next-token entropy proxy)
        H_final = compute_h_final(model, tokenizer, prompt)

        # Semantic entropy (k stochastic samples)
        sem_entropy = compute_semantic_entropy(
            model, tokenizer, prompt, scorer,
            k=args.k, max_new_tokens=args.max_new_tokens,
        )

        # ECR-based route: CoVe only if H_final > tau_H AND medical
        if H_final > args.tau_H and domain_medical:
            ecr_text = cove_generate(model, tokenizer, question, args.max_new_tokens)
            ecr_strategy = "cove"
        else:
            ecr_text = greedy_text
            ecr_strategy = "greedy"
        ecr_correct = int(score_response(scorer, ecr_text, reference, args.scoring))

        # SE-based route: CoVe only if sem_entropy > tau_SE AND medical
        if sem_entropy > args.tau_SE and domain_medical:
            se_text = cove_generate(model, tokenizer, question, args.max_new_tokens)
            se_strategy = "cove"
        else:
            se_text = greedy_text
            se_strategy = "greedy"
        se_correct = int(score_response(scorer, se_text, reference, args.scoring))

        records.append({
            "i": i,
            "domain": domain,
            "H_final": round(H_final, 4),
            "sem_entropy": round(sem_entropy, 4),
            "greedy_correct": greedy_correct,
            "ecr_strategy": ecr_strategy,
            "ecr_correct": ecr_correct,
            "se_strategy": se_strategy,
            "se_correct": se_correct,
        })
        print(
            f"H={H_final:.2f} SE={sem_entropy:.2f} "
            f"greedy={greedy_correct} ecr({ecr_strategy})={ecr_correct} "
            f"se({se_strategy})={se_correct}"
        )

    n = len(records)
    greedy_acc = sum(r["greedy_correct"] for r in records) / n
    ecr_acc = sum(r["ecr_correct"] for r in records) / n
    se_acc = sum(r["se_correct"] for r in records) / n

    summary = {
        "model": args.model,
        "benchmark": args.benchmark,
        "n": n,
        "k_stochastic": args.k,
        "tau_H": args.tau_H,
        "tau_SE": args.tau_SE,
        "greedy_accuracy": round(greedy_acc, 4),
        "ecr_based_accuracy": round(ecr_acc, 4),
        "se_based_accuracy": round(se_acc, 4),
        "ecr_gain_pp": round((ecr_acc - greedy_acc) * 100, 2),
        "se_gain_pp": round((se_acc - greedy_acc) * 100, 2),
        "ecr_cove_rate": round(sum(1 for r in records if r["ecr_strategy"] == "cove") / n, 4),
        "se_cove_rate": round(sum(1 for r in records if r["se_strategy"] == "cove") / n, 4),
        "per_question": records,
    }

    print(f"\n=== Results ===")
    print(f"  Greedy acc:    {greedy_acc:.1%}")
    print(f"  ECR-gate acc:  {ecr_acc:.1%}  ({summary['ecr_gain_pp']:+.1f} pp)")
    print(f"  SE-gate acc:   {se_acc:.1%}  ({summary['se_gain_pp']:+.1f} pp)")
    print(f"  ECR CoVe rate: {summary['ecr_cove_rate']:.1%}")
    print(f"  SE  CoVe rate: {summary['se_cove_rate']:.1%}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
