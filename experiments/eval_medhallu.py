"""
eval_medhallu.py
MedHallu evaluation with true multiple-choice framing.

Each sample provides a question and two candidates:
  - Ground Truth
  - Hallucinated Answer

The model does not generate a free-form answer. Instead, we score both
candidates by average token log-likelihood and choose the higher-likelihood one.

Usage:
    python -u experiments/eval_medhallu.py
    python -u experiments/eval_medhallu.py --n 50 --alpha1 0.3 --alpha2 0.3
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generate_instruct import (
    compute_delta_dola_logits,
    format_instruct_prompt,
    get_layer_logits_cached,
    model,
    tokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="", help="HF dataset id")
    parser.add_argument("--subset", type=str, default="", help="HF config/subset")
    parser.add_argument("--split", type=str, default="", help="split name")
    parser.add_argument("--n", type=int, default=50, help="number of usable samples")
    parser.add_argument("--alpha1", type=float, default=0.3, help="DeLTa blend weight")
    parser.add_argument("--alpha2", type=float, default=0.3, help="DoLa blend weight")
    parser.add_argument("--early-layer", type=int, default=7, dest="early_layer_idx")
    parser.add_argument("--mid-layer", type=int, default=14, dest="mid_layer_idx")
    parser.add_argument("--top-k", type=int, default=200, dest="top_k")
    parser.add_argument(
        "--abstain-band",
        type=float,
        default=0.05,
        help="count margin ties when |score_gt-score_hallu| is below this value",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "medhallu_results.json"),
        help="output JSON path",
    )
    return parser.parse_args()


def try_load_dataset(dataset_id: str, subset: str, split: str):
    if subset:
        return load_dataset(dataset_id, subset, split=split)
    return load_dataset(dataset_id, split=split)


def load_medhallu_dataset(dataset_arg: str, subset_arg: str, split_arg: str):
    if dataset_arg:
        split = split_arg or "train"
        ds = try_load_dataset(dataset_arg, subset_arg, split)
        return ds, dataset_arg, subset_arg or None, split

    candidates = [
        ("UTAustin-AIHealth/MedHallu", "pqa_artificial", "train"),
        ("UTAustin-AIHealth/MedHallu", "pqa_labeled", "train"),
        ("medhallu", "", "test"),
        ("medhallu", "", "validation"),
        ("medhallu", "", "train"),
        ("openlifescienceai/medhallu", "", "test"),
        ("FreedomIntelligence/MedHallu", "", "test"),
        ("hirundo-io/medhallu", "default", "train"),
        ("Lizong/MedHallu", "pqa_artificial", "train"),
        ("Lizong/MedHallu", "pqa_labeled", "train"),
    ]

    errors = []
    for dataset_id, subset, split in candidates:
        try:
            ds = try_load_dataset(dataset_id, subset, split)
            return ds, dataset_id, subset or None, split
        except Exception as exc:
            errors.append(f"{dataset_id}:{subset}:{split} -> {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Could not load a MedHallu dataset automatically. "
        "Pass --dataset/--subset/--split explicitly.\n"
        + "\n".join(errors)
    )


def extract_question(sample: dict) -> str:
    for key in ["Question", "question", "query", "prompt", "input", "instruction", "claim"]:
        if key in sample and isinstance(sample[key], str) and sample[key].strip():
            return sample[key].strip()
    return ""


def _norm_log_probs(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    return shifted - np.log(np.sum(np.exp(shifted)))


def _candidate_avg_logprob(question: str, candidate: str, mode: str, cfg: dict) -> float:
    prompt = format_instruct_prompt(f"Question: {question}\nAnswer:")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    cand_ids = tokenizer.encode(" " + candidate.strip(), add_special_tokens=False)
    if not cand_ids:
        return -1e9

    full_ids = prompt_ids + cand_ids
    input_ids = torch.tensor([full_ids], device=model.device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states[1:]  # skip embedding layer
    norm = model.model.norm
    lm_head = model.lm_head
    prompt_len = len(prompt_ids)

    total_lp = 0.0
    for offset, tok_id in enumerate(cand_ids):
        pred_pos = prompt_len + offset - 1
        layer_logits = []
        for h in hidden_states:
            hs = h[:, pred_pos, :]
            logits = lm_head(norm(hs)).squeeze(0).detach().cpu().float().numpy()
            layer_logits.append(logits)
        layer_logits = np.array(layer_logits, dtype=np.float32)

        if mode == "delta_dola":
            step_logits = compute_delta_dola_logits(
                layer_logits,
                alpha1=cfg["alpha1"],
                alpha2=cfg["alpha2"],
                early_layer_idx=cfg["early_layer_idx"],
                mid_layer_idx=cfg["mid_layer_idx"],
                top_k=cfg["top_k"],
            )
        else:
            step_logits = layer_logits[-1]

        total_lp += float(_norm_log_probs(step_logits)[int(tok_id)])

    return total_lp / max(len(cand_ids), 1)


def evaluate_as_mc_chooser(dataset, label: str, mode: str, n: int, cfg: dict, abstain_band: float) -> dict:
    correct = 0
    seen = 0
    skipped = 0
    margins = []
    difficulty_counts = defaultdict(lambda: {"correct": 0, "total": 0})

    for sample in dataset:
        if seen >= n:
            break

        question = extract_question(sample)
        gt_answer = str(sample.get("Ground Truth", "") or "").strip()
        hallu_answer = str(sample.get("Hallucinated Answer", "") or "").strip()
        difficulty = str(sample.get("Difficulty Level", "unknown") or "unknown").strip().lower()

        if not question or not gt_answer or not hallu_answer:
            skipped += 1
            continue

        s_gt = _candidate_avg_logprob(question, gt_answer, mode, cfg)
        s_ha = _candidate_avg_logprob(question, hallu_answer, mode, cfg)
        margin = s_gt - s_ha

        is_correct = s_gt >= s_ha
        if is_correct:
            correct += 1

        difficulty_counts[difficulty]["total"] += 1
        if is_correct:
            difficulty_counts[difficulty]["correct"] += 1

        margins.append(float(margin))
        seen += 1

    if seen == 0:
        raise RuntimeError("No usable MedHallu rows found with Question/Ground Truth/Hallucinated Answer.")

    by_difficulty = {}
    for difficulty, counts in difficulty_counts.items():
        total_d = counts["total"]
        by_difficulty[difficulty] = {
            "n": total_d,
            "accuracy": round(counts["correct"] / max(total_d, 1), 4),
        }

    abstain_hits = sum(1 for m in margins if abs(m) < abstain_band)
    mean_margin = float(sum(margins) / len(margins)) if margins else 0.0

    return {
        "label": label,
        "mode": mode,
        "n_used": seen,
        "n_skipped": skipped,
        "accuracy": round(correct / max(seen, 1), 4),
        "mean_margin": round(mean_margin, 4),
        "abstain_band": abstain_band,
        "abstain_band_rate": round(abstain_hits / max(len(margins), 1), 4),
        "by_difficulty": by_difficulty,
    }


def main(args=None):
    args = args or parse_args()
    start = time.time()

    print("Loading MedHallu dataset...", flush=True)
    dataset, ds_id, ds_subset, ds_split = load_medhallu_dataset(args.dataset, args.subset, args.split)
    print(
        f"Loaded dataset={ds_id} subset={ds_subset} split={ds_split} rows={len(dataset)}",
        flush=True,
    )

    cfg = {
        "alpha1": args.alpha1,
        "alpha2": args.alpha2,
        "early_layer_idx": args.early_layer_idx,
        "mid_layer_idx": args.mid_layer_idx,
        "top_k": args.top_k,
    }

    strategies = [
        ("greedy_mc", "greedy"),
        (f"delta_dola_mc_a1{args.alpha1}_a2{args.alpha2}", "delta_dola"),
    ]

    results = []
    for label, mode in strategies:
        print(f"Running {label}...", flush=True)
        r = evaluate_as_mc_chooser(
            dataset,
            label=label,
            mode=mode,
            n=args.n,
            cfg=cfg,
            abstain_band=args.abstain_band,
        )
        results.append(r)
        print(
            f"  {label}: acc={r['accuracy']:.1%} margin={r['mean_margin']:+.3f} "
            f"|margin|<{args.abstain_band:.2f} on {r['abstain_band_rate']:.1%} "
            f"used={r['n_used']} skipped={r['n_skipped']}",
            flush=True,
        )

    payload = {
        "dataset": {
            "id": ds_id,
            "subset": ds_subset,
            "split": ds_split,
        },
        "n_target": args.n,
        "chooser": "multiple_choice_by_candidate_loglikelihood",
        "delta_dola": cfg,
        "results": results,
        "runtime_min": round((time.time() - start) / 60, 2),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nMedHallu MC evaluation complete.", flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved results to {out_path}", flush=True)
    return payload


if __name__ == "__main__":
    main()
