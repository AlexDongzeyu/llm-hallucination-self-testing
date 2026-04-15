#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def lin_r2(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=float)
    coef = np.polyfit(x, y, 1)
    y_hat = coef[0] * x + coef[1]
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-12:
        return 1.0
    return 1.0 - (ss_res / ss_tot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    prompts_seed = [
        "The capital of France is",
        "The speed of light is",
        "Water boils at",
        "The author of Hamlet is",
        "The largest planet is",
        "DNA stands for",
        "The first president was",
        "Gravity pulls at",
        "The symbol for gold is",
        "Humans have",
    ]
    prompts = (prompts_seed * ((args.n_prompts + len(prompts_seed) - 1) // len(prompts_seed)))[: args.n_prompts]

    if torch.cuda.is_available():
        # Allow TensorFloat-32 kernels for faster matmul on supported GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    kwargs = {"device_map": "auto"}
    if args.load_in_4bit:
        kwargs["load_in_4bit"] = True
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()

    n_layers = int(model.config.num_hidden_layers)
    start_layer = int(n_layers * 0.5)

    r2_scores = []
    entropies = []

    lm_head = model.lm_head

    i = 0
    batch_size = max(1, int(args.batch_size))
    while i < len(prompts):
        batch_prompts = prompts[i : i + batch_size]
        try:
            inp = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            inp = {k: v.to(model.device) for k, v in inp.items()}

            with torch.inference_mode():
                out = model(**inp, output_hidden_states=True)

            hidden = out.hidden_states
            bsz = hidden[-1].shape[0]
            seq_len = hidden[-1].shape[1]

            if "attention_mask" in inp:
                last_idx = (inp["attention_mask"].sum(dim=1) - 1).clamp(min=0)
            else:
                last_idx = torch.full((bsz,), seq_len - 1, dtype=torch.long, device=hidden[-1].device)

            b_idx = torch.arange(bsz, device=hidden[-1].device)
            trajs = [[] for _ in range(bsz)]

            for layer_idx in range(start_layer, n_layers):
                h = hidden[layer_idx][b_idx, last_idx, :]
                logits = lm_head(h)
                max_vals = torch.amax(logits, dim=-1).detach().cpu().numpy()
                for j, v in enumerate(max_vals):
                    trajs[j].append(float(v))

            for traj in trajs:
                y = np.asarray(traj, dtype=float)
                r2_scores.append(lin_r2(y))

            final_h = hidden[-1][b_idx, last_idx, :]
            final_logits = lm_head(final_h)
            probs = torch.softmax(final_logits, dim=-1)
            entropy_batch = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).detach().cpu().numpy()
            entropies.extend(float(x) for x in entropy_batch)

            i += bsz
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"OOM during calibration batch, retrying with batch_size={batch_size}")
                continue
            raise

    mean_r2 = float(np.mean(r2_scores))
    std_r2 = float(np.std(r2_scores))
    mean_entropy = float(np.mean(entropies))
    r2_cutoff = round(mean_r2 - 0.05, 3)

    payload = {
        "model": args.model,
        "n_layers": n_layers,
        "start_layer": start_layer,
        "r2_mean": round(mean_r2, 4),
        "r2_std": round(std_r2, 4),
        "r2_cutoff_used": r2_cutoff,
        "entropy_final_mean": round(mean_entropy, 4),
        "n_prompts": len(prompts),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
