"""Compute late-layer logit linearity (R²), curvature (κ), and ECR for any model.

Measures whether the DeLTa linearity assumption holds at a given model scale.
Run this for each model before setting tau_R2 / tau_ECR in router_thresholds.json.

Output: results/CANONICAL_v2/profile_{size}.json

Usage examples:
  python experiments/compute_logit_linearity.py \\
      --model meta-llama/Llama-3.2-3B-Instruct --n 50 \\
      --compute-curvature --compute-ecr \\
      --out results/CANONICAL_v2/profile_3b.json

  python experiments/compute_logit_linearity.py \\
      --model Qwen/Qwen2.5-32B-Instruct --load-in-4bit --n 50 \\
      --compute-curvature --compute-ecr \\
      --out results/CANONICAL_v2/profile_32b.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure late-layer logit linearity")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model id or local path",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use bitsandbytes 4-bit loading (for 32B on 80GB GPU)",
    )
    parser.add_argument("--n", type=int, default=50, help="Number of TruthfulQA prompts")
    parser.add_argument(
        "--start-layer-ratio",
        type=float,
        default=0.7,
        dest="start_layer_ratio",
        help="Fraction of layers to use as start of regression window (default: 0.7)",
    )
    parser.add_argument(
        "--end-layer-ratio",
        type=float,
        default=1.0,
        dest="end_layer_ratio",
        help="Fraction of layers to use as end of regression window (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        dest="top_k",
        help="Top-k final-layer tokens for R² aggregation",
    )
    parser.add_argument(
        "--compute-curvature",
        action="store_true",
        dest="compute_curvature",
        help="Also compute κ (quadratic-gain fraction) per question",
    )
    parser.add_argument(
        "--compute-ecr",
        action="store_true",
        dest="compute_ecr",
        help="Also compute ECR = H_final/H_peak per question (needs all layers)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON path (default: results/CANONICAL_v2/profile_<model>.json)",
    )
    return parser.parse_args()


def _safe_model_tag(model_name: str) -> str:
    return model_name.replace("/", "__").replace("-", "_").lower()


def _r2_linear(x: np.ndarray, y: np.ndarray) -> float:
    x_m = float(np.mean(x))
    y_m = float(np.mean(y))
    denom = float(np.sum((x - x_m) ** 2))
    if denom < 1e-12:
        return 0.0
    b1 = float(np.sum((x - x_m) * (y - y_m)) / denom)
    y_hat = y_m + b1 * (x - x_m)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y_m) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - ss_res / ss_tot))


def compute_r2_for_prompt(
    model: Any,
    tokenizer: Any,
    norm: Any,
    lm_head: Any,
    prompt: str,
    start_layer: int,
    end_layer: int,
    top_k: int,
    compute_curvature: bool,
    compute_ecr: bool,
) -> dict:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)

    # hidden_states[0] = embedding; [1:] = transformer layers
    hs = out.hidden_states[1:]
    n_layers = len(hs)

    # Regression window
    s = max(0, min(start_layer, n_layers - 1))
    e = max(s + 1, min(end_layer, n_layers))
    reg_hs = hs[s:e]

    layer_logits_list = []
    for h in reg_hs:
        logits = lm_head(norm(h[:, -1, :])).squeeze(0).detach().cpu().float().numpy()
        layer_logits_list.append(logits)
    logit_matrix = np.asarray(layer_logits_list, dtype=np.float32)  # (L, vocab)
    L = int(logit_matrix.shape[0])

    result: dict = {
        "n_layers_total": n_layers,
        "layer_start": s,
        "layer_end": e,
        "n_layers_used": L,
    }

    if L < 2:
        result.update({"mean_r2": 0.0, "median_r2": 0.0, "std_r2": 0.0})
        if compute_curvature:
            result["mean_kappa"] = 0.0
        if compute_ecr:
            result.update({"ecr": 0.0, "h_final": 0.0, "h_peak": 0.0})
        return result

    topk_idx = np.argsort(logit_matrix[-1])[-top_k:]
    x_n = np.arange(L, dtype=np.float32)
    x_n = (x_n - x_n.mean()) / max(float(x_n.std()), 1e-8)

    r2s: list[float] = []
    kappas: list[float] = []
    for tok in topk_idx:
        y = logit_matrix[:, int(tok)]
        ss_tot = float(np.var(y)) * L
        if ss_tot < 1e-8:
            continue

        # Linear fit
        A_lin = np.column_stack([x_n, np.ones(L)])
        b_l, _, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
        ss_lin = float(np.sum((y - A_lin @ b_l) ** 2))
        r2_lin = max(0.0, 1.0 - ss_lin / ss_tot)
        r2s.append(r2_lin)

        if compute_curvature and L >= 3:
            A_quad = np.column_stack([x_n ** 2, x_n, np.ones(L)])
            b_q, _, _, _ = np.linalg.lstsq(A_quad, y, rcond=None)
            ss_quad = float(np.sum((y - A_quad @ b_q) ** 2))
            r2_quad = max(0.0, 1.0 - ss_quad / ss_tot)
            kappas.append(max(0.0, (r2_quad - r2_lin) / (1.0 - r2_lin + 1e-8)))

    r2_arr = np.array(r2s) if r2s else np.array([0.0])
    result.update({
        "mean_r2": float(r2_arr.mean()),
        "median_r2": float(np.median(r2_arr)),
        "std_r2": float(r2_arr.std()),
    })

    if compute_curvature:
        result["mean_kappa"] = float(np.mean(kappas)) if kappas else 0.0

    if compute_ecr:
        # ECR needs ALL transformer layers, not just the regression window
        entropies = []
        for h in hs:
            lg = lm_head(norm(h[:, -1, :])).squeeze(0).detach().cpu().float()
            probs = torch.softmax(lg, dim=-1).numpy()
            H = float(-np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0))))
            entropies.append(H)
        H_final = entropies[-1]
        H_peak = max(entropies)
        result.update({
            "ecr": float(H_final / (H_peak + 1e-8)),
            "h_final": float(H_final),
            "h_peak": float(H_peak),
        })

    return result


# Type hint for model parameter — avoids importing torch at top for Any usage
from typing import Any


def format_prompt(tokenizer: Any, question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful, honest assistant. Answer questions accurately and concisely."},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"Question: {question}\nAnswer:"


if __name__ == "__main__":
    args = parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Build output path
    if args.out:
        out_path = Path(args.out)
    else:
        tag = _safe_model_tag(args.model)
        out_path = ROOT / "results" / "CANONICAL_v2" / f"profile_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # NOTE: For 4-bit models, device_map="auto" can dispatch some modules to CPU/disk and
    # transformers will raise unless fp32 CPU offload is explicitly enabled.
    # For profiling, we prefer "GPU-only" placement when possible.
    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                # If transformers decides to offload, allow it (prevents hard crash).
                llm_int8_enable_fp32_cpu_offload=True,
            )
            load_kwargs.pop("torch_dtype", None)
            # Strongly prefer GPU-only placement for 4-bit when VRAM allows.
            # Using an explicit device_map avoids the \"modules dispatched to CPU/disk\" error.
            load_kwargs["device_map"] = {"": 0}
            # Give the mapper an explicit budget (A800 40GB).
            load_kwargs["max_memory"] = {0: "39000MiB"}
            print("  Using 4-bit quantization (GPU-only device_map)", flush=True)
        except ImportError:
            print("  WARNING: bitsandbytes not installed, ignoring --load-in-4bit", flush=True)

    use_sdpa = os.environ.get("CURED_DISABLE_SDPA", "").strip() != "1"
    model = None
    if use_sdpa:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, attn_implementation="sdpa", **load_kwargs
            )
        except Exception as exc:
            print(
                f"  attn_implementation=sdpa failed ({type(exc).__name__}); using default attention",
                flush=True,
            )
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    n_layers = model.config.num_hidden_layers
    start_layer = int(n_layers * args.start_layer_ratio)
    end_layer = int(n_layers * args.end_layer_ratio)
    print(
        f"Loaded | device={next(model.parameters()).device} "
        f"| layers={n_layers} | regression window=[{start_layer},{end_layer})",
        flush=True,
    )

    # Access norm and lm_head (standard for Llama/Qwen/Mistral families)
    norm = model.model.norm
    lm_head = model.lm_head

    print(f"Loading TruthfulQA (n={args.n})...", flush=True)
    ds = load_dataset("truthful_qa", "generation", split="validation")

    rows: list[dict] = []
    n_use = min(args.n, len(ds))
    for i, sample in enumerate(ds.select(range(n_use))):
        prompt = format_prompt(tokenizer, sample["question"])
        r = compute_r2_for_prompt(
            model=model,
            tokenizer=tokenizer,
            norm=norm,
            lm_head=lm_head,
            prompt=prompt,
            start_layer=start_layer,
            end_layer=end_layer,
            top_k=args.top_k,
            compute_curvature=args.compute_curvature,
            compute_ecr=args.compute_ecr,
        )
        rows.append(r)
        ecr_str = f" ecr={r['ecr']:.3f}" if args.compute_ecr and "ecr" in r else ""
        kappa_str = f" kappa={r.get('mean_kappa', 0.0):.3f}" if args.compute_curvature else ""
        print(
            f"  [{i+1}/{n_use}] R2={r['mean_r2']:.3f}{kappa_str}{ecr_str} "
            f"layers={r['layer_start']}..{r['layer_end']}",
            flush=True,
        )

    payload = {
        "model": args.model,
        "n_questions": n_use,
        "start_layer_ratio": args.start_layer_ratio,
        "end_layer_ratio": args.end_layer_ratio,
        "top_k": args.top_k,
        "mean_r2": float(np.mean([r["mean_r2"] for r in rows])) if rows else 0.0,
        "median_r2": float(np.median([r["mean_r2"] for r in rows])) if rows else 0.0,
        "std_r2": float(np.std([r["mean_r2"] for r in rows])) if rows else 0.0,
    }
    if args.compute_curvature:
        payload["mean_kappa"] = float(np.mean([r.get("mean_kappa", 0.0) for r in rows]))
    if args.compute_ecr:
        payload["mean_ecr"] = float(np.mean([r.get("ecr", 0.0) for r in rows]))
        payload["mean_h_final"] = float(np.mean([r.get("h_final", 0.0) for r in rows]))
        payload["mean_h_peak"] = float(np.mean([r.get("h_peak", 0.0) for r in rows]))
    payload["per_question"] = rows

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n=== LINEARITY PROFILE: {args.model} ===", flush=True)
    print(f"  Mean R²:   {payload['mean_r2']:.3f}", flush=True)
    print(f"  Median R²: {payload['median_r2']:.3f}", flush=True)
    print(f"  Std R²:    {payload['std_r2']:.3f}", flush=True)
    if args.compute_curvature:
        print(f"  Mean κ:    {payload['mean_kappa']:.3f}", flush=True)
    if args.compute_ecr:
        print(f"  Mean ECR:  {payload['mean_ecr']:.3f}  (H_final={payload['mean_h_final']:.2f} / H_peak={payload['mean_h_peak']:.2f})", flush=True)
    print(f"  Saved → {out_path}", flush=True)
