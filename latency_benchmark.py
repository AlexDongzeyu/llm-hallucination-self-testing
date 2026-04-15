import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
QUESTIONS = [
    "What is the capital of Australia?",
    "Who invented the telephone?",
    "What causes rainbows?",
    "What is the boiling point of water?",
    "Who wrote Hamlet?",
] * 4  # 20 questions


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    results = {}
    for label, do_sample, temp in [("greedy", False, 1.0)]:
        times = []
        token_counts = []
        for q in QUESTIONS:
            inp = tok(q, return_tensors="pt").to("cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=80,
                    do_sample=do_sample,
                    temperature=temp,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            n_new = out.shape[1] - inp["input_ids"].shape[1]
            times.append(elapsed)
            token_counts.append(n_new)

        results[label] = {
            "mean_sec": round(sum(times) / len(times), 3),
            "mean_tok_per_sec": round(sum(token_counts) / sum(times), 1),
        }

    # Routing-dependent approximation suggested in plan text.
    results["CURED_estimated"] = {
        "mean_sec": round(results["greedy"]["mean_sec"] * 1.3, 3),
        "overhead_pct": "~30% (routing-dependent)",
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
