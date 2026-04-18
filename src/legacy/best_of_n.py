"""
best_of_n.py
Generates N answers by temperature sampling and returns the most
self-consistent one via pairwise cosine similarity in embedding space.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from generate_base import model, tokenizer

scorer = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def sample_answer(prompt: str, temperature: float = 0.7,
                  max_new_tokens: int = 80) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


def best_of_n(prompt: str, n: int = 3, temperature: float = 0.7,
              max_new_tokens: int = 80) -> dict:
    candidates = [sample_answer(prompt, temperature, max_new_tokens)
                  for _ in range(n)]
    candidates = [c for c in candidates if len(c.strip()) > 5]
    if len(candidates) <= 1:
        return {"text": candidates[0] if candidates else "", "strategy": "fallback"}

    embeddings = scorer.encode(candidates, convert_to_tensor=True, device="cpu")
    scores = np.zeros(len(candidates))
    for i in range(len(candidates)):
        for j in range(len(candidates)):
            if i != j:
                scores[i] += float(util.cos_sim(embeddings[i], embeddings[j]).item())

    return {
        "text": candidates[int(np.argmax(scores))],
        "strategy": "best_of_n",
        "n_generated": len(candidates),
    }
