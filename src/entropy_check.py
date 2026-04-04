"""Quick entropy sanity-check on a small TruthfulQA instruct subset."""

from datasets import load_dataset
from generate_instruct import format_instruct_prompt, get_layer_logits_cached, compute_entropy
import numpy as np
from generate_instruct import tokenizer, model

def main() -> None:
    dataset = load_dataset("truthful_qa", "generation", split="validation")

    entropies = []
    for i, sample in enumerate(dataset.select(range(20))):
        prompt = format_instruct_prompt(sample["question"])
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        layer_logits, _ = get_layer_logits_cached(input_ids, None)
        H = compute_entropy(layer_logits[-1])
        entropies.append(H)
        print(f"  Q{i+1}: entropy={H:.3f} | {sample['question'][:50]}")

    arr = np.array(entropies)
    print(f"\nMean entropy: {arr.mean():.3f}")
    print(f"Max entropy:  {arr.max():.3f}")
    print(f"Min entropy:  {arr.min():.3f}")
    print(f"% above 3.5:  {np.mean(arr > 3.5):.1%}")
    print(f"% above 2.0:  {np.mean(arr > 2.0):.1%}")
    print(f"% above 1.0:  {np.mean(arr > 1.0):.1%}")
    print(f"% above 0.5:  {np.mean(arr > 0.5):.1%}")


if __name__ == "__main__":
    main()
