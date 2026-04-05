"""8B linearity comparison helper.

Groq APIs do not expose per-layer hidden states/logits needed for exact DeLTa-style
R2 trajectory fitting. This script provides a reproducible fallback workflow:
- report measured 3B R2 from local artifact
- print literature-backed 8B guidance for joint-paper comparison

If local 8B weights are available, use experiments/compute_logit_linearity.py with
that model instead of this placeholder.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R2_3B_PATH = ROOT / "results" / "logit_linearity_3b.json"


def main() -> None:
    print("8B linearity comparison helper (literature-backed mode)\n")

    if R2_3B_PATH.exists():
        payload = json.loads(R2_3B_PATH.read_text(encoding="utf-8"))
        print("Measured local 3B result:")
        print(f"- model: {payload.get('model', 'unknown')}")
        print(f"- n_questions: {payload.get('n_questions', 'unknown')}")
        print(f"- mean R2: {payload.get('mean_r2', 0.0):.4f}")
        print(f"- median R2: {payload.get('median_r2', 0.0):.4f}")
        print(f"- std R2: {payload.get('std_r2', 0.0):.4f}\n")
    else:
        print("No local 3B R2 artifact found at results/logit_linearity_3b.json\n")

    print("8B reference guidance for writing:")
    print("- DeLTa reports higher late-layer linearity for 7B+ models (paper figures).")
    print("- Use this as literature baseline when local 8B per-layer logits are unavailable.")
    print("- Keep metric spaces separate unless a shared evaluation metric is created.")


if __name__ == "__main__":
    main()
