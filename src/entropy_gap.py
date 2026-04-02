"""
entropy_gap.py — Step 2B: Document what features we have vs. what's missing.

Key constraint: entropy needs the full vocab distribution (128,256 logits).
Our CSV only saved one logit per layer per candidate. The full distribution
was projected in eval.py but discarded before saving.
"""

import pandas as pd

df = pd.read_csv("trajectories_dataset.csv")

print("Columns available:", list(df.columns))
print()
print("What we HAVE for gating:")
print("  curvature  -- quadratic coefficient of logit trajectory [OK]")
print("  slope      -- linear trend across layers [OK]")
print("  stability  -- std of step-wise differences [OK]")
print()
print("What we are MISSING:")
print("  entropy    — needs full vocab distribution (128,256 values)")
print("               NOT in this CSV. Must extract live from model.")
print()

# Check what final_logit looks like — this is the only vocab-adjacent info
print("Sample final_logit values (NOT entropy, just one token's final logit):")
print(df["final_logit"].describe())
print()
print("Conclusion: entropy requires re-running the model with")
print("            output_hidden_states=True and projecting hidden states")
print("            through lm_head to get the full vocab distribution.")
