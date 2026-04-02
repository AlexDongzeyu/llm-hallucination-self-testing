"""
audit.py — Step 1: Understand the structure of trajectories_dataset.csv
before writing any new code.

Questions to answer:
  1. What does one row represent?
  2. How are the layer logits stored?
  3. Are the values raw logits or probabilities?
  4. What is the vocabulary dimension?
"""

import pandas as pd
import numpy as np

# ── Load the dataset ──
df = pd.read_csv("trajectories_dataset.csv")

# ── 1. Basic shape ──
print("=" * 60)
print("BASIC SHAPE")
print("=" * 60)
print(f"Rows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print()

# ── 2. Column names and types ──
print("=" * 60)
print("COLUMN NAMES AND TYPES")
print("=" * 60)
print(df.dtypes)
print()

# ── 3. All column names (one per line for clarity) ──
print("=" * 60)
print("ALL COLUMN NAMES")
print("=" * 60)
for col in df.columns:
    print(f"  - {col}")
print()

# ── 4. First 3 rows ──
print("=" * 60)
print("FIRST 3 ROWS")
print("=" * 60)
print(df.head(3).to_string())
print()

# ── 5. Missing values per column ──
print("=" * 60)
print("MISSING VALUES PER COLUMN")
print("=" * 60)
print(df.isnull().sum())
print()

# ── 6. First row (all fields) ──
print("=" * 60)
print("FULL FIRST ROW")
print("=" * 60)
print(df.iloc[0])
print()

# ── 7. Unique values for key categorical columns ──
print("=" * 60)
print("UNIQUE VALUE COUNTS")
print("=" * 60)
for col in ["prompt", "prompt_type", "candidate_token", "correct_or_wrong", "layer"]:
    if col in df.columns:
        print(f"  {col}: {df[col].nunique()} unique values")
print()

# ── 8. Layer range (helps answer: is each row one layer?) ──
if "layer" in df.columns:
    print("=" * 60)
    print("LAYER RANGE")
    print("=" * 60)
    print(f"  Min layer: {df['layer'].min()}")
    print(f"  Max layer: {df['layer'].max()}")
    print(f"  Unique layers: {sorted(df['layer'].unique())}")
    print()

# ── 9. Logit value range (raw logits or probabilities?) ──
if "logit" in df.columns:
    print("=" * 60)
    print("LOGIT VALUE RANGE (raw logits vs. probabilities)")
    print("=" * 60)
    print(f"  Min logit:  {df['logit'].min():.4f}")
    print(f"  Max logit:  {df['logit'].max():.4f}")
    print(f"  Mean logit: {df['logit'].mean():.4f}")
    print(f"  Std logit:  {df['logit'].std():.4f}")
    # If values go below 0 or above 1, they are raw logits (not probabilities)
    has_negative = (df['logit'] < 0).any()
    has_above_one = (df['logit'] > 1).any()
    if has_negative or has_above_one:
        print("  >> Conclusion: These are RAW LOGITS (not probabilities).")
    else:
        print("  >> Conclusion: Could be probabilities. Check if they sum to 1.")
    print()

# ── 10. Vocab dimension check ──
# This CSV does NOT store full distributions; it stores only the logit
# for one target candidate token per row. So vocab dimension is NOT
# represented here — each row has a single logit for one token.
print("=" * 60)
print("VOCABULARY DIMENSION NOTE")
print("=" * 60)
print("  This CSV stores one logit value per row (for a specific candidate")
print("  token), NOT a full distribution over the vocabulary.")
print("  Therefore, the vocab dimension (e.g. 32,000) is not directly")
print("  observable in this file.")
print()

print("=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
