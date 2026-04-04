"""
verify_features.py — Step 2A: Verify that curvature, slope, and stability
stored in data/trajectories_dataset.csv match a manual recomputation.

If any "Match" line prints False, there's a bug in the original eval.py
that will corrupt everything downstream.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("data/trajectories_dataset.csv")

# ── Pick ONE specific example to verify manually ──
# Isolate one prompt + one candidate + one prompt type = 29 rows in order
example = df[
    (df["prompt"] == df["prompt"].iloc[0]) &
    (df["candidate_token"] == df["candidate_token"].iloc[0]) &
    (df["prompt_type"] == "original")
].sort_values("layer")

print("Number of layers found:", len(example))  # should be 29
print()

# The logit trajectory for this candidate across all 29 layers
trajectory = example["logit"].values
print("Trajectory (logits across layers 0-28):")
print(trajectory)
print()

# ── Recompute curvature manually and compare to stored value ──
# eval.py uses np.polyfit(x, layer_logits, 2)[0] — the 'a' in ax^2 + bx + c
layers = np.arange(len(trajectory))
coefficients = np.polyfit(layers, trajectory, deg=2)
quadratic_coef = coefficients[0]  # the 'a' in ax^2 + bx + c

stored_curvature = example["curvature"].iloc[0]
print(f"Recomputed quadratic curvature: {quadratic_coef:.6f}")
print(f"Stored curvature in CSV:        {stored_curvature:.6f}")
print(f"Match: {np.isclose(quadratic_coef, stored_curvature, atol=1e-4)}")
print()

# ── Recompute slope ──
# eval.py uses np.polyfit(x, layer_logits, 1)[0]
stored_slope = example["slope"].iloc[0]
recomputed_slope = np.polyfit(layers, trajectory, deg=1)[0]
print(f"Recomputed slope: {recomputed_slope:.6f}")
print(f"Stored slope:     {stored_slope:.6f}")
print(f"Match: {np.isclose(recomputed_slope, stored_slope, atol=1e-4)}")
print()

# ── Recompute stability ──
# eval.py uses np.std(np.diff(layer_logits))
stored_stability = example["stability"].iloc[0]
recomputed_stability = np.std(np.diff(trajectory))
print(f"Recomputed stability: {recomputed_stability:.6f}")
print(f"Stored stability:     {stored_stability:.6f}")
print(f"Match: {np.isclose(recomputed_stability, stored_stability, atol=1e-4)}")
