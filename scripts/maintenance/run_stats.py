from scipy import stats
import numpy as np

# Verified n=50 numbers. Fill n=100 counts when available.
data = {
    "TruthfulQA (n=50)": {"greedy": (35, 50), "cured": (37, 50)},
    "MedHallu (n=50)": {"greedy": (25, 50), "cured": (27, 50)},
    "TruthfulQA (n=100)": {"greedy": (None, 100), "cured": (None, 100)},
}

for bench, d in data.items():
    g_k, g_n = d["greedy"]
    c_k, c_n = d["cured"]
    if g_k is None:
        print(f"\n{bench}: PENDING - fill in when n=100 results land")
        continue

    p1, p2 = g_k / g_n, c_k / c_n
    p_pool = (g_k + c_k) / (g_n + c_n)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / g_n + 1 / c_n))
    z = (p2 - p1) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    print(f"\n{bench}: greedy={p1:.1%}, CURED={p2:.1%}, d=+{p2-p1:.1%}")
    print(
        f"  z={z:.3f}, p={p_val:.4f}  "
        f"{'OK p<0.05' if p_val < 0.05 else 'NOT significant'}"
    )
