import json, numpy as np
from scipy import stats

profile_r2 = {'3B': 0.501, '8B': 0.582, '14B': 0.444, '32B': 0.473}
alta_gains = {'3B': 2.5, '8B': 12.6, '14B': -6.6, '32B': 0.4}

scales = ['3B', '8B', '14B', '32B']
x = [profile_r2[s] for s in scales]
y = [alta_gains[s] for s in scales]
r, p = stats.pearsonr(x, y)

result = {
    "analysis": "scale-level R² profile vs ALTA TruthfulQA gain",
    "n_scales": 4,
    "scales": scales,
    "profile_r2": profile_r2,
    "alta_gains_pp": alta_gains,
    "pearson_r": round(r, 4),
    "pearson_p": round(p, 4),
    "note": "distinct from per-question R² analysis (r=0.039, p=0.58 in r2_stratified_analysis.json)"
}
import os
os.makedirs('results/CANONICAL_v2', exist_ok=True)
with open('results/CANONICAL_v2/r2_scale_correlation.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"Scale-level r={r:.4f}, p={p:.4f}")
