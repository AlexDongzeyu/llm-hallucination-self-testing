import json
from evaluate import evaluate_config

configs = [
    ("Baseline (greedy)",           999.0, 999.0, 12, 0.0, "joint"),
    ("SLED + entropy gate (H=3.0)", 0.0,   3.0,  12, 0.3, "sled_entropy"),
    ("SLED + entropy gate (H=3.5)", 0.0,   3.5,  12, 0.3, "sled_entropy"),
    ("SLED + entropy gate (H=4.0)", 0.0,   4.0,  12, 0.3, "sled_entropy"),
]
N = 100

results = []
for label, ct, et, el, alpha, gmode in configs:
    print("\n" + "="*60, flush=True)
    print(f"Running: {label}", flush=True)
    print("="*60, flush=True)
    r = evaluate_config(label, ct, et, el, alpha, gate_mode=gmode, n_samples=N)
    results.append(r)
    print(f"  DONE -- acc={r['accuracy']:.1%} | fire={r['gate_fire_rate']:.1%} | rep={r['repetition_rate']:.1%}", flush=True)

with open("generation_results_n100_4configs.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("RESULTS TABLE (N=100, 4 requested configs)")
print("="*60)
print(f"{'Method':<35} {'Acc':>6} {'Fire%':>7} {'Rep%':>6}")
print("-"*60)
for r in results:
    print(f"{r['label']:<35} {r['accuracy']:>6.1%} {r['gate_fire_rate']:>7.1%} {r['repetition_rate']:>6.1%}")
