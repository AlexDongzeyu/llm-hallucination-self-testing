"""
test_low_thresholds.py -- Verify the gate fires with lower thresholds.
curve_threshold=0.01, entropy_threshold=2.0
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generate_base import gated_generate

result = gated_generate(
    "The capital of Canada is",
    max_new_tokens=20,
    curve_threshold=0.01,
    entropy_threshold=2.0
)

print(f"Generated text: {result['text']}")
print(f"Gate fire rate: {result['gate_fire_rate']:.1%}")
print()

print(f"{'Step':<5} {'Token':<15} {'Curvature':<12} {'Entropy':<10} {'Gate':<6} {'Before Gate':<15}")
print("-" * 65)
for m in result["metadata"]:
    gate_str = "FIRED" if m["gate_fired"] else "-"
    top_tok = m.get("top_token", "")
    changed = "*" if m["gate_fired"] and m["token"] != top_tok else ""
    print(f"{m['step']:<5} {repr(m['token']):<15} {m['curvature']:<12} "
          f"{m['entropy']:<10} {gate_str:<6} {repr(top_tok):<15} {changed}")
