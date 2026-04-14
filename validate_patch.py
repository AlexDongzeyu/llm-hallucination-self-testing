#!/usr/bin/env python3
"""
validate_patch.py — smoke-tests the patched cured.py without running any model.

Checks:
  1. cured.py parses as valid Python
  2. All new symbols exist at module level
  3. ALTA_R2_CUTOFF is 0.50
  4. mc_score_sample accepts mc_protocol kwarg
  5. _average_choice_log_prob_alta is defined
  6. _mc_proto logic is present in run_protocol
  7. CLI --scoring and --force-recalibrate flags exist
"""
import ast
import sys
from pathlib import Path

src_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cured.py")
if not src_path.exists():
    print(f"Error: {src_path} not found")
    sys.exit(1)

src = src_path.read_text(encoding="utf-8")
print(f"Validating: {src_path} ({len(src)} chars, {src.count(chr(10))} lines)")

ok = True

# 1. Valid Python syntax
try:
    tree = ast.parse(src)
    print("  ✓ Valid Python syntax")
except SyntaxError as e:
    print(f"  ✗ Syntax error: {e}")
    ok = False

# 2. ALTA_R2_CUTOFF value
if "ALTA_R2_CUTOFF = 0.50" in src:
    print("  ✓ ALTA_R2_CUTOFF = 0.50")
elif "ALTA_R2_CUTOFF = 0.55" in src:
    print("  ✗ ALTA_R2_CUTOFF still 0.55 — patch step 1 failed")
    ok = False
else:
    print("  ✗ ALTA_R2_CUTOFF not found")
    ok = False

# 3. _average_choice_log_prob_alta defined
if "def _average_choice_log_prob_alta(" in src:
    print("  ✓ _average_choice_log_prob_alta() defined")
else:
    print("  ✗ _average_choice_log_prob_alta() missing — patch step 2 failed")
    ok = False

# 4. mc_score_sample accepts mc_protocol
if "mc_protocol: str" in src:
    print("  ✓ mc_score_sample() has mc_protocol param")
else:
    print("  ✗ mc_protocol param missing — patch step 3 failed")
    ok = False

# 5. _mc_score_fn branching in mc_score_sample
if "_mc_score_fn = _average_choice_log_prob_alta if mc_protocol" in src:
    print("  ✓ MC1/MC2 scorer branches on mc_protocol")
else:
    print("  ✗ _mc_score_fn branch missing — patch steps 4-5 failed")
    ok = False

# 6. Strategy-faithful _mc_proto in run_protocol
if "_mc_proto" in src and "_alta_strategies" in src:
    print("  ✓ Strategy-faithful _mc_proto threading present in run_protocol")
else:
    print("  ✗ _mc_proto threading missing — patch step 6 failed")
    ok = False

# 7. CLI flags
if '"--scoring"' in src or "'--scoring'" in src:
    print("  ✓ --scoring CLI flag present")
else:
    print("  ✗ --scoring flag missing")
    ok = False

if '"--force-recalibrate"' in src or "'--force-recalibrate'" in src:
    print("  ✓ --force-recalibrate CLI flag present")
else:
    print("  ✗ --force-recalibrate flag missing")
    ok = False

# 8. mc_score_sample call site passes mc_protocol
if "mc_protocol=_mc_proto" in src:
    print("  ✓ mc_score_sample() call site passes mc_protocol=_mc_proto")
else:
    print("  ✗ mc_score_sample call site missing mc_protocol kwarg")
    ok = False

print()
if ok:
    print("✅ All checks passed — patch is ready.")
    sys.exit(0)
else:
    print("❌ Some checks failed — review unmatched patch steps above.")
    sys.exit(1)
