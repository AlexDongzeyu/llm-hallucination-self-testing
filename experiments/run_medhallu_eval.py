"""
run_medhallu_eval.py
Compatibility entrypoint for MedHallu evaluation.

This now runs the corrected multiple-choice (candidate likelihood) evaluation
implemented in experiments/eval_medhallu.py.

By default it writes to the canonical MedHallu output file. If you need to
preserve legacy detector-style naming, pass --out explicitly.
"""

import argparse
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from eval_medhallu import main as run_mc_eval

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="", help="HF dataset id")
    parser.add_argument("--subset", type=str, default="", help="HF config/subset")
    parser.add_argument("--split", type=str, default="", help="split name")
    parser.add_argument("--n", type=int, default=50, help="number of usable samples")
    parser.add_argument("--alpha1", type=float, default=0.3, help="DeLTa blend weight")
    parser.add_argument("--alpha2", type=float, default=0.3, help="DoLa blend weight")
    parser.add_argument("--early-layer", type=int, default=7, dest="early_layer_idx")
    parser.add_argument("--mid-layer", type=int, default=14, dest="mid_layer_idx")
    parser.add_argument("--top-k", type=int, default=200, dest="top_k")
    parser.add_argument("--abstain-band", type=float, default=0.05)
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "medhallu_results.json"),
        help="output JSON path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_mc_eval(parse_args())
