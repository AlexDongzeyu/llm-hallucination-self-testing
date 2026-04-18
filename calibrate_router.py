"""calibrate_router.py — Learn routing thresholds from Phase 2 ablation data.

Reads all ablation_*.json files from results/CANONICAL_v2/, pivots per-question
correctness to find which protocol wins per question, trains a shallow decision tree,
and extracts the root + depth-2 splits as calibrated threshold values.

Production Phase 4 thresholds in ``configs/router_thresholds.json`` were finalized
with manual inspection; ``tau_kappa`` / ``tau_ECR`` were corrected from broken
defaults (0.08 / 0.10) — see the CRITICAL BUG HISTORY header in ``cured.py``.

IMPORTANT: Manual review of the printed tree is required for deeper nodes.
The auto-extracted thresholds cover only the root and depth-2 splits.

Usage:
  python calibrate_router.py \\
      --results-dir results/CANONICAL_v2 \\
      --pattern "ablation_*.json" \\
      --out configs/router_thresholds.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text


FEATURE_COLS = ["r2_q", "var_r2_q", "kappa_q", "ecr_q", "h_final", "domain_medical"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate CURED router thresholds from ablation data")
    parser.add_argument("--results-dir", default="results/CANONICAL_v2")
    parser.add_argument("--pattern", default="ablation_*.json")
    parser.add_argument("--out", default="configs/router_thresholds.json")
    return parser.parse_args()


def load_ablation_files(results_dir: str, pattern: str) -> pd.DataFrame:
    """Load all ablation JSON files and flatten per_question records."""
    records: list[dict] = []
    fpaths = glob.glob(str(Path(results_dir) / pattern))
    if not fpaths:
        raise FileNotFoundError(
            f"No files matching {pattern!r} in {results_dir!r}. "
            "Run Phase 2 ablations first."
        )

    for fpath in sorted(fpaths):
        fname = Path(fpath).stem  # e.g. ablation_8b_alta_truthfulqa_n200
        parts = fname.split("_")
        # Expected format: ablation_{size}_{protocol}_{benchmark}_n{n}
        if len(parts) < 4:
            continue
        model_size = parts[1]
        protocol = parts[2]
        benchmark = "_".join(parts[3:-1]) if len(parts) > 4 else parts[3]

        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        # per_question is either at top-level or nested under results
        pq_list: list[dict] = []
        if "per_question" in data:
            pq_list = data["per_question"]
        else:
            for bench_data in data.get("results", {}).values():
                for proto_data in bench_data.values():
                    if isinstance(proto_data, dict) and "per_question" in proto_data:
                        pq_list.extend(proto_data["per_question"])

        for q in pq_list:
            records.append({
                "q_id": q.get("q_id", q.get("i", -1)),
                "model_size": model_size,
                "benchmark": benchmark,
                "protocol": protocol,
                "correct": int(q.get("correct") or 0),
                "r2_q": q.get("r2_q"),
                "var_r2_q": q.get("var_r2_q"),
                "kappa_q": q.get("kappa_q"),
                "ecr_q": q.get("ecr_q"),
                "h_final": q.get("h_final"),
                "sc_q": q.get("sc_q"),
                "domain_medical": q.get("domain_medical"),
            })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} per-question records from {len(fpaths)} files.")
    return df


def pivot_best_protocol(df: pd.DataFrame) -> pd.DataFrame:
    """For each (q_id, model_size, benchmark), find the protocol that got it right."""
    pivot = df.pivot_table(
        index=["q_id", "model_size", "benchmark"],
        columns="protocol",
        values="correct",
        aggfunc="first",
    ).reset_index()

    available_protocols = [c for c in ["greedy", "alta", "cove", "iti"] if c in pivot.columns]
    if not available_protocols:
        raise ValueError("No protocol columns found in pivot table.")

    # Best protocol: first correct one in priority order, or greedy as fallback
    def best_proto(row: pd.Series) -> str:
        for proto in ["alta", "iti", "cove", "greedy"]:
            if proto in row and row[proto] == 1:
                return proto
        return "greedy"

    pivot["best_protocol"] = pivot[available_protocols].apply(best_proto, axis=1)
    return pivot


def train_and_print_tree(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> DecisionTreeClassifier:
    """Train a shallow decision tree and print it for manual threshold inspection."""
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    clf.fit(X, y)

    print(f"\nCross-val accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"Classes: {list(clf.classes_)}")
    print("\n=== DECISION TREE (read splits to set thresholds) ===")
    print(export_text(clf, feature_names=feature_names))
    print("=== END TREE ===\n")
    print("# Manual review required — inspect the tree above and update")
    print("# configs/router_thresholds.json accordingly for splits beyond depth 2.\n")
    return clf


def extract_thresholds_from_tree(
    clf: DecisionTreeClassifier,
    feature_names: list[str],
) -> dict[str, float]:
    """Auto-extract root and depth-2 node thresholds only.

    A max_depth=4 tree has up to 15 internal nodes. Mapping all splits to named
    tau values is error-prone. Only the root (most predictive) and its immediate
    children (depth-2) are auto-extracted here. Inspect the full tree printout
    for deeper nodes and update configs/router_thresholds.json manually.
    """
    tree = clf.tree_
    thresholds: dict[str, float] = {}

    # Root node (index 0) — most important split
    if tree.feature[0] >= 0:  # not a leaf
        root_feat = feature_names[tree.feature[0]]
        thresholds[f"tau_{root_feat}_root"] = float(tree.threshold[0])
        print(f"[auto] Root split: {root_feat} <= {tree.threshold[0]:.4f}")

    # Depth-2 nodes: left child = children_left[0], right child = children_right[0]
    for child_idx in [tree.children_left[0], tree.children_right[0]]:
        if child_idx > 0 and tree.feature[child_idx] >= 0:  # valid internal node
            feat = feature_names[tree.feature[child_idx]]
            thresholds[f"tau_{feat}_depth2"] = float(tree.threshold[child_idx])
            print(f"[auto] Depth-2 split: {feat} <= {tree.threshold[child_idx]:.4f}")

    return thresholds


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_ablation_files(args.results_dir, args.pattern)
    pivot = pivot_best_protocol(df)

    # Join feature values from ALTA runs (they compute per-question features)
    alta_features = (
        df[df["protocol"] == "alta"][["q_id", "model_size", "benchmark"] + FEATURE_COLS]
        .drop_duplicates(subset=["q_id", "model_size", "benchmark"])
    )
    cal_df = pivot.merge(alta_features, on=["q_id", "model_size", "benchmark"], how="inner")

    n_before = len(pivot)
    n_after = len(cal_df)
    print(f"Joined {n_after}/{n_before} questions with ALTA feature vectors.")
    if n_after == 0:
        print(
            "WARNING: No feature data found. Ensure Phase 2 ablations ran with "
            "--protocols alta --save-per-question and --router new."
        )
        return

    # IMPORTANT: force plain NumPy arrays to avoid pandas/pyarrow-backed indexing
    # issues inside sklearn's CV splitter on some server images.
    X = cal_df[FEATURE_COLS].fillna(0.0).astype(float).to_numpy()
    y = cal_df["best_protocol"].astype(str).to_numpy(dtype=object)

    clf = train_and_print_tree(X, y, FEATURE_COLS)
    auto_thresholds = extract_thresholds_from_tree(clf, FEATURE_COLS)

    # Load existing thresholds (if any) and merge auto-extracted ones
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        existing = {
            "tau_R2": 0.65,
            "tau_kappa": 0.08,
            "tau_ECR": 0.10,
            "tau_H_easy": 0.5,
            "tau_H_hard": 3.0,
            "tau_SC_easy": 0.90,
            "tau_SC_hard": 0.60,
            "beta1": 3.0,
            "beta2": 0.5,
            "beta3": 5.0,
            "beta4": 2.0,
        }

    # Map auto-extracted names to canonical tau names
    name_map = {
        "tau_r2_q_root": "tau_R2",
        "tau_r2_q_depth2": "tau_R2",
        "tau_kappa_q_root": "tau_kappa",
        "tau_kappa_q_depth2": "tau_kappa",
        "tau_ecr_q_root": "tau_ECR",
        "tau_ecr_q_depth2": "tau_ECR",
        "tau_h_final_root": "tau_H_easy",
        "tau_h_final_depth2": "tau_H_hard",
    }
    for auto_key, auto_val in auto_thresholds.items():
        canonical = name_map.get(auto_key)
        if canonical:
            existing[canonical] = round(auto_val, 4)

    # Always write auto_extracted section for traceability
    existing["_auto_extracted"] = {k: round(v, 4) for k, v in auto_thresholds.items()}
    existing["_tree_cross_val_accuracy"] = round(float(cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()), 4)

    out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Saved calibrated thresholds → {out_path}")
    print("\nNEXT STEPS:")
    print("  1. Inspect the tree printout above for depth-3/4 splits")
    print("  2. Manually update tau_H_hard, tau_SC_* in", out_path)
    print("  3. Re-run smoke test with --router new --router-config", out_path)


if __name__ == "__main__":
    main()
