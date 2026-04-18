"""compute_final_stats.py — Publication-quality statistical tests for CURED results.

Primary test: exact binomial (sign test via scipy.stats.binomtest).
Secondary: chi-squared without continuity correction (for comparability with older literature).
Also computes: bootstrap 95% CI, power analysis, formatted paper table.

Requires paired runs: both greedy and CURED must have been run with
--seed 42 --no-shuffle --save-per-question so that q_id[i] refers to the
same question in both files.

Usage:
  python compute_final_stats.py \\
      --results-dir results/CANONICAL_v2 \\
      --output results/CANONICAL_v2/statistics_table.json

  # Single comparison:
  python compute_final_stats.py \\
      --file-a results/CANONICAL_v2/main_greedy_8b_truthfulqa_n817.json \\
      --file-b results/CANONICAL_v2/main_cured_8b_truthfulqa_n500.json \\
      --label "8B TruthfulQA: greedy vs CURED"
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import binomtest


def _strip_n_suffix(key: str) -> str:
    """Normalise a filename stem key for greedy/CURED pairing.

    Strips both ``_n<digits>`` *and* any trailing ``_v<digits>`` suffix so that
    ``3b_truthfulqa_n500_v2`` and ``3b_truthfulqa_n817`` both reduce to the
    canonical key ``3b_truthfulqa``.

    Without this, v2 files (ending in ``_n500_v2``) never match greedy files
    (ending in ``_n817``) in the auto-scan mode because ``re.sub(r"_n\d+$")``
    only matches at end-of-string and ``_v2`` sits after the ``_n`` token.
    """
    # Strip _v<digits> first, then _n<digits>
    key = re.sub(r"_v\d+$", "", key)
    key = re.sub(r"_n\d+$", "", key)
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute McNemar + bootstrap CI for CURED comparisons")
    parser.add_argument("--results-dir", default="results/CANONICAL_v2",
                        help="Directory to scan for paired result files")
    parser.add_argument("--output", default="results/CANONICAL_v2/statistics_table.json")
    parser.add_argument("--file-a", default="", help="Baseline file (e.g. greedy)")
    parser.add_argument("--file-b", default="", help="Method file (e.g. cured)")
    parser.add_argument("--label", default="", help="Label for single comparison mode")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    return parser.parse_args()


def load_per_question(fpath: str) -> list[int]:
    """Load per-question correctness list from a result JSON, sorted by q_id."""
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    pq: list[dict] = []
    # Per-question may be embedded directly or under results[bench][protocol]
    if "per_question" in data:
        pq = data["per_question"]
    else:
        for bench_data in data.get("results", {}).values():
            for proto_data in bench_data.values():
                if isinstance(proto_data, dict) and "per_question" in proto_data:
                    pq.extend(proto_data["per_question"])

    # Sort by q_id for consistent pairing
    pq_sorted = sorted(pq, key=lambda x: int(x.get("q_id", x.get("i", 0))))
    return [int(q.get("correct") or 0) for q in pq_sorted]


def mcnemar_test(
    correct_a: list[int],
    correct_b: list[int],
) -> dict:
    """Paired comparison: method B (correct_b) vs baseline A (correct_a).

    Primary: exact binomial (sign test) — preferred for publication.
    Secondary: chi-squared without continuity correction — for older-literature comparability.

    Returns dict with p_exact, p_chi2, net_gain, b, c, n_discordant, significant.
    """
    assert len(correct_a) == len(correct_b), (
        f"Length mismatch: {len(correct_a)} vs {len(correct_b)}. "
        "Both runs must use --seed 42 --no-shuffle --save-per-question."
    )
    n = len(correct_a)
    # b = A correct, B wrong (B failed)
    # c = A wrong, B correct (B succeeded)
    b = sum(1 for a, bv in zip(correct_a, correct_b) if a == 1 and bv == 0)
    c = sum(1 for a, bv in zip(correct_a, correct_b) if a == 0 and bv == 1)

    if b + c == 0:
        return {
            "p_exact": 1.0, "p_chi2": 1.0,
            "net_gain": 0, "b": 0, "c": 0,
            "n_discordant": 0, "n_total": n,
            "significant": False,
        }

    # Primary: exact binomial (scipy >= 1.7.0 API)
    binom_result = binomtest(c, b + c, 0.5, alternative="two-sided")
    p_exact = float(binom_result.pvalue)

    # Secondary: chi-squared without continuity correction
    chi2 = float((b - c) ** 2 / (b + c))
    p_chi2 = float(1.0 - stats.chi2.cdf(chi2, df=1))

    return {
        "p_exact": round(p_exact, 6),
        "p_chi2": round(p_chi2, 6),
        "net_gain": c - b,   # positive → method B wins
        "b": b,
        "c": c,
        "n_discordant": b + c,
        "n_total": n,
        "significant": p_exact < 0.05,
    }


def bootstrap_ci(
    correct_list: list[int],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Non-parametric bootstrap 95% CI for accuracy."""
    n = len(correct_list)
    arr = np.array(correct_list, dtype=np.float32)
    rng = np.random.default_rng(42)
    boot_means = [float(rng.choice(arr, n, replace=True).mean()) for _ in range(n_bootstrap)]
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def power_analysis(n: int, delta: float = 0.04, alpha: float = 0.05) -> float:
    """Approximate power for paired McNemar at given n and effect size delta.

    Assumes ~30% baseline accuracy and delta pp gain.
    Uses normal approximation: power = Φ(|z| - z_{α/2}).
    """
    # Discordant pairs ≈ p(1-p) * n where p ≈ baseline accuracy
    p_baseline = 0.65  # conservative estimate
    p_discordant = p_baseline * (1 - p_baseline + delta)
    n_disc = n * p_discordant
    if n_disc < 1:
        return 0.0
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    # Effect size in discordant pairs: delta*n / sqrt(n_disc)
    z_effect = abs(delta * n) / math.sqrt(n_disc)
    power = float(stats.norm.cdf(z_effect - z_alpha))
    return round(power, 3)


def compare_pair(fpath_a: str, fpath_b: str, label: str, n_bootstrap: int) -> dict:
    """Full comparison between two result files."""
    correct_a = load_per_question(fpath_a)
    correct_b = load_per_question(fpath_b)

    # Use overlapping prefix if lengths differ (both must be seed-aligned)
    n = min(len(correct_a), len(correct_b))
    if len(correct_a) != len(correct_b):
        print(f"  WARNING: length mismatch ({len(correct_a)} vs {len(correct_b)}), using first {n}")
    correct_a = correct_a[:n]
    correct_b = correct_b[:n]

    acc_a = round(float(np.mean(correct_a)), 4)
    acc_b = round(float(np.mean(correct_b)), 4)
    ci_a = bootstrap_ci(correct_a, n_bootstrap)
    ci_b = bootstrap_ci(correct_b, n_bootstrap)
    mc = mcnemar_test(correct_a, correct_b)

    return {
        "label": label,
        "n": n,
        "baseline": {
            "file": fpath_a,
            "accuracy": acc_a,
            "ci_95": [round(ci_a[0], 4), round(ci_a[1], 4)],
        },
        "method": {
            "file": fpath_b,
            "accuracy": acc_b,
            "ci_95": [round(ci_b[0], 4), round(ci_b[1], 4)],
        },
        "delta_pp": round((acc_b - acc_a) * 100, 2),
        "mcnemar": mc,
        "power_at_n": power_analysis(n),
        "interpretation": (
            f"delta={acc_b - acc_a:+.1%} | p_exact={mc['p_exact']:.4f} "
            f"({'sig' if mc['significant'] else 'not sig'}) | "
            f"discordant={mc['n_discordant']}/{n}"
        ),
    }


def print_table(comparisons: list[dict]) -> None:
    """Print a formatted paper-ready table."""
    lines = [
        "\n" + "=" * 80,
        f"{'Label':<40} {'Base':>6} {'Method':>6} {'Dpp':>6} {'p':>8} {'sig':>4}",
        "-" * 80,
    ]
    for c in comparisons:
        lines.append(
            f"{c['label'][:39]:<40} "
            f"{c['baseline']['accuracy']:>6.1%} "
            f"{c['method']['accuracy']:>6.1%} "
            f"{c['delta_pp']:>+6.1f} "
            f"{c['mcnemar']['p_exact']:>8.4f} "
            f"{'*' if c['mcnemar']['significant'] else '':>4}"
        )
    lines += ["=" * 80, "* p < 0.05 (exact binomial). Primary test: scipy.stats.binomtest.\n"]
    print("\n".join(lines).encode("ascii", errors="replace").decode("ascii"))


def r2_stratified_alta_analysis(results_dir: str, output_path: str) -> dict:
    """For each question with saved per-question features (from --save-per-question),
    correlate R2_q with ALTA accuracy gain over greedy.

    Requires ablation_{scale}_alta_{bench}_n200.json and
    ablation_{scale}_greedy_{bench}_n200.json both run with
    --save-per-question --router new.

    Returns dict of statistics; also writes r2_stratified_analysis.json.
    """
    import pandas as pd
    from scipy import stats as scipy_stats

    results_path = Path(results_dir)
    records = []

    def get_pq(data: dict) -> list:
        if "per_question" in data:
            return data["per_question"]
        for bv in data.get("results", {}).values():
            for pv in bv.values():
                if isinstance(pv, dict) and "per_question" in pv:
                    return pv["per_question"]
        return []

    for scale in ["3b", "8b", "14b", "32b"]:
        for bench in ["truthfulqa", "medhallu"]:
            alta_f = results_path / f"ablation_{scale}_alta_{bench}_n200.json"
            greedy_f = results_path / f"ablation_{scale}_greedy_{bench}_n200.json"
            if not alta_f.exists() or not greedy_f.exists():
                print(f"  Missing: {scale} {bench} ablation files")
                continue

            with open(alta_f) as f:
                alta_data = json.load(f)
            with open(greedy_f) as f:
                greedy_data = json.load(f)

            alta_pq = {q.get("q_id", q.get("i")): q for q in get_pq(alta_data)}
            greedy_pq = {q.get("q_id", q.get("i")): q for q in get_pq(greedy_data)}

            common_ids = set(alta_pq) & set(greedy_pq)
            if not common_ids:
                print(
                    f"  No overlapping q_ids for {scale} {bench}. "
                    "Re-run ablations with --save-per-question --seed 42 --no-shuffle."
                )
                continue

            for qid in common_ids:
                a = alta_pq[qid]
                g = greedy_pq[qid]
                r2_q = a.get("r2_q")
                if r2_q is None:
                    continue
                records.append({
                    "scale": scale,
                    "benchmark": bench,
                    "q_id": qid,
                    "r2_q": float(r2_q),
                    "kappa_q": float(a.get("kappa_q") or 0),
                    "ecr_q": float(a.get("ecr_q") or 0),
                    "h_final": float(a.get("h_final") or 0),
                    "alta_correct": int(a.get("correct") or 0),
                    "greedy_correct": int(g.get("correct") or 0),
                })

    if not records:
        msg = (
            "No per-question feature data found. "
            "Re-run ablations with --save-per-question."
        )
        print(f"  WARNING: {msg}")
        return {"error": msg}

    df = pd.DataFrame(records)
    df["alta_gain"] = df["alta_correct"] - df["greedy_correct"]

    results_out: dict = {}

    print("\n=== R² vs ALTA Accuracy (per-question, within scale) ===")
    for scale in sorted(df["scale"].unique()):
        sub = df[df["scale"] == scale]
        r, p = scipy_stats.pearsonr(sub["r2_q"], sub["alta_correct"])
        results_out[f"pearson_r_scale_{scale}"] = round(r, 4)
        results_out[f"pearson_p_scale_{scale}"] = round(p, 4)
        print(f"  {scale}: r={r:.3f}, p={p:.4f}, n={len(sub)}")

    df["r2_quartile"] = pd.qcut(
        df["r2_q"], q=4, labels=["Q1 (R²<p25)", "Q2", "Q3", "Q4 (R²>p75)"]
    )
    q_stats = (
        df.groupby("r2_quartile")
        .agg(
            n=("r2_q", "count"),
            r2_mean=("r2_q", "mean"),
            alta_acc=("alta_correct", "mean"),
            greedy_acc=("greedy_correct", "mean"),
            mean_gain=("alta_gain", "mean"),
        )
        .round(4)
    )
    print("\n=== Quartile Analysis (pooled) ===")
    print(q_stats.to_string())
    results_out["quartile_analysis"] = q_stats.to_dict()

    r_pb, p_pb = scipy_stats.pointbiserialr(df["alta_correct"], df["r2_q"])
    results_out["point_biserial_r2_vs_alta_correct"] = round(r_pb, 4)
    results_out["point_biserial_p"] = round(p_pb, 6)
    print(
        f"\nPoint-biserial r(R2_q, alta_correct) = {r_pb:.4f}, p={p_pb:.6f}"
        "\n(Positive = higher R² → more likely ALTA is correct)"
    )

    out = Path(output_path).parent / "r2_stratified_analysis.json"
    out.write_text(json.dumps(results_out, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out}")
    return results_out


def main() -> None:
    args = parse_args()

    comparisons: list[dict] = []

    if args.file_a and args.file_b:
        # Single-pair mode
        label = args.label or f"{Path(args.file_a).stem} vs {Path(args.file_b).stem}"
        print(f"Comparing: {label}")
        comp = compare_pair(args.file_a, args.file_b, label, args.n_bootstrap)
        comparisons.append(comp)
        print(f"  {comp['interpretation']}")
    else:
        # Auto-scan mode: pair greedy_* with cured_* files by (model_size, benchmark)
        # Strip _n<digits> suffix so "8b_truthfulqa_n817" matches "8b_truthfulqa_n500".
        results_dir = Path(args.results_dir)
        greedy_files = {
            _strip_n_suffix(Path(f).stem.replace("main_greedy_", "")): f
            for f in glob.glob(str(results_dir / "main_greedy_*.json"))
        }
        # Skip old-router files for automatic pairing (keep new router only).
        cured_files = {
            _strip_n_suffix(Path(f).stem.replace("main_cured_", "")): f
            for f in glob.glob(str(results_dir / "main_cured_*.json"))
            if "old_" not in Path(f).stem
        }

        if not greedy_files or not cured_files:
            print(
                "No main_greedy_* or main_cured_* files found in "
                f"{results_dir}. Run Phase 4 first."
            )
            return

        for key in sorted(set(greedy_files) & set(cured_files)):
            label = f"greedy vs CURED | {key}"
            print(f"Comparing: {label}")
            try:
                comp = compare_pair(greedy_files[key], cured_files[key], label, args.n_bootstrap)
                comparisons.append(comp)
                print(f"  {comp['interpretation']}")
            except Exception as exc:
                print(f"  ERROR: {exc}")

    if not comparisons:
        print("No comparisons completed.")
        return

    print_table(comparisons)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    print(f"Saved statistics -> {out_path}")

    print("\n=== R² Stratified ALTA Analysis ===")
    r2_stratified_alta_analysis(args.results_dir, args.output)


if __name__ == "__main__":
    main()
