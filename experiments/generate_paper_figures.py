"""Generate exactly four manuscript figures from canonical result files.

Figure 1: Mean R^2 vs ALTA gain over greedy (8 points: 4 scales x 2 benchmarks)
Figure 2: 3B per-layer entropy profile from results/entropy_by_layer.json
Figure 3: TruthfulQA grouped bars (Greedy vs CURED v2) with significance stars
Figure 4: 3B transparency plot (native-profile vs cross-scale-profile CURED)

Usage:
    python experiments/generate_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results" / "CANONICAL_v2"
FIGURES_RESULTS = ROOT / "results" / "figures"
FIGURES_PAPER = ROOT / "paper" / "figures"
FIGURES_RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES_PAPER.mkdir(parents=True, exist_ok=True)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    print("matplotlib is not installed. Install it with pip install matplotlib")
    HAS_MPL = False

try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_accuracy(payload: dict, benchmark: str, protocol: str) -> float:
    return float(payload["results"][benchmark][protocol]["accuracy"])


def save_figure(fig, filename: str) -> None:
    out_results = FIGURES_RESULTS / filename
    out_paper = FIGURES_PAPER / filename
    fig.savefig(out_results, dpi=220, bbox_inches="tight")
    fig.savefig(out_paper, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_results}")
    print(f"Saved: {out_paper}")


def scale_label(scale: str) -> str:
    return scale.upper().replace("B", "B")


def load_p_values_by_scale() -> dict[str, float]:
    pvals: dict[str, float] = {}

    table_path = CANONICAL / "statistics_table.json"
    if table_path.exists():
        rows = load_json(table_path)
        for row in rows:
            label = str(row.get("label", "")).lower()
            for scale in ("3b", "8b", "14b", "32b"):
                if f"{scale}_truthfulqa" in label:
                    pvals[scale] = float(row["mcnemar"]["p_exact"])

    # Prefer v2 significance for 3B and 8B headline comparisons.
    for scale in ("3b", "8b"):
        stats_path = CANONICAL / f"stats_{scale}_tqa_v2.json"
        if stats_path.exists():
            rows = load_json(stats_path)
            if rows:
                pvals[scale] = float(rows[0]["mcnemar"]["p_exact"])

    return pvals


def figure1_r2_vs_alta_gain() -> None:
    scales = ["3b", "8b", "14b", "32b"]
    benches = [("truthfulqa", "TQA", "#1565C0"), ("medhallu", "Med", "#C62828")]

    points = []
    for scale in scales:
        r2_payload = load_json(CANONICAL / f"profile_{scale}.json")
        mean_r2 = float(r2_payload["mean_r2"])

        for bench, bench_short, color in benches:
            greedy_payload = load_json(CANONICAL / f"ablation_{scale}_greedy_{bench}_n200.json")
            alta_payload = load_json(CANONICAL / f"ablation_{scale}_alta_{bench}_n200.json")
            greedy_acc = extract_accuracy(greedy_payload, bench, "greedy")
            alta_acc = extract_accuracy(alta_payload, bench, "alta")
            gain_pp = (alta_acc - greedy_acc) * 100.0

            points.append(
                {
                    "x": mean_r2,
                    "y": gain_pp,
                    "label": f"{scale_label(scale)}-{bench_short}",
                    "color": color,
                }
            )

    x = np.array([p["x"] for p in points], dtype=float)
    y = np.array([p["y"] for p in points], dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.linspace(x.min() - 0.01, x.max() + 0.01, 100)
    line_y = slope * line_x + intercept

    if HAS_SCIPY:
        r_val, p_val = scipy_stats.pearsonr(x, y)
    else:
        r_val = float(np.corrcoef(x, y)[0, 1])
        p_val = float("nan")

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for p in points:
        ax.scatter(p["x"], p["y"], s=70, color=p["color"], edgecolors="white", linewidth=0.8, zorder=3)
        ax.annotate(p["label"], (p["x"], p["y"]), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.plot(line_x, line_y, color="#37474F", linewidth=2.0, label="Linear regression")

    p_text = "n/a" if np.isnan(p_val) else (f"{p_val:.1e}" if p_val < 1e-3 else f"{p_val:.3f}")
    ax.text(
        0.02,
        0.98,
        f"r = {r_val:.3f}\np = {p_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#90A4AE", alpha=0.95),
    )

    ax.set_xlabel("Mean R^2 (mechanistic profile)", fontsize=11)
    ax.set_ylabel("ALTA gain over greedy (percentage points)", fontsize=11)
    ax.set_title("Figure 1: Mechanistic Linearity Predicts ALTA Gain", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="#B0BEC5", linewidth=1)
    ax.legend(loc="lower right", fontsize=9)

    save_figure(fig, "fig1_r2_vs_alta_gain.png")


def figure2_entropy_curve() -> None:
    entropy = load_json(ROOT / "results" / "entropy_by_layer.json")
    curves = np.array(entropy["all_curves"], dtype=float)
    means = np.array(entropy["layer_means"], dtype=float)
    layers = np.arange(1, len(means) + 1)

    peak_idx = int(np.argmax(means))
    peak_layer = int(layers[peak_idx])
    peak_val = float(means[peak_idx])
    final_layer = int(layers[-1])
    final_val = float(means[-1])

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for curve in curves:
        ax.plot(layers, curve, color="#90CAF9", alpha=0.35, linewidth=0.9)

    ax.plot(layers, means, color="#0D47A1", linewidth=2.8, label=f"Mean (n={entropy['n_questions']})")

    ax.annotate(
        f"peak ~ {peak_val:.2f}",
        xy=(peak_layer, peak_val),
        xytext=(peak_layer + 2, peak_val + 0.5),
        fontsize=9,
        color="#0D47A1",
        arrowprops=dict(arrowstyle="->", color="#0D47A1", lw=1),
    )
    ax.annotate(
        f"final ~ {final_val:.2f}",
        xy=(final_layer, final_val),
        xytext=(final_layer - 8, final_val + 1.2),
        fontsize=9,
        color="#0D47A1",
        arrowprops=dict(arrowstyle="->", color="#0D47A1", lw=1),
    )

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Entropy (nats)", fontsize=11)
    ax.set_title(
        "Figure 2: 3B Per-Layer Entropy Curves (RLHF Compression)\n"
        f"Early peak ~{peak_val:.2f} -> H{final_layer} ~{final_val:.2f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(1, final_layer)
    ax.set_ylim(0, max(11.5, float(np.max(curves)) + 0.2))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    save_figure(fig, "fig2_entropy_curve_3b.png")


def figure3_grouped_truthfulqa() -> None:
    scales = ["3b", "8b", "14b", "32b"]
    pvals = load_p_values_by_scale()

    greedy_files = {
        "3b": "main_greedy_3b_truthfulqa_n817.json",
        "8b": "main_greedy_8b_truthfulqa_n817.json",
        "14b": "main_greedy_14b_truthfulqa_n817.json",
        "32b": "main_greedy_32b_truthfulqa_n817.json",
    }
    cured_v2_files = {
        "3b": "main_cured_3b_truthfulqa_n500_v2.json",
        "8b": "main_cured_8b_truthfulqa_n500_v2.json",
        "14b": "main_cured_14b_truthfulqa_n500.json",
        "32b": "main_cured_32b_truthfulqa_n500.json",
    }

    greedy = []
    cured = []
    for scale in scales:
        g_payload = load_json(CANONICAL / greedy_files[scale])
        c_payload = load_json(CANONICAL / cured_v2_files[scale])
        greedy.append(100.0 * extract_accuracy(g_payload, "truthfulqa", "greedy"))
        cured.append(100.0 * extract_accuracy(c_payload, "truthfulqa", "cured"))

    x = np.arange(len(scales))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    bars_g = ax.bar(x - width / 2, greedy, width, label="Greedy", color="#90A4AE", edgecolor="white", linewidth=0.7)
    bars_c = ax.bar(x + width / 2, cured, width, label="CURED v2", color="#1565C0", edgecolor="white", linewidth=0.7)

    for idx, (bg, bc) in enumerate(zip(bars_g, bars_c)):
        ax.text(bg.get_x() + bg.get_width() / 2, bg.get_height() + 0.4, f"{bg.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5)
        ax.text(bc.get_x() + bc.get_width() / 2, bc.get_height() + 0.4, f"{bc.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5, color="#0D47A1")
        scale = scales[idx]
        p_val = pvals.get(scale)
        if p_val is not None and p_val < 0.05:
            ax.text(bc.get_x() + bc.get_width() / 2, bc.get_height() + 2.0, "★", ha="center", va="bottom", fontsize=14, color="#0D47A1")

    ax.set_xticks(x)
    ax.set_xticklabels([scale_label(s) for s in scales], fontsize=10)
    ax.set_ylabel("TruthfulQA accuracy (%)", fontsize=11)
    ax.set_title("Figure 3: TruthfulQA Headline Result (Greedy vs CURED v2)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.99, 0.02, "★ p < 0.05", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

    y_top = max(max(greedy), max(cured)) + 6.0
    y_bottom = min(min(greedy), min(cured)) - 4.0
    ax.set_ylim(y_bottom, y_top)

    save_figure(fig, "fig3_truthfulqa_greedy_vs_cured_v2.png")


def figure4_transparency() -> None:
    greedy_payload = load_json(CANONICAL / "main_greedy_3b_truthfulqa_n817.json")
    native_payload = load_json(CANONICAL / "main_cured_3b_truthfulqa_n500_v2_native_profile.json")
    cross_payload = load_json(CANONICAL / "main_cured_3b_truthfulqa_n500_v2.json")

    values = [
        100.0 * extract_accuracy(greedy_payload, "truthfulqa", "greedy"),
        100.0 * extract_accuracy(native_payload, "truthfulqa", "cured"),
        100.0 * extract_accuracy(cross_payload, "truthfulqa", "cured"),
    ]
    labels = ["Greedy", "CURED-native-3B-profile", "CURED-8B-profile"]
    colors = ["#90A4AE", "#EF9A9A", "#1565C0"]

    fig, ax = plt.subplots(figsize=(8.8, 5.9))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("TruthfulQA accuracy (%)", fontsize=11)
    ax.set_title("Figure 4: 3B Transparency Ablation", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(min(values) - 4.0, max(values) + 6.5)

    fig.text(
        0.5,
        0.01,
        "Caption: CURED gain at 3B depends on cross-scale R^2 calibration. "
        "With native 3B profiling, gain collapses.",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_figure(fig, "fig4_3b_profile_transparency.png")


if __name__ == "__main__":
    if not HAS_MPL:
        raise SystemExit("matplotlib is required. Install it first.")

    print("Generating exactly 4 requested figures...")
    figure1_r2_vs_alta_gain()
    figure2_entropy_curve()
    figure3_grouped_truthfulqa()
    figure4_transparency()

    print("\nDone. Generated files:")
    for name in [
        "fig1_r2_vs_alta_gain.png",
        "fig2_entropy_curve_3b.png",
        "fig3_truthfulqa_greedy_vs_cured_v2.png",
        "fig4_3b_profile_transparency.png",
    ]:
        print(f"  {name}")
