#!/usr/bin/env python3
"""Build a clean, organized all_results.md from results/**/*.json.

This is meant to be re-runnable as new JSONs appear (e.g., during the remote pipeline).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_MD = ROOT / "all_results.md"


@dataclass(frozen=True)
class FileInfo:
    rel: str
    size_kb: float
    mtime: str


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.1f}%"


def _fmt_num(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    try:
        xf = float(x)
        if abs(xf) < 1e-9:
            return "0"
        if abs(xf) < 1:
            return f"{xf:.3f}"
        return f"{xf:.2f}"
    except Exception:
        return str(x)


def _file_info(path: Path) -> FileInfo:
    st = path.stat()
    mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return FileInfo(rel=str(path.relative_to(ROOT)).replace("\\", "/"), size_kb=st.st_size / 1024.0, mtime=mtime)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out: list[str] = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _extract_eval_rows(payload: dict[str, Any], file_rel: str) -> list[list[str]]:
    """Extract rows for a CURED eval-dict style JSON."""
    model = str(payload.get("model", payload.get("api_model", payload.get("api_mode", "")) or ""))
    scoring = str(payload.get("scoring", payload.get("scoring_mode", "")) or "")
    n_target = payload.get("n_target", payload.get("n", ""))
    out: list[list[str]] = []
    results = payload.get("results")
    if not isinstance(results, dict):
        return out
    for bench, proto_map in results.items():
        if not isinstance(proto_map, dict):
            continue
        for proto, res in proto_map.items():
            if not isinstance(res, dict):
                continue
            out.append(
                [
                    file_rel,
                    model,
                    str(bench),
                    str(proto),
                    scoring,
                    str(n_target) if n_target != "" else str(res.get("n_total", "—")),
                    _fmt_pct(res.get("accuracy")),
                    _fmt_pct(res.get("rep_rate")),
                    _fmt_num(res.get("n_scored")),
                    _fmt_num(res.get("runtime_min")),
                ]
            )
    return out


def _infer_scale_tag(payload: dict[str, Any], filename: str) -> str:
    b = payload.get("model_params_b")
    try:
        if b is not None:
            bf = float(b)
            if bf >= 31:
                return "32B"
            if bf >= 13:
                return "14B"
            if bf >= 7:
                return "8B"
            if bf > 0:
                return "3B"
    except Exception:
        pass
    m = re.search(r"(?:^|_)((?:3|8|14|32)b)(?:_|\.)", filename.lower())
    if m:
        return m.group(1).upper()
    return "?"


def _main_cured_priority(path: Path) -> int:
    """Rank main CURED files for headline tables."""
    stem = path.stem
    if "old_" in stem or "native_profile" in stem:
        return -1
    if stem.endswith("_v2"):
        return 2
    return 1


def _pivot_table(
    title: str,
    rows: list[dict[str, Any]],
    *,
    scales: list[str] = ["3B", "8B", "14B", "32B"],
    methods: list[str],
) -> str:
    """Create a wide table: method × scale -> acc."""
    grid: dict[tuple[str, str], str] = {}
    for r in rows:
        grid[(r["method"], r["scale"])] = r.get("acc", "—")
    out_rows: list[list[str]] = []
    for m in methods:
        out_rows.append([m] + [grid.get((m, s), "running…") for s in scales])
    return "\n".join(
        [
            f"### {title}",
            "",
            _md_table(["Method"] + scales, out_rows),
        ]
    )


def _extract_profile_row(payload: dict[str, Any], file_rel: str) -> list[str] | None:
    if "mean_r2" not in payload and "mean_kappa" not in payload and "mean_ecr" not in payload:
        return None
    model = str(payload.get("model", ""))
    return [
        file_rel,
        model,
        _fmt_num(payload.get("n_questions")),
        _fmt_num(payload.get("mean_r2")),
        _fmt_num(payload.get("mean_kappa")),
        _fmt_num(payload.get("mean_ecr")),
        _fmt_num(payload.get("mean_h_final")),
        _fmt_num(payload.get("mean_h_peak")),
    ]


def _extract_experimental_rows(payload: Any, file_rel: str) -> list[list[str]]:
    """Extract one or more rows that summarize a result JSON in a common table schema."""
    rows: list[list[str]] = []
    if payload is None:
        return [["—", "invalid-json", "—", "—", "—", "—", "—", "—", "—", "—", "—", "Unable to parse JSON"]]

    kind = _kind(payload)
    if not isinstance(payload, dict):
        if kind == "list":
            return [[
                file_rel,
                kind,
                "—",
                "—",
                "—",
                "—",
                str(len(payload)),
                "—",
                "—",
                "—",
                "—",
                "List-only JSON payload",
            ]]
        return [[
            file_rel,
            kind,
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            f"Unhandled JSON kind: {kind}",
        ]]

    # Reuse canonical parser for full eval-dict style files.
    eval_rows = _extract_eval_rows(payload, file_rel)
    if eval_rows:
        for row in eval_rows:
            rows.append([
                row[0],  # file
                "eval-dict",
                row[1],  # model
                row[2],  # benchmark
                row[3],  # protocol
                row[4],  # scoring
                row[5],  # n
                row[6],  # acc
                row[7],  # rep
                row[8],  # n_scored
                row[9],  # runtime
                "Parsed from results dictionary",
            ])
        return rows

    # Profile-style outputs
    if "mean_r2" in payload or "mean_kappa" in payload or "mean_ecr" in payload:
        rows.append([
            file_rel,
            "profile",
            str(payload.get("model", "")),
            "profile",
            "profile",
            "—",
            _fmt_num(payload.get("n_questions", "")),
            "—",
            _fmt_num(payload.get("n_questions", "")),
            _fmt_num(payload.get("runtime_min", "")),
            (
                f"mean_r2={_fmt_num(payload.get('mean_r2'))}; "
                f"mean_kappa={_fmt_num(payload.get('mean_kappa'))}; "
                f"mean_ecr={_fmt_num(payload.get('mean_ecr'))}; "
                f"mean_h_final={_fmt_num(payload.get('mean_h_final'))}; "
                f"mean_h_peak={_fmt_num(payload.get('mean_h_peak'))}"
            ),
        ])
        return rows

    # Generic score-bearing JSON
    model = str(payload.get("model", payload.get("api_model", payload.get("api_mode", "")) or ""))
    bench = str(payload.get("benchmark", payload.get("bench", "")) or "—")
    proto = str(payload.get("protocol", payload.get("method", "")) or "—")
    scoring = str(payload.get("scoring", payload.get("scoring_mode", "")) or "—")
    n_value = payload.get("n_target", payload.get("n", payload.get("n_total", payload.get("n_scored", ""))))
    acc = payload.get("accuracy", payload.get("acc", None))
    rep = payload.get("rep_rate", payload.get("rep", None))
    n_scored = payload.get("n_scored")
    runtime = payload.get("runtime_min")

    rows.append([
        file_rel,
        kind,
        model if model else "—",
        bench,
        proto,
        scoring,
        str(n_value) if n_value != "" else "—",
        _fmt_pct(acc) if isinstance(acc, (int, float)) or isinstance(acc, str) else "—",
        _fmt_pct(rep) if isinstance(rep, (int, float)) or isinstance(rep, str) else "—",
        _fmt_num(n_scored),
        _fmt_num(runtime),
        "Generic JSON result container",
    ])
    return rows


def _kind(payload: Any) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
        return "eval-dict"
    if isinstance(payload, dict) and any(k in payload for k in ("mean_r2", "mean_kappa", "mean_ecr")):
        return "profile"
    if isinstance(payload, list):
        return "list"
    if isinstance(payload, dict):
        return "dict"
    return type(payload).__name__


def main() -> None:
    json_files = sorted(RESULTS_DIR.rglob("*.json"))
    infos = {p: _file_info(p) for p in json_files}
    payloads = {p: _read_json(p) for p in json_files}

    # ── Canonical v2: profiles / ablations / main runs / other eval dicts ──
    canonical_dir = RESULTS_DIR / "CANONICAL_v2"
    canonical_files = [p for p in json_files if canonical_dir in p.parents]

    profile_rows: list[list[str]] = []
    canonical_eval_rows: list[list[str]] = []
    ablation_eval_rows: list[list[str]] = []
    main_eval_rows: list[list[str]] = []
    misc_canonical_rows: list[list[str]] = []

    for p in canonical_files:
        info = infos[p]
        payload = payloads[p]
        if not isinstance(payload, dict):
            continue
        if p.name.startswith("profile_"):
            r = _extract_profile_row(payload, info.rel)
            if r:
                profile_rows.append(r)
            continue
        rows = _extract_eval_rows(payload, info.rel)
        if not rows:
            continue
        if p.name.startswith("ablation_"):
            ablation_eval_rows.extend(rows)
        elif p.name.startswith("main_"):
            main_eval_rows.extend(rows)
        elif p.name.startswith("results_"):
            canonical_eval_rows.extend(rows)
        else:
            misc_canonical_rows.extend(rows)

    # ── Build paper-style pivot tables (Canonical v2 only) ─────────────────
    pivot_truthfulqa_n200: list[dict[str, Any]] = []
    pivot_medhallu_n200: list[dict[str, Any]] = []
    pivot_profiles: list[dict[str, Any]] = []
    # Phase 4: CURED accuracy by (scale, benchmark) and greedy accuracy by scale
    pivot_main_cured: dict[tuple[str, str], str] = {}   # (scale, bench) -> acc
    pivot_main_cured_priority: dict[tuple[str, str], int] = {}
    pivot_main_greedy: dict[tuple[str, str], str] = {}   # (scale, bench) -> acc

    for p in canonical_files:
        payload = payloads[p]
        if not isinstance(payload, dict):
            continue
        scale = _infer_scale_tag(payload, p.name)

        # profiles
        if p.name.startswith("profile_"):
            pivot_profiles.append(
                {
                    "scale": scale,
                    "mean_r2": _fmt_num(payload.get("mean_r2")),
                    "mean_kappa": _fmt_num(payload.get("mean_kappa")),
                    "mean_ecr": _fmt_num(payload.get("mean_ecr")),
                    "h_final": _fmt_num(payload.get("mean_h_final")),
                    "h_peak": _fmt_num(payload.get("mean_h_peak")),
                }
            )
            continue

        results = payload.get("results")
        if not isinstance(results, dict):
            continue

        for bench, proto_map in results.items():
            if not isinstance(proto_map, dict):
                continue
            for proto, res in proto_map.items():
                if not isinstance(res, dict):
                    continue
                acc = res.get("accuracy")

                # Phase 2 ablation pivot (greedy/alta/cove/iti only, cosine scored)
                if (p.name.startswith("ablation_") and p.name.endswith("_n200.json")
                        and str(payload.get("scoring", "")) == "cosine"):
                    item = {"scale": scale, "method": str(proto), "acc": _fmt_pct(acc)}
                    if str(bench) == "truthfulqa":
                        pivot_truthfulqa_n200.append(item)
                    if str(bench) == "medhallu":
                        pivot_medhallu_n200.append(item)

                # Phase 4 CURED new-router pivot
                if p.name.startswith("main_cured_") and str(proto) == "cured":
                    priority = _main_cured_priority(p)
                    if priority < 0:
                        continue
                    bench_label = "strategyqa" if str(bench) == "custom" else str(bench)
                    key = (scale, bench_label)
                    if priority > pivot_main_cured_priority.get(key, -1):
                        pivot_main_cured[key] = _fmt_pct(acc)
                        pivot_main_cured_priority[key] = priority

                # Phase 4 greedy baseline pivot (truthfulqa only)
                if p.name.startswith("main_greedy_") and str(proto) == "greedy":
                    bench_label = "strategyqa" if str(bench) == "custom" else str(bench)
                    pivot_main_greedy[(scale, bench_label)] = _fmt_pct(acc)

    # ── Non-canonical: group by provider-ish prefixes, otherwise inventory ──
    provider_eval_rows: list[list[str]] = []
    other_inventory_rows: list[list[str]] = []

    for p in json_files:
        if canonical_dir in p.parents:
            continue
        info = infos[p]
        payload = payloads[p]
        if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
            provider_eval_rows.extend(_extract_eval_rows(payload, info.rel))
        else:
            k = _kind(payload)
            other_inventory_rows.append(
                [
                    info.rel,
                    k,
                    f"{info.size_kb:.1f}",
                    info.mtime,
                ]
            )

    # Full ledger: include every JSON under results/ in a unified row format.
    all_result_rows: list[list[str]] = []
    for p in json_files:
        info = infos[p]
        payload = payloads[p]
        all_result_rows.extend(_extract_experimental_rows(payload, info.rel))

    # ── Write markdown ─────────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md: list[str] = []
    md.append("# All Results (auto-generated)")
    md.append("")
    md.append(f"Last updated: {now}")
    md.append("")
    md.append("This file is generated by `scripts/build_all_results_md.py` from `results/**/*.json`.")
    md.append("")

    md.append("## Canonical v2 — Paper-style tables (auto)")
    md.append("")
    # Table 1/2: ablation grid snapshots (n=200). Phase 2 ran greedy/alta/cove/iti only.
    ablation_methods = ["greedy", "alta", "cove", "iti"]
    if pivot_truthfulqa_n200:
        md.append(_pivot_table("TABLE 1 — Ablations: TruthfulQA (cosine, n=200)", pivot_truthfulqa_n200, methods=ablation_methods))
        md.append("")
    else:
        md.append("### TABLE 1 — Ablations: TruthfulQA (cosine, n=200)\n\n_running…_\n")

    if pivot_medhallu_n200:
        md.append(_pivot_table("TABLE 2 — Ablations: MedHallu (cosine, n=200)", pivot_medhallu_n200, methods=ablation_methods))
        md.append("")
    else:
        md.append("### TABLE 2 — Ablations: MedHallu (cosine, n=200)\n\n_running…_\n")

    # Table 3: mechanistic profile by scale
    if pivot_profiles:
        scale_order = {"3B": 0, "8B": 1, "14B": 2, "32B": 3}
        pivot_profiles.sort(key=lambda x: scale_order.get(str(x["scale"]), 9))
        md.append("### TABLE 3 — Mechanistic profile by model scale")
        md.append("")
        md.append(
            _md_table(
                ["Scale", "Mean R²", "Mean κ", "Mean ECR", "H_final", "H_peak"],
                [[p["scale"], p["mean_r2"], p["mean_kappa"], p["mean_ecr"], p["h_final"], p["h_peak"]] for p in pivot_profiles],
            )
        )
        md.append("")
    else:
        md.append("### TABLE 3 — Mechanistic profile by model scale\n\n_running…_\n")

    # Table 4: Phase 4 main results — CURED accuracy by scale × benchmark
    if pivot_main_cured:
        scales = ["3B", "8B", "14B", "32B"]
        benchmarks = ["truthfulqa", "medhallu", "strategyqa"]
        md.append("### TABLE 4 — Phase 4 Main: CURED (new router, n=500) accuracy")
        md.append("")
        header = ["Benchmark"] + scales
        rows4: list[list[str]] = []
        for bench in benchmarks:
            row = [bench]
            for sc in scales:
                row.append(pivot_main_cured.get((sc, bench), "—"))
            rows4.append(row)
        # Greedy baseline row (truthfulqa only, n=817)
        greedy_row = ["greedy baseline (n=817)"]
        for sc in scales:
            greedy_row.append(pivot_main_greedy.get((sc, "truthfulqa"), "—"))
        rows4.append(greedy_row)
        md.append(_md_table(header, rows4))
        md.append("")
    else:
        md.append("### TABLE 4 — Phase 4 Main results\n\n_No main_cured_*.json found yet._\n")

    # Table 5: Phase 5 statistics (from statistics_table.json if present)
    stats_path = RESULTS_DIR / "CANONICAL_v2" / "statistics_table.json"
    if stats_path.exists():
        try:
            stats_data: list[dict[str, Any]] = json.loads(stats_path.read_text(encoding="utf-8"))
            if stats_data:
                md.append("### TABLE 5 — Phase 5 Statistical Tests (CURED vs Greedy, paired McNemar)")
                md.append("")
                stat_rows: list[list[str]] = []
                for c in stats_data:
                    mc = c.get("mcnemar", {})
                    sig = "YES" if mc.get("significant") else "NO"
                    base_acc = c.get("baseline", {}).get("accuracy", 0)
                    meth_acc = c.get("method", {}).get("accuracy", 0)
                    stat_rows.append([
                        str(c.get("label", "")).replace("|", "\\|"),
                        str(c.get("n", "")),
                        f"{base_acc:.1%}",
                        f"{meth_acc:.1%}",
                        f"{c.get('delta_pp', 0):+.1f}pp",
                        f"{c.get('delta_pp', 0) / 100:+.2f}",
                        f"{mc.get('p_exact', 1):.4f}",
                        f"{mc.get('p_chi2', 1):.4f}",
                        sig,
                        f"{mc.get('b', 0)}->{mc.get('c', 0)}",
                        f"{mc.get('n_discordant', 0)}/{c.get('n', '')}",
                        f"{c.get('power_at_n', 0):.2f}",
                    ])
                md.append(_md_table(
                    [
                        "Comparison",
                        "n",
                        "Greedy",
                        "CURED",
                        "Delta_pp",
                        "Delta (proportion)",
                        "p_exact",
                        "p_chi2",
                        "sig",
                        "b->c",
                        "discordant",
                        "power",
                    ],
                    stat_rows,
                ))
                md.append("")
                md.append("_\\* p < 0.05 (exact binomial, scipy.stats.binomtest). Primary test._")
                md.append("")
                md.append("_p_chi2: chi-square statistic p-value without continuity correction._")
                md.append("")
        except Exception:
            pass

    # Full experimental ledger: every result JSON under results/ (including archived and misc).
    md.append("## All experimental results (`results/**/*.json`)")
    md.append("")
    if all_result_rows:
        md.append(
            _md_table(
                [
                    "file",
                    "kind",
                    "model",
                    "benchmark",
                    "protocol",
                    "scoring",
                    "n",
                    "acc",
                    "rep",
                    "n_scored",
                    "runtime_min",
                    "notes",
                ],
                all_result_rows,
            )
        )
    else:
        md.append("_No JSON files were found under results/._")
    md.append("")

    md.append("## Canonical v2 — Long-form tables (all rows)")
    md.append("")
    md.append("These tables list every protocol row extracted from each JSON, useful for debugging and audit.")
    md.append("")

    md.append("### Canonical v2 — Mechanistic Profiles (raw files)")
    md.append("")
    md.append(_md_table(["file", "model", "n_q", "mean_R2", "mean_kappa", "mean_ECR", "H_final", "H_peak"], profile_rows) if profile_rows else "_None._")
    md.append("")

    md.append("## Canonical v2 — Main result summaries (`results_*.json`)")
    md.append("")
    if canonical_eval_rows:
        md.append(
            _md_table(
                ["file", "model", "benchmark", "protocol", "scoring", "n", "acc", "rep", "n_scored", "runtime_min"],
                canonical_eval_rows,
            )
        )
    else:
        md.append("_No `results_*.json` found in `results/CANONICAL_v2/`._")
    md.append("")

    md.append("## Canonical v2 — Phase 2 Ablations (`ablation_*_n200.json`)")
    md.append("")
    if ablation_eval_rows:
        md.append(
            _md_table(
                ["file", "model", "benchmark", "protocol", "scoring", "n", "acc", "rep", "n_scored", "runtime_min"],
                ablation_eval_rows,
            )
        )
    else:
        md.append("_No ablation JSONs found._")
    md.append("")

    md.append("## Canonical v2 — Phase 4 Main runs (`main_*`)")
    md.append("")
    if main_eval_rows:
        md.append(
            _md_table(
                ["file", "model", "benchmark", "protocol", "scoring", "n", "acc", "rep", "n_scored", "runtime_min"],
                main_eval_rows,
            )
        )
    else:
        md.append("_No main run JSONs found yet (pipeline still running)._")
    md.append("")

    if misc_canonical_rows:
        md.append("## Canonical v2 — Other eval JSONs")
        md.append("")
        md.append(
            _md_table(
                ["file", "model", "benchmark", "protocol", "scoring", "n", "acc", "rep", "n_scored", "runtime_min"],
                misc_canonical_rows,
            )
        )
        md.append("")

    md.append("## Provider / API runs (non-canonical eval dicts)")
    md.append("")
    if provider_eval_rows:
        md.append(
            _md_table(
                ["file", "model", "benchmark", "protocol", "scoring", "n", "acc", "rep", "n_scored", "runtime_min"],
                provider_eval_rows,
            )
        )
    else:
        md.append("_No provider/API eval-dict JSONs detected._")
    md.append("")

    md.append("## Other JSON inventory (analysis, lists, archives, misc)")
    md.append("")
    md.append(
        _md_table(
            ["file", "kind", "size_kb", "modified"],
            other_inventory_rows,
        )
    )
    md.append("")

    OUT_MD.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()

