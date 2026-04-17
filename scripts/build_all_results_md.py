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

        # eval dicts
        results = payload.get("results")
        if not isinstance(results, dict):
            continue
        for bench, proto_map in results.items():
            if not isinstance(proto_map, dict):
                continue
            for proto, res in proto_map.items():
                if not isinstance(res, dict):
                    continue
                if str(payload.get("scoring", "")) != "cosine":
                    continue
                if not p.name.startswith("ablation_") or not p.name.endswith("_n200.json"):
                    continue
                acc = res.get("accuracy")
                item = {"scale": scale, "method": str(proto), "acc": _fmt_pct(acc)}
                if str(bench) == "truthfulqa":
                    pivot_truthfulqa_n200.append(item)
                if str(bench) == "medhallu":
                    pivot_medhallu_n200.append(item)

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
    # Table 1/2: ablation grid snapshots (n=200)
    methods = ["greedy", "alta", "cove", "iti", "delta_dola", "cured"]
    if pivot_truthfulqa_n200:
        md.append(_pivot_table("TABLE 1 — Ablations: TruthfulQA (cosine, n=200)", pivot_truthfulqa_n200, methods=methods))
        md.append("")
    else:
        md.append("### TABLE 1 — Ablations: TruthfulQA (cosine, n=200)\n\n_running…_\n")

    if pivot_medhallu_n200:
        md.append(_pivot_table("TABLE 2 — Ablations: MedHallu (cosine, n=200)", pivot_medhallu_n200, methods=methods))
        md.append("")
    else:
        md.append("### TABLE 2 — Ablations: MedHallu (cosine, n=200)\n\n_running…_\n")

    # Table 3: mechanistic profile by scale
    if pivot_profiles:
        # deterministic order
        order = {"3B": 0, "8B": 1, "14B": 2, "32B": 3}
        pivot_profiles.sort(key=lambda x: order.get(str(x["scale"]), 9))
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

