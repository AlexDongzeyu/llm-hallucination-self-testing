#!/usr/bin/env python3
"""
Pull all JSON under results/CANONICAL_v2 from A800 and refresh all_results.md + RESULTS.md.

Uses the same SSH host/port as scripts/monitor_a800.py (port 14036).

Usage (from repo root):
  python scripts/sync_a800_canonical_to_local.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import paramiko

HOST = "js4.blockelite.cn"
PORT = 14036
USER = "root"
PASS = "aiPh9chu"
REMOTE_REPO = "/root/llm-hallucination-self-testing"
REMOTE_CANON = f"{REMOTE_REPO}/results/CANONICAL_v2"

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CANON = ROOT / "results" / "CANONICAL_v2"
ALL_RESULTS = ROOT / "all_results.md"
RESULTS_MD = ROOT / "RESULTS.md"

SYNC_START = "<!-- A800_CANONICAL_SYNC -->"
SYNC_END = "<!-- /A800_CANONICAL_SYNC -->"


def parse_result_json(path: Path) -> list[dict]:
    """Same row shape as scripts/parse_v2_results.parse_file."""
    d = json.loads(path.read_text(encoding="utf-8"))
    model = d.get("model", "?")
    bench_global = d.get("benchmark", "?")
    scoring = d.get("scoring", "cosine")
    n_target = d.get("n_target")
    rows: list[dict] = []
    results = d.get("results", {})
    if not isinstance(results, dict):
        return rows
    for bk, bval in results.items():
        if not isinstance(bval, dict):
            continue
        for proto, pval in bval.items():
            if not isinstance(pval, dict):
                continue
            acc = pval.get("accuracy")
            n = pval.get("n_questions", pval.get("n", n_target))
            rt = pval.get("runtime_min")
            if rt is None and pval.get("runtime_s"):
                rt = round(pval["runtime_s"] / 60, 2)
            rep = pval.get("repetition_rate", 0)
            n_scored = pval.get("n_scored", n)
            rows.append(
                {
                    "file": "results/CANONICAL_v2/" + path.name,
                    "model": model,
                    "benchmark": bk,
                    "protocol": proto,
                    "scoring": scoring,
                    "n": n,
                    "acc": f"{acc * 100:.1f}%" if acc is not None else "?",
                    "rep": f"{rep * 100:.1f}%" if rep is not None else "0.0%",
                    "n_scored": n_scored,
                    "runtime_min": round(rt, 2) if rt else "?",
                }
            )
    return rows


def row_md(r: dict) -> str:
    return (
        f"| {r['file']} | {r['model']} | {r['benchmark']} | {r['protocol']} | "
        f"{r['scoring']} | {r['n']} | {r['acc']} | {r['rep']} | {r['n_scored']} | {r['runtime_min']} |"
    )


def table_md(rows: list[dict]) -> str:
    head = (
        "| file | model | benchmark | protocol | scoring | n | acc | rep | n_scored | runtime_min |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    body = "\n".join(row_md(r) for r in rows)
    return head + "\n" + body


def pull_all_json() -> list[str]:
    LOCAL_CANON.mkdir(parents=True, exist_ok=True)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=60)
    sftp = c.open_sftp()
    pulled: list[str] = []
    for attr in sftp.listdir_attr(REMOTE_CANON):
        name = attr.filename
        if not name.endswith(".json"):
            continue
        remote = f"{REMOTE_CANON}/{name}"
        local = LOCAL_CANON / name
        sftp.get(remote, str(local))
        pulled.append(name)
    sftp.close()
    c.close()
    return sorted(pulled)


def build_sync_block() -> str:
    lines: list[str] = [
        "",
        SYNC_START,
        "",
        "## Canonical v2 — A800 pipeline outputs (synced)!",
        "",
        "_Pulled from A800 `results/CANONICAL_v2/` — includes FACTOR, semantic-entropy ablation, "
        "final statistics, and refreshed 8B TruthfulQA ablations (save-per-question + r2_q)._",
        "",
    ]

    # FACTOR + any multi-protocol result files not covered elsewhere
    factor_paths = sorted(LOCAL_CANON.glob("results_8b_factor_*_n200.json"))
    if factor_paths:
        lines += [
            "### FACTOR (8B, letter, max_new_tokens=5)!",
            "",
        ]
        all_rows: list[dict] = []
        for p in factor_paths:
            all_rows.extend(parse_result_json(p))
        lines.append(table_md(all_rows))
        lines.append("")

    sem_p = LOCAL_CANON / "semantic_entropy_gate_comparison.json"
    if sem_p.is_file():
        sem = json.loads(sem_p.read_text(encoding="utf-8"))
        lines += [
            "### Semantic entropy vs ECR gate (8B, MedHallu)!",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| model | `{sem.get('model', '')}` |",
            f"| benchmark | {sem.get('benchmark', '')} |",
            f"| n | {sem.get('n', '')} |",
            f"| k (stochastic) | {sem.get('k_stochastic', '')} |",
            f"| greedy_accuracy | {sem.get('greedy_accuracy', '')} |",
            f"| ecr_based_accuracy | {sem.get('ecr_based_accuracy', '')} |",
            f"| se_based_accuracy | {sem.get('se_based_accuracy', '')} |",
            f"| ecr_gain_pp | {sem.get('ecr_gain_pp', '')} |",
            f"| se_gain_pp | {sem.get('se_gain_pp', '')} |",
            f"| ecr_cove_rate | {sem.get('ecr_cove_rate', '')} |",
            f"| se_cove_rate | {sem.get('se_cove_rate', '')} |",
            f"| file | `results/CANONICAL_v2/semantic_entropy_gate_comparison.json` |",
            "",
        ]

    for label, name in (
        ("Final paired statistics", "statistics_table.json"),
        ("R² stratified analysis", "r2_stratified_analysis.json"),
    ):
        p = LOCAL_CANON / name
        if p.is_file():
            lines += [
                f"### {label}!",
                "",
                f"- **File:** `results/CANONICAL_v2/{name}`",
                f"- **Size:** {p.stat().st_size} bytes",
                "",
            ]

    lines.append(SYNC_END)
    lines.append("")
    return "\n".join(lines)


def replace_ablation_rows(content: str) -> str:
    for fname in (
        "ablation_8b_alta_truthfulqa_n200.json",
        "ablation_8b_greedy_truthfulqa_n200.json",
    ):
        p = LOCAL_CANON / fname
        if not p.is_file():
            continue
        rows = parse_result_json(p)
        if not rows:
            continue
        new_line = row_md(rows[0])
        pat = re.compile(
            r"^(\| results/CANONICAL_v2/" + re.escape(fname) + r" \|[^\n]*)\s*$",
            re.MULTILINE,
        )
        if not pat.search(content):
            print(f"WARN: no markdown row to replace for {fname}", file=sys.stderr)
            continue
        content = pat.sub(new_line, content)
    return content


def insert_or_replace_sync_section(content: str, block: str) -> str:
    anchor = "## Provider / API runs (non-canonical eval dicts)\n"
    if SYNC_START in content and SYNC_END in content:
        inner = re.compile(
            re.escape(SYNC_START) + r"[\s\S]*?" + re.escape(SYNC_END),
            re.MULTILINE,
        )
        content = inner.sub(block.strip(), content, count=1)
        return content
    if anchor not in content:
        print("WARN: Provider anchor not found; appending sync block to end.", file=sys.stderr)
        return content.rstrip() + "\n\n" + block
    return content.replace(anchor, block + "\n" + anchor, 1)


def patch_results_md() -> None:
    text = RESULTS_MD.read_text(encoding="utf-8")

    p_g = LOCAL_CANON / "ablation_8b_greedy_truthfulqa_n200.json"
    if p_g.is_file():
        rows = parse_result_json(p_g)
        if rows:
            acc = rows[0]["acc"]

            def _greedy_row(m: re.Match) -> str:
                return f"{m.group(1)}{acc}{m.group(3)}"

            text = re.sub(
                r"(\| Greedy \| )(\d+\.\d+%)( \| \d+\.\d+% \|)",
                _greedy_row,
                text,
                count=1,
            )
    p_a = LOCAL_CANON / "ablation_8b_alta_truthfulqa_n200.json"
    if p_a.is_file():
        rows = parse_result_json(p_a)
        if rows:
            acc = rows[0]["acc"]

            def _alta_row(m: re.Match) -> str:
                return f"{m.group(1)}{acc}{m.group(3)}"

            text = re.sub(
                r"(\| ALTA \| )(\d+\.\d+%)( \| \d+\.\d+% \|)",
                _alta_row,
                text,
                count=1,
            )

    sync_heading = "## A800 sync — FACTOR, semantic entropy, stats!"
    factor_paths = sorted(LOCAL_CANON.glob("results_8b_factor_*_n200.json"))
    sem_p = LOCAL_CANON / "semantic_entropy_gate_comparison.json"
    has_stats = (LOCAL_CANON / "statistics_table.json").is_file()
    has_r2 = (LOCAL_CANON / "r2_stratified_analysis.json").is_file()

    if sync_heading not in text and (
        factor_paths or sem_p.is_file() or has_stats or has_r2
    ):
        extra_lines = ["", sync_heading, ""]
        if factor_paths:
            for p in factor_paths:
                rows = parse_result_json(p)
                for r in rows:
                    extra_lines.append(
                        f"- **`{p.name}`** ({r['benchmark']} / {r['protocol']}): "
                        f"acc **{r['acc']}**, n_scored={r['n_scored']}, runtime={r['runtime_min']} min"
                    )
            extra_lines.append("")
        if sem_p.is_file():
            sem = json.loads(sem_p.read_text(encoding="utf-8"))
            ga = float(sem.get("greedy_accuracy") or 0) * 100
            ea = float(sem.get("ecr_based_accuracy") or 0) * 100
            sa = float(sem.get("se_based_accuracy") or 0) * 100
            extra_lines += [
                "### Semantic entropy ablation (MedHallu, n=50, k=5)!",
                "",
                f"- Greedy **{ga:.1f}%** vs ECR-gate **{ea:.1f}%** vs SE-gate **{sa:.1f}%** "
                f"(`semantic_entropy_gate_comparison.json`).",
                "",
            ]
        if has_stats:
            extra_lines.append(
                "- Paired tests: `results/CANONICAL_v2/statistics_table.json`."
            )
        if has_r2:
            extra_lines.append(
                "- R² stratified ALTA analysis: `results/CANONICAL_v2/r2_stratified_analysis.json`."
            )
        extra_lines.append("")
        insert_at = text.find("## Reproducing Results")
        block = "\n".join(extra_lines)
        if insert_at == -1:
            text = text.rstrip() + block + "\n"
        else:
            text = text[:insert_at].rstrip() + "\n" + block + "\n" + text[insert_at:]
    elif sync_heading in text:
        # Refresh FACTOR / sem bullets if section exists: replace block between heading and Reproducing
        # (simple re-insert not implemented; second run updates all_results + JSON only)
        pass

    RESULTS_MD.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-pull",
        action="store_true",
        help="Only refresh markdown using existing local results/CANONICAL_v2/*.json",
    )
    args = ap.parse_args()

    if args.skip_pull:
        pulled = sorted(p.name for p in LOCAL_CANON.glob("*.json"))
        print(f"Skip pull; using {len(pulled)} local JSON files in {LOCAL_CANON}")
    else:
        print("Pulling JSON from A800...")
        pulled = pull_all_json()
        print(f"  {len(pulled)} files -> {LOCAL_CANON}")

    block = build_sync_block()
    ar = ALL_RESULTS.read_text(encoding="utf-8")
    ar = replace_ablation_rows(ar)
    ar = insert_or_replace_sync_section(ar, block)
    ALL_RESULTS.write_text(ar, encoding="utf-8")
    print(f"Updated {ALL_RESULTS.relative_to(ROOT)}")

    patch_results_md()
    print(f"Updated {RESULTS_MD.relative_to(ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
