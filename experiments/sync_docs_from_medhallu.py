"""Sync MedHallu metrics/timestamps into README.md and raw_results.md.

This script only updates the known table rows and timestamp fields so manual
notes in the documents remain intact.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MED_PATH = ROOT / "results" / "medhallu_results.json"
README_PATH = ROOT / "README.md"
RAW_PATH = ROOT / "raw_results.md"


def load_medhallu():
    payload = json.loads(MED_PATH.read_text(encoding="utf-8"))
    by_label = {r["label"]: r for r in payload["results"]}
    return payload, by_label


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def update_readme(text: str, by_label: dict, payload: dict) -> str:
    text = re.sub(
        r"(\| greedy \| )[^|]+( \| )[^|]+( \|)",
        rf"\g<1>{pct(by_label['greedy']['accuracy'])}\g<2>{pct(by_label['greedy']['rep_rate'])}\g<3>",
        text,
    )
    text = re.sub(
        r"(\| cove \| )[^|]+( \| )[^|]+( \|)",
        rf"\g<1>{pct(by_label['cove']['accuracy'])}\g<2>{pct(by_label['cove']['rep_rate'])}\g<3>",
        text,
    )
    text = re.sub(
        r"(\| dynamic \| )[^|]+( \| )[^|]+( \|)",
        rf"\g<1>{pct(by_label['dynamic']['accuracy'])}\g<2>{pct(by_label['dynamic']['rep_rate'])}\g<3>",
        text,
    )
    text = re.sub(
        r"(\| gadr2 \| )[^|]+( \| )[^|]+( \|)",
        rf"\g<1>{pct(by_label['gadr2']['accuracy'])}\g<2>{pct(by_label['gadr2']['rep_rate'])}\g<3>",
        text,
    )

    mtime = MED_PATH.stat().st_mtime
    # Keep format aligned with existing text style.
    from datetime import datetime

    stamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    text = re.sub(
        r"(Latest standalone MedHallu refresh: `results/medhallu_results\.json` last written `)[^`]+(`)",
        rf"\g<1>{stamp}\g<2>",
        text,
    )
    return text


def update_raw_results(text: str, by_label: dict, payload: dict) -> str:
    text = re.sub(
        r"(## 10\. MedHallu Evaluation \(`results/medhallu_results\.json`\)[\s\S]*?- runtime_min: )[^\n]+",
        rf"\g<1>{payload['runtime_min']}",
        text,
        count=1,
    )

    text = re.sub(
        r"(\| greedy \| 50 \| 0 \| )[^|]+( \| )[^|]+( \|)",
        rf"\g<1>{by_label['greedy']['accuracy']:.2f}\g<2>{by_label['greedy']['rep_rate']:.2f}\g<3>",
        text,
    )
    text = re.sub(
        r"(\| cove \| 50 \| 0 \| )[^|]+( \| )[^|]+( \|)",
        rf"\g<1>{by_label['cove']['accuracy']:.2f}\g<2>{by_label['cove']['rep_rate']:.2f}\g<3>",
        text,
    )
    text = re.sub(
        r"(\| dynamic \| 50 \| 0 \| )[^|]+( \| )[^|]+( \|)",
        rf"\g<1>{by_label['dynamic']['accuracy']:.2f}\g<2>{by_label['dynamic']['rep_rate']:.2f}\g<3>",
        text,
    )
    text = re.sub(
        r"(\| gadr2 \| 50 \| 0 \| )[^|]+( \| )[^|]+( \|)",
        rf"\g<1>{by_label['gadr2']['accuracy']:.2f}\g<2>{by_label['gadr2']['rep_rate']:.2f}\g<3>",
        text,
    )

    from datetime import datetime

    stamp = datetime.fromtimestamp(MED_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    text = re.sub(
        r"(Latest standalone artifact refresh:\n\n- `results/medhallu_results\.json` last written `)[^`]+(`)",
        rf"\g<1>{stamp}\g<2>",
        text,
    )

    return text


def main() -> None:
    payload, by_label = load_medhallu()

    readme_text = README_PATH.read_text(encoding="utf-8")
    raw_text = RAW_PATH.read_text(encoding="utf-8")

    readme_new = update_readme(readme_text, by_label, payload)
    raw_new = update_raw_results(raw_text, by_label, payload)

    README_PATH.write_text(readme_new, encoding="utf-8")
    RAW_PATH.write_text(raw_new, encoding="utf-8")

    print("Updated README.md and raw_results.md from medhallu_results.json")


if __name__ == "__main__":
    main()
