#!/usr/bin/env python3
"""
add_scoring_to_cured.py

Compatibility helper for the requested workflow.
If cured.py already contains scoring support, this script reports success and exits.
"""

from __future__ import annotations

from pathlib import Path
import sys

TARGET = Path("cured.py")


def main() -> None:
    if not TARGET.exists():
        sys.exit("cured.py not found. Run this from repo root.")

    code = TARGET.read_text(encoding="utf-8")
    required_tokens = [
        "--scoring",
        "def letter_match(",
        "def yesno_match(",
        "def reference_match(",
        "scoring=str(args.scoring)",
    ]

    missing = [tok for tok in required_tokens if tok not in code]
    if missing:
        print("Scoring patch is incomplete.")
        print("Missing tokens:")
        for tok in missing:
            print(f"  - {tok}")
        sys.exit(1)

    print("cured.py already contains scoring patch. No changes needed.")


if __name__ == "__main__":
    main()
