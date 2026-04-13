#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CANONICAL_DIR="results/CANONICAL_v2"
ARCHIVE_DIR="results/archive"
mkdir -p "$CANONICAL_DIR" "$ARCHIVE_DIR"

copy_if_exists() {
  local src="$1"
  local dst_dir="$2"
  if [ -f "$src" ]; then
    cp -f "$src" "$dst_dir/"
    echo "[COPY] $src -> $dst_dir/"
  else
    echo "[SKIP] missing: $src"
  fi
}

move_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    mv -f "$src" "$dst"
    echo "[MOVE] $src -> $dst"
  else
    echo "[SKIP] missing: $src"
  fi
}

echo "[INFO] Populating canonical result bundle..."
copy_if_exists "results/results_3b_truthfulqa_full_mc.json" "$CANONICAL_DIR"
copy_if_exists "results/results_8b_truthfulqa_full_mc.json" "$CANONICAL_DIR"
copy_if_exists "results/results_3b_medhallu_n100.json" "$CANONICAL_DIR"
copy_if_exists "results_8b_medhallu_v2.json" "$CANONICAL_DIR"
copy_if_exists "results_8b_pubmedqa_v2.json" "$CANONICAL_DIR"
copy_if_exists "results_8b_medqa_v3_fixed.json" "$CANONICAL_DIR"
copy_if_exists "results_8b_both.json" "$CANONICAL_DIR"

for f in results/results_openrouter_*_v2.json; do
  if [ -f "$f" ]; then
    cp -f "$f" "$CANONICAL_DIR/"
    echo "[COPY] $f -> $CANONICAL_DIR/"
  fi
done

echo "[INFO] Archiving legacy/invalid artifacts..."
move_if_exists "results/results_openrouter_medqa.json" "$ARCHIVE_DIR/results_openrouter_medqa.json"
move_if_exists "results/results_openrouter_pubmedqa.json" "$ARCHIVE_DIR/results_openrouter_pubmedqa.json"
move_if_exists "results/pre_v2_invalid_401" "$ARCHIVE_DIR/pre_v2_invalid_401"

if [ -f "raw_results.md" ]; then
  rm -f "raw_results.md"
  echo "[DELETE] raw_results.md"
fi

if [ -f "results/online_openrouter_status.md" ]; then
  rm -f "results/online_openrouter_status.md"
  echo "[DELETE] results/online_openrouter_status.md"
fi

echo "[DONE] Result organization complete."
