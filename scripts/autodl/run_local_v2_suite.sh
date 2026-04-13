#!/usr/bin/env bash
set -euo pipefail

# Run the three local 8B custom v2 jobs with the repository's canonical outputs.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/llm-env/bin/python}"
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
N_CUSTOM="${N_CUSTOM:-100}"
PROTOCOLS="${PROTOCOLS:-greedy,alta,cove,cured}"
CANONICAL_DIR="results/CANONICAL_v2"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
SKIP_ITI="${SKIP_ITI:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
TS="$(date +%Y%m%d_%H%M%S)"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
  echo "[HINT] Run scripts/autodl/bootstrap_gpu_env.sh first." >&2
  exit 1
fi

mkdir -p logs "${CANONICAL_DIR}"

run_job() {
  local name="$1"
  local csv_path="$2"
  local scoring="$3"
  local max_tokens="$4"
  local out_json="$5"
  local log_file="logs/${name}_${TS}.log"

  if [ -f "${out_json}" ] && [ "${FORCE_RERUN}" != "1" ]; then
    echo "[SKIP] ${out_json} already exists (set FORCE_RERUN=1 to overwrite)."
    return 0
  fi

  local -a cmd=(
    "${PYTHON_BIN}" -u cured.py
    --model "${MODEL_ID}"
    --protocols "${PROTOCOLS}"
    --benchmark custom
    --custom-csv "${csv_path}"
    --question-col question
    --answer-col answer
    --n "${N_CUSTOM}"
    --scoring "${scoring}"
    --max-new-tokens "${max_tokens}"
    --out "${out_json}"
  )

  if [ "${LOAD_IN_4BIT}" = "1" ]; then
    cmd+=(--load-in-4bit)
  fi

  if [ "${SKIP_ITI}" = "1" ]; then
    cmd+=(--skip-iti)
  fi

  echo "[RUN] ${name}"
  echo "      output=${out_json}"
  echo "      log=${log_file}"
  "${cmd[@]}" > "${log_file}" 2>&1
  echo "[DONE] ${name}"
}

run_job "local_medqa_v2" "benchmarks/medqa_usmle_n200.csv" "letter" "40" "${CANONICAL_DIR}/results_8b_medqa_v2.json"
run_job "local_pubmedqa_v2" "benchmarks/pubmedqa_n200.csv" "yesno" "10" "${CANONICAL_DIR}/results_8b_pubmedqa_v2.json"
run_job "local_medhallu_v2" "benchmarks/medhallu_n200.csv" "cosine" "80" "${CANONICAL_DIR}/results_8b_medhallu_v2.json"

echo "[DONE] Local v2 suite complete."
