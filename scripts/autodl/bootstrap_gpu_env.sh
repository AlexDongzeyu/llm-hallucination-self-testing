#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a Linux GPU instance for this repo without changing project layout.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/llm-env"

cd "${ROOT_DIR}"

PY_BOOTSTRAP="${PY_BOOTSTRAP:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SKIP_TORCH_INSTALL="${SKIP_TORCH_INSTALL:-0}"

if ! command -v "${PY_BOOTSTRAP}" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: ${PY_BOOTSTRAP}" >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "[INFO] Creating virtual environment at ${VENV_DIR}"
  "${PY_BOOTSTRAP}" -m venv "${VENV_DIR}"
else
  echo "[INFO] Reusing existing virtual environment at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [ "${SKIP_TORCH_INSTALL}" != "1" ]; then
  echo "[INFO] Installing PyTorch from ${TORCH_INDEX_URL}"
  python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
else
  echo "[INFO] Skipping torch install because SKIP_TORCH_INSTALL=1"
fi

echo "[INFO] Installing project requirements"
python -m pip install -r requirements.txt

echo "[INFO] Verifying GPU visibility"
python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device_name={torch.cuda.get_device_name(0)}")
PY

echo "[DONE] Environment is ready."
