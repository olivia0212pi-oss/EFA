#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTODL_DATA_DIR="${AUTODL_DATA_DIR:-/root/autodl-tmp}"

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx reason; then
  conda create -n reason python=3.10 -y
fi
conda activate reason

mkdir -p "${AUTODL_DATA_DIR}/huggingface"
export HF_HOME="${AUTODL_DATA_DIR}/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "${PROJECT_DIR}"
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -m generation.check_gpu

echo
echo "Environment ready. Activate it with: conda activate reason"
echo "Set model cache with: export HF_HOME=${AUTODL_DATA_DIR}/huggingface"

