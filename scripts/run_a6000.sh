#!/usr/bin/env bash
# Run on train.8b.is (A6000 48 GiB). Assumes `uv` and CUDA wheels installed.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-config/base-27b.yaml}"

echo "[a6000] GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[a6000] QLoRA SFT..."
uv run train.py --config "$CONFIG"

echo "[a6000] (merge adapter -> fp weights here; use peft merge_and_unload)"
echo "[a6000] p/q ternary seal..."
uv run pqseal.py --weights "${MERGED_WEIGHTS:-/workspace/out/eightb-27b/merged.safetensors}" --out deck/

echo "[a6000] done."
