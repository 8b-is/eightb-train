#!/usr/bin/env bash
# Full end-to-end run on train.8b.is (A6000 48 GiB): SFT -> merge -> p/q seal.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-config/base-27b.yaml}"

echo "[a6000] GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[a6000] (1/3) QLoRA SFT..."
uv run train.py --config "$CONFIG"

OUT_DIR="$(uv run --no-project python -c 'import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$CONFIG")"
BASE="$(uv run --no-project python -c 'import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))["base_model"])' "$CONFIG")"

echo "[a6000] (2/3) merge adapter -> dense weights..."
uv run merge_adapter.py --base "$BASE" --adapter "$OUT_DIR/lora-adapter" --out "$OUT_DIR/merged.safetensors"

echo "[a6000] (3/3) p/q ternary seal..."
SHARD="$(ls "$OUT_DIR"/merged/*.safetensors 2>/dev/null | head -1 || true)"
uv run pqseal.py --weights "${MERGED_WEIGHTS:-$SHARD}" --out "$OUT_DIR/deck"

echo "[a6000] done -> $OUT_DIR/deck/deck_manifest.json"
