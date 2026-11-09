#!/usr/bin/env python3
"""Merge a trained LoRA adapter into the dense base to produce fp weights.

Feeds pqseal.py: QLoRA training writes a small adapter; to publish (and to run
the p/q ternary seal on real weights) we fold the adapter back into the base and
save a single safetensors file.

    uv run eightb-train/merge_adapter.py \
        --base Qwen/Qwen3.8-27B \
        --adapter /workspace/out/eightb-27b/lora-adapter \
        --out /workspace/out/eightb-27b/merged.safetensors

Notes:
- The base is loaded in bf16 (not 4-bit); merging 4-bit quantized weights is lossy.
- Uses peft `merge_and_unload`, then `save_pretrained(..., safe_serialization=True)`.
- --dry-run validates import wiring only.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="dense base model id/path")
    ap.add_argument("--adapter", required=True, help="trained LoRA adapter dir")
    ap.add_argument("--out", required=True, help="output merged .safetensors path")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv[1:])

    if args.dry_run:
        for name in ("torch", "transformers", "peft", "safetensors"):
            try:
                __import__(name)
                print(f"  import {name}: ok")
            except Exception as e:  # noqa: BLE001
                print(f"  import {name}: MISSING ({e.__class__.__name__})")
        print("[merge] dry-run complete (no merge performed)")
        return 0

    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[merge] loading base {args.base} (bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    print(f"[merge] attaching adapter {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("[merge] merge_and_unload")
    model = model.merge_and_unload()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = str(out)[: -len(".safetensors")] if str(out).endswith(".safetensors") else str(out)
    model.save_pretrained(stem, safe_serialization=True)
    print(f"[merge] saved merged weights -> {stem}/*.safetensors")

    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tok.save_pretrained(stem)
    print("[merge] done. Next: pqseal.py --weights <merged shard> --out deck/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
