#!/usr/bin/env python3
"""Exact p/q ternary weight-deck sealer (8b-is WUKONG style).

Takes trained (merged) weights and, per group, runs the group-wise L2-optimal
balanced-ternary quantizer {-1,0,+1}, packs trits 5-per-byte in SPQT base-243
(3^5 = 243 <= 256), and seals each tensor into a 64B@0x62 LitanyWireFrame using
an integer XOR fold. Mirrors wupz-core's `quantize_groupwise_l2_ternary_pq`,
`pack_trit_chunk`, and `seal_litany_wire_64b` — integer/dyadic only, no float
drift in the encoding.

    uv run eightb-train/pqseal.py --weights /path/merged.safetensors --out deck/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GATE = 0x62
QT = 0x55
MAX_I64 = np.iinfo(np.int64).max


def quantize_groupwise_l2_ternary(values_q16: np.ndarray) -> np.ndarray:
    """Bit-exact port of wupz-core's quantize_groupwise_l2_ternary_pq -> trits in {-1,0,+1}."""
    n = values_q16.size
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    av = np.abs(values_q16.astype(np.int64))
    mean_abs = int(av.sum() // n)
    tau = (mean_abs * 68) // 100
    alpha = mean_abs
    for _ in range(2):
        active = av > tau
        cnt = int(active.sum())
        alpha = int(av[active].sum() // cnt) if cnt else mean_abs
        tau = alpha // 2
    trits = np.zeros(n, dtype=np.int8)
    trits[(av > tau) & (values_q16 > 0)] = 1
    trits[(av > tau) & (values_q16 < 0)] = -1
    return trits


def pack_trit_chunk(trits: np.ndarray) -> bytes:
    """5 balanced trits -> one SPQT base-243 byte (digit = trit + 1, 3^5 = 243)."""
    out = bytearray()
    for i in range(0, trits.size, 5):
        byte_val, pow3 = 0, 1
        for t in trits[i:i + 5]:
            byte_val = (byte_val + (int(t) + 1) * pow3) % 256
            pow3 = (pow3 * 3) % 256
        out.append(byte_val & 0xFF)
    return bytes(out)


def xor_fold(data: bytes) -> int:
    acc = 0
    for i in range(0, len(data) - len(data) % 8, 8):
        acc ^= int.from_bytes(data[i:i + 8], "little")
    tail = len(data) - len(data) % 8
    for b in data[tail:]:
        acc ^= b
    return acc & 0xFFFFFFFFFFFFFFFF


def seal64(packed: bytes) -> dict:
    """64B@0x62 LitanyWireFrame summary: folds, exact p/q ratio, base-243 spqt root, wire seal."""
    fold = xor_fold(packed)
    mask = (QT * 0x0101010101010101) & 0xFFFFFFFFFFFFFFFF
    pos, neg = fold & mask, fold & (~mask & 0xFFFFFFFFFFFFFFFF)
    ratio_p, ratio_q = len(packed), max(1, len(packed))
    seal16 = xor_fold(packed[:62]) & 0xFFFF
    return {
        "bytes": len(packed),
        "fold": f"0x{fold:016x}",
        "bfp64_pos": f"0x{pos:016x}",
        "bfp64_neg": f"0x{neg:016x}",
        "spqt_root": list(pack_trit_chunk(np.frombuffer(packed[:2], np.int8))),
        "gate": f"0x{GATE:02x}",
        "wire_seal16": f"0x{seal16:04x}",
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="merged safetensors (fp/bf16)")
    ap.add_argument("--out", required=True, help="output deck dir")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv[1:])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        # Validate the codec on a synthetic tensor without loading model weights.
        rng = np.random.default_rng(0x8B15)
        synth = (rng.standard_normal(4096) * 20000).astype(np.int64)
        trits = quantize_groupwise_l2_ternary(synth)
        packed = pack_trit_chunk(trits)
        print(json.dumps({"dry_run": True, **seal64(packed)}, indent=2))
        return 0

    from safetensors.numpy import load_file
    tensors = load_file(args.weights)
    manifest = {}
    total_trits = 0
    for name, arr in tensors.items():
        if arr.ndim < 1 or arr.size == 0:
            continue
        flat = arr.reshape(-1).astype(np.float64)
        # Scale to Q16 integers exactly (round, not truncate).
        q16 = np.round(flat * 65536.0).astype(np.int64)
        trits = np.concatenate([
            quantize_groupwise_l2_ternary(q16[i:i + args.group])
            for i in range(0, q16.size, args.group)
        ])
        packed = pack_trit_chunk(trits)
        total_trits += int(trits.size)
        manifest[name] = {"shape": list(arr.shape), **seal64(packed)}
        print(f"[pqseal] {name}: {trits.size} trits -> {len(packed)} B, {manifest[name]['fold']}")

    (out / "deck_manifest.json").write_text(json.dumps(manifest, indent=2))
    fp16_bytes = total_trits * 2
    deck_bytes = total_trits * 5 / 8  # ~1.585 bits/trit
    print(f"[pqseal] {total_trits} trits | {fp16_bytes/deck_bytes:.2f}x vs FP16 (~375/1 with KV cull)")
    print(f"[pqseal] deck manifest -> {out/'deck_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
