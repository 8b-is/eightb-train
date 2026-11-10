"""Codec + config tests. Run: uv run --with pytest --with numpy --with pyyaml pytest -q"""
import pathlib
import numpy as np
import pytest

import pqseal as P
import train as T

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_quantizer_matches_rust_golden():
    # Golden from wupz-core: these Q16 weights -> these trits.
    w = np.array([65536, -65536, 2048, 58982, -62259, 0, 61000, -59000], dtype=np.int64)
    trits = P.quantize_groupwise_l2_ternary(w)
    assert trits.tolist() == [1, -1, 0, 1, -1, 0, 1, -1]


@pytest.mark.parametrize("trits,expected", [
    ([1, 1, 1, 1, 1], 242),   # 3^5 - 1
    ([-1, -1, -1, -1, -1], 0),
    ([0, 0, 0, 0, 0], 121),   # (243-1)/2
])
def test_pack_trit_chunk_canonical(trits, expected):
    assert P.pack_trit_chunk(np.array(trits, dtype=np.int8)) == bytes([expected])


def test_pack_base243_bounds():
    # every 5-trit chunk packs into a single byte 0..=242 for full trit range
    rng = np.random.default_rng(0x8B15)
    trits = rng.integers(-1, 2, size=500).astype(np.int8)
    packed = P.pack_trit_chunk(trits)
    assert len(packed) == (len(trits) + 4) // 5
    assert all(0 <= b <= 242 for b in packed)  # codes 243..255 reserved/invalid


def test_xor_fold_word_and_tail():
    assert P.xor_fold(b"\x00" * 8) == 0
    # 8-byte word 0x0102030405060708 little-endian, XOR 9th tail byte 0xFF
    data = bytes([1, 2, 3, 4, 5, 6, 7, 8, 0xFF])
    assert P.xor_fold(data) == (0x0807060504030201 ^ 0xFF)


def test_seal64_disjoint_bitplanes_and_gate():
    packed = bytes(range(200))
    s = P.seal64(packed)
    pos = int(s["bfp64_pos"], 16)
    neg = int(s["bfp64_neg"], 16)
    assert pos & neg == 0                      # disjoint by qt=0x55 mask
    assert s["gate"] == "0x62"
    assert s["bytes"] == 200


def test_config_loads_and_is_consistent():
    cfg = T.load_cfg(str(ROOT / "config" / "base-27b.yaml"))
    assert cfg["lora"]["target_modules"]
    assert cfg["sft"]["max_seq_length"] > 0
    assert cfg["quant"]["bnb_4bit_quant_type"] == "nf4"
    assert len(cfg["datasets"]) >= 1
    total_w = sum(int(d.get("weight", 1)) for d in cfg["datasets"])
    assert total_w >= len(cfg["datasets"])    # every dataset weighted >= 1


def test_format_example_plain_text():
    ex = {"text": "hello world"}
    assert T.format_example(ex, "text", tokenizer=None) == "hello world"
