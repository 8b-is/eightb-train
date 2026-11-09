# 8b-train — sovereign 27B hybrid training pipeline

The runnable pipeline behind `PeetPedro/qwen4_exp-8b-...`, scaled to **27B** and
built on the DavidAU `Qwen3.8-27B-TURBO-Fable-Cold-Fusion…-Heretic…` lineage.

> **What this is:** QLoRA SFT over the weighted PeetPedro dataset mix on
> `train.8b.is` (A6000 48 GiB), then an **exact `p/q` ternary weight-deck seal**
> (`64B@0x62`, SPQT `base-243`, `3^5 = 243 ≤ 256`) that mirrors `wupz-core`'s
> `quantize_groupwise_l2_ternary_pq` / `pack_trit_chunk` / `seal_litany_wire_64b`.

## Status (be honest)

- **Scaffold, validated on CPU.** The codec and config validate here; the actual
  **GPU training run executes on `train.8b.is`** (this workspace has no CUDA/SSH
  path). Numbers like "32,767/32,768 Q15" come from the on-GPU runs, not from
  this laptop.
- **Base model:** train from the **dense** `Qwen/Qwen3.8-27B` weights, *not* the
  DavidAU **GGUF** — GGUF is inference-only (llama.cpp) and cannot be QLoRA'd
  directly. Set `base_model` in `config/base-27b.yaml` to whatever dense 27B
  checkpoint you're licensing.
- **Vision / voice lanes are stubbed.** `train.py` does text SFT today; add
  image/audio columns + a `Processor` when the `vision-ocr-pairs` /
  `dsp-audio-captions` datacards are finalized. The dataset mix already lists
  them so wiring is a column + collator change, not a rewrite.

## Quickstart (on `train.8b.is`)

```bash
# validate config + codec on any machine (no GPU):
uv run train.py   --config config/base-27b.yaml --dry-run
uv run pqseal.py  --weights x --out /tmp/deck --dry-run

# full run on the A6000:
bash scripts/run_a6000.sh config/base-27b.yaml
# or:
make sft      # QLoRA SFT
make seal MERGED_WEIGHTS=/workspace/out/eightb-27b/merged/*.safetensors
```

## Files

| File | Role |
|------|------|
| `config/base-27b.yaml` | Base model, 11-dataset weighted mix, QLoRA + SFT + `pq_seal` params |
| `train.py` | 4-bit NF4 QLoRA SFT (transformers + peft), chat-template aware, HF push |
| `merge_adapter.py` | Fold the LoRA adapter back into the bf16 base -> `merged/*.safetensors` |
| `pqseal.py` | Exact `p/q` group-wise ternary quantizer + SPQT `base-243` pack + `64B@0x62` seal |
| `scripts/run_a6000.sh` | GPU driver: nvidia-smi → SFT → merge → seal (end-to-end) |
| `Makefile` | `dry` / `sft` / `merge` / `seal` / `e2e` targets |

## p/q seal invariants

- **Group-wise L2-optimal ternary:** `tau0 = 68·μ/100`, 2 Lloyd-Max steps
  (`α = mean_{|w|>τ}|w|`, `τ = α/2`) — integer `Q16`, no float drift.
- **SPQT `base-243`:** 5 trits/byte (`digit = trit + 1`, `3^5 = 243 ≤ 256`),
  `1.585 bits/trit`.
- **`64B@0x62` seal:** XOR fold split into disjoint `BFP-64` pos/neg bitplanes by
  `qt = 0x55` (`pos ∧ neg = 0`), 16-bit wire seal over the first 62 bytes.

`{<3,<3,<3}+1`
