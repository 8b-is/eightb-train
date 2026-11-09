#!/usr/bin/env python3
"""8b-is sovereign 27B hybrid — QLoRA SFT trainer.

Trains an 8B-SOTA-style hybrid (language now; vision/voice lanes stubbed) from a
dense Qwen base with 4-bit NF4 QLoRA, over the weighted PeetPedro dataset mix,
then optionally runs the exact p/q ternary seal (see pqseal.py).

Run on train.8b.is (A6000 48 GiB). Requires: torch, transformers, peft, datasets,
accelerate, bitsandbytes (Linux+CUDA).

    uv run eightb-train/train.py --config eightb-train/config/base-27b.yaml
    uv run eightb-train/train.py --config ... --dry-run   # validate, no GPU

This module is import-safe (no torch import at module load) so --dry-run works
on a laptop.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import yaml


def load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def format_example(ex: dict, text_field: str, tokenizer) -> str:
    """Turn a dataset row into a single training string."""
    if text_field == "messages" or "messages" in ex:
        msgs = ex.get("messages") or ex
        # Prefer the model's chat template; fall back to a plain join.
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            return "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in msgs)
    return ex.get(text_field) or ex.get("text") or ex.get("content") or ""


def build_mixture(cfg: dict, tokenizer, seed: int):
    """Load + weight-sample the dataset mix into one interleaved list of strings."""
    from datasets import load_dataset, concatenate_datasets

    rng = random.Random(seed)
    shards = []
    for spec in cfg["datasets"]:
        ds = load_dataset(spec["path"], split=spec.get("split", "train"))
        tf = spec.get("text_field", "text")
        ds = ds.map(lambda ex, tf=tf: {"__text": format_example(ex, tf, tokenizer)},
                    remove_columns=ds.column_names)
        ds = ds.filter(lambda ex: bool(ex["__text"] and ex["__text"].strip()))
        # Repeat by weight so the concatenation respects the mixture ratio.
        for _ in range(int(spec.get("weight", 1))):
            shards.append(ds)
    mixed = concatenate_datasets(shards).shuffle(seed=seed)
    return mixed, rng


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="validate config/imports only")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args(argv[1:])

    cfg = load_cfg(args.config)
    print(f"[eightb-train] run={cfg['run_name']} base={cfg['base_model']}")
    print(f"[eightb-train] datasets={len(cfg['datasets'])} lora.r={cfg['lora']['r']} "
          f"seq={cfg['sft']['max_seq_length']}")

    if args.dry_run:
        # Validate the mixture math without touching CUDA.
        total_w = sum(int(d.get("weight", 1)) for d in cfg["datasets"])
        print(f"[eightb-train] mixture weight total = {total_w}")
        for name in ("torch", "transformers", "peft", "datasets", "bitsandbytes", "trl"):
            try:
                __import__(name)
                print(f"  import {name}: ok")
            except Exception as e:  # noqa: BLE001
                print(f"  import {name}: MISSING ({e.__class__.__name__})")
        print("[eightb-train] dry-run complete (no training performed)")
        return 0

    # ---- heavy imports only after dry-run gate ----
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    seed = 0x8B15
    random.seed(seed)
    torch.manual_seed(seed)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    q = cfg["quant"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"], bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    mixed, _ = build_mixture(cfg, tok, seed)
    max_len = cfg["sft"]["max_seq_length"]

    def tokenize(batch):
        out = tok(batch["__text"], truncation=True, max_length=max_len)
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        return out

    tok_ds = mixed.map(tokenize, batched=True, remove_columns=mixed.column_names)

    s = cfg["sft"]
    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=s["per_device_train_batch_size"],
        gradient_accumulation_steps=s["gradient_accumulation_steps"],
        learning_rate=float(s["learning_rate"]),
        num_train_epochs=float(s["num_train_epochs"]),
        warmup_ratio=s["warmup_ratio"],
        lr_scheduler_type=s["lr_scheduler_type"],
        logging_steps=s["logging_steps"],
        save_steps=s["save_steps"],
        gradient_checkpointing=s["gradient_checkpointing"],
        bf16=s["bf16"],
        optim=s["optim"],
        report_to=["none"],
        save_total_limit=3,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tok_ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()

    adapter_dir = f"{cfg['output_dir']}/lora-adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"[eightb-train] adapter saved -> {adapter_dir}")

    if not args.no_push:
        from huggingface_hub import HfApi
        HfApi().create_repo(cfg["target_repo"], exist_ok=True, private=True)
        HfApi().upload_folder(folder_path=adapter_dir, repo_id=cfg["target_repo"])
        print(f"[eightb-train] pushed -> https://huggingface.co/{cfg['target_repo']}")

    print("[eightb-train] done. Next: run pqseal.py on merged weights for the 64B@0x62 deck.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
