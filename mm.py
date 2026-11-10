"""Multimodal (vision / voice) lane for the 8b-is trainer.

Text-only training lives in `train.py`. This module adds the *wiring* for the
image and audio lanes so that, when the `vision-ocr-pairs` / `dsp-audio-captions`
datacards and a multimodal base (e.g. a Qwen-VL / Qwen-Audio checkpoint) are
ready, training becomes a config switch — not a rewrite.

Design:
- One `resolve_modality` reads `modality` + the dataset rows and decides whether
  rows carry `__image` / `__audio`.
- `load_processor` wraps `AutoProcessor` (which exposes `.tokenizer` + image /
  audio preprocessing). Requires `transformers`; import-safe otherwise.
- `MultimodalCollator` batches `{"__text", "__image"?, "__audio"?}` rows into
  model-ready tensors via the processor.

All heavy imports are lazy and the module is import-safe (for --dry-run / tests).
"""
from __future__ import annotations

from typing import Any

VALID_MODALITIES = ("text", "vision", "audio", "multimodal")


def resolve_modality(cfg: dict) -> str:
    m = str(cfg.get("modality", "text")).lower()
    if m not in VALID_MODALITIES:
        raise ValueError(f"modality must be one of {VALID_MODALITIES}, got {m!r}")
    return m


def wants_images(m: str) -> bool:
    return m in ("vision", "multimodal")


def wants_audio(m: str) -> bool:
    return m in ("audio", "multimodal")


def load_processor(base_model: str):
    """AutoProcessor for a multimodal base; exposes .tokenizer + image/audio prep."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(base_model, trust_remote_code=True)


def format_multimodal(ex: dict, text_field: str, processor) -> dict:
    """Turn one dataset row into {__text, __image?, __audio?} using the processor tokenizer."""
    tok = getattr(processor, "tokenizer", processor)
    text = ex.get(text_field) or ex.get("text") or ex.get("content") or ""
    if text_field == "messages" or "messages" in ex:
        msgs = ex.get("messages") or ex
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs)
    row: dict[str, Any] = {"__text": text}
    # Image/audio columns are copied through if present; the collator decides.
    if "image" in ex:
        row["__image"] = ex["image"]
    if "audio" in ex:
        row["__audio"] = ex["audio"]
    return row


class MultimodalCollator:
    """Batches multimodal rows into model-ready tensors via an AutoProcessor."""

    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        texts = [r["__text"] for r in batch]
        kwargs: dict[str, Any] = {
            "text": texts,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
        }
        images = [r.get("__image") for r in batch]
        audio = [r.get("__audio") for r in batch]
        if any(i is not None for i in images):
            kwargs["images"] = images
        if any(a is not None for a in audio):
            kwargs["audio"] = audio
        out = self.processor(**kwargs)
        # Causal-LM labels = input ids (loss ignores padding via -100 if configured).
        if "input_ids" in out:
            labels = out["input_ids"].clone()
            pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                labels[labels == pad_id] = -100
            out["labels"] = labels
        return out
