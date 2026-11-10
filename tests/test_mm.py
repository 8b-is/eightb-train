"""Multimodal lane (mm.py) tests — wiring only, no model download."""
import pytest

import mm


def test_resolve_modality_valid_and_invalid():
    assert mm.resolve_modality({}) == "text"
    assert mm.resolve_modality({"modality": "vision"}) == "vision"
    assert mm.resolve_modality({"modality": "AUDIO"}) == "audio"
    with pytest.raises(ValueError):
        mm.resolve_modality({"modality": "telepathy"})


def test_wants_image_audio():
    assert mm.wants_images("vision") and mm.wants_images("multimodal")
    assert not mm.wants_images("audio") and not mm.wants_images("text")
    assert mm.wants_audio("audio") and mm.wants_audio("multimodal")
    assert not mm.wants_audio("vision")


class _StubTok:
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        raise RuntimeError("no template")  # exercise the fallback join


class _StubProc:
    tokenizer = _StubTok()


def test_format_multimodal_plain_and_messages():
    p = _StubProc()
    assert mm.format_multimodal({"text": "hi"}, "text", p) == {"__text": "hi"}
    row = {"messages": [{"role": "user", "content": "yo"}]}
    out = mm.format_multimodal(row, "messages", p)
    assert out["__text"] == "user: yo"


def test_format_multimodal_passthrough_media():
    p = _StubProc()
    out = mm.format_multimodal({"text": "t", "image": b"img", "audio": b"wav"}, "text", p)
    assert out["__image"] == b"img" and out["__audio"] == b"wav"
