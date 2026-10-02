"""Worker operation for deterministic TinyLM tokenizer training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..tiny_v2.data import ordered_texts, parse_jsonl_documents
from ..tiny_v2.tokenizer import (
    serialize_tokenizer,
    tokenizer_type,
    train_tokenizer,
)


def handle_tokenizer_train(
    request: Mapping[str, Any],
    context: Any,
) -> Mapping[str, Any]:
    """Train from the verified train split and commit the exact tokenizer bytes."""

    source = context.input("dataset.train")
    documents = parse_jsonl_documents(source.path.read_bytes(), split="train")
    tokenizer = train_tokenizer(
        ordered_texts(documents, "train"),
        tokenizer_profile_id=request["tokenizer_profile_id"],
        vocab_size=request["vocab_size"],
        seed=request["seed"],
    )
    artifact = context.stage_artifact(
        "tokenizer_json",
        serialize_tokenizer(tokenizer),
        filename="tokenizer.json",
    )
    tokenizer_id = context.output_allocations["tokenizer_id"]
    if tokenizer_id is None:
        raise ValueError("tokenizer_train has no parent-allocated tokenizer identity")
    return {
        "operation": "tokenizer_train",
        "tokenizer_id": tokenizer_id,
        "tokenizer_type": tokenizer_type(tokenizer),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": tokenizer.fingerprint(),
        "artifact_ids": [artifact["artifact_id"]],
    }


__all__ = ["handle_tokenizer_train"]
