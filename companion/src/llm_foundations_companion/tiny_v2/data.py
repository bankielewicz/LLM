"""Strict document inputs and protected shifted-window mechanics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
from typing import Any

from .tokenizer import Tokenizer


SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Document:
    record_id: str
    split: str
    text: str
    scenario_group_id: str | None = None
    slice_name: str | None = None

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError("Document split must be train, validation, or test.")
        _require_identifier(self.record_id, "record_id")
        _require_text(self.text)
        if self.scenario_group_id is not None:
            _require_identifier(self.scenario_group_id, "scenario_group_id")
        if self.slice_name is not None:
            _require_identifier(self.slice_name, "slice")


def _require_identifier(value: Any, field: str) -> None:
    if (
        not isinstance(value, str)
        or "\x00" in value
        or not value.strip()
        or len(value) > 120
    ):
        raise ValueError(
            f"{field} must be a nonblank identifier of at most 120 scalars."
        )
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"{field} must contain valid Unicode scalars."
        ) from exc


def _require_text(value: Any) -> None:
    if not isinstance(value, str) or "\x00" in value or not value.strip():
        raise ValueError("text must be a nonblank string without NUL.")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("text must contain valid Unicode scalars.") from exc


def _strict_object(line: str, number: int) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"line {number}: duplicate JSON key {key!r}.")
            result[key] = value
        return result

    try:
        value = json.loads(
            line,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"line {number}: non-finite JSON value {token}.")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"line {number}: invalid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"line {number}: record must be a JSON object.")
    return value


def parse_jsonl_documents(
    source: bytes | str,
    *,
    split: str,
) -> tuple[Document, ...]:
    """Parse registered document_text_v1 bytes while preserving stored order."""

    if split not in SPLITS:
        raise ValueError("split must be train, validation, or test.")
    if isinstance(source, bytes):
        try:
            text = source.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("Document JSONL must be valid UTF-8.") from exc
    elif isinstance(source, str):
        try:
            source.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError("Document JSONL must contain valid Unicode scalars.") from exc
        text = source
    else:
        raise TypeError("Document JSONL must be bytes or text.")

    documents: list[Document] = []
    seen: set[str] = set()
    required = {"record_id", "scenario_group_id", "text"}
    allowed = required | {"slice"}
    # Split only on the JSONL delimiter.  str.splitlines() would incorrectly
    # split a valid JSON string containing U+2028 or U+2029.
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        value = _strict_object(line, number)
        if set(value) not in (required, allowed):
            raise ValueError(
                f"line {number}: document_text_v1 has missing or unknown fields."
            )
        record_id = value["record_id"]
        document = Document(
            record_id=record_id,
            split=split,
            text=value["text"],
            scenario_group_id=value["scenario_group_id"],
            slice_name=value.get("slice"),
        )
        if document.record_id in seen:
            raise ValueError(
                f"line {number}: duplicate record_id {document.record_id!r}."
            )
        seen.add(document.record_id)
        documents.append(document)
    if not documents:
        raise ValueError("Document JSONL must contain at least one record.")
    return tuple(documents)


def ordered_texts(
    documents: Iterable[Document],
    split: str,
) -> tuple[str, ...]:
    if split not in SPLITS:
        raise ValueError("split must be train, validation, or test.")
    result: list[str] = []
    for document in documents:
        if not isinstance(document, Document):
            raise TypeError("ordered_texts requires Document values.")
        if document.split == split:
            result.append(document.text)
    return tuple(result)


def training_text_bytes(documents: Iterable[Document]) -> int:
    return sum(
        len(text.encode("utf-8", errors="strict"))
        for text in ordered_texts(documents, "train")
    )


def _stream_texts(
    documents: Iterable[Document | str],
    split: str | None,
) -> tuple[str, ...]:
    result: list[str] = []
    for document in documents:
        if isinstance(document, Document):
            if split is None or document.split == split:
                result.append(document.text)
        elif isinstance(document, str):
            result.append(document)
        else:
            raise TypeError("Token streams require Document or text values.")
    return tuple(result)


def encode_documents(
    documents: Iterable[Document | str],
    tokenizer: Tokenizer,
    context: int,
    *,
    split: str | None = None,
) -> Any:
    """Concatenate stored-order encodings plus EOS and require C+1 tokens."""

    if (
        isinstance(context, bool)
        or not isinstance(context, int)
        or context < 1
    ):
        raise ValueError("context must be a positive integer.")
    if split is not None and split not in SPLITS:
        raise ValueError("split must be train, validation, test, or None.")
    texts = _stream_texts(documents, split)
    ids = [
        token
        for text in texts
        for token in tokenizer.encode(text) + [tokenizer.eos_id]
    ]
    if len(ids) <= context:
        raise ValueError(
            f"Need at least {context + 1} tokens in each split; found {len(ids)}."
        )
    import torch

    return torch.tensor(ids, dtype=torch.long)


def batch(
    data: Any,
    context: int,
    size: int,
    rng: Any,
    device: str,
) -> tuple[Any, Any]:
    """Sample protected uniformly shifted windows using a CPU generator."""

    import torch

    starts = torch.randint(len(data) - context, (size,), generator=rng)
    inputs = torch.stack([data[index : index + context] for index in starts])
    targets = torch.stack(
        [data[index + 1 : index + context + 1] for index in starts]
    )
    return inputs.to(device), targets.to(device)


__all__ = [
    "Document",
    "SPLITS",
    "batch",
    "encode_documents",
    "ordered_texts",
    "parse_jsonl_documents",
    "training_text_bytes",
]
