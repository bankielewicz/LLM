"""Protected teaching byte/BPE mechanics for the fixed tiny-v2 worker."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from typing import Any


BYTE_TOKENIZER_PROFILE = "byte-v1"
BPE_TOKENIZER_PROFILE = "byte-bpe-v1"
TOKENIZER_FORMAT = "teaching-byte-bpe-v1"
MAX_MERGES = 1024 - 257
MAX_PIECE_BYTES = 200_000


def merge(seq: Sequence[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace non-overlapping adjacent pairs from left to right."""

    out: list[int] = []
    index = 0
    while index < len(seq):
        if (
            index + 1 < len(seq)
            and (seq[index], seq[index + 1]) == pair
        ):
            out.append(new_id)
            index += 2
        else:
            out.append(seq[index])
            index += 1
    return out


class Tokenizer:
    """The protected whole-document teaching tokenizer."""

    eos_id = 256

    def __init__(self, merges: Iterable[Sequence[int]] = ()) -> None:
        normalized: list[tuple[int, int]] = []
        for index, value in enumerate(merges):
            if index >= MAX_MERGES:
                raise ValueError("Tokenizer has more than 767 merges.")
            if (
                isinstance(value, (str, bytes, bytearray))
                or not isinstance(value, Sequence)
                or len(value) != 2
            ):
                raise ValueError("Each tokenizer merge must contain two token IDs.")
            left, right = value
            if (
                isinstance(left, bool)
                or not isinstance(left, int)
                or isinstance(right, bool)
                or not isinstance(right, int)
            ):
                raise ValueError("Tokenizer merge IDs must be integers.")
            upper = 257 + index
            for token_id in (left, right):
                if token_id < 0 or token_id == self.eos_id or token_id >= upper:
                    raise ValueError("Tokenizer merge references an unavailable token ID.")
            normalized.append((left, right))

        self.merges = normalized
        self.pieces = {token_id: bytes([token_id]) for token_id in range(256)}
        for new_id, (left, right) in enumerate(self.merges, 257):
            left_piece = self.pieces[left]
            right_piece = self.pieces[right]
            if len(left_piece) > MAX_PIECE_BYTES - len(right_piece):
                raise ValueError(
                    "Tokenizer merge piece exceeds 200000 UTF-8 bytes."
                )
            self.pieces[new_id] = left_piece + right_piece

    @property
    def vocab_size(self) -> int:
        return 257 + len(self.merges)

    @classmethod
    def train(
        cls, documents: Iterable[str], vocab_size: int = 320
    ) -> "Tokenizer":
        if (
            isinstance(vocab_size, bool)
            or not isinstance(vocab_size, int)
            or not 257 <= vocab_size <= 1024
        ):
            raise ValueError("Teaching tokenizer vocabulary must be 257..1024.")
        sequences = [list(document.encode("utf-8")) for document in documents]
        if not any(sequences):
            raise ValueError("Tokenizer training needs nonempty text.")
        merges: list[tuple[int, int]] = []
        while 257 + len(merges) < vocab_size:
            counts = Counter(
                pair
                for sequence in sequences
                for pair in zip(sequence, sequence[1:])
            )
            if not counts:
                break
            pair, count = min(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
            if count < 2:
                break
            new_id = 257 + len(merges)
            merges.append(pair)
            sequences = [
                merge(sequence, pair, new_id) for sequence in sequences
            ]
        return cls(merges)

    def encode(self, text: str) -> list[int]:
        sequence = list(text.encode("utf-8"))
        for new_id, pair in enumerate(self.merges, 257):
            sequence = merge(sequence, pair, new_id)
        return sequence

    def decode(self, ids: Iterable[int], errors: str = "strict") -> str:
        return b"".join(
            self.pieces[token_id]
            for token_id in ids
            if token_id != self.eos_id
        ).decode("utf-8", errors)

    def to_dict(self) -> dict[str, Any]:
        return {"format": TOKENIZER_FORMAT, "merges": self.merges}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Tokenizer":
        if not isinstance(value, Mapping) or set(value) != {"format", "merges"}:
            raise ValueError("Tokenizer data must contain exactly format and merges.")
        if value.get("format") != TOKENIZER_FORMAT:
            raise ValueError("Unsupported tokenizer format.")
        merges = value.get("merges")
        if not isinstance(merges, list):
            raise ValueError("Tokenizer merges must be a JSON array.")
        return cls(merges)

    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()


def train_tokenizer(
    documents: Iterable[str],
    tokenizer_profile_id: str,
    vocab_size: int,
    seed: int,
) -> Tokenizer:
    """Train the fixed deterministic tokenizer; seed is retained metadata."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2_147_483_647
    ):
        raise ValueError("Tokenizer seed must be an integer from 0 through 2147483647.")
    if tokenizer_profile_id == BYTE_TOKENIZER_PROFILE:
        if vocab_size != 257:
            raise ValueError("byte-v1 requires vocabulary size 257.")
    elif tokenizer_profile_id != BPE_TOKENIZER_PROFILE:
        raise ValueError("Unsupported tokenizer profile.")
    return Tokenizer.train(documents, vocab_size=vocab_size)


def serialize_tokenizer(tokenizer: Tokenizer) -> bytes:
    """Return the exact protected tokenizer.json bytes (with no trailing LF)."""

    return json.dumps(tokenizer.to_dict(), indent=2).encode("utf-8")


def tokenizer_type(tokenizer: Tokenizer) -> str:
    return "byte" if not tokenizer.merges else "byte_bpe"


def load_tokenizer_bytes(raw: bytes) -> Tokenizer:
    """Load a strict tokenizer artifact without accepting duplicate JSON keys."""

    if not isinstance(raw, bytes):
        raise TypeError("Tokenizer artifact must be bytes.")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError("Tokenizer JSON contains a duplicate key.")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"Tokenizer JSON contains non-finite value {token}.")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Tokenizer artifact is not valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError("Tokenizer artifact must contain a JSON object.")
    return Tokenizer.from_dict(value)


__all__ = [
    "BPE_TOKENIZER_PROFILE",
    "BYTE_TOKENIZER_PROFILE",
    "MAX_MERGES",
    "MAX_PIECE_BYTES",
    "TOKENIZER_FORMAT",
    "Tokenizer",
    "load_tokenizer_bytes",
    "merge",
    "serialize_tokenizer",
    "tokenizer_type",
    "train_tokenizer",
]
