from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from llm_foundations_companion.tiny_v2.tokenizer import (
    Tokenizer,
    load_tokenizer_bytes,
    merge,
    serialize_tokenizer,
    tokenizer_type,
    train_tokenizer,
)


REPOSITORY = Path(__file__).resolve().parents[2]
PROTECTED_PATH = REPOSITORY / "course" / "labs" / "tokenizer.py"
SPEC = importlib.util.spec_from_file_location(
    "protected_teaching_tokenizer", PROTECTED_PATH
)
assert SPEC is not None and SPEC.loader is not None
protected = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protected)


def test_nonoverlapping_merge_matches_protected_left_to_right_rule() -> None:
    assert merge([97, 97, 97, 97, 97], (97, 97), 257) == [
        257,
        257,
        97,
    ]


@pytest.mark.parametrize("vocab_size", [257, 258, 280, 320])
def test_training_encoding_and_fingerprint_match_protected_source(
    vocab_size: int,
) -> None:
    documents = [
        "banana bandana",
        "banana\n",
        "naïve snowman ☃",
        " whitespace\tstays ",
    ]
    expected = protected.Tokenizer.train(documents, vocab_size=vocab_size)
    actual = Tokenizer.train(documents, vocab_size=vocab_size)
    assert actual.merges == expected.merges
    assert actual.vocab_size == expected.vocab_size
    assert actual.fingerprint() == expected.fingerprint()
    for text in documents + ["unseen Ω text"]:
        assert actual.encode(text) == expected.encode(text)
        assert actual.decode(actual.encode(text)) == text


def test_tokenizer_artifact_bytes_and_digest_roles_are_exact() -> None:
    tokenizer = Tokenizer([(97, 98), (257, 99)])
    artifact = serialize_tokenizer(tokenizer)
    assert artifact == json.dumps(tokenizer.to_dict(), indent=2).encode("utf-8")
    assert not artifact.endswith(b"\n")
    assert load_tokenizer_bytes(artifact).to_dict() == tokenizer.to_dict()
    fingerprint_bytes = json.dumps(
        tokenizer.to_dict(), sort_keys=True
    ).encode()
    assert tokenizer.fingerprint() == hashlib.sha256(fingerprint_bytes).hexdigest()
    assert hashlib.sha256(artifact).hexdigest() != tokenizer.fingerprint()


def test_profiles_preserve_seed_independence_and_report_actual_type() -> None:
    byte = train_tokenizer(["abab"], "byte-v1", 257, seed=1)
    bpe_a = train_tokenizer(["abab"], "byte-bpe-v1", 320, seed=1)
    bpe_b = train_tokenizer(["abab"], "byte-bpe-v1", 320, seed=999)
    assert tokenizer_type(byte) == "byte"
    assert tokenizer_type(bpe_a) == "byte_bpe"
    assert bpe_a.merges == bpe_b.merges
    assert bpe_a.vocab_size < 320


def test_invalid_profile_and_strict_artifact_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="requires vocabulary"):
        train_tokenizer(["text"], "byte-v1", 258, seed=0)
    with pytest.raises(ValueError, match="Unsupported tokenizer profile"):
        train_tokenizer(["text"], "unknown", 257, seed=0)
    with pytest.raises(ValueError, match="duplicate key"):
        load_tokenizer_bytes(
            b'{"format":"teaching-byte-bpe-v1","format":"x","merges":[]}'
        )
    with pytest.raises(ValueError, match="unavailable token"):
        Tokenizer([(256, 1)])
    with pytest.raises(ValueError, match="unavailable token"):
        Tokenizer([(257, 1)])


def test_eos_is_metadata_and_replacement_decode_is_explicit() -> None:
    tokenizer = Tokenizer()
    assert tokenizer.decode([65, tokenizer.eos_id, 66]) == "AB"
    with pytest.raises(UnicodeDecodeError):
        tokenizer.decode([0xF0])
    assert tokenizer.decode([0xF0], errors="replace") == "\ufffd"


def test_malformed_merge_chain_is_bounded_before_oversized_piece_allocation() -> None:
    merges: list[tuple[int, int]] = []
    previous = 97
    for index in range(18):
        merges.append((previous, previous))
        previous = 257 + index
    with pytest.raises(ValueError, match="exceeds 200000"):
        Tokenizer(merges)


def test_loaded_tokenizer_cannot_exceed_supported_vocabulary() -> None:
    merges = [(97, 98)] * 768
    with pytest.raises(ValueError, match="more than 767"):
        Tokenizer(merges)
