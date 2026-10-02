from __future__ import annotations

import pytest

from llm_foundations_companion.tiny_v2.data import (
    Document,
    ordered_texts,
    parse_jsonl_documents,
    training_text_bytes,
)


def test_document_jsonl_preserves_line_order_and_split_identity() -> None:
    raw = (
        '{"record_id":"b","scenario_group_id":"g2","text":"snowman \\u2603"}\n'
        '{"record_id":"a","scenario_group_id":"g1","slice":"s1","text":"two"}\n'
    ).encode()
    documents = parse_jsonl_documents(raw, split="train")
    assert [document.record_id for document in documents] == ["b", "a"]
    assert ordered_texts(documents, "train") == ("snowman ☃", "two")
    assert ordered_texts(documents, "validation") == ()
    assert documents[1].slice_name == "s1"
    assert training_text_bytes(documents) == len("snowman ☃two".encode("utf-8"))


def test_split_helpers_do_not_reorder_combined_registered_records() -> None:
    documents = (
        Document("v2", "validation", "V2", "vg2"),
        Document("t2", "train", "T2", "tg2"),
        Document("t1", "train", "T1", "tg1"),
        Document("v1", "validation", "V1", "vg1"),
    )
    assert ordered_texts(documents, "train") == ("T2", "T1")
    assert ordered_texts(documents, "validation") == ("V2", "V1")
    assert training_text_bytes(documents) == 4


@pytest.mark.parametrize(
    "raw,match",
    [
        (
            b'{"record_id":"r","scenario_group_id":"g","text":"x","extra":1}\n',
            "missing or unknown",
        ),
        (
            b'{"record_id":"r","record_id":"x","scenario_group_id":"g","text":"x"}\n',
            "duplicate JSON key",
        ),
        (
            b'{"record_id":"r","scenario_group_id":"g","text":"x"}\n'
            b'{"record_id":"r","scenario_group_id":"g2","text":"y"}\n',
            "duplicate record_id",
        ),
        (
            b'{"record_id":"r","scenario_group_id":"g","text":" \\t"}\n',
            "nonblank",
        ),
        (
            b'{"record_id":"r","scenario_group_id":"g","text":"a\\u0000b"}\n',
            "without NUL",
        ),
        (b"\xff", "valid UTF-8"),
        (b"\n\n", "at least one"),
    ],
)
def test_document_jsonl_rejects_invalid_registered_shapes(
    raw: bytes, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        parse_jsonl_documents(raw, split="train")


def test_document_jsonl_ignores_blank_lines_but_keeps_text_bytes_exact() -> None:
    raw = (
        '\n{"record_id":"r1","scenario_group_id":"g1","text":"  keep  "}\n\n'
    )
    documents = parse_jsonl_documents(raw, split="validation")
    assert documents == (
        Document("r1", "validation", "  keep  ", "g1"),
    )
    assert training_text_bytes(documents) == 0


def test_document_jsonl_does_not_treat_unicode_line_separator_as_jsonl() -> None:
    raw = (
        '{"record_id":"r1","scenario_group_id":"g1",'
        '"text":"before after"}\n'
    )
    documents = parse_jsonl_documents(raw, split="train")
    assert documents[0].text == "before after"


def test_nonstring_identifier_is_rejected_as_validation_error() -> None:
    raw = (
        b'{"record_id":[],"scenario_group_id":"g1","text":"text"}\n'
    )
    with pytest.raises(ValueError, match="record_id"):
        parse_jsonl_documents(raw, split="train")
