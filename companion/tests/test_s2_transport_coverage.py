from __future__ import annotations

import asyncio
import errno
import io
import os
import stat
import uuid
import zipfile
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import llm_foundations_companion.datasets as datasets_module
import llm_foundations_companion.transport as transport_module
from llm_foundations_companion.database import Database
from llm_foundations_companion.datasets import DatasetRegistry
from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.registry import Registry, StagedArtifact
from llm_foundations_companion.schema import canonical_json
from llm_foundations_companion.transport import (
    StagedMultipart,
    inspect_zip_archive,
    receive_json,
    receive_multipart,
    require_media_type,
)


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def _receive(chunks: Iterable[bytes]):
    remaining = list(chunks)

    async def receive():
        if not remaining:
            raise AssertionError("body receiver called after final chunk")
        body = remaining.pop(0)
        return {
            "type": "http.request",
            "body": body,
            "more_body": bool(remaining),
        }

    return receive


def _multipart(boundary: str, parts: list[tuple[str, str, bytes]]) -> bytes:
    body = bytearray()
    for name, media_type, payload in parts:
        body.extend(f"--{boundary}\r\n".encode("ascii"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n'.encode("ascii")
        )
        body.extend(f"Content-Type: {media_type}\r\n\r\n".encode("ascii"))
        body.extend(payload)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("ascii"))
    return bytes(body)


def _raw_part(boundary: str, headers: bytes, payload: bytes = b"x") -> bytes:
    return (
        f"--{boundary}\r\n".encode("ascii")
        + headers
        + b"\r\n\r\n"
        + payload
        + f"\r\n--{boundary}--\r\n".encode("ascii")
    )


def _stack(tmp_path: Path) -> tuple[Database, Registry, DatasetRegistry]:
    root = tmp_path / "root"
    database = Database(root)
    database.initialize()
    registry = Registry(database, root, str(uuid.uuid4()), os.urandom(32))
    return database, registry, DatasetRegistry(registry)


def _reserve(registry: Registry) -> str:
    owner = str(uuid.uuid4())
    registry.reserve_capacity(
        owner_kind="upload",
        owner_id=owner,
        byte_count=64 * 1024 * 1024,
        artifact_rows=5,
        dataset_rows=1,
        reservation_id=owner,
    )
    return owner


def _document_row(**changes: Any) -> bytes:
    value: dict[str, Any] = {
        "record_id": "record-1",
        "scenario_group_id": "group-1",
        "text": "safe text",
    }
    value.update(changes)
    return canonical_json(value) + b"\n"


def _write_zip(path: Path, entries: list[tuple[zipfile.ZipInfo | str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)


def test_public_header_json_and_staged_lookup_fail_closed() -> None:
    with pytest.raises(KeyError):
        StagedMultipart({}, "0" * 64, (), "1" * 64).part("absent")

    invalid_headers = [
        {},
        {"Content-Type": "application/json", "content-type": "application/json"},
        {"Content-Type": "application/json\r\nX: value"},
        {"Content-Type": "application/json; charset=utf-8; charset=utf-8"},
        {"Content-Type": "application/json; charset=latin-1"},
        {"Content-Type": "text/plain; charset=utf-8"},
    ]
    for headers in invalid_headers:
        with pytest.raises(ApiError) as caught:
            require_media_type(headers, "application/json")
        assert caught.value.code == "UNSUPPORTED_MEDIA_TYPE"

    async def wrong_message():
        return {"type": "http.disconnect"}

    async def wrong_body():
        return {"type": "http.request", "body": "{}", "more_body": False}

    for receiver in (wrong_message, wrong_body):
        with pytest.raises(ApiError) as caught:
            _run(receive_json(receiver, content_length=None))
        assert caught.value.code == "INVALID_REQUEST"

    with pytest.raises(ApiError) as exceeded:
        _run(receive_json(_receive([b"{}"]) , content_length=1))
    assert exceeded.value.code == "INVALID_REQUEST"
    with pytest.raises(ApiError) as declared:
        _run(receive_json(_receive([b"{}"]) , content_length=3, max_bytes=2))
    assert declared.value.code == "PAYLOAD_TOO_LARGE"


@pytest.mark.parametrize(
    "content_type",
    [
        "application/json; boundary=x",
        "multipart/form-data",
        "multipart/form-data; boundary=x; charset=utf-8",
        "multipart/form-data; boundary=é",
        'multipart/form-data; boundary="x "',
    ],
)
def test_multipart_boundary_grammar_is_exact(
    tmp_path: Path,
    content_type: str,
) -> None:
    with pytest.raises(ApiError) as caught:
        _run(
            receive_multipart(
                _receive([b""]),
                content_type=content_type,
                content_length=0,
                staging_dir=tmp_path / uuid.uuid4().hex,
                upload_kind="dataset",
            )
        )
    assert caught.value.code == "UNSUPPORTED_MEDIA_TYPE"


def test_multipart_rejects_invalid_kind_length_and_declared_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = {
        "content_type": "multipart/form-data; boundary=x",
        "content_length": 0,
        "staging_dir": tmp_path / "stage",
    }
    with pytest.raises(ValueError):
        _run(
            receive_multipart(
                _receive([b""]), **base, upload_kind="other"  # type: ignore[arg-type]
            )
        )
    for invalid in (-1, True):
        with pytest.raises(ApiError) as caught:
            _run(
                receive_multipart(
                    _receive([b""]),
                    **{**base, "content_length": invalid},
                    upload_kind="dataset",
                )
            )
        assert caught.value.code == "INVALID_REQUEST"

    monkeypatch.setattr(transport_module, "DATASET_TOTAL_BYTES", 0)
    monkeypatch.setattr(transport_module, "JSON_BODY_BYTES", 0)
    monkeypatch.setattr(transport_module, "MAX_MULTIPART_OVERHEAD_BYTES", 0)
    with pytest.raises(ApiError) as oversized:
        _run(
            receive_multipart(
                _receive([b"x"]),
                content_type="multipart/form-data; boundary=x",
                content_length=1,
                staging_dir=tmp_path / "oversized",
                upload_kind="dataset",
            )
        )
    assert oversized.value.code == "PAYLOAD_TOO_LARGE"


@pytest.mark.parametrize(
    "headers",
    [
        b"X" * (transport_module.MAX_PART_HEADER_BYTES + 1),
        b"\r\n".join(b"X-%02d: value" % index for index in range(33)),
        b" Content-Disposition: form-data; name=\"bundle\"\r\nContent-Type: application/vnd.llm-foundations.bundle+zip",
        b"\xff: value\r\nContent-Type: application/vnd.llm-foundations.bundle+zip",
        b"Content-Type: application/octet-stream\r\nContent-Type: application/octet-stream",
        b"Content-Disposition: form-data; name=\"bundle\"",
        b"Content-Disposition: attachment; name=\"bundle\"\r\nContent-Type: application/vnd.llm-foundations.bundle+zip",
        b"Content-Disposition: form-data; name=\"bundle\"; size=1\r\nContent-Type: application/vnd.llm-foundations.bundle+zip",
        b"Content-Disposition: form-data\r\nContent-Type: application/vnd.llm-foundations.bundle+zip",
    ],
)
def test_multipart_part_headers_reject_noncanonical_shapes(
    tmp_path: Path,
    headers: bytes,
) -> None:
    boundary = "header-grammar"
    body = _raw_part(boundary, headers)
    with pytest.raises(ApiError) as caught:
        _run(
            receive_multipart(
                _receive([body]),
                content_type=f"multipart/form-data; boundary={boundary}",
                content_length=len(body),
                staging_dir=tmp_path / uuid.uuid4().hex,
                upload_kind="bundle",
            )
        )
    assert caught.value.code in {"INVALID_REQUEST", "UNSUPPORTED_MEDIA_TYPE"}


def test_multipart_stream_rejects_truncation_and_length_mismatch(
    tmp_path: Path,
) -> None:
    boundary = "stream-shape"
    headers = (
        b'Content-Disposition: form-data; name="bundle"\r\n'
        b"Content-Type: application/vnd.llm-foundations.bundle+zip"
    )
    cases = [
        f"--{boundary}\r\nunfinished".encode("ascii"),
        f"--{boundary}\r\n".encode("ascii")
        + b"X" * (transport_module.MAX_PART_HEADER_BYTES + 16),
        f"--{boundary}\r\n".encode("ascii")
        + headers
        + b"\r\n\r\npayload-without-boundary",
    ]
    for body in cases:
        with pytest.raises(ApiError) as caught:
            _run(
                receive_multipart(
                    _receive([body]),
                    content_type=f"multipart/form-data; boundary={boundary}",
                    content_length=len(body),
                    staging_dir=tmp_path / uuid.uuid4().hex,
                    upload_kind="bundle",
                )
            )
        assert caught.value.code == "INVALID_REQUEST"

    complete = _multipart(
        boundary,
        [("bundle", "application/vnd.llm-foundations.bundle+zip", b"payload")],
    )
    with pytest.raises(ApiError) as mismatch:
        _run(
            receive_multipart(
                _receive([complete]),
                content_type=f"multipart/form-data; boundary={boundary}",
                content_length=len(complete) + 1,
                staging_dir=tmp_path / "mismatch",
                upload_kind="bundle",
            )
        )
    assert mismatch.value.code == "INVALID_REQUEST"


def test_multipart_boundary_prefix_inside_payload_is_data(tmp_path: Path) -> None:
    boundary = "payload-prefix"
    payload = b"left\r\n--payload-prefixXXright"
    body = _multipart(
        boundary,
        [("bundle", "application/vnd.llm-foundations.bundle+zip", payload)],
    )
    result = _run(
        receive_multipart(
            _receive([body]),
            content_type=f"multipart/form-data; boundary={boundary}",
            content_length=len(body),
            staging_dir=tmp_path / "stage",
            upload_kind="bundle",
        )
    )
    assert result.part("bundle").path.read_bytes() == payload


def test_multipart_metadata_shape_and_required_parts_are_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = "required-parts"
    cases = [
        (
            _multipart(boundary, [("metadata", "application/json", b"[]")]),
            "SCHEMA_INVALID",
            [""],
        ),
        (
            _multipart(
                boundary,
                [("train", "application/x-ndjson", b"{}\n")],
            ),
            "SCHEMA_INVALID",
            ["/metadata"],
        ),
        (
            _multipart(boundary, [("metadata", "application/json", b"{}")]),
            "SCHEMA_INVALID",
            [""],
        ),
    ]
    for index, (body, reason, fields) in enumerate(cases):
        with pytest.raises(ApiError) as caught:
            _run(
                receive_multipart(
                    _receive([body]),
                    content_type=f"multipart/form-data; boundary={boundary}",
                    content_length=len(body),
                    staging_dir=tmp_path / f"case-{index}",
                    upload_kind="dataset",
                )
            )
        assert caught.value.reason_code == reason
        assert [item["field_path"] for item in caught.value.field_errors] == fields

    monkeypatch.setattr(transport_module, "JSON_BODY_BYTES", 1)
    body = _multipart(boundary, [("metadata", "application/json", b"{}")])
    with pytest.raises(ApiError) as too_large:
        _run(
            receive_multipart(
                _receive([body]),
                content_type=f"multipart/form-data; boundary={boundary}",
                content_length=len(body),
                staging_dir=tmp_path / "large-metadata",
                upload_kind="dataset",
            )
        )
    assert too_large.value.code == "PAYLOAD_TOO_LARGE"


def test_multipart_part_and_total_size_limits_remove_staged_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = "part-limits"
    metadata = ("metadata", "application/json", b"{}")
    one_file = _multipart(
        boundary,
        [metadata, ("train", "application/x-ndjson", b"12")],
    )
    monkeypatch.setattr(transport_module, "DATASET_FILE_BYTES", 1)
    with pytest.raises(ApiError) as part_limit:
        _run(
            receive_multipart(
                _receive([one_file]),
                content_type=f"multipart/form-data; boundary={boundary}",
                content_length=len(one_file),
                staging_dir=tmp_path / "part",
                upload_kind="dataset",
            )
        )
    assert part_limit.value.code == "PAYLOAD_TOO_LARGE"
    assert list((tmp_path / "part").iterdir()) == []

    monkeypatch.setattr(transport_module, "DATASET_FILE_BYTES", 10)
    monkeypatch.setattr(transport_module, "DATASET_TOTAL_BYTES", 3)
    two_files = _multipart(
        boundary,
        [
            metadata,
            ("train", "application/x-ndjson", b"12"),
            ("test", "application/x-ndjson", b"34"),
        ],
    )
    with pytest.raises(ApiError) as total_limit:
        _run(
            receive_multipart(
                _receive([two_files]),
                content_type=f"multipart/form-data; boundary={boundary}",
                content_length=len(two_files),
                staging_dir=tmp_path / "total",
                upload_kind="dataset",
            )
        )
    assert total_limit.value.code == "PAYLOAD_TOO_LARGE"
    assert list((tmp_path / "total").iterdir()) == []


@pytest.mark.parametrize(
    "failure_errno,expected_code",
    [(errno.ENOSPC, "DISK_FULL"), (errno.EACCES, "STORAGE_UNAVAILABLE")],
)
def test_multipart_staging_open_errors_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_errno: int,
    expected_code: str,
) -> None:
    boundary = "open-error"
    body = _multipart(
        boundary,
        [("bundle", "application/vnd.llm-foundations.bundle+zip", b"zip")],
    )

    def fail_open(*args: Any, **kwargs: Any) -> int:
        raise OSError(failure_errno, "injected")

    monkeypatch.setattr(transport_module.os, "open", fail_open)
    with pytest.raises(ApiError) as caught:
        _run(
            receive_multipart(
                _receive([body]),
                content_type=f"multipart/form-data; boundary={boundary}",
                content_length=len(body),
                staging_dir=tmp_path / uuid.uuid4().hex,
                upload_kind="bundle",
            )
        )
    assert caught.value.code == expected_code


def test_multipart_write_and_flush_failures_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = "write-error"
    body = _multipart(
        boundary,
        [("bundle", "application/vnd.llm-foundations.bundle+zip", b"zip")],
    )
    target = tmp_path / "write.part"
    target.write_bytes(b"")

    class FailingWriter:
        def write(self, chunk: bytes) -> None:
            target.unlink()
            raise OSError(errno.EIO, "injected write failure")

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        transport_module,
        "_staging_file",
        lambda staging_dir, name: (target, FailingWriter()),
    )
    with pytest.raises(ApiError) as write_failure:
        _run(
            receive_multipart(
                _receive([body]),
                content_type=f"multipart/form-data; boundary={boundary}",
                content_length=len(body),
                staging_dir=tmp_path / "write-stage",
                upload_kind="bundle",
            )
        )
    assert write_failure.value.code == "STORAGE_UNAVAILABLE"
    assert not target.exists()

    monkeypatch.undo()
    stage = tmp_path / "flush-stage"
    monkeypatch.setattr(
        transport_module.os,
        "fsync",
        lambda descriptor: (_ for _ in ()).throw(
            OSError(errno.EIO, "injected fsync")
        ),
    )
    with pytest.raises(ApiError) as flush_failure:
        _run(
            receive_multipart(
                _receive([body]),
                content_type=f"multipart/form-data; boundary={boundary}",
                content_length=len(body),
                staging_dir=stage,
                upload_kind="bundle",
            )
        )
    assert flush_failure.value.code == "STORAGE_UNAVAILABLE"
    assert list(stage.iterdir()) == []


def test_multipart_cleanup_tolerates_unlink_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = "cleanup-error"
    body = _multipart(
        boundary,
        [("train", "application/x-ndjson", b"{}\n")],
    )
    stage = tmp_path / "stage"
    original_unlink = Path.unlink

    def fail_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.parent == stage:
            raise OSError(errno.EACCES, "injected cleanup failure")
        original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_unlink)
        with pytest.raises(ApiError) as caught:
            _run(
                receive_multipart(
                    _receive([body]),
                    content_type=f"multipart/form-data; boundary={boundary}",
                    content_length=len(body),
                    staging_dir=stage,
                    upload_kind="dataset",
                )
            )
        assert caught.value.reason_code == "SCHEMA_INVALID"
    for path in stage.iterdir():
        path.unlink()


def test_zip_storage_eocd_comment_and_size_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ApiError) as missing:
        inspect_zip_archive(tmp_path / "missing.zip")
    assert missing.value.code == "STORAGE_UNAVAILABLE"

    malformed = tmp_path / "malformed.zip"
    malformed.write_bytes(b"not-a-zip")
    with pytest.raises(ApiError) as no_eocd:
        inspect_zip_archive(malformed)
    assert no_eocd.value.reason_code == "ARCHIVE_UNSAFE"

    commented = tmp_path / "commented.zip"
    _write_zip(commented, [("llm-foundations-bundle/manifest.json", b"{}")])
    with zipfile.ZipFile(commented, "a") as archive:
        archive.comment = b"forbidden"
    with pytest.raises(ApiError) as comment:
        inspect_zip_archive(commented)
    assert comment.value.reason_code == "ARCHIVE_UNSAFE"

    monkeypatch.setattr(transport_module, "BUNDLE_ARCHIVE_BYTES", 1)
    with pytest.raises(ApiError) as archive_size:
        inspect_zip_archive(commented)
    assert archive_size.value.code == "PAYLOAD_TOO_LARGE"


def _fake_info(**changes: Any) -> SimpleNamespace:
    values = {
        "filename": "llm-foundations-bundle/manifest.json",
        "comment": b"",
        "flag_bits": 0,
        "compress_type": zipfile.ZIP_STORED,
        "extra": b"",
        "external_attr": stat.S_IFREG << 16,
        "file_size": 1,
        "compress_size": 1,
    }
    values.update(changes)
    return SimpleNamespace(**values)


class _FakeArchive:
    comment = b""

    def __init__(self, info: SimpleNamespace, payload: bytes | BaseException) -> None:
        self.info = info
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def infolist(self) -> list[SimpleNamespace]:
        return [self.info]

    def open(self, info: SimpleNamespace, mode: str = "r"):
        if isinstance(self.payload, BaseException):
            return _FailingReader(self.payload)
        return io.BytesIO(self.payload)


class _FailingReader:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def read(self, size: int = -1) -> bytes:
        raise self.error


@pytest.mark.parametrize(
    "extra",
    [b"x", b"\x02\x00\x00\x00", b"\x01\x00\x00\x00"],
)
def test_zip_extra_fields_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: bytes,
) -> None:
    fake = _FakeArchive(_fake_info(extra=extra), b"x")
    monkeypatch.setattr(transport_module, "_validate_eocd", lambda path: None)
    monkeypatch.setattr(
        transport_module.zipfile,
        "ZipFile",
        lambda *args, **kwargs: fake,
    )
    path = tmp_path / "fake.zip"
    path.write_bytes(b"x")
    with pytest.raises(ApiError) as caught:
        inspect_zip_archive(path)
    assert caught.value.reason_code == "ARCHIVE_UNSAFE"


@pytest.mark.parametrize(
    "info,payload",
    [
        (_fake_info(file_size=1, compress_size=0), b""),
        (_fake_info(file_size=2, compress_size=2), b"x"),
        (_fake_info(), RuntimeError("injected stream failure")),
    ],
)
def test_zip_declared_and_actual_stream_sizes_are_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    info: SimpleNamespace,
    payload: bytes | BaseException,
) -> None:
    fake = _FakeArchive(info, payload)
    monkeypatch.setattr(transport_module, "_validate_eocd", lambda path: None)
    monkeypatch.setattr(
        transport_module.zipfile,
        "ZipFile",
        lambda *args, **kwargs: fake,
    )
    path = tmp_path / "fake.zip"
    path.write_bytes(b"x")
    with pytest.raises(ApiError) as caught:
        inspect_zip_archive(path)
    assert caught.value.reason_code == "ARCHIVE_UNSAFE"


def test_zip_entry_comment_compression_path_and_entry_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commented = zipfile.ZipInfo("llm-foundations-bundle/manifest.json")
    commented.comment = b"forbidden"
    commented_path = tmp_path / "entry-comment.zip"
    _write_zip(commented_path, [(commented, b"{}")])
    with pytest.raises(ApiError) as entry_comment:
        inspect_zip_archive(commented_path)
    assert entry_comment.value.reason_code == "ARCHIVE_UNSAFE"

    bzip = tmp_path / "bzip.zip"
    with zipfile.ZipFile(bzip, "w", compression=zipfile.ZIP_BZIP2) as archive:
        archive.writestr("llm-foundations-bundle/manifest.json", b"{}")
    with pytest.raises(ApiError) as compression:
        inspect_zip_archive(bzip)
    assert compression.value.reason_code == "ARCHIVE_UNSAFE"

    long_name = "llm-foundations-bundle/objects/note/" + "x" * 241
    long_path = tmp_path / "long.zip"
    _write_zip(long_path, [(long_name, b"x")])
    with pytest.raises(ApiError) as path_length:
        inspect_zip_archive(long_path)
    assert path_length.value.reason_code == "ARCHIVE_UNSAFE"

    size_path = tmp_path / "entry-size.zip"
    _write_zip(size_path, [("llm-foundations-bundle/manifest.json", b"xx")])
    monkeypatch.setattr(transport_module, "MAX_ARCHIVE_ENTRY_BYTES", 1)
    with pytest.raises(ApiError) as entry_size:
        inspect_zip_archive(size_path)
    assert entry_size.value.reason_code == "ARCHIVE_UNSAFE"


def _valid_materialized_rows() -> list[dict[str, Any]]:
    families = [
        "data-clinic-v1",
        "data-clinic-leaky-v1",
        "applied-intents-v1",
        "capstone-support-v1",
        "retrieval-manual-v1",
    ]
    return [
        {
            "path": f"{family}/train.jsonl",
            "records": 1,
            "sha256": f"{index + 1:064x}",
            "utf8_bytes": 1,
        }
        for index, family in enumerate(families)
    ]


@pytest.mark.parametrize(
    "document",
    [
        {"format": "wrong", "files": []},
        {"format": "llm-foundations-materialized-fixtures-v1", "files": [None]},
        {
            "format": "llm-foundations-materialized-fixtures-v1",
            "files": [
                {
                    "path": "x/train.jsonl",
                    "records": 0,
                    "sha256": "x",
                    "utf8_bytes": 0,
                }
            ],
        },
        {
            "format": "llm-foundations-materialized-fixtures-v1",
            "files": [
                {
                    "path": "too/many/pieces.jsonl",
                    "records": 1,
                    "sha256": "1" * 64,
                    "utf8_bytes": 1,
                }
            ],
        },
        {
            "format": "llm-foundations-materialized-fixtures-v1",
            "files": [
                {
                    "path": "family/test.jsonl",
                    "records": 1,
                    "sha256": "1" * 64,
                    "utf8_bytes": 1,
                },
                {
                    "path": "family/sealed_test.jsonl",
                    "records": 1,
                    "sha256": "2" * 64,
                    "utf8_bytes": 1,
                },
            ],
        },
        {"format": "llm-foundations-materialized-fixtures-v1", "files": []},
    ],
)
def test_dataset_bundled_fixture_registry_rejects_malformed_authority(
    monkeypatch: pytest.MonkeyPatch,
    document: dict[str, Any],
) -> None:
    monkeypatch.setattr(datasets_module, "load_document", lambda name: document)
    with pytest.raises(RuntimeError):
        DatasetRegistry(object())  # type: ignore[arg-type]


def test_dataset_metadata_validation_guards_schema_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, datasets = _stack(tmp_path)
    with pytest.raises(TypeError):
        datasets.register([], {})  # type: ignore[arg-type]

    monkeypatch.setattr(datasets_module, "validate", lambda name, value: [])
    with pytest.raises(RuntimeError):
        datasets.register(
            {"name": "x", "record_format": "document_text_v1"},
            {"train": io.BytesIO(_document_row())},
        )


@pytest.mark.parametrize(
    "record_format,row,reason",
    [
        ("document_text_v1", [], "SCHEMA_INVALID"),
        ("document_text_v1", {"unexpected": 1}, "SCHEMA_INVALID"),
        (
            "document_text_v1",
            {
                "record_id": "",
                "scenario_group_id": "",
                "text": 3,
                "slice": "",
            },
            "SCHEMA_INVALID",
        ),
        (
            "instruction_intent_v1",
            {
                "record_id": "",
                "scenario_group_id": "",
                "messages": "x",
                "expected": [],
                "slice": "",
            },
            "SCHEMA_INVALID",
        ),
        (
            "instruction_intent_v1",
            {
                "record_id": "r",
                "scenario_group_id": "g",
                "messages": [{}],
                "expected": {
                    "intent": "intent",
                    "response": "INTENT=intent\nREPLY=ok",
                },
                "slice": "intent",
            },
            "SCHEMA_INVALID",
        ),
        (
            "instruction_intent_v1",
            {
                "record_id": "r",
                "scenario_group_id": "g",
                "messages": [{"role": "user", "content": 3}],
                "expected": {"intent": "intent", "response": 3},
                "slice": "intent",
            },
            "SCHEMA_INVALID",
        ),
        (
            "instruction_intent_v1",
            {
                "record_id": "r",
                "scenario_group_id": "g",
                "messages": [{"role": "assistant", "content": "hello"}],
                "expected": {"intent": "other", "response": " "},
                "slice": "intent",
            },
            "SEMANTIC_INVALID",
        ),
        (
            "instruction_intent_v1",
            {
                "record_id": "r",
                "scenario_group_id": "g",
                "messages": [{"role": "user", "content": "hello"}],
                "expected": {"intent": "intent", "response": "not canonical"},
                "slice": "intent",
            },
            "SEMANTIC_INVALID",
        ),
    ],
)
def test_dataset_closed_record_grammar_rejects_invalid_rows(
    tmp_path: Path,
    record_format: str,
    row: Any,
    reason: str,
) -> None:
    _, registry, datasets = _stack(tmp_path)
    with pytest.raises(ApiError) as caught:
        datasets.register(
            {"name": "invalid", "record_format": record_format},
            {"train": io.BytesIO(canonical_json(row) + b"\n")},
            upload_id=_reserve(registry),
        )
    assert caught.value.reason_code == reason


def test_dataset_decode_empty_and_staged_read_failures(tmp_path: Path) -> None:
    _, registry, datasets = _stack(tmp_path / "decode")
    for raw, reason in [
        (b"\xff\n", "INVALID_ENCODING"),
        (b"\n \n", "SCHEMA_INVALID"),
    ]:
        with pytest.raises(ApiError) as caught:
            datasets.register(
                {"name": "invalid", "record_format": "document_text_v1"},
                {"train": io.BytesIO(raw)},
                upload_id=_reserve(registry),
            )
        assert caught.value.reason_code == reason

    owner = str(uuid.uuid4())
    missing = tmp_path / "missing-stage.part"
    fake_registry = SimpleNamespace(
        stage_stream=lambda source, upload_id, max_bytes: StagedArtifact(
            owner,
            missing,
            1,
            "1" * 64,
        )
    )
    guarded = DatasetRegistry.__new__(DatasetRegistry)
    guarded.registry = fake_registry
    guarded._release_families = {}
    with pytest.raises(ApiError) as unavailable:
        guarded.register(
            {"name": "missing", "record_format": "document_text_v1"},
            {"train": io.BytesIO(_document_row())},
            upload_id=owner,
        )
    assert unavailable.value.code == "STORAGE_UNAVAILABLE"


def test_dataset_parts_ids_and_closed_format_are_validated_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, datasets = _stack(tmp_path)
    metadata = {"name": "x", "record_format": "document_text_v1"}
    invalid_part_maps = [
        {},
        {
            "train": io.BytesIO(b""),
            "validation": io.BytesIO(b""),
            "test": io.BytesIO(b""),
            "extra": io.BytesIO(b""),
        },
    ]
    for parts in invalid_part_maps:
        with pytest.raises(ApiError) as count:
            datasets.register(metadata, parts)
        assert count.value.reason_code == "SCHEMA_INVALID"

    with pytest.raises(ApiError) as names:
        datasets.register(metadata, {"other": io.BytesIO(b"x")})
    assert names.value.reason_code == "SCHEMA_INVALID"

    with pytest.raises(ValueError):
        datasets.register(
            metadata,
            {"train": io.BytesIO(_document_row())},
            upload_id=str(uuid.uuid4()).upper(),
        )

    monkeypatch.setattr(
        datasets,
        "_validate_metadata",
        lambda value: {"name": "x", "record_format": "outside-vocabulary"},
    )
    with pytest.raises(AssertionError):
        datasets.register(metadata, {"train": io.BytesIO(_document_row())})


def test_dataset_combined_staged_size_guard_cleans_temporary_file(
    tmp_path: Path,
) -> None:
    owner = str(uuid.uuid4())
    staged_path = tmp_path / "oversized.part"
    staged_path.write_bytes(b"x")
    fake_registry = SimpleNamespace(
        stage_stream=lambda source, upload_id, max_bytes: StagedArtifact(
            owner,
            staged_path,
            30 * 1024 * 1024 + 1,
            "1" * 64,
        )
    )
    datasets = DatasetRegistry.__new__(DatasetRegistry)
    datasets.registry = fake_registry
    datasets._release_families = {}
    with pytest.raises(ApiError) as caught:
        datasets.register(
            {"name": "too large", "record_format": "document_text_v1"},
            {"train": io.BytesIO(_document_row())},
            upload_id=owner,
        )
    assert caught.value.code == "PAYLOAD_TOO_LARGE"
    assert not staged_path.exists()


@pytest.mark.parametrize(
    "record_format,record_count,text_bytes,reason",
    [
        ("document_text_v1", 100_001, 1, None),
        (
            "instruction_intent_v1",
            1,
            20 * 1024 * 1024 + 1,
            "PAYLOAD_TOO_LARGE",
        ),
    ],
)
def test_dataset_parsed_count_and_instruction_text_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_format: str,
    record_count: int,
    text_bytes: int,
    reason: str | None,
) -> None:
    owner = str(uuid.uuid4())
    staged_path = tmp_path / f"{record_format}.part"
    staged_path.write_bytes(b"x")
    staged = StagedArtifact(owner, staged_path, 1, "1" * 64)
    fake_registry = SimpleNamespace(stage_stream=lambda *args, **kwargs: staged)
    datasets = DatasetRegistry.__new__(DatasetRegistry)
    datasets.registry = fake_registry
    datasets._release_families = {}
    record = datasets_module._Record(
        "train",
        1,
        "record",
        "group",
        "intent" if record_format == "instruction_intent_v1" else None,
        b"x",
        "x",
        text_bytes,
    )
    parsed = datasets_module._ParsedPart(
        "train",
        staged,
        (record,) * record_count,
        1,
    )
    monkeypatch.setattr(datasets, "_parse_parts", lambda mapping, fmt: [parsed])
    with pytest.raises(ApiError) as caught:
        datasets.register(
            {"name": "bounded", "record_format": record_format},
            {"train": io.BytesIO(b"x")},
            upload_id=owner,
        )
    expected_code = "VALIDATION_FAILED" if reason is not None else "PAYLOAD_TOO_LARGE"
    assert caught.value.code == expected_code
    assert caught.value.reason_code == reason
    assert not staged_path.exists()


def test_dataset_sealed_train_is_rejected_and_idempotency_success_is_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, registry, datasets = _stack(tmp_path / "sealed")
    monkeypatch.setattr(registry, "is_sealed_digest", lambda digest: True)
    with pytest.raises(ApiError) as sealed:
        datasets.register(
            {"name": "sealed", "record_format": "document_text_v1"},
            {"train": io.BytesIO(_document_row())},
            upload_id=_reserve(registry),
        )
    assert sealed.value.reason_code == "SEMANTIC_INVALID"
    assert [item["field_path"] for item in sealed.value.field_errors] == ["/train"]

    _, registry, datasets = _stack(tmp_path / "success")
    calls: list[tuple[int, dict[str, Any]]] = []

    class Commit:
        def record_success(
            self,
            connection: Any,
            *,
            status_code: int,
            response_body: dict[str, Any],
        ) -> None:
            calls.append((status_code, response_body))

    manifest = datasets.register(
        {"name": "success", "record_format": "document_text_v1"},
        {"train": io.BytesIO(_document_row())},
        upload_id=_reserve(registry),
        idempotency_commit=Commit(),  # type: ignore[arg-type]
    )
    assert calls == [(201, manifest)]
