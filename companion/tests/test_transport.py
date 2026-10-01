from __future__ import annotations

import asyncio
import hashlib
import os
import stat
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pytest

from llm_foundations_companion.errors import ApiError
from llm_foundations_companion.platform_security import check_private_file
from llm_foundations_companion.schema import canonical_json
from llm_foundations_companion.transport import (
    inspect_zip_archive,
    parse_content_length,
    receive_json,
    receive_multipart,
    require_media_type,
)


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


def _chunks(payload: bytes, sizes: tuple[int, ...] = (1, 7, 2, 31, 5)) -> list[bytes]:
    result: list[bytes] = []
    offset = 0
    index = 0
    while offset < len(payload):
        size = sizes[index % len(sizes)]
        result.append(payload[offset : offset + size])
        offset += size
        index += 1
    return result or [b""]


def _multipart(boundary: str, parts: list[tuple[str, str, bytes, str | None]]) -> bytes:
    body = bytearray()
    for name, media_type, payload, filename in parts:
        body.extend(f"--{boundary}\r\n".encode("ascii"))
        disposition = f'Content-Disposition: form-data; name="{name}"'
        if filename is not None:
            disposition += f'; filename="{filename}"'
        body.extend(disposition.encode("utf-8") + b"\r\n")
        body.extend(f"Content-Type: {media_type}\r\n\r\n".encode("ascii"))
        body.extend(payload)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("ascii"))
    return bytes(body)


def _run(coroutine):
    return asyncio.run(coroutine)


def test_content_headers_reject_ambiguity_encoding_and_wrong_media() -> None:
    assert parse_content_length({"Content-Length": "12"}) == 12
    assert parse_content_length({}) is None
    with pytest.raises(ApiError) as missing:
        parse_content_length({}, required=True)
    assert missing.value.code == "INVALID_REQUEST"
    with pytest.raises(ApiError):
        parse_content_length({"Content-Length": "-1"})

    assert require_media_type({"Content-Type": "application/json"}, "application/json") == "application/json"
    assert require_media_type(
        {"Content-Type": "application/json; charset=utf-8"}, "application/json"
    ) == "application/json"
    with pytest.raises(ApiError) as encoded:
        require_media_type(
            {"Content-Type": "application/json", "Content-Encoding": "gzip"},
            "application/json",
        )
    assert encoded.value.code == "UNSUPPORTED_MEDIA_TYPE"
    with pytest.raises(ApiError) as wrong:
        require_media_type({"Content-Type": "text/plain"}, "application/json")
    assert wrong.value.code == "UNSUPPORTED_MEDIA_TYPE"


def test_receive_json_is_stream_bounded_strict_and_length_checked() -> None:
    payload = canonical_json({"message": "héllo", "value": 4})
    parsed = _run(
        receive_json(
            _receive(_chunks(payload)),
            content_length=len(payload),
        )
    )
    assert parsed == {"message": "héllo", "value": 4}

    with pytest.raises(ApiError) as mismatch:
        _run(receive_json(_receive([payload]), content_length=len(payload) + 1))
    assert mismatch.value.code == "INVALID_REQUEST"

    with pytest.raises(ApiError) as oversize:
        _run(receive_json(_receive([b"1234"]), content_length=None, max_bytes=3))
    assert oversize.value.code == "PAYLOAD_TOO_LARGE"

    with pytest.raises(ApiError) as duplicate:
        _run(receive_json(_receive([b'{"x":1,"x":2}']), content_length=None))
    assert duplicate.value.code == "VALIDATION_FAILED"
    assert duplicate.value.reason_code == "INVALID_JSON"


def test_receive_json_rejects_invalid_declared_length_and_stream_shape() -> None:
    async def invalid_more_body():
        return {
            "type": "http.request",
            "body": b"{}",
            "more_body": "false",
        }

    for invalid_length in (-1, True):
        with pytest.raises(ApiError) as length:
            _run(
                receive_json(
                    _receive([b"{}"]),
                    content_length=invalid_length,
                )
            )
        assert length.value.code == "INVALID_REQUEST"

    with pytest.raises(ApiError) as stream:
        _run(receive_json(invalid_more_body, content_length=None))
    assert stream.value.code == "INVALID_REQUEST"


def test_dataset_multipart_streams_to_server_paths_and_has_order_independent_digest(
    tmp_path: Path,
) -> None:
    boundary = "dataset-boundary"
    metadata = canonical_json(
        {"name": "small", "provenance_note": "local", "record_format": "text"}
    )
    train = b'{"text":"one"}\n{"text":"two"}\n'
    validation = b'{"text":"check"}\n'
    first_parts = [
        ("metadata", "application/json", metadata, None),
        ("train", "application/x-ndjson", train, "../../secret.jsonl"),
        ("validation", "application/x-ndjson", validation, "v.jsonl"),
    ]
    second_parts = [first_parts[2], first_parts[0], first_parts[1]]
    first_body = _multipart(boundary, first_parts)
    second_body = _multipart(boundary, second_parts)

    first = _run(
        receive_multipart(
            _receive(_chunks(first_body)),
            content_type=f'multipart/form-data; boundary="{boundary}"',
            content_length=len(first_body),
            staging_dir=tmp_path / "first",
            upload_kind="dataset",
        )
    )
    second = _run(
        receive_multipart(
            _receive(_chunks(second_body, (13, 3, 29))),
            content_type=f"multipart/form-data; boundary={boundary}",
            content_length=len(second_body),
            staging_dir=tmp_path / "second",
            upload_kind="dataset",
        )
    )

    assert first.metadata == second.metadata
    assert first.canonical_sha256 == second.canonical_sha256
    assert [part.name for part in first.parts] == ["train", "validation"]
    assert first.part("train").path.read_bytes() == train
    assert first.part("train").sha256 == hashlib.sha256(train).hexdigest()
    assert "secret" not in first.part("train").path.name
    assert first.part("train").path.parent == tmp_path / "first"
    if os.name == "nt":
        check_private_file(first.part("train").path)
    else:
        assert stat.S_IMODE(first.part("train").path.stat().st_mode) == 0o600


def test_multipart_ignores_rfc_filename_and_rejects_nonexact_file_media_type(
    tmp_path: Path,
) -> None:
    boundary = "strict-parts"
    metadata = canonical_json({"name": "x", "record_format": "document_text_v1"})
    train = b'{"text":"one"}\n'
    body = _multipart(
        boundary,
        [
            ("metadata", "application/json; charset=utf-8", metadata, None),
            ("train", "application/x-ndjson", train, None),
        ],
    )
    body = body.replace(
        b'Content-Disposition: form-data; name="train"',
        b"Content-Disposition: form-data; name=\"train\"; "
        b"filename*=UTF-8''..%2Fsecret.jsonl",
    )
    staged = _run(
        receive_multipart(
            _receive(_chunks(body)),
            content_type=f"multipart/form-data; boundary={boundary}",
            content_length=len(body),
            staging_dir=tmp_path / "filename",
            upload_kind="dataset",
        )
    )
    assert staged.part("train").path.read_bytes() == train
    assert "secret" not in staged.part("train").path.name

    wrong_media = _multipart(
        boundary,
        [
            ("metadata", "application/json", metadata, None),
            (
                "train",
                "application/x-ndjson; charset=utf-8",
                train,
                "train.jsonl",
            ),
        ],
    )
    with pytest.raises(ApiError) as caught:
        _run(
            receive_multipart(
                _receive([wrong_media]),
                content_type=f"multipart/form-data; boundary={boundary}",
                content_length=len(wrong_media),
                staging_dir=tmp_path / "media-parameter",
                upload_kind="dataset",
            )
        )
    assert caught.value.code == "UNSUPPORTED_MEDIA_TYPE"


def test_multipart_rejects_boundary_outside_rfc_token_set(tmp_path: Path) -> None:
    with pytest.raises(ApiError) as caught:
        _run(
            receive_multipart(
                _receive([b""]),
                content_type='multipart/form-data; boundary="bad[boundary"',
                content_length=0,
                staging_dir=tmp_path / "bad-boundary",
                upload_kind="dataset",
            )
        )
    assert caught.value.code == "UNSUPPORTED_MEDIA_TYPE"


def test_multipart_rejects_duplicate_unknown_bad_media_and_bad_final_boundary(
    tmp_path: Path,
) -> None:
    boundary = "closed"
    metadata = canonical_json({"name": "x", "record_format": "text"})
    train = b'{"text":"one"}\n'
    cases = [
        _multipart(
            boundary,
            [
                ("metadata", "application/json", metadata, None),
                ("train", "application/x-ndjson", train, "a"),
                ("train", "application/x-ndjson", train, "b"),
            ],
        ),
        _multipart(
            boundary,
            [
                ("metadata", "application/json", metadata, None),
                ("other", "application/x-ndjson", train, "a"),
            ],
        ),
    ]
    for index, body in enumerate(cases):
        staging = tmp_path / f"case-{index}"
        with pytest.raises(ApiError) as caught:
            _run(
                receive_multipart(
                    _receive(_chunks(body)),
                    content_type=f"multipart/form-data; boundary={boundary}",
                    content_length=len(body),
                    staging_dir=staging,
                    upload_kind="dataset",
                )
            )
        assert caught.value.code == "INVALID_REQUEST"
        assert list(staging.iterdir()) == []

    wrong_media = _multipart(
        boundary,
        [
            ("metadata", "text/plain", metadata, None),
            ("train", "application/x-ndjson", train, "a"),
        ],
    )
    with pytest.raises(ApiError) as media:
        _run(
            receive_multipart(
                _receive([wrong_media]),
                content_type=f"multipart/form-data; boundary={boundary}",
                content_length=len(wrong_media),
                staging_dir=tmp_path / "media",
                upload_kind="dataset",
            )
        )
    assert media.value.code == "UNSUPPORTED_MEDIA_TYPE"

    valid = _multipart(
        boundary,
        [
            ("metadata", "application/json", metadata, None),
            ("train", "application/x-ndjson", train, "a"),
        ],
    )
    malformed = valid[:-2]
    with pytest.raises(ApiError) as final:
        _run(
            receive_multipart(
                _receive([malformed]),
                content_type=f"multipart/form-data; boundary={boundary}",
                content_length=len(malformed),
                staging_dir=tmp_path / "final",
                upload_kind="dataset",
            )
        )
    assert final.value.code == "INVALID_REQUEST"


def test_bundle_multipart_uses_empty_metadata_digest_and_exact_media_type(
    tmp_path: Path,
) -> None:
    boundary = "bundle-boundary"
    bundle_bytes = b"PK-not-yet-semantically-validated"
    body = _multipart(
        boundary,
        [
            (
                "bundle",
                "application/vnd.llm-foundations.bundle+zip",
                bundle_bytes,
                "state.lfbundle",
            )
        ],
    )
    staged = _run(
        receive_multipart(
            _receive(_chunks(body)),
            content_type=f"multipart/form-data; boundary={boundary}",
            content_length=len(body),
            staging_dir=tmp_path / "bundle",
            upload_kind="bundle",
        )
    )
    assert staged.metadata == {}
    assert staged.metadata_sha256 == hashlib.sha256(canonical_json({})).hexdigest()
    assert staged.part("bundle").path.read_bytes() == bundle_bytes


def _write_zip(path: Path, entries: list[tuple[zipfile.ZipInfo | str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)


def test_archive_inspection_reads_without_extracting_and_rejects_expansion(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "safe.lfbundle"
    _write_zip(
        archive_path,
        [
            ("llm-foundations-bundle/manifest.json", b"{}\n"),
            ("llm-foundations-bundle/objects/note/id/note.json", b'{"note":1}\n'),
        ],
    )
    inspection = inspect_zip_archive(archive_path)
    assert inspection.entries == 2
    assert inspection.total_uncompressed_bytes == len(b"{}\n") + len(b'{"note":1}\n')
    assert sorted(path.name for path in tmp_path.iterdir()) == ["safe.lfbundle"]

    with pytest.raises(ApiError) as expansion:
        inspect_zip_archive(archive_path, max_expanded_bytes=3)
    assert expansion.value.code == "VALIDATION_FAILED"
    assert expansion.value.reason_code == "ARCHIVE_UNSAFE"


@pytest.mark.parametrize("unsafe_name", ["../escape", "/absolute", "C:/drive", "a\\b", "a//b"])
def test_archive_inspection_rejects_unsafe_names(tmp_path: Path, unsafe_name: str) -> None:
    path = tmp_path / f"unsafe-{abs(hash(unsafe_name))}.zip"
    _write_zip(path, [(unsafe_name, b"x")])
    with pytest.raises(ApiError) as caught:
        inspect_zip_archive(path)
    assert caught.value.reason_code == "ARCHIVE_UNSAFE"


def test_archive_inspection_rejects_symlink_duplicate_ratio_and_trailing_bytes(
    tmp_path: Path,
) -> None:
    symlink = zipfile.ZipInfo("llm-foundations-bundle/objects/link")
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    symlink_path = tmp_path / "symlink.zip"
    _write_zip(symlink_path, [(symlink, b"target")])

    duplicate_path = tmp_path / "duplicate.zip"
    _write_zip(
        duplicate_path,
        [
            ("llm-foundations-bundle/objects/note/A.txt", b"a"),
            ("llm-foundations-bundle/objects/note/a.txt", b"b"),
        ],
    )

    ratio_path = tmp_path / "ratio.zip"
    _write_zip(
        ratio_path,
        [("llm-foundations-bundle/objects/note/large.txt", b"A" * 100_000)],
    )

    trailing_path = tmp_path / "trailing.zip"
    _write_zip(
        trailing_path,
        [("llm-foundations-bundle/objects/note/ok.txt", b"ok")],
    )
    with trailing_path.open("ab") as output:
        output.write(b"trailing")

    for path in [symlink_path, duplicate_path, ratio_path, trailing_path]:
        with pytest.raises(ApiError) as caught:
            inspect_zip_archive(path)
        assert caught.value.reason_code == "ARCHIVE_UNSAFE"


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "outside.txt",
        "llm-foundations-bundle/objects/note/CON",
        "llm-foundations-bundle/objects/note/trailing.",
        "llm-foundations-bundle/objects/note/not:portable",
        "llm-foundations-bundle/objects/note/nonascii-é",
    ],
)
def test_archive_inspection_rejects_nonportable_bundle_names(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    path = tmp_path / f"unsafe-name-{abs(hash(unsafe_name))}.zip"
    _write_zip(path, [(unsafe_name, b"x")])
    with pytest.raises(ApiError) as caught:
        inspect_zip_archive(path)
    assert caught.value.reason_code == "ARCHIVE_UNSAFE"


def test_archive_inspection_rejects_executable_regular_file(tmp_path: Path) -> None:
    executable = zipfile.ZipInfo(
        "llm-foundations-bundle/objects/note/id/script.txt"
    )
    executable.create_system = 3
    executable.external_attr = (stat.S_IFREG | 0o755) << 16
    path = tmp_path / "executable.zip"
    _write_zip(path, [(executable, b"not executable content")])
    with pytest.raises(ApiError) as caught:
        inspect_zip_archive(path)
    assert caught.value.reason_code == "ARCHIVE_UNSAFE"
