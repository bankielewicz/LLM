"""Bounded HTTP body, multipart staging, and inert ZIP inspection helpers."""

from __future__ import annotations

import errno
import hashlib
import os
import re
import stat
import struct
import uuid
import zipfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from email.message import Message
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from .errors import ApiError
from .limits import LIMITS
from .platform_security import ensure_private_directory
from .schema import canonical_json, strict_json


Receive = Callable[[], Awaitable[Mapping[str, Any]]]
UploadKind = Literal["dataset", "bundle"]

JSON_BODY_BYTES = LIMITS["json_body_bytes"]
DATASET_FILE_BYTES = LIMITS["dataset_file_bytes"]
DATASET_TOTAL_BYTES = LIMITS["dataset_total_bytes"]
BUNDLE_ARCHIVE_BYTES = LIMITS["bundle_archive_bytes"]
BUNDLE_EXPANDED_BYTES = LIMITS["bundle_expanded_bytes"]
MAX_PART_HEADER_BYTES = 16 * 1024
MAX_PART_HEADERS = 32
MAX_MULTIPART_OVERHEAD_BYTES = 256 * 1024
MAX_ARCHIVE_ENTRIES = 10_000
MAX_ARCHIVE_ENTRY_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_PATH_BYTES = 240
MAX_COMPRESSION_RATIO = 100

_DATASET_FILE_PARTS = frozenset({"train", "validation", "test"})
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$", re.ASCII)
_BOUNDARY_RE = re.compile(rb"^[0-9A-Za-z'()+_,./:=? -]{1,70}$", re.ASCII)
_CONTENT_LENGTH_RE = re.compile(r"^[0-9]+$", re.ASCII)
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_BUNDLE_PATH_RE = re.compile(r"^[A-Za-z0-9._/-]+$", re.ASCII)
_WINDOWS_DEVICE_RE = re.compile(
    r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)",
    re.ASCII | re.IGNORECASE,
)


@dataclass(frozen=True)
class StagedPart:
    name: str
    path: Path
    size: int
    sha256: str
    media_type: str


@dataclass(frozen=True)
class StagedMultipart:
    metadata: Mapping[str, Any]
    metadata_sha256: str
    parts: tuple[StagedPart, ...]
    canonical_sha256: str

    def part(self, name: str) -> StagedPart:
        for candidate in self.parts:
            if candidate.name == name:
                return candidate
        raise KeyError(name)


@dataclass(frozen=True)
class ArchiveInspection:
    entries: int
    total_compressed_bytes: int
    total_uncompressed_bytes: int


def _header_values(headers: Mapping[str, str], name: str) -> list[str]:
    target = name.casefold()
    return [str(value) for key, value in headers.items() if str(key).casefold() == target]


def parse_content_length(
    headers: Mapping[str, str],
    *,
    required: bool = False,
) -> int | None:
    """Parse one non-negative Content-Length without accepting ambiguity."""

    values = _header_values(headers, "content-length")
    if not values:
        if required:
            raise ApiError("INVALID_REQUEST", "Content-Length is required.")
        return None
    if len(values) != 1 or _CONTENT_LENGTH_RE.fullmatch(values[0]) is None:
        raise ApiError("INVALID_REQUEST", "Content-Length is invalid.")
    return int(values[0])


def _parse_content_type(value: str) -> tuple[str, list[tuple[str, str]]]:
    if "\r" in value or "\n" in value:
        raise ApiError("UNSUPPORTED_MEDIA_TYPE", "The Content-Type is invalid.")
    message = Message()
    message["content-type"] = value
    media_type = message.get_content_type().lower()
    raw_params = message.get_params(header="content-type", unquote=True) or []
    params: list[tuple[str, str]] = []
    for key, item in raw_params[1:]:
        if not isinstance(key, str) or not isinstance(item, str):
            raise ApiError("UNSUPPORTED_MEDIA_TYPE", "The Content-Type is invalid.")
        params.append((key.casefold(), item))
    if len({key for key, _ in params}) != len(params):
        raise ApiError("UNSUPPORTED_MEDIA_TYPE", "The Content-Type is invalid.")
    return media_type, params


def require_media_type(headers: Mapping[str, str], expected: str) -> str:
    """Reject content coding and require one expected media type."""

    if _header_values(headers, "content-encoding"):
        raise ApiError(
            "UNSUPPORTED_MEDIA_TYPE",
            "Content encoding is not supported.",
        )
    values = _header_values(headers, "content-type")
    if len(values) != 1:
        raise ApiError("UNSUPPORTED_MEDIA_TYPE", "The Content-Type is required.")
    media_type, params = _parse_content_type(values[0])
    if media_type != expected.casefold():
        raise ApiError("UNSUPPORTED_MEDIA_TYPE", "The Content-Type is not supported.")
    if media_type == "application/json":
        if any(key != "charset" or value.casefold() != "utf-8" for key, value in params):
            raise ApiError("UNSUPPORTED_MEDIA_TYPE", "The Content-Type is not supported.")
    elif params:
        raise ApiError("UNSUPPORTED_MEDIA_TYPE", "The Content-Type is not supported.")
    return media_type


def _multipart_boundary(content_type: str) -> bytes:
    media_type, params = _parse_content_type(content_type)
    if media_type != "multipart/form-data":
        raise ApiError("UNSUPPORTED_MEDIA_TYPE", "The Content-Type is not supported.")
    if len(params) != 1 or params[0][0] != "boundary":
        raise ApiError("UNSUPPORTED_MEDIA_TYPE", "A single multipart boundary is required.")
    boundary = params[0][1]
    try:
        encoded = boundary.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise ApiError("UNSUPPORTED_MEDIA_TYPE", "The multipart boundary is invalid.") from exc
    if _BOUNDARY_RE.fullmatch(encoded) is None or encoded[-1:] == b" ":
        raise ApiError("UNSUPPORTED_MEDIA_TYPE", "The multipart boundary is invalid.")
    return encoded


class _BodyStream:
    def __init__(
        self,
        receive: Receive,
        *,
        content_length: int | None,
        max_transport_bytes: int,
    ) -> None:
        self.receive = receive
        self.content_length = content_length
        self.max_transport_bytes = max_transport_bytes
        self.buffer = bytearray()
        self.actual_bytes = 0
        self.eof = False

    async def fill(self) -> None:
        if self.eof:
            return
        message = await self.receive()
        if not isinstance(message, Mapping) or message.get("type") != "http.request":
            raise ApiError("INVALID_REQUEST", "The request body stream ended unexpectedly.")
        body = message.get("body", b"")
        more_body = message.get("more_body", False)
        if not isinstance(more_body, bool):
            raise ApiError("INVALID_REQUEST", "The request body stream is invalid.")
        if not isinstance(body, (bytes, bytearray, memoryview)):
            raise ApiError("INVALID_REQUEST", "The request body stream is invalid.")
        chunk = bytes(body)
        self.actual_bytes += len(chunk)
        if self.actual_bytes > self.max_transport_bytes:
            raise ApiError("PAYLOAD_TOO_LARGE", "The request body is too large.")
        if self.content_length is not None and self.actual_bytes > self.content_length:
            raise ApiError("INVALID_REQUEST", "The request body length does not match Content-Length.")
        self.buffer.extend(chunk)
        self.eof = not more_body

    async def expect(self, expected: bytes) -> None:
        while len(self.buffer) < len(expected) and not self.eof:
            await self.fill()
        if not self.buffer.startswith(expected):
            raise ApiError("INVALID_REQUEST", "The multipart body is malformed.")
        del self.buffer[: len(expected)]

    async def read_until(self, marker: bytes, *, max_bytes: int) -> bytes:
        while True:
            position = self.buffer.find(marker)
            if position >= 0:
                if position > max_bytes:
                    raise ApiError("INVALID_REQUEST", "Multipart part headers are too large.")
                result = bytes(self.buffer[:position])
                del self.buffer[: position + len(marker)]
                return result
            if len(self.buffer) > max_bytes + len(marker):
                raise ApiError("INVALID_REQUEST", "Multipart part headers are too large.")
            if self.eof:
                raise ApiError("INVALID_REQUEST", "The multipart body is truncated.")
            await self.fill()

    async def part_chunks(self, marker: bytes):
        """Yield bytes until a marker with a valid boundary suffix is found."""

        keep = len(marker) + 2
        while True:
            position = self.buffer.find(marker)
            if position >= 0:
                while len(self.buffer) < position + keep and not self.eof:
                    await self.fill()
                suffix = bytes(
                    self.buffer[position + len(marker) : position + len(marker) + 2]
                )
                if suffix in {b"\r\n", b"--"}:
                    if position:
                        yield bytes(self.buffer[:position])
                    del self.buffer[: position + len(marker)]
                    return
                # A boundary-prefix byte sequence inside the payload is data.
                end = position + 1
                yield bytes(self.buffer[:end])
                del self.buffer[:end]
                continue
            if self.eof:
                raise ApiError("INVALID_REQUEST", "The multipart body is truncated.")
            safe = max(0, len(self.buffer) - keep)
            if safe:
                yield bytes(self.buffer[:safe])
                del self.buffer[:safe]
            await self.fill()

    async def finish(self) -> None:
        while not self.eof:
            await self.fill()
        if self.buffer:
            raise ApiError("INVALID_REQUEST", "The multipart final boundary is invalid.")
        if self.content_length is not None and self.actual_bytes != self.content_length:
            raise ApiError("INVALID_REQUEST", "The request body length does not match Content-Length.")


async def receive_json(
    receive: Receive,
    *,
    content_length: int | None,
    max_bytes: int = JSON_BODY_BYTES,
) -> Any:
    """Receive one bounded ASGI body and parse strict UTF-8 JSON."""

    if content_length is not None and (
        not isinstance(content_length, int)
        or isinstance(content_length, bool)
        or content_length < 0
    ):
        raise ApiError("INVALID_REQUEST", "Content-Length is invalid.")
    if content_length is not None and content_length > max_bytes:
        raise ApiError("PAYLOAD_TOO_LARGE", "The request body is too large.")
    stream = _BodyStream(
        receive,
        content_length=content_length,
        max_transport_bytes=max_bytes,
    )
    while not stream.eof:
        await stream.fill()
    if content_length is not None and stream.actual_bytes != content_length:
        raise ApiError("INVALID_REQUEST", "The request body length does not match Content-Length.")
    return strict_json(bytes(stream.buffer))


def _parse_part_headers(block: bytes) -> tuple[str, str]:
    if len(block) > MAX_PART_HEADER_BYTES:
        raise ApiError("INVALID_REQUEST", "Multipart part headers are too large.")
    lines = block.split(b"\r\n") if block else []
    if not lines or len(lines) > MAX_PART_HEADERS:
        raise ApiError("INVALID_REQUEST", "Multipart part headers are invalid.")
    headers: dict[str, str] = {}
    for line in lines:
        if not line or line[:1] in {b" ", b"\t"} or b":" not in line or b"\n" in line or b"\r" in line:
            raise ApiError("INVALID_REQUEST", "Multipart part headers are invalid.")
        raw_name, raw_value = line.split(b":", 1)
        try:
            name = raw_name.decode("ascii", errors="strict").lower()
            value = raw_value.decode("latin-1").strip()
        except UnicodeDecodeError as exc:
            raise ApiError("INVALID_REQUEST", "Multipart part headers are invalid.") from exc
        if _HEADER_NAME_RE.fullmatch(name) is None or name in headers:
            raise ApiError("INVALID_REQUEST", "Multipart part headers are invalid.")
        headers[name] = value
    if set(headers) != {"content-disposition", "content-type"}:
        raise ApiError("INVALID_REQUEST", "Multipart part headers are invalid.")

    disposition = Message()
    disposition["content-disposition"] = headers["content-disposition"]
    if disposition.get_content_disposition() != "form-data":
        raise ApiError("INVALID_REQUEST", "Multipart Content-Disposition is invalid.")
    raw_params = disposition.get_params(header="content-disposition", unquote=True) or []
    params: dict[str, Any] = {}
    for key, value in raw_params[1:]:
        folded = str(key).casefold()
        if folded in params or folded not in {"name", "filename", "filename*"}:
            raise ApiError("INVALID_REQUEST", "Multipart Content-Disposition is invalid.")
        if folded == "name" and not isinstance(value, str):
            raise ApiError("INVALID_REQUEST", "Multipart Content-Disposition is invalid.")
        # Filename values, including RFC 2231 tuples, are deliberately ignored.
        params[folded] = value if folded == "name" else None
    name = params.get("name")
    if not isinstance(name, str) or not name:
        raise ApiError("INVALID_REQUEST", "Multipart part name is required.")
    # filename and filename* are intentionally ignored and never become paths.
    media_type, media_params = _parse_content_type(headers["content-type"])
    if media_params and (
        media_type != "application/json"
        or any(
            key != "charset" or value.casefold() != "utf-8"
            for key, value in media_params
        )
    ):
        raise ApiError("UNSUPPORTED_MEDIA_TYPE", "The part Content-Type is not supported.")
    return name, media_type


def _staging_file(staging_dir: Path, part_name: str) -> tuple[Path, Any]:
    path = staging_dir / f"upload-{part_name}-{uuid.uuid4().hex}.part"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        return path, os.fdopen(descriptor, "wb")
    except OSError as exc:
        raise _storage_error(exc) from exc


def _storage_error(error: OSError) -> ApiError:
    if error.errno in {errno.ENOSPC, getattr(errno, "EDQUOT", -1)}:
        return ApiError("DISK_FULL", "The upload could not be staged because storage is full.")
    return ApiError("STORAGE_UNAVAILABLE", "The upload staging area is unavailable.")


def _multipart_digest(
    metadata: Mapping[str, Any],
    parts: tuple[StagedPart, ...],
) -> tuple[str, str]:
    metadata_sha256 = hashlib.sha256(canonical_json(metadata)).hexdigest()
    digest_input = {
        "metadata_sha256": metadata_sha256,
        "parts": [
            {"name": part.name, "size": part.size, "sha256": part.sha256}
            for part in sorted(parts, key=lambda item: item.name)
        ],
    }
    return metadata_sha256, hashlib.sha256(canonical_json(digest_input)).hexdigest()


async def receive_multipart(
    receive: Receive,
    *,
    content_type: str,
    content_length: int,
    staging_dir: Path,
    upload_kind: UploadKind,
) -> StagedMultipart:
    """Stream one closed-shape multipart upload into private staging files."""

    if upload_kind not in {"dataset", "bundle"}:
        raise ValueError("upload_kind must be dataset or bundle")
    if not isinstance(content_length, int) or isinstance(content_length, bool) or content_length < 0:
        raise ApiError("INVALID_REQUEST", "Content-Length is invalid.")
    limit = (
        DATASET_TOTAL_BYTES + JSON_BODY_BYTES + MAX_MULTIPART_OVERHEAD_BYTES
        if upload_kind == "dataset"
        else BUNDLE_ARCHIVE_BYTES + MAX_MULTIPART_OVERHEAD_BYTES
    )
    if content_length > limit:
        raise ApiError("PAYLOAD_TOO_LARGE", "The multipart body is too large.")
    boundary = _multipart_boundary(content_type)
    ensure_private_directory(staging_dir)
    stream = _BodyStream(
        receive,
        content_length=content_length,
        max_transport_bytes=limit,
    )
    marker = b"\r\n--" + boundary
    created: list[Path] = []
    seen: set[str] = set()
    staged: list[StagedPart] = []
    metadata: Mapping[str, Any] | None = None
    dataset_total = 0

    try:
        await stream.expect(b"--" + boundary + b"\r\n")
        while True:
            header_block = await stream.read_until(
                b"\r\n\r\n", max_bytes=MAX_PART_HEADER_BYTES
            )
            name, media_type = _parse_part_headers(header_block)
            allowed = (
                {"metadata", "train", "validation", "test"}
                if upload_kind == "dataset"
                else {"bundle"}
            )
            if name not in allowed or name in seen:
                raise ApiError("INVALID_REQUEST", "Multipart parts are duplicated or unknown.")
            seen.add(name)

            if name == "metadata":
                if media_type != "application/json":
                    raise ApiError("UNSUPPORTED_MEDIA_TYPE", "Metadata must be application/json.")
                collected = bytearray()
                async for chunk in stream.part_chunks(marker):
                    collected.extend(chunk)
                    if len(collected) > JSON_BODY_BYTES:
                        raise ApiError("PAYLOAD_TOO_LARGE", "Dataset metadata is too large.")
                parsed = strict_json(bytes(collected))
                if not isinstance(parsed, dict):
                    raise ApiError(
                        "VALIDATION_FAILED",
                        "Dataset metadata must be an object.",
                        reason_code="SCHEMA_INVALID",
                        field_errors=[{"field_path": "", "message": "Expected an object."}],
                    )
                metadata = parsed
            else:
                expected_media = (
                    "application/x-ndjson"
                    if upload_kind == "dataset"
                    else "application/vnd.llm-foundations.bundle+zip"
                )
                if media_type != expected_media:
                    raise ApiError("UNSUPPORTED_MEDIA_TYPE", "The part Content-Type is not supported.")
                part_limit = DATASET_FILE_BYTES if upload_kind == "dataset" else BUNDLE_ARCHIVE_BYTES
                path, output = _staging_file(staging_dir, name)
                created.append(path)
                digest = hashlib.sha256()
                size = 0
                try:
                    async for chunk in stream.part_chunks(marker):
                        size += len(chunk)
                        if size > part_limit:
                            raise ApiError("PAYLOAD_TOO_LARGE", "An upload part is too large.")
                        if upload_kind == "dataset" and dataset_total + size > DATASET_TOTAL_BYTES:
                            raise ApiError("PAYLOAD_TOO_LARGE", "Dataset upload parts are too large.")
                        try:
                            output.write(chunk)
                        except OSError as exc:
                            raise _storage_error(exc) from exc
                        digest.update(chunk)
                    try:
                        output.flush()
                        os.fsync(output.fileno())
                    except OSError as exc:
                        raise _storage_error(exc) from exc
                finally:
                    output.close()
                if upload_kind == "dataset":
                    dataset_total += size
                staged.append(StagedPart(name, path, size, digest.hexdigest(), media_type))

            if stream.buffer.startswith(b"--"):
                await stream.expect(b"--\r\n")
                await stream.finish()
                break
            await stream.expect(b"\r\n")

        if upload_kind == "dataset":
            missing_errors = []
            if metadata is None:
                missing_errors.append({"field_path": "/metadata", "message": "Required part missing."})
            file_names = seen & _DATASET_FILE_PARTS
            if not file_names:
                missing_errors.append({"field_path": "", "message": "At least one dataset file is required."})
            if missing_errors:
                raise ApiError(
                    "VALIDATION_FAILED",
                    "The multipart dataset shape is invalid.",
                    reason_code="SCHEMA_INVALID",
                    field_errors=missing_errors,
                )
            resolved_metadata = metadata
        else:
            if seen != {"bundle"}:
                raise ApiError(
                    "VALIDATION_FAILED",
                    "The bundle part is required.",
                    reason_code="SCHEMA_INVALID",
                    field_errors=[{"field_path": "/bundle", "message": "Required part missing."}],
                )
            resolved_metadata = {}

        parts = tuple(sorted(staged, key=lambda item: item.name))
        metadata_sha256, request_sha256 = _multipart_digest(resolved_metadata, parts)
        return StagedMultipart(
            metadata=resolved_metadata,
            metadata_sha256=metadata_sha256,
            parts=parts,
            canonical_sha256=request_sha256,
        )
    except BaseException:
        for path in created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        raise


def _archive_unsafe() -> ApiError:
    return ApiError(
        "VALIDATION_FAILED",
        "The archive structure is unsafe or unsupported.",
        reason_code="ARCHIVE_UNSAFE",
    )


def _validate_eocd(path: Path) -> None:
    try:
        size = path.stat().st_size
        with path.open("rb") as source:
            tail_size = min(size, 65_535 + 22)
            source.seek(size - tail_size)
            tail = source.read(tail_size)
    except OSError as exc:
        raise _storage_error(exc) from exc
    position = tail.rfind(b"PK\x05\x06")
    if position < 0 or len(tail) - position < 22:
        raise _archive_unsafe()
    try:
        (
            signature,
            disk_number,
            directory_disk,
            disk_entries,
            total_entries,
            directory_size,
            directory_offset,
            comment_length,
        ) = struct.unpack_from("<4s4H2LH", tail, position)
    except struct.error as exc:
        raise _archive_unsafe() from exc
    absolute_position = size - tail_size + position
    if (
        signature != b"PK\x05\x06"
        or disk_number != 0
        or directory_disk != 0
        or disk_entries != total_entries
        or comment_length != 0
        or absolute_position + 22 != size
        or directory_offset + directory_size != absolute_position
        or total_entries > MAX_ARCHIVE_ENTRIES
    ):
        raise _archive_unsafe()


def _validate_extra(extra: bytes) -> None:
    offset = 0
    while offset < len(extra):
        if len(extra) - offset < 4:
            raise _archive_unsafe()
        identifier, length = struct.unpack_from("<HH", extra, offset)
        offset += 4
        if offset + length > len(extra) or identifier != 0x0001:
            raise _archive_unsafe()
        # No ZIP64 size is required under the 1 GiB/512 MiB structural limits.
        raise _archive_unsafe()


def _validate_archive_name(name: str) -> str:
    if (
        not name
        or "\x00" in name
        or _BUNDLE_PATH_RE.fullmatch(name) is None
        or "\\" in name
        or name.startswith("/")
        or name.endswith("/")
        or name.startswith("//")
        or _DRIVE_RE.match(name)
    ):
        raise _archive_unsafe()
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise _archive_unsafe()
    if any(
        part.endswith((".", " ")) or _WINDOWS_DEVICE_RE.match(part)
        for part in parts
    ):
        raise _archive_unsafe()
    try:
        encoded = name.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise _archive_unsafe() from exc
    if len(encoded) > MAX_ARCHIVE_PATH_BYTES or str(PurePosixPath(*parts)) != name:
        raise _archive_unsafe()
    if name != "llm-foundations-bundle/manifest.json" and not name.startswith(
        "llm-foundations-bundle/objects/"
    ):
        raise _archive_unsafe()
    return name


def inspect_zip_archive(
    path: Path,
    *,
    max_expanded_bytes: int = BUNDLE_EXPANDED_BYTES,
) -> ArchiveInspection:
    """Inspect and fully read a ZIP without extracting or registering content."""

    try:
        if path.stat().st_size > BUNDLE_ARCHIVE_BYTES:
            raise ApiError("PAYLOAD_TOO_LARGE", "The bundle archive is too large.")
    except OSError as exc:
        raise _storage_error(exc) from exc
    _validate_eocd(path)
    try:
        archive = zipfile.ZipFile(path, mode="r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise _archive_unsafe() from exc
    with archive:
        if archive.comment:
            raise _archive_unsafe()
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_ENTRIES:
            raise _archive_unsafe()
        names: set[str] = set()
        folded_names: set[str] = set()
        total_compressed = 0
        total_declared = 0
        for info in infos:
            name = _validate_archive_name(info.filename)
            folded = name.casefold()
            if name in names or folded in folded_names:
                raise _archive_unsafe()
            names.add(name)
            folded_names.add(folded)
            if info.comment or info.flag_bits & 0x1 or info.flag_bits & 0x8:
                raise _archive_unsafe()
            if any(ord(character) > 127 for character in name) and not info.flag_bits & 0x800:
                raise _archive_unsafe()
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise _archive_unsafe()
            _validate_extra(info.extra)
            mode = (info.external_attr >> 16) & 0xFFFF
            kind = stat.S_IFMT(mode)
            if (
                info.external_attr & 0x10
                or (kind and kind != stat.S_IFREG)
                or mode & 0o111
            ):
                raise _archive_unsafe()
            if info.file_size > MAX_ARCHIVE_ENTRY_BYTES:
                raise _archive_unsafe()
            if info.file_size and info.compress_size == 0:
                raise _archive_unsafe()
            if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
                raise _archive_unsafe()
            total_compressed += info.compress_size
            total_declared += info.file_size
            if total_declared > max_expanded_bytes:
                raise _archive_unsafe()
        if total_compressed and total_declared / total_compressed > MAX_COMPRESSION_RATIO:
            raise _archive_unsafe()

        total_actual = 0
        try:
            for info in infos:
                entry_actual = 0
                with archive.open(info, mode="r") as source:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        entry_actual += len(chunk)
                        total_actual += len(chunk)
                        if entry_actual > info.file_size or total_actual > max_expanded_bytes:
                            raise _archive_unsafe()
                if entry_actual != info.file_size:
                    raise _archive_unsafe()
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise _archive_unsafe() from exc
        return ArchiveInspection(
            entries=len(infos),
            total_compressed_bytes=total_compressed,
            total_uncompressed_bytes=total_actual,
        )


__all__ = [
    "ArchiveInspection",
    "StagedMultipart",
    "StagedPart",
    "inspect_zip_archive",
    "parse_content_length",
    "receive_json",
    "receive_multipart",
    "require_media_type",
]
