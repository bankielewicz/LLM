"""Closed, role-aware public error envelopes for the local companion."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

from .contract_data import load_json


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _vocabulary() -> Mapping[str, Mapping[str, Any]]:
    document = load_json("error-vocabulary.json")
    if not isinstance(document, dict) or not isinstance(document.get("codes"), list):
        raise RuntimeError("bundled error vocabulary is malformed")
    rows: dict[str, Mapping[str, Any]] = {}
    for row in document["codes"]:
        if not isinstance(row, dict) or not isinstance(row.get("code"), str):
            raise RuntimeError("bundled error vocabulary row is malformed")
        code = row["code"]
        if code in rows:
            raise RuntimeError(f"duplicate bundled error code: {code}")
        rows[code] = MappingProxyType(row)
    return MappingProxyType(rows)


ERROR_VOCABULARY = _vocabulary()


def _binding(row: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    kinds = row.get("kinds")
    if not isinstance(kinds, list) or kind not in kinds:
        raise ValueError(f"{row.get('code', '<unknown>')} is not permitted as {kind}")
    bindings = row.get("bindings")
    candidate: Any
    if bindings is None:
        candidate = row
    elif isinstance(bindings, dict):
        candidate = bindings.get(kind)
    else:
        candidate = None
    if not isinstance(candidate, Mapping):
        raise RuntimeError(
            f"bundled error vocabulary has no {kind} binding for {row['code']}"
        )
    return candidate


def _default_message(code: str, reason_code: str | None) -> str:
    row = ERROR_VOCABULARY[reason_code or code]
    meaning = row.get("meaning")
    if isinstance(meaning, str) and meaning and meaning != "Top-level HTTP error code":
        return meaning
    return code.replace("_", " ").capitalize() + "."


def _normalize_field_errors(
    field_errors: Iterable[Mapping[str, Any]] | None,
) -> tuple[Mapping[str, str], ...]:
    normalized: list[Mapping[str, str]] = []
    for item in field_errors or ():
        if not isinstance(item, Mapping) or set(item) != {"field_path", "message"}:
            raise ValueError("field errors must contain exactly field_path and message")
        path = item["field_path"]
        message = item["message"]
        if not isinstance(path, str) or len(path) > 500:
            raise ValueError("field error path must be a string of at most 500 scalars")
        if not isinstance(message, str) or not (1 <= len(message) <= 1_000):
            raise ValueError("field error message must contain 1 to 1000 scalars")
        normalized.append(
            MappingProxyType({"field_path": path, "message": message})
        )
    if len(normalized) > 100:
        raise ValueError("at most 100 field errors are permitted")
    normalized.sort(key=lambda item: (item["field_path"], item["message"]))
    return tuple(normalized)


class ApiError(Exception):
    """A validated synchronous error from the sealed public vocabulary."""

    def __init__(
        self,
        code: str,
        message: str | None = None,
        *,
        reason_code: str | None = None,
        field_errors: Iterable[Mapping[str, Any]] | None = None,
        http_status: int | None = None,
    ) -> None:
        if not isinstance(code, str) or code not in ERROR_VOCABULARY:
            raise ValueError(f"unknown public error code: {code!r}")
        top_binding = _binding(ERROR_VOCABULARY[code], "top_level")
        expected_status = top_binding.get("http_status")
        retryable = top_binding.get("retryable")
        if not isinstance(expected_status, int) or isinstance(expected_status, bool):
            raise RuntimeError(f"invalid top-level HTTP binding for {code}")
        if not isinstance(retryable, bool):
            raise RuntimeError(f"invalid top-level retry binding for {code}")

        if reason_code is not None:
            if not isinstance(reason_code, str) or reason_code not in ERROR_VOCABULARY:
                raise ValueError(f"unknown public reason code: {reason_code!r}")
            reason_binding = _binding(ERROR_VOCABULARY[reason_code], "reason")
            if reason_binding.get("top_level_code") != code:
                raise ValueError(
                    f"reason code {reason_code} is not bound to top-level code {code}"
                )
            if reason_binding.get("http_status") != expected_status:
                raise RuntimeError(
                    f"inconsistent HTTP binding for {code}/{reason_code}"
                )
            reason_retryable = reason_binding.get("retryable")
            if not isinstance(reason_retryable, bool):
                raise RuntimeError(f"invalid retry binding for {reason_code}")
            retryable = reason_retryable

        if http_status is not None and http_status != expected_status:
            raise ValueError(
                f"HTTP status {http_status} conflicts with {code}'s sealed status "
                f"{expected_status}"
            )
        resolved_message = message or _default_message(code, reason_code)
        if not isinstance(resolved_message, str) or not (1 <= len(resolved_message) <= 2_000):
            raise ValueError("public error message must contain 1 to 2000 scalars")

        self.code = code
        self.message = resolved_message
        self.reason_code = reason_code
        self.field_errors = _normalize_field_errors(field_errors)
        self.retryable = retryable
        self.status_code = expected_status
        super().__init__(resolved_message)

    @property
    def http_status(self) -> int:
        """Compatibility alias for callers that name the sealed HTTP binding."""

        return self.status_code

    def as_error(self, request_id: str) -> dict[str, Any]:
        """Return the inner ``Error.error`` object."""

        if not isinstance(request_id, str) or _UUID_RE.fullmatch(request_id) is None:
            raise ValueError("request_id must be a lowercase RFC 4122 UUID")
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "field_errors": [dict(item) for item in self.field_errors],
            "retryable": self.retryable,
            "request_id": request_id,
        }
        if self.reason_code is not None:
            result["reason_code"] = self.reason_code
        return result

    def as_envelope(self, request_id: str) -> dict[str, Any]:
        """Return the complete OpenAPI ``Error`` object."""

        return {"error": self.as_error(request_id)}
