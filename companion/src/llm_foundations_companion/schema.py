"""Strict JSON, canonical JSON, and stdlib-only frozen-schema validation."""

from __future__ import annotations

import json
import math
import posixpath
import re
import unicodedata
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import unquote
from uuid import UUID

from .contract_data import ContractDataError, _load_json_cached, declared_files, load_json
from .errors import ApiError
from .limits import LIMITS


JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
Schema = bool | Mapping[str, Any]

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_TIME_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z$"
)
_ABSOLUTE_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/)")
_ANNOTATION_KEYS = {
    "$schema",
    "$id",
    "$defs",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "discriminator",
    "contentEncoding",
    "contentMediaType",
}
_VALIDATION_KEYS = {
    "$ref",
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "prefixItems",
    "enum",
    "const",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "minContains",
    "maxContains",
    "dependentRequired",
    "propertyNames",
}
_DEFERRED_SEMANTIC_RULES = {
    "TINY_PARAMETER_CAP",
    "EVALUATE_TINY_TEXT_BYTES",
    "ADAPTER_LINEAGE_UPDATES",
    "ASSESSMENT_DERIVATION",
    "PROGRESS_IMPORT_RESOLUTIONS",
    "CAPSTONE_REASON_CHAIN",
    "BACKUP_EXPORT_REFERENCES",
    "BUNDLE_RESUME_COMPLETE",
    "BUNDLE_BASE_IDENTITY",
    "BUNDLE_ENTRY_DIGEST",
    "EXERCISE_DERIVED_PROGRESS",
}


class SchemaDefinitionError(RuntimeError):
    """A bundled or caller-supplied schema uses unsupported or invalid syntax."""


class _DuplicateKey(ValueError):
    pass


class _NonFinite(ValueError):
    pass


@dataclass(frozen=True)
class _Issue:
    path: tuple[str | int, ...]
    message: str
    reason_code: str = "SCHEMA_INVALID"

    @property
    def pointer(self) -> str:
        if not self.path:
            return ""
        return "".join(
            "/" + str(part).replace("~", "~0").replace("/", "~1")
            for part in self.path
        )


@dataclass(frozen=True)
class _Context:
    document: str
    root: Mapping[str, Any]
    depth: int = 0

    def deeper(self, *, document: str | None = None, root: Mapping[str, Any] | None = None) -> "_Context":
        if self.depth >= 256:
            raise SchemaDefinitionError("schema reference depth exceeds 256")
        return _Context(document or self.document, root or self.root, self.depth + 1)


def _reject_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _parse_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise _NonFinite(token)
    return value


def _reject_constant(token: str) -> Any:
    raise _NonFinite(token)


def _ensure_json_value(value: Any, path: tuple[str | int, ...] = ()) -> None:
    if value is None or isinstance(value, (bool, str)):
        if isinstance(value, str):
            try:
                value.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise ValueError(f"lone surrogate at {_pointer(path)}") from exc
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number at {_pointer(path)}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_json_value(item, path + (index,))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string object key at {_pointer(path)}")
            _ensure_json_value(key, path + (key,))
            _ensure_json_value(item, path + (key,))
        return
    raise ValueError(f"non-JSON value at {_pointer(path)}")


def strict_json(raw: bytes | bytearray | memoryview | str) -> JSONValue:
    """Parse one strict UTF-8 JSON value without duplicates or non-finite numbers."""

    if isinstance(raw, (bytes, bytearray, memoryview)):
        try:
            text = bytes(raw).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ApiError(
                "VALIDATION_FAILED",
                "JSON input is not strict UTF-8.",
                reason_code="INVALID_ENCODING",
                field_errors=[{"field_path": "", "message": "Invalid UTF-8 input."}],
            ) from exc
    elif isinstance(raw, str):
        text = raw
    else:
        raise TypeError("strict_json accepts bytes-like input or str")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_pairs,
            parse_float=_parse_float,
            parse_constant=_reject_constant,
        )
        _ensure_json_value(value)
    except (json.JSONDecodeError, _DuplicateKey, _NonFinite, ValueError) as exc:
        reason = "INVALID_ENCODING" if "surrogate" in str(exc) else "INVALID_JSON"
        message = (
            "JSON input contains an invalid Unicode scalar."
            if reason == "INVALID_ENCODING"
            else "JSON input is malformed or contains a duplicate/non-finite value."
        )
        raise ApiError(
            "VALIDATION_FAILED",
            message,
            reason_code=reason,
            field_errors=[{"field_path": "", "message": message}],
        ) from exc
    return value


def canonical_json(value: JSONValue) -> bytes:
    """Return the exact INT-008 Python 3.12 canonical JSON bytes."""

    try:
        _ensure_json_value(value)
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return text.encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("value cannot be encoded as canonical JSON") from exc


def load_document(name: str) -> dict[str, Any]:
    """Load a wheel-local contract document as a defensive copy."""

    resolved = _resolve_document_name(name)
    document = load_json(resolved)
    if not isinstance(document, dict):
        raise ContractDataError(f"bundled contract document is not an object: {resolved}")
    return document


def _resolve_document_name(name: str, current: str | None = None) -> str:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or "?" in name
        or "#" in name
        or name.startswith("/")
        or ":" in name.split("/", 1)[0]
    ):
        raise ContractDataError(f"invalid contract document name: {name!r}")

    raw_candidates: list[str] = []
    if current:
        current_directory = posixpath.dirname(current)
        raw_candidates.append(posixpath.join(current_directory, name))
    raw_candidates.append(name)
    if "/" not in name:
        raw_candidates.append(posixpath.join("schemas", name))

    candidates: list[str] = []
    for candidate in raw_candidates:
        normalized = posixpath.normpath(candidate)
        if normalized in ("", ".", "..") or normalized.startswith("../"):
            continue
        if normalized not in candidates:
            candidates.append(normalized)

    if not candidates:
        raise ContractDataError(f"invalid contract document name: {name!r}")
    available = set(declared_files())
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ContractDataError(f"contract document is not bundled: {name}")


def _pointer(path: tuple[str | int, ...]) -> str:
    if not path:
        return ""
    return "".join(
        "/" + str(part).replace("~", "~0").replace("/", "~1") for part in path
    )


def _resolve_pointer(document: Any, fragment: str) -> Any:
    if not fragment:
        return document
    decoded = unquote(fragment)
    if not decoded.startswith("/"):
        raise SchemaDefinitionError(f"unsupported JSON reference fragment: #{fragment}")
    current = document
    for raw_token in decoded[1:].split("/"):
        if re.search(r"~(?![01])", raw_token):
            raise SchemaDefinitionError(f"invalid JSON pointer escape: {raw_token}")
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isascii() and token.isdigit():
            index = int(token)
            if index >= len(current):
                raise SchemaDefinitionError(f"JSON reference index is out of range: {token}")
            current = current[index]
        else:
            raise SchemaDefinitionError(f"unresolved JSON reference token: {token}")
    return current


def _resolve_ref(ref: str, context: _Context) -> tuple[Schema, _Context]:
    if not isinstance(ref, str) or ref.count("#") > 1:
        raise SchemaDefinitionError("$ref must be a string with at most one fragment")
    base, marker, fragment = ref.partition("#")
    if base:
        name = _resolve_document_name(base, context.document)
        root = _load_json_cached(name)
        if not isinstance(root, dict):
            raise SchemaDefinitionError(f"referenced contract is not an object: {name}")
        target_context = context.deeper(document=name, root=root)
    else:
        target_context = context.deeper()
    target = _resolve_pointer(target_context.root, fragment if marker else "")
    if not isinstance(target, (dict, bool)):
        raise SchemaDefinitionError(f"$ref does not resolve to a schema: {ref}")
    return target, target_context


def _assert_schema(schema: Schema) -> None:
    if isinstance(schema, bool):
        return
    if not isinstance(schema, Mapping):
        raise SchemaDefinitionError("schema must be an object or boolean")
    unknown = {
        key
        for key in schema
        if key not in _ANNOTATION_KEYS
        and key not in _VALIDATION_KEYS
        and not key.startswith("x-")
    }
    if unknown:
        raise SchemaDefinitionError(
            "unsupported JSON Schema keyword(s): " + ", ".join(sorted(unknown))
        )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _matches_type(expected: Any, value: Any) -> bool:
    if isinstance(expected, list):
        return any(_matches_type(item, value) for item in expected)
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return _is_number(value) and math.isfinite(float(value))
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    raise SchemaDefinitionError(f"unsupported JSON Schema type: {expected!r}")


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if _is_number(left) and _is_number(right):
        return Decimal(str(left)) == Decimal(str(right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _valid_format(name: str, value: str) -> bool:
    if name == "binary":
        return True
    if name == "uuid":
        if _UUID_RE.fullmatch(value) is None:
            return False
        try:
            return str(UUID(value)) == value
        except ValueError:
            return False
    if name == "date-time":
        if _TIME_RE.fullmatch(value) is None:
            return False
        try:
            datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            return False
        return True
    raise SchemaDefinitionError(f"unsupported asserted format: {name}")


def _number_issue(schema: Mapping[str, Any], value: int | float, path: tuple[str | int, ...], reason: str) -> list[_Issue]:
    issues: list[_Issue] = []
    for keyword, comparison, phrase in (
        ("minimum", lambda a, b: a < b, "below the minimum"),
        ("maximum", lambda a, b: a > b, "above the maximum"),
        ("exclusiveMinimum", lambda a, b: a <= b, "not above the exclusive minimum"),
        ("exclusiveMaximum", lambda a, b: a >= b, "not below the exclusive maximum"),
    ):
        if keyword in schema and comparison(value, schema[keyword]):
            issues.append(_Issue(path, f"Number is {phrase} {schema[keyword]!r}.", reason))
    if "multipleOf" in schema:
        divisor = schema["multipleOf"]
        if not _is_number(divisor) or divisor <= 0:
            raise SchemaDefinitionError("multipleOf must be a positive finite number")
        try:
            quotient = Decimal(str(value)) / Decimal(str(divisor))
        except (InvalidOperation, ZeroDivisionError) as exc:
            raise SchemaDefinitionError("invalid multipleOf value") from exc
        if quotient != quotient.to_integral_value():
            issues.append(_Issue(path, f"Number is not a multiple of {divisor!r}.", reason))
    return issues


def _collect(schema: Schema, value: Any, context: _Context, path: tuple[str | int, ...] = (), inherited_reason: str = "SCHEMA_INVALID") -> list[_Issue]:
    _assert_schema(schema)
    if schema is True:
        return []
    if schema is False:
        return [_Issue(path, "Value is rejected by the schema.", inherited_reason)]
    reason = schema.get("x-reason-code", inherited_reason)
    if not isinstance(reason, str):
        raise SchemaDefinitionError("x-reason-code must be a string")
    issues: list[_Issue] = []

    if "$ref" in schema:
        target, target_context = _resolve_ref(schema["$ref"], context)
        issues.extend(_collect(target, value, target_context, path, reason))

    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(expected_type, value):
        return issues + [_Issue(path, f"Expected type {expected_type!r}.", reason)]

    if "const" in schema and not _json_equal(value, schema["const"]):
        issues.append(_Issue(path, f"Value must equal {schema['const']!r}.", reason))
    if "enum" in schema:
        choices = schema["enum"]
        if not isinstance(choices, list) or not choices:
            raise SchemaDefinitionError("enum must be a nonempty array")
        if not any(_json_equal(value, choice) for choice in choices):
            issues.append(_Issue(path, "Value is not in the permitted enum.", reason))

    if _is_number(value):
        issues.extend(_number_issue(schema, value, path, reason))

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            issues.append(_Issue(path, f"String has fewer than {schema['minLength']} scalars.", reason))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            issues.append(_Issue(path, f"String has more than {schema['maxLength']} scalars.", reason))
        if "pattern" in schema:
            try:
                matched = re.search(schema["pattern"], value) is not None
            except (TypeError, re.error) as exc:
                raise SchemaDefinitionError("invalid schema pattern") from exc
            if not matched:
                issues.append(_Issue(path, "String does not match the required pattern.", reason))
        if "format" in schema and not _valid_format(schema["format"], value):
            issues.append(_Issue(path, f"String is not valid {schema['format']}.", reason))

    if isinstance(value, dict):
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            issues.append(_Issue(path, f"Object has fewer than {schema['minProperties']} properties.", reason))
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            issues.append(_Issue(path, f"Object has more than {schema['maxProperties']} properties.", reason))
        required = schema.get("required", [])
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            raise SchemaDefinitionError("required must be an array of property names")
        for name in required:
            if name not in value:
                issues.append(_Issue(path + (name,), "Required property is missing.", reason))
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise SchemaDefinitionError("properties must be an object")
        for name, child_schema in properties.items():
            if name in value:
                issues.extend(_collect(child_schema, value[name], context.deeper(), path + (name,), reason))
        extras = sorted(set(value) - set(properties))
        additional = schema.get("additionalProperties", True)
        if additional is False:
            for name in extras:
                issues.append(_Issue(path + (name,), "Unknown property is not permitted.", reason))
        elif isinstance(additional, (dict, bool)):
            for name in extras:
                issues.extend(_collect(additional, value[name], context.deeper(), path + (name,), reason))
        else:
            raise SchemaDefinitionError("additionalProperties must be a schema")
        if "propertyNames" in schema:
            for name in sorted(value):
                issues.extend(_collect(schema["propertyNames"], name, context.deeper(), path + (name,), reason))
        dependencies = schema.get("dependentRequired", {})
        if not isinstance(dependencies, Mapping):
            raise SchemaDefinitionError("dependentRequired must be an object")
        for trigger, names in dependencies.items():
            if trigger in value:
                if not isinstance(names, list) or any(not isinstance(item, str) for item in names):
                    raise SchemaDefinitionError("dependentRequired values must be string arrays")
                for name in names:
                    if name not in value:
                        issues.append(_Issue(path + (name,), f"Property is required when {trigger!r} is present.", reason))

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            issues.append(_Issue(path, f"Array has fewer than {schema['minItems']} items.", reason))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            issues.append(_Issue(path, f"Array has more than {schema['maxItems']} items.", reason))
        if schema.get("uniqueItems") is True:
            for index, item in enumerate(value):
                if any(_json_equal(item, earlier) for earlier in value[:index]):
                    issues.append(_Issue(path + (index,), "Array item is not unique.", reason))
        prefix = schema.get("prefixItems", [])
        if not isinstance(prefix, list):
            raise SchemaDefinitionError("prefixItems must be an array")
        for index, child_schema in enumerate(prefix[: len(value)]):
            issues.extend(_collect(child_schema, value[index], context.deeper(), path + (index,), reason))
        if "items" in schema:
            start = len(prefix)
            for index in range(start, len(value)):
                issues.extend(_collect(schema["items"], value[index], context.deeper(), path + (index,), reason))
        if "contains" in schema:
            matches = [
                index
                for index, item in enumerate(value)
                if not _collect(schema["contains"], item, context.deeper(), path + (index,), reason)
            ]
            minimum = schema.get("minContains", 1)
            maximum = schema.get("maxContains")
            if len(matches) < minimum or (maximum is not None and len(matches) > maximum):
                issues.append(_Issue(path, "Array does not satisfy contains cardinality.", reason))

    for child_schema in schema.get("allOf", []):
        issues.extend(_collect(child_schema, value, context.deeper(), path, reason))
    for keyword, exact in (("anyOf", False), ("oneOf", True)):
        if keyword not in schema:
            continue
        branches = schema[keyword]
        if not isinstance(branches, list) or not branches:
            raise SchemaDefinitionError(f"{keyword} must be a nonempty array")
        branch_issues = [
            _collect(branch, value, context.deeper(), path, reason) for branch in branches
        ]
        matches = [index for index, found in enumerate(branch_issues) if not found]
        if (exact and len(matches) != 1) or (not exact and not matches):
            if not matches:
                best = min(branch_issues, key=lambda found: (len(found), tuple((item.pointer, item.message) for item in found)))
                issues.extend(best or [_Issue(path, f"Value does not match any {keyword} branch.", reason)])
            else:
                issues.append(_Issue(path, "Value matches more than one oneOf branch.", reason))
    if "not" in schema and not _collect(schema["not"], value, context.deeper(), path, reason):
        issues.append(_Issue(path, "Value matches a forbidden schema.", reason))
    if "if" in schema:
        condition_matches = not _collect(schema["if"], value, context.deeper(), path, reason)
        selected = schema.get("then") if condition_matches else schema.get("else")
        if selected is not None:
            issues.extend(_collect(selected, value, context.deeper(), path, reason))
    return issues


def _normalize(schema: Schema, value: Any, context: _Context) -> Any:
    if isinstance(schema, bool):
        return value
    current = value
    if "$ref" in schema:
        target, target_context = _resolve_ref(schema["$ref"], context)
        current = _normalize(target, current, target_context)
    expected = schema.get("type")
    if expected == "number" and isinstance(current, int) and not isinstance(current, bool):
        current = float(current)
    if isinstance(current, dict):
        properties = schema.get("properties", {})
        for name, child_schema in properties.items():
            if name in current:
                current[name] = _normalize(child_schema, current[name], context.deeper())
        additional = schema.get("additionalProperties", True)
        if isinstance(additional, dict):
            for name in set(current) - set(properties):
                current[name] = _normalize(additional, current[name], context.deeper())
    if isinstance(current, list):
        prefix = schema.get("prefixItems", [])
        for index, child_schema in enumerate(prefix[: len(current)]):
            current[index] = _normalize(child_schema, current[index], context.deeper())
        if "items" in schema:
            for index in range(len(prefix), len(current)):
                current[index] = _normalize(schema["items"], current[index], context.deeper())
    for child_schema in schema.get("allOf", []):
        current = _normalize(child_schema, current, context.deeper())
    for keyword in ("oneOf", "anyOf"):
        for branch in schema.get(keyword, []):
            if not _collect(branch, value, context.deeper()):
                current = _normalize(branch, current, context.deeper())
                break
    if "if" in schema:
        selected = schema.get("then") if not _collect(schema["if"], value, context.deeper()) else schema.get("else")
        if selected is not None:
            current = _normalize(selected, current, context.deeper())
    return current


def _semantic_rules(schema: Schema, value: Any, context: _Context) -> list[str]:
    if isinstance(schema, bool):
        return []
    rules: list[str] = []
    declared = schema.get("x-semantic-rules", [])
    if not isinstance(declared, list) or any(not isinstance(item, str) for item in declared):
        raise SchemaDefinitionError("x-semantic-rules must be a string array")
    rules.extend(declared)
    if "$ref" in schema:
        target, target_context = _resolve_ref(schema["$ref"], context)
        rules.extend(_semantic_rules(target, value, target_context))
    for child in schema.get("allOf", []):
        rules.extend(_semantic_rules(child, value, context.deeper()))
    for keyword in ("oneOf", "anyOf"):
        for branch in schema.get(keyword, []):
            if not _collect(branch, value, context.deeper()):
                rules.extend(_semantic_rules(branch, value, context.deeper()))
                if keyword == "oneOf":
                    break
    if "if" in schema:
        selected = schema.get("then") if not _collect(schema["if"], value, context.deeper()) else schema.get("else")
        if selected is not None:
            rules.extend(_semantic_rules(selected, value, context.deeper()))
    return list(dict.fromkeys(rules))


def _semantic_issue(path: str, rule: str, reason: str | None = None) -> _Issue:
    normalized = path if path.startswith("/") or not path else "/" + path
    parts: tuple[str | int, ...] = tuple(
        token.replace("~1", "/").replace("~0", "~")
        for token in normalized.lstrip("/").split("/")
        if token != ""
    )
    return _Issue(parts, f"Semantic rule {rule} is not satisfied.", reason or "")


def _utf8_size(value: str) -> int:
    return len(value.encode("utf-8", errors="strict"))


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or _TIME_RE.fullmatch(value) is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return None


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def _apply_semantic_rule(rule: str, value: Any) -> list[_Issue]:
    if rule in _DEFERRED_SEMANTIC_RULES:
        return []
    if not isinstance(value, dict):
        return []
    if rule == "TINY_EVAL_CADENCE":
        if value.get("eval_every", 0) > value.get("steps", 0):
            return [_semantic_issue("eval_every", rule)]
    elif rule == "TINY_HEAD_DIVISIBILITY":
        heads = value.get("heads")
        width = value.get("width")
        if isinstance(heads, int) and heads and isinstance(width, int) and width % heads:
            return [_semantic_issue("heads", rule)]
    elif rule == "DIAGNOSIS_HOLD_SHAPE":
        profile = value.get("exercise_profile_id")
        hold = value.get("curriculum_hold_after_step")
        valid = profile is None and hold is None
        if profile is not None or hold is not None:
            valid = (
                profile == "tiny-v2-diagnosis-v1"
                and hold == 25
                and value.get("steps", 0) > 25
                and 25 % value.get("eval_every", 26) == 0
                and value.get("architecture_profile_id") == "tiny-v2-standard-v1"
                and "e01_verification_id" not in value
            )
        if not valid:
            return [_semantic_issue("curriculum_hold_after_step", rule, "DIAGNOSIS_HOLD_INVALID")]
    elif rule == "CHAT_MESSAGE_BYTES":
        messages = value.get("messages")
        if value.get("backend") == "chat" and messages is None:
            messages = value.get("messages", [])
        if isinstance(messages, list):
            return [
                _semantic_issue(f"messages/{index}/content", rule, "PAYLOAD_TOO_LARGE")
                for index, message in enumerate(messages)
                if isinstance(message, dict)
                and isinstance(message.get("content"), str)
                and _utf8_size(message["content"]) > LIMITS["chat_message_bytes"]
            ]
    elif rule == "CHAT_TOTAL_BYTES":
        messages = value.get("messages")
        if isinstance(messages, list):
            total = sum(
                _utf8_size(message.get("content", ""))
                for message in messages
                if isinstance(message, dict) and isinstance(message.get("content", ""), str)
            )
            if total > LIMITS["chat_total_bytes"]:
                return [_semantic_issue("messages", rule, "PAYLOAD_TOO_LARGE")]
    elif rule == "TINY_PROMPT_BYTES":
        prompt = value.get("prompt")
        if isinstance(prompt, str) and _utf8_size(prompt) > LIMITS["tiny_prompt_bytes"]:
            return [_semantic_issue("prompt", rule, "PAYLOAD_TOO_LARGE")]
    elif rule == "RETRIEVAL_QUERY_BYTES":
        query = value.get("query")
        if isinstance(query, str) and _utf8_size(query) > LIMITS["retrieval_query_bytes"]:
            return [_semantic_issue("query", rule, "PAYLOAD_TOO_LARGE")]
    elif rule == "EXPORT_SELECTION_COUNT":
        count = sum(
            len(value.get(name, []))
            for name in (
                "run_ids",
                "model_ids",
                "checkpoint_options",
                "evidence_ids",
                "note_ids",
                "conversation_ids",
            )
            if isinstance(value.get(name, []), list)
        )
        if not 1 <= count <= 100:
            return [_semantic_issue("checkpoint_options", rule, "SEMANTIC_INVALID")]
    elif rule == "EXPORT_RESUME_REQUIRES_DATASET":
        checkpoint_options = value.get("checkpoint_options", [])
        wants_resume = any(
            isinstance(item, dict) and item.get("include_resume_state") is True
            for item in checkpoint_options
        )
        includes_dataset = value.get("content_options", {}).get("include_dataset_bytes") is True
        if wants_resume and not includes_dataset:
            return [_semantic_issue("content_options/include_dataset_bytes", rule, "SEMANTIC_INVALID")]
    elif rule == "EVALUATE_SUBJECT_PROFILE":
        profile = value.get("evaluation_profile_id")
        subjects = value.get("subjects", [])
        kinds = [item.get("kind") for item in subjects if isinstance(item, dict)]
        valid = (
            profile == "tiny-nll-per-byte-v1" and kinds and all(kind == "tiny_checkpoint" for kind in kinds)
        ) or (
            profile == "applied-intents-greedy-v1" and kinds and all(kind in {"base_model", "adapter"} for kind in kinds)
        )
        if "release_token" in value:
            valid = valid and kinds == ["base_model", "adapter"]
        if not valid:
            return [_semantic_issue("subjects", rule, "SUBJECT_INCOMPATIBLE")]
    elif rule == "PROGRESS_DIRECT_FLAGS":
        dimension = value.get("dimension")
        setting = value.get("value")
        if (dimension in {"practiced", "self_checked"} and setting is True) or (
            setting is False and value.get("confirm_reset") is not True
        ):
            return [_semantic_issue("dimension", rule, "PROGRESS_DERIVED_FLAG")]
    elif rule == "BACKUP_EXPORT_SELECTION":
        for noun in ("notes", "evidence", "conversations"):
            include = value.get(f"include_{noun}") is True
            ids = value.get(f"{noun[:-1] if noun.endswith('s') else noun}_ids")
            if ids is not None and (include or not ids):
                return [_semantic_issue(f"{noun}_ids", rule, "SEMANTIC_INVALID")]
        if not value.get("confirm_sensitive_text") and (
            value.get("include_notes") or value.get("include_conversations")
        ):
            return [_semantic_issue("confirm_sensitive_text", rule, "SEMANTIC_INVALID")]
    elif rule == "JOB_FAILED_REASON":
        if value.get("state") == "failed":
            error = value.get("error")
            if not isinstance(error, dict) or value.get("terminal_reason") != error.get("code"):
                return [_semantic_issue("terminal_reason", rule, "SEMANTIC_INVALID")]
    elif rule == "JOB_STEP_BOUNDS":
        step = value.get("step")
        final = value.get("requested_final_step")
        if step is not None and final is not None and step > final:
            return [_semantic_issue("step", rule, "SEMANTIC_INVALID")]
    elif rule == "JOB_TIME_ORDER":
        created = _parse_time(value.get("created_at"))
        updated = _parse_time(value.get("updated_at"))
        started = _parse_time(value.get("started_at"))
        finished = _parse_time(value.get("finished_at"))
        if created and updated and (
            created > updated
            or (started is not None and created > started)
            or (finished is not None and started is not None and started > finished)
        ):
            return [_semantic_issue("updated_at", rule, "SEMANTIC_INVALID")]
    elif rule == "ASSESSMENT_TOTAL":
        rows = value.get("rubric_rows", [])
        if isinstance(rows, list):
            score = sum(item.get("score", 0) for item in rows if isinstance(item, dict))
            if value.get("total_score") != score:
                return [_semantic_issue("total_score", rule, "SEMANTIC_INVALID")]
    elif rule == "PROGRESS_TIMESTAMPS":
        modules = value.get("modules", {})
        for module_id, module in modules.items() if isinstance(modules, dict) else ():
            if isinstance(module, dict):
                for dimension in ("reading", "read", "practiced", "self_checked"):
                    if (f"{dimension}_at" in module) != (module.get(dimension) is True):
                        return [_semantic_issue(f"modules/{module_id}/{dimension}_at", rule, "SEMANTIC_INVALID")]
    elif rule == "EVIDENCE_IMPORT_STATE":
        if value.get("claim_status") == "imported_claim" and not (
            value.get("verification_status") == "unverified"
            and value.get("review_status") == "not_reviewed"
            and "verifier" not in value
            and value.get("assessment_ids") == []
        ):
            return [_semantic_issue("claim_status", rule, "SEMANTIC_INVALID")]
    elif rule == "EVIDENCE_VERIFIED_STATE":
        if value.get("verification_status") == "artifact_verified":
            checks = value.get("verification_checks", [])
            if not (
                value.get("claim_status") == "local_submission"
                and "verifier" in value
                and checks
                and all(isinstance(item, dict) and item.get("result") == "pass" for item in checks)
            ):
                return [_semantic_issue("verification_status", rule, "SEMANTIC_INVALID")]
    elif rule == "TINY_SAVE_CADENCE":
        if value.get("backend") == "tiny_v2":
            identity = value.get("training_identity", {})
            if identity.get("save_every") != identity.get("eval_every"):
                return [_semantic_issue("training_identity/save_every", rule, "SEMANTIC_INVALID")]
    elif rule == "BUNDLE_UNIQUE_PATHS":
        seen: set[str] = set()
        for index, entry in enumerate(value.get("entries", [])):
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                key = unicodedata.normalize("NFC", entry["path"]).casefold()
                if key in seen:
                    return [_semantic_issue(f"entries/{index}/path", rule, "ARCHIVE_UNSAFE")]
                seen.add(key)
    elif rule == "BUNDLE_NO_ABSOLUTE_PATHS":
        if any(_ABSOLUTE_PATH_RE.match(text) for text in _walk_strings(value)):
            return [_semantic_issue("entries", rule, "SEMANTIC_INVALID")]
    else:
        raise SchemaDefinitionError(f"unsupported semantic rule: {rule}")
    return []


def _raise_issues(issues: list[_Issue]) -> None:
    unique: dict[tuple[str, str], _Issue] = {}
    for issue in issues:
        unique[(issue.pointer, issue.message)] = issue
    ordered = sorted(unique.values(), key=lambda item: (item.pointer, item.message))[:100]
    reasons = {item.reason_code for item in ordered if item.reason_code}
    reason = next(iter(reasons)) if len(reasons) == 1 else None
    raise ApiError(
        "VALIDATION_FAILED",
        "Value does not satisfy the frozen contract.",
        reason_code=reason,
        field_errors=[
            {"field_path": item.pointer, "message": item.message} for item in ordered
        ],
    )


def validate_schema(schema: Schema, value: JSONValue, *, document: str = "openapi.json") -> JSONValue:
    """Validate one inline schema with offline refs and return type-normalized JSON."""

    name = _resolve_document_name(document)
    root = _load_json_cached(name)
    if not isinstance(root, dict):
        raise SchemaDefinitionError(f"schema document is not an object: {name}")
    try:
        _ensure_json_value(value)
    except ValueError as exc:
        _raise_issues([_Issue((), "Value is not strict JSON.")])
        raise AssertionError("unreachable") from exc
    context = _Context(name, root)
    issues = _collect(schema, value, context)
    if issues:
        _raise_issues(issues)
    normalized = _normalize(schema, deepcopy(value), context)
    semantic_issues: list[_Issue] = []
    for rule in _semantic_rules(schema, normalized, context):
        semantic_issues.extend(_apply_semantic_rule(rule, normalized))
    if semantic_issues:
        _raise_issues(semantic_issues)
    return normalized


def validate(schema_name: str, value: JSONValue) -> JSONValue:
    """Validate an OpenAPI component or bundled standalone schema by name."""

    if not isinstance(schema_name, str) or not schema_name:
        raise ValueError("schema_name must be a nonempty string")
    openapi = _load_json_cached("openapi.json")
    components = openapi.get("components", {}).get("schemas", {})
    if schema_name in components:
        return validate_schema(components[schema_name], value, document="openapi.json")
    document = _resolve_document_name(schema_name)
    schema = _load_json_cached(document)
    if not isinstance(schema, (dict, bool)):
        raise SchemaDefinitionError(f"contract document is not a schema: {document}")
    return validate_schema(schema, value, document=document)
