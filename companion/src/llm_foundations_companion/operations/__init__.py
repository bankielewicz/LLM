"""Closed operation dispatch for the S1 runtime boundary.

The concrete model and bundle operations are implemented by later slices. S1
still ships the complete closed operation vocabulary so an accepted worker can
never fall through to dynamic imports or learner-provided code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NoReturn


OperationHandler = Callable[[Mapping[str, Any], Any], Mapping[str, Any]]

OPERATIONS = (
    "tokenizer_train",
    "tiny_train",
    "tiny_resume",
    "evaluate",
    "generate",
    "model_prepare",
    "adapter_train",
    "adapter_resume",
    "context_preview",
    "chat_generate",
    "retrieval_build",
    "retrieval_query",
    "export_bundle",
    "validate_bundle",
    "import_bundle",
)


class OperationUnavailable(RuntimeError):
    """A closed operation whose implementation is outside this build slice."""

    code = "CAPABILITY_UNAVAILABLE"


def _unavailable(request: Mapping[str, Any], context: Any) -> NoReturn:
    operation = request.get("operation", "unknown")
    raise OperationUnavailable(
        f"The {operation} operation is not available in this build slice."
    )


# Later slices replace values with fixed imports. Request data never selects a
# module or callable name.
HANDLERS: Mapping[str, OperationHandler] = {
    name: _unavailable for name in OPERATIONS
}


def handler_for(operation: str) -> OperationHandler:
    try:
        return HANDLERS[operation]
    except KeyError as exc:
        raise OperationUnavailable("The requested operation is not supported.") from exc


__all__ = [
    "HANDLERS",
    "OPERATIONS",
    "OperationHandler",
    "OperationUnavailable",
    "handler_for",
]
