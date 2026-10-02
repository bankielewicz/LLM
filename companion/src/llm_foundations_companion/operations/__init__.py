"""Closed operation dispatch.

S2 handlers are imported only inside literal wrapper functions.  The service
process imports this module to validate operation names, so eager handler
imports here would also import the model framework into the parent process.
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


class OperationFailure(RuntimeError):
    """A handler failure in the closed asynchronous job-error vocabulary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field_errors: tuple[Mapping[str, str], ...] = (),
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field_errors = tuple(dict(item) for item in field_errors)
        self.retryable = retryable


def _unavailable(request: Mapping[str, Any], context: Any) -> NoReturn:
    operation = request.get("operation", "unknown")
    raise OperationUnavailable(
        f"The {operation} operation is not available in this build slice."
    )


def _tokenizer_train(request: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    from .tokenizer_training import handle_tokenizer_train

    return handle_tokenizer_train(request, context)


def _tiny_train(request: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    from .tiny_training import handle_tiny_train

    return handle_tiny_train(request, context)


def _tiny_resume(request: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    from .tiny_training import handle_tiny_resume

    return handle_tiny_resume(request, context)


def _evaluate(request: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    from .tiny_inference import handle_evaluate

    return handle_evaluate(request, context)


def _context_preview(request: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    from .tiny_inference import handle_context_preview

    return handle_context_preview(request, context)


def _generate(request: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
    from .tiny_inference import handle_generate

    return handle_generate(request, context)


# Later slices replace values with fixed imports. Request data never selects a
# module or callable name.
HANDLERS: Mapping[str, OperationHandler] = {
    "tokenizer_train": _tokenizer_train,
    "tiny_train": _tiny_train,
    "tiny_resume": _tiny_resume,
    "evaluate": _evaluate,
    "generate": _generate,
    "model_prepare": _unavailable,
    "adapter_train": _unavailable,
    "adapter_resume": _unavailable,
    "context_preview": _context_preview,
    "chat_generate": _unavailable,
    "retrieval_build": _unavailable,
    "retrieval_query": _unavailable,
    "export_bundle": _unavailable,
    "validate_bundle": _unavailable,
    "import_bundle": _unavailable,
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
    "OperationFailure",
    "OperationUnavailable",
    "handler_for",
]
