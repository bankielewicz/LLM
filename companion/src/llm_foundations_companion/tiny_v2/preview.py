"""Framework-free TinyLM prompt tokenization and cropping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TinyPromptPreview:
    original_input_token_count: int
    input_token_ids: tuple[int, ...]
    input_token_count: int
    serialized_text: str
    cropped_input_tokens: int
    effective_context_budget: int


def preview_prompt(tokenizer: Any, prompt: str, *, context: int) -> TinyPromptPreview:
    """Tokenize and crop a TinyLM prompt without appending EOS."""

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a nonempty string")
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        raise ValueError("context must be a positive integer")
    token_ids = tuple(int(value) for value in tokenizer.encode(prompt))
    if not token_ids:
        raise ValueError("prompt tokenization must be nonempty")
    if tokenizer.decode(token_ids, errors="strict") != prompt:
        raise ValueError("prompt tokenization did not round-trip exactly")
    retained = token_ids[-context:]
    serialized = tokenizer.decode(retained, errors="replace")
    return TinyPromptPreview(
        original_input_token_count=len(token_ids),
        input_token_ids=retained,
        input_token_count=len(retained),
        serialized_text=serialized,
        cropped_input_tokens=len(token_ids) - len(retained),
        effective_context_budget=context,
    )


__all__ = ["TinyPromptPreview", "preview_prompt"]
