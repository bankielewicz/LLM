"""Deterministic TinyLM continuation and top-p sampling mechanics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .preview import TinyPromptPreview, preview_prompt


EOS_ID = 256


@dataclass(frozen=True)
class SamplingDistribution:
    """The ordered, truncated distribution used for one sampled token."""

    token_ids: tuple[int, ...]
    probabilities: torch.Tensor


@dataclass(frozen=True)
class TinyGeneration:
    preview: TinyPromptPreview
    generated_token_ids: tuple[int, ...]
    generated_text: str
    generated_token_count: int
    stop_reason: str
    replacement_occurred: bool


def _one_dimensional_logits(logits: torch.Tensor) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("logits must be a nonempty one-dimensional tensor")
    values = logits.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("logits must be finite")
    return values


def top_p_distribution(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> SamplingDistribution:
    """Return the exact float64 TinyLM top-p sampling distribution.

    Probabilities are sorted by descending value and then ascending token ID.
    The probability tensor stays on the logits device so a matching job-local
    Torch generator can be passed directly to torch.multinomial.
    """

    if isinstance(temperature, bool) or not 0.01 <= float(temperature) <= 2.0:
        raise ValueError("positive temperature must be within [0.01, 2]")
    if isinstance(top_p, bool) or not 0.0 < float(top_p) <= 1.0:
        raise ValueError("top_p must be within (0, 1]")

    values = _one_dimensional_logits(logits) / float(temperature)
    probabilities = torch.softmax(values, dim=-1, dtype=torch.float64)
    token_ids = tuple(
        sorted(
            range(probabilities.numel()),
            key=lambda token_id: (-float(probabilities[token_id].item()), token_id),
        )
    )
    order = torch.tensor(token_ids, dtype=torch.long, device=probabilities.device)
    ordered = probabilities.index_select(0, order)
    cumulative = torch.cumsum(ordered, dim=0, dtype=torch.float64)
    if float(top_p) == 1.0:
        retained_count = len(token_ids)
    else:
        threshold = torch.nonzero(cumulative >= float(top_p), as_tuple=False)
        retained_count = (
            int(threshold[0].item()) + 1
            if threshold.numel()
            else len(token_ids)
        )
    retained_ids = token_ids[:retained_count]
    retained = ordered[:retained_count]
    retained = retained / retained.sum(dtype=torch.float64)
    return SamplingDistribution(retained_ids, retained)


def select_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
) -> int:
    """Select one token using the frozen TinyLM greedy or top-p rule."""

    values = _one_dimensional_logits(logits)
    if float(temperature) == 0.0:
        if float(top_p) != 1.0:
            raise ValueError("top_p must equal 1 when temperature is zero")
        maximum = values.max()
        maxima = torch.nonzero(values == maximum, as_tuple=False).flatten()
        return int(maxima.min().item())

    if generator is None:
        raise ValueError("positive-temperature selection requires a job-local generator")
    distribution = top_p_distribution(values, temperature=temperature, top_p=top_p)
    sampled_index = int(
        torch.multinomial(distribution.probabilities, 1, generator=generator).item()
    )
    return distribution.token_ids[sampled_index]


def _model_logits(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    output = model(input_ids)
    logits = output[0] if isinstance(output, tuple) else output
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("TinyLM forward must return rank-three logits")
    return logits[0, -1, :]


def generate_tokens(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    context: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    device: str | torch.device,
    cancelled: Callable[[], bool] | None = None,
    eos_id: int = EOS_ID,
) -> TinyGeneration:
    """Generate a TinyLM continuation, polling cancellation before each step."""

    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 4_294_967_295
    ):
        raise ValueError("seed is outside the uint32 range")
    preview = preview_prompt(tokenizer, prompt, context=context)
    selected_device = torch.device(device)
    generator = None
    if float(temperature) != 0.0:
        generator = torch.Generator(device=selected_device).manual_seed(seed)

    generated: list[int] = []
    stop_reason = "length"
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if cancelled is not None and cancelled():
                stop_reason = "cancelled"
                break
            history = (*preview.input_token_ids, *generated)[-context:]
            input_ids = torch.tensor([history], dtype=torch.long, device=selected_device)
            token_id = select_next_token(
                _model_logits(model, input_ids),
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            generated.append(token_id)
            if token_id == eos_id:
                stop_reason = "eos"
                break

    visible_ids: Sequence[int]
    visible_ids = generated[:-1] if generated and generated[-1] == eos_id else generated
    try:
        generated_text = tokenizer.decode(visible_ids, errors="strict")
        replacement_occurred = False
    except UnicodeDecodeError:
        generated_text = tokenizer.decode(visible_ids, errors="replace")
        replacement_occurred = True
    return TinyGeneration(
        preview=preview,
        generated_token_ids=tuple(generated),
        generated_text=generated_text,
        generated_token_count=len(generated),
        stop_reason=stop_reason,
        replacement_occurred=replacement_occurred,
    )
