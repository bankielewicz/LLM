"""Fixed tiny-v2 causal transformer.

The standard profile preserves the protected lesson architecture, module names,
initialization order, and already-shifted target contract.  The tied profile
changes only output/token weight identity and initializes shared storage once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


STANDARD_PROFILE = "tiny-v2-standard-v1"
TIED_PROFILE = "tiny-v2-weight-tied-v1"
UNTIED_VARIANT = "tiny-v2-untied-v1"
TIED_VARIANT = "tiny-v2-tied-v1"


@dataclass(frozen=True)
class Config:
    vocab_size: int = 257
    context: int = 64
    width: int = 64
    heads: int = 4
    layers: int = 2

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values.values()):
            raise ValueError("All model sizes must be integers.")
        if min(values.values()) < 1 or self.width % self.heads:
            raise ValueError(
                "All sizes must be positive; width must divide evenly by heads."
            )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Config":
        expected = {"vocab_size", "context", "width", "heads", "layers"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("Tiny model config has missing or unknown fields.")
        return cls(**{name: value[name] for name in expected})


class Attention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.heads = cfg.heads
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.output = nn.Linear(cfg.width, cfg.width)
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(cfg.context, cfg.context, dtype=torch.bool)
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [
            part.view(
                batch_size,
                time,
                self.heads,
                width // self.heads,
            ).transpose(1, 2)
            for part in (q, k, v)
        ]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(width // self.heads)
        scores = scores.masked_fill(
            ~self.mask[:time, :time], float("-inf")
        )
        mixed = (
            (scores.softmax(-1) @ v)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, time, width)
        )
        return self.output(mixed)


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.width)
        self.norm2 = nn.LayerNorm(cfg.width)
        self.attention = Attention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.width, 4 * cfg.width),
            nn.GELU(),
            nn.Linear(4 * cfg.width, cfg.width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(self, cfg: Config, *, tied_weights: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.tied_weights = tied_weights
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.width)
        self.positions = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.Sequential(*(Block(cfg) for _ in range(cfg.layers)))
        self.norm = nn.LayerNorm(cfg.width)
        self.output = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        if tied_weights:
            self.output.weight = self.tokens.weight

        initialized: set[int] = set()

        def initialize_once(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                identity = id(module.weight)
                if identity not in initialized:
                    nn.init.normal_(module.weight, std=0.02)
                    initialized.add(identity)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

        # Module.apply uses the protected post-order traversal.  De-duplicating
        # by Parameter identity is inert for the standard profile and prevents
        # the tied profile from initializing its shared storage twice.
        self.apply(initialize_once)

    @property
    def architecture_variant(self) -> str:
        return TIED_VARIANT if self.tied_weights else UNTIED_VARIANT

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        length = inputs.shape[1]
        if not 1 <= length <= self.cfg.context:
            raise ValueError("Input length must be within the model context.")
        positions = torch.arange(length, device=inputs.device)
        x = self.tokens(inputs) + self.positions(positions)
        logits = self.output(self.norm(self.blocks(x)))
        loss = (
            None
            if targets is None
            else F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
            )
        )
        return logits, loss


def profile_is_tied(architecture_profile_id: str) -> bool:
    if architecture_profile_id == STANDARD_PROFILE:
        return False
    if architecture_profile_id == TIED_PROFILE:
        return True
    raise ValueError("Unsupported tiny-v2 architecture profile.")


def architecture_variant(architecture_profile_id: str) -> str:
    return TIED_VARIANT if profile_is_tied(architecture_profile_id) else UNTIED_VARIANT


def build_model(
    config: Config,
    architecture_profile_id: str = STANDARD_PROFILE,
) -> TinyLM:
    if not isinstance(config, Config):
        raise TypeError("config must be a tiny-v2 Config.")
    return TinyLM(
        config,
        tied_weights=profile_is_tied(architecture_profile_id),
    )


def unique_parameter_count(
    config: Config,
    architecture_profile_id: str = STANDARD_PROFILE,
) -> int:
    if not isinstance(config, Config):
        raise TypeError("config must be a tiny-v2 Config.")
    vocab = config.vocab_size
    width = config.width
    count = (
        2 * vocab * width
        + config.context * width
        + config.layers * (12 * width * width + 13 * width)
        + 2 * width
    )
    if profile_is_tied(architecture_profile_id):
        count -= vocab * width
    return count


__all__ = [
    "Attention",
    "Block",
    "Config",
    "STANDARD_PROFILE",
    "TIED_PROFILE",
    "TIED_VARIANT",
    "TinyLM",
    "UNTIED_VARIANT",
    "architecture_variant",
    "build_model",
    "profile_is_tied",
    "unique_parameter_count",
]
