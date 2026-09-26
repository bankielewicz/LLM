"""A small causal transformer with explicit attention for learning.

Learned positions, LayerNorm, GELU, separate output weights, no dropout.
It is a teaching architecture, not an implementation of a current named LLM.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int
    context: int = 64
    width: int = 64
    heads: int = 4
    layers: int = 2

    def __post_init__(self):
        if min(asdict(self).values()) < 1 or self.width % self.heads:
            raise ValueError("All sizes must be positive; width must divide evenly by heads.")


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.output = nn.Linear(cfg.width, cfg.width)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.context, cfg.context, dtype=torch.bool)))

    def forward(self, x):
        batch, time, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [part.view(batch, time, self.heads, width // self.heads).transpose(1, 2) for part in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(width // self.heads)
        scores = scores.masked_fill(~self.mask[:time, :time], float("-inf"))
        mixed = (scores.softmax(-1) @ v).transpose(1, 2).contiguous().view(batch, time, width)
        return self.output(mixed)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.attention = Attention(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.width, 4 * cfg.width), nn.GELU(), nn.Linear(4 * cfg.width, cfg.width))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.width)
        self.positions = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.Sequential(*(Block(cfg) for _ in range(cfg.layers)))
        self.norm = nn.LayerNorm(cfg.width)
        self.output = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, inputs, targets=None):
        length = inputs.shape[1]
        if not 1 <= length <= self.cfg.context:
            raise ValueError("Input length must be within the model context.")
        positions = torch.arange(length, device=inputs.device)
        x = self.tokens(inputs) + self.positions(positions)
        logits = self.output(self.norm(self.blocks(x)))
        # Targets arrive already shifted by the data loader. Do not shift twice.
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1))
        return logits, loss
