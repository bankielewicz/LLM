"""Worker-only deterministic Torch setup for tiny-v2 operations."""

from __future__ import annotations

import os
import random
import sys
from typing import Any


_THREADS_CONFIGURED = False


def configure_torch_determinism(seed: int, device: str) -> Any:
    """Configure the selected worker process before model construction.

    The CUDA workspace environment is established before importing Torch.
    Callers should invoke this as the first Torch-bearing action in a worker.
    """

    global _THREADS_CONFIGURED
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 4_294_967_295
    ):
        raise ValueError("seed must be an integer from 0 through 4294967295.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda.")

    if device == "cuda":
        required = ":4096:8"
        if "torch" in sys.modules and os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ) != required:
            raise RuntimeError(
                "CUDA deterministic workspace must be configured before Torch import."
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = required

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The selected CUDA device is unavailable.")
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    if not _THREADS_CONFIGURED:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        _THREADS_CONFIGURED = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return torch


__all__ = ["configure_torch_determinism"]
