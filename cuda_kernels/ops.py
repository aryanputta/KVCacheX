from __future__ import annotations

import time
from typing import Any, Callable

import torch


def profile_callable(function: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        return result, float(start.elapsed_time(end))
    wall_start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, (time.perf_counter() - wall_start) * 1000.0


def prune_and_compact_sequence(tensor: torch.Tensor, keep_indices: torch.Tensor) -> tuple[torch.Tensor, float]:
    def _apply() -> torch.Tensor:
        return tensor.index_select(2, keep_indices).contiguous()

    return profile_callable(_apply)
