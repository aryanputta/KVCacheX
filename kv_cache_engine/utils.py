from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_json(path: str | Path, data: Any) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_device(preference: str) -> str:
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return preference


def resolve_dtype(device: str, preference: str) -> torch.dtype:
    if preference == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if preference not in mapping:
        raise ValueError(f"Unsupported dtype preference: {preference}")
    return mapping[preference]


def process_memory_bytes() -> int:
    return psutil.Process(os.getpid()).memory_info().rss


def percent_change(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return ((candidate - baseline) / baseline) * 100.0


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    magnitude = min(int(math.log(num_bytes, 1024)), len(suffixes) - 1)
    scaled = num_bytes / (1024**magnitude)
    return f"{scaled:.2f} {suffixes[magnitude]}"


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0
