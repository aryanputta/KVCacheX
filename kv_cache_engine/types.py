from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class QuantizedTensor:
    data: torch.Tensor
    scale: torch.Tensor
    offset: torch.Tensor | None
    bits: int
    shape: tuple[int, ...]
    original_dtype: torch.dtype
    scheme: str = "symmetric_per_head"

    def estimated_bytes(self) -> int:
        total = (
            self.data.numel() * self.data.element_size()
            + self.scale.numel() * self.scale.element_size()
        )
        if self.offset is not None:
            total += self.offset.numel() * self.offset.element_size()
        return total


@dataclass
class LowRankTensor:
    u: torch.Tensor
    s: torch.Tensor
    vh: torch.Tensor
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype

    def estimated_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() for tensor in (self.u, self.s, self.vh)
        )


@dataclass
class SegmentedTensor:
    compressed_chunks: list["TensorLike"]
    raw_tail: torch.Tensor
    axis: int = 2

    def estimated_bytes(self) -> int:
        return sum(estimate_tensor_like_bytes(chunk) for chunk in self.compressed_chunks) + (
            self.raw_tail.numel() * self.raw_tail.element_size()
        )

    def compressed_prefix_tokens(self) -> int:
        return sum(tensor_like_seq_len(chunk) for chunk in self.compressed_chunks)

    def total_tokens(self) -> int:
        return self.compressed_prefix_tokens() + int(self.raw_tail.shape[self.axis])


TensorLike = torch.Tensor | QuantizedTensor | LowRankTensor | SegmentedTensor


def tensor_like_seq_len(tensor_like: TensorLike) -> int:
    if isinstance(tensor_like, torch.Tensor):
        return int(tensor_like.shape[2])
    if isinstance(tensor_like, QuantizedTensor):
        return int(tensor_like.shape[2])
    if isinstance(tensor_like, LowRankTensor):
        return int(tensor_like.original_shape[2])
    if isinstance(tensor_like, SegmentedTensor):
        return tensor_like.total_tokens()
    raise TypeError(f"Unsupported tensor type: {type(tensor_like)!r}")


def estimate_tensor_like_bytes(tensor_like: TensorLike) -> int:
    if isinstance(tensor_like, torch.Tensor):
        return tensor_like.numel() * tensor_like.element_size()
    if isinstance(tensor_like, SegmentedTensor):
        return tensor_like.estimated_bytes()
    return tensor_like.estimated_bytes()


@dataclass
class LayerKVCache:
    key: TensorLike
    value: TensorLike
    compression_method: str = "raw"

    def estimated_bytes(self) -> int:
        return estimate_tensor_like_bytes(self.key) + estimate_tensor_like_bytes(self.value)


@dataclass
class SchedulePlan:
    target_cache_tokens: int
    collect_attentions: bool
    apply_compression: bool
    compression_method: str
    memory_pressure: float
    latency_pressure: float


@dataclass
class KVCacheState:
    layers: list[LayerKVCache]
    positions: torch.Tensor
    token_ids: torch.Tensor
    cumulative_attention: torch.Tensor
    recent_attention: torch.Tensor
    embedding_norms: torch.Tensor
    step_index: int = 0
    compression_ratio: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        return int(self.positions.numel())

    def estimated_bytes(self) -> int:
        return sum(layer.estimated_bytes() for layer in self.layers)

    def is_compressed(self) -> bool:
        return any(layer.compression_method != "raw" for layer in self.layers)

    def slice(self, keep_indices: torch.Tensor) -> "KVCacheState":
        if keep_indices.dtype != torch.long:
            keep_indices = keep_indices.to(dtype=torch.long)
        keep_indices = torch.unique(keep_indices, sorted=True)
        sliced_layers: list[LayerKVCache] = []
        for layer in self.layers:
            if not isinstance(layer.key, torch.Tensor) or not isinstance(layer.value, torch.Tensor):
                raise TypeError("KVCacheState.slice expects raw tensor layers.")
            sliced_layers.append(
                LayerKVCache(
                    key=layer.key.index_select(dim=2, index=keep_indices),
                    value=layer.value.index_select(dim=2, index=keep_indices),
                    compression_method="raw",
                )
            )
        return KVCacheState(
            layers=sliced_layers,
            positions=self.positions.index_select(0, keep_indices),
            token_ids=self.token_ids.index_select(0, keep_indices),
            cumulative_attention=self.cumulative_attention.index_select(0, keep_indices),
            recent_attention=self.recent_attention.index_select(0, keep_indices),
            embedding_norms=self.embedding_norms.index_select(0, keep_indices),
            step_index=self.step_index,
            compression_ratio=self.compression_ratio,
            metadata=dict(self.metadata),
        )


@dataclass
class StepMetrics:
    step: int
    latency_ms: float
    process_memory_bytes: int
    cache_bytes: int
    compression_ratio: float
    nll: float | None = None
    token_agreement: float | None = None
    kernel_time_ms: float | None = None
    bandwidth_bytes: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    workload_name: str
    mode: str
    run_kind: str
    prompt_tokens: int
    generated_tokens: int
    output_text: str
    mean_latency_ms: float
    p95_latency_ms: float
    tokens_per_sec: float
    peak_process_memory_bytes: int
    peak_cache_bytes: int
    mean_nll: float | None
    perplexity: float | None
    token_agreement: float | None
    compression_ratio: float
    extra_metrics: dict[str, Any] = field(default_factory=dict)
