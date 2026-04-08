from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import torch

from kv_cache_engine.config import CompressionConfig
from kv_cache_engine.types import (
    KVCacheState,
    LayerKVCache,
    LowRankTensor,
    QuantizedTensor,
    SegmentedTensor,
    TensorLike,
    tensor_like_seq_len,
)

from .clustering import cluster_state
from .low_rank import compress_tensor_low_rank, decompress_low_rank_tensor
from .quantization import dequantize_tensor, quantize_tensor


class CompressionAgent:
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.reset_runtime_cache()

    def reset_runtime_cache(self) -> None:
        self._chunk_materialization_cache: OrderedDict[
            tuple[object, ...], tuple[QuantizedTensor | LowRankTensor, torch.Tensor]
        ] = OrderedDict()
        self._prefix_materialization_cache: OrderedDict[
            tuple[tuple[object, ...], ...], tuple[tuple[TensorLike, ...], torch.Tensor]
        ] = OrderedDict()
        self._materialization_stats = {
            "chunk_cache_hits": 0,
            "chunk_cache_misses": 0,
            "prefix_cache_hits": 0,
            "prefix_cache_misses": 0,
            "prefix_cache_extensions": 0,
        }

    def materialization_stats(self) -> dict[str, int]:
        return dict(self._materialization_stats)

    def _quantization_scheme(self, tensor_kind: str) -> str:
        if not self.config.asymmetric_kv_quantization:
            return "symmetric_per_head"
        if tensor_kind == "key":
            return self.config.key_quantization_scheme
        return self.config.value_quantization_scheme

    def compress_state(
        self,
        state: KVCacheState,
        method_override: str | None = None,
        previous_state: KVCacheState | None = None,
    ) -> KVCacheState:
        if not self.config.enabled:
            return state
        raw_state = self.decompress_state(state)
        method = method_override or self.config.method
        if (
            method in {"clustering", "hybrid"}
            or method.startswith("hybrid")
        ) and raw_state.seq_len >= self.config.min_seq_for_clustering:
            raw_state = cluster_state(
                raw_state,
                cluster_ratio=self.config.cluster_ratio,
                prefix_fraction=self.config.cluster_prefix_fraction,
            )
        if method == "clustering":
            raw_state.compression_ratio = 1.0
            raw_state.metadata["compression_method"] = method
            return raw_state
        if self._should_segment_prefix(method, raw_state):
            return self._compress_state_segmented(raw_state, method, previous_state)

        original_bytes = raw_state.estimated_bytes()
        compressed_layers = []
        for layer in raw_state.layers:
            if not isinstance(layer.key, torch.Tensor) or not isinstance(layer.value, torch.Tensor):
                raise TypeError("Compression expects raw tensors prior to quantization or factorization.")
            if method == "low_rank" and raw_state.seq_len >= self.config.min_seq_for_low_rank:
                key = compress_tensor_low_rank(layer.key, self.config.low_rank_ratio)
                value = compress_tensor_low_rank(layer.value, self.config.low_rank_ratio)
                layer_method = "low_rank"
            else:
                bits = 4 if "4" in method else self.config.quantization_bits
                key = quantize_tensor(
                    layer.key,
                    bits=bits,
                    scheme=self._quantization_scheme("key"),
                )
                value = quantize_tensor(
                    layer.value,
                    bits=bits,
                    scheme=self._quantization_scheme("value"),
                )
                layer_method = f"quantized_{key.scheme}_{value.scheme}_int{bits}"
            compressed_layers.append(
                LayerKVCache(key=key, value=value, compression_method=layer_method)
            )
        compressed_state = replace(
            raw_state,
            layers=compressed_layers,
            metadata={
                **raw_state.metadata,
                "compression_method": method,
                "asymmetric_kv_quantization": self.config.asymmetric_kv_quantization,
                "key_quantization_scheme": self._quantization_scheme("key"),
                "value_quantization_scheme": self._quantization_scheme("value"),
            },
        )
        compressed_bytes = max(1, compressed_state.estimated_bytes())
        compressed_state.compression_ratio = original_bytes / compressed_bytes
        return compressed_state

    def _should_segment_prefix(self, method: str, state: KVCacheState) -> bool:
        if not self.config.segmented_prefix_enabled:
            return False
        if "quantization" not in method:
            return False
        return state.seq_len > self.config.segment_tail_tokens

    def _compress_state_segmented(
        self,
        raw_state: KVCacheState,
        method: str,
        previous_state: KVCacheState | None,
    ) -> KVCacheState:
        original_bytes = raw_state.estimated_bytes()
        previous_layers = (
            previous_state.layers
            if previous_state is not None
            and previous_state.metadata.get("compression_method") == method
            and len(previous_state.layers) == len(raw_state.layers)
            else None
        )
        compressed_layers = []
        for layer_index, layer in enumerate(raw_state.layers):
            if not isinstance(layer.key, torch.Tensor) or not isinstance(layer.value, torch.Tensor):
                raise TypeError("Segmented compression expects raw tensors.")
            previous_key = previous_layers[layer_index].key if previous_layers is not None else None
            previous_value = previous_layers[layer_index].value if previous_layers is not None else None
            key = self._compress_tensor_segmented(
                layer.key,
                previous_key,
                method,
                tensor_kind="key",
            )
            value = self._compress_tensor_segmented(
                layer.value,
                previous_value,
                method,
                tensor_kind="value",
            )
            bits = 4 if "4" in method else self.config.quantization_bits
            compressed_layers.append(
                LayerKVCache(
                    key=key,
                    value=value,
                    compression_method=(
                        f"segmented_quantized_{self._quantization_scheme('key')}_"
                        f"{self._quantization_scheme('value')}_int{bits}"
                    ),
                )
            )
        compressed_state = replace(
            raw_state,
            layers=compressed_layers,
            metadata={
                **raw_state.metadata,
                "compression_method": method,
                "asymmetric_kv_quantization": self.config.asymmetric_kv_quantization,
                "key_quantization_scheme": self._quantization_scheme("key"),
                "value_quantization_scheme": self._quantization_scheme("value"),
                "segmented_prefix": True,
                "segment_tail_tokens": self.config.segment_tail_tokens,
                "segment_flush_tokens": self.config.segment_flush_tokens,
            },
        )
        compressed_state.compression_ratio = original_bytes / max(1, compressed_state.estimated_bytes())
        return compressed_state

    def _compress_tensor_segmented(
        self,
        tensor: torch.Tensor,
        previous_tensor: torch.Tensor | QuantizedTensor | LowRankTensor | SegmentedTensor | None,
        method: str,
        tensor_kind: str,
    ) -> SegmentedTensor:
        if tensor.dim() != 4:
            raise ValueError("Segmented KV compression expects a [B, H, S, D] tensor.")
        bits = 4 if "4" in method else self.config.quantization_bits
        scheme = self._quantization_scheme(tensor_kind)
        tail_tokens = min(self.config.segment_tail_tokens, tensor.shape[2])
        flush_tokens = max(1, min(self.config.segment_flush_tokens, tail_tokens))

        if isinstance(previous_tensor, SegmentedTensor):
            prefix_tokens = previous_tensor.compressed_prefix_tokens()
            if prefix_tokens <= tensor.shape[2]:
                new_chunks = list(previous_tensor.compressed_chunks)
                overflow = tensor.shape[2] - prefix_tokens - tail_tokens
                if overflow > 0:
                    flush_tensor = tensor[:, :, prefix_tokens : prefix_tokens + overflow, :].contiguous()
                    for chunk_start in range(0, flush_tensor.shape[2], flush_tokens):
                        chunk = flush_tensor[:, :, chunk_start : chunk_start + flush_tokens, :].contiguous()
                        new_chunks.append(quantize_tensor(chunk, bits=bits, scheme=scheme))
                    prefix_tokens += overflow
                raw_tail = tensor[:, :, prefix_tokens:, :].contiguous()
                return SegmentedTensor(compressed_chunks=new_chunks, raw_tail=raw_tail)

        if tensor.shape[2] <= tail_tokens:
            return SegmentedTensor(compressed_chunks=[], raw_tail=tensor.contiguous())

        prefix = tensor[:, :, : tensor.shape[2] - tail_tokens, :].contiguous()
        raw_tail = tensor[:, :, tensor.shape[2] - tail_tokens :, :].contiguous()
        compressed_chunks = []
        for chunk_start in range(0, prefix.shape[2], flush_tokens):
            chunk = prefix[:, :, chunk_start : chunk_start + flush_tokens, :].contiguous()
            compressed_chunks.append(quantize_tensor(chunk, bits=bits, scheme=scheme))
        return SegmentedTensor(compressed_chunks=compressed_chunks, raw_tail=raw_tail)

    def decompress_state(self, state: KVCacheState) -> KVCacheState:
        if not state.is_compressed():
            return state
        decompressed_layers = []
        for layer in state.layers:
            key = self._decompress_tensor(layer.key)
            value = self._decompress_tensor(layer.value)
            decompressed_layers.append(
                LayerKVCache(key=key.contiguous(), value=value.contiguous(), compression_method="raw")
            )
        return replace(state, layers=decompressed_layers)

    def _cache_insert(
        self,
        cache: OrderedDict,
        key,
        value,
        limit: int,
    ):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max(0, limit):
            cache.popitem(last=False)
        return value

    def _cache_key(
        self,
        tensor_like: torch.Tensor | QuantizedTensor | LowRankTensor | SegmentedTensor,
    ) -> tuple[object, ...]:
        if isinstance(tensor_like, torch.Tensor):
            return ("raw", tensor_like.data_ptr(), tuple(int(dim) for dim in tensor_like.shape), str(tensor_like.dtype))
        if isinstance(tensor_like, QuantizedTensor):
            return ("quantized", id(tensor_like), tensor_like.bits, tensor_like.scheme)
        if isinstance(tensor_like, LowRankTensor):
            return ("low_rank", id(tensor_like.u), id(tensor_like.s), id(tensor_like.vh))
        if isinstance(tensor_like, SegmentedTensor):
            return (
                "segmented",
                tuple(self._cache_key(chunk) for chunk in tensor_like.compressed_chunks),
                tensor_like.raw_tail.data_ptr(),
            )
        raise TypeError(f"Unsupported tensor type: {type(tensor_like)!r}")

    def _materialize_cached_tensor(
        self,
        tensor_like: QuantizedTensor | LowRankTensor,
    ) -> torch.Tensor:
        if not self.config.materialization_cache_enabled:
            self._materialization_stats["chunk_cache_misses"] += 1
            if isinstance(tensor_like, QuantizedTensor):
                return dequantize_tensor(tensor_like).contiguous()
            return decompress_low_rank_tensor(tensor_like).contiguous()

        cache_key = self._cache_key(tensor_like)
        if cache_key in self._chunk_materialization_cache:
            cached_tensor_like, cached_materialized = self._chunk_materialization_cache[cache_key]
            if cached_tensor_like is tensor_like:
                self._materialization_stats["chunk_cache_hits"] += 1
                self._chunk_materialization_cache.move_to_end(cache_key)
                return cached_materialized
            self._chunk_materialization_cache.pop(cache_key, None)
        self._materialization_stats["chunk_cache_misses"] += 1
        if isinstance(tensor_like, QuantizedTensor):
            materialized = dequantize_tensor(tensor_like).contiguous()
        else:
            materialized = decompress_low_rank_tensor(tensor_like).contiguous()
        return self._cache_insert(
            self._chunk_materialization_cache,
            cache_key,
            (tensor_like, materialized),
            self.config.materialization_chunk_cache_size,
        )[1]

    @staticmethod
    def _same_chunk_sequence(
        cached_chunks: tuple[TensorLike, ...],
        current_chunks: list[TensorLike],
    ) -> bool:
        if len(cached_chunks) != len(current_chunks):
            return False
        return all(cached is current for cached, current in zip(cached_chunks, current_chunks))

    def _prefix_from_cache(
        self,
        prefix_key: tuple[tuple[object, ...], ...],
        current_chunks: list[TensorLike],
    ) -> torch.Tensor | None:
        if prefix_key not in self._prefix_materialization_cache:
            return None
        cached_chunks, cached_prefix = self._prefix_materialization_cache[prefix_key]
        if not self._same_chunk_sequence(cached_chunks, current_chunks):
            self._prefix_materialization_cache.pop(prefix_key, None)
            return None
        self._materialization_stats["prefix_cache_hits"] += 1
        self._prefix_materialization_cache.move_to_end(prefix_key)
        return cached_prefix

    def _cache_prefix(
        self,
        prefix_key: tuple[tuple[object, ...], ...],
        current_chunks: list[TensorLike],
        prefix_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self._cache_insert(
            self._prefix_materialization_cache,
            prefix_key,
            (tuple(current_chunks), prefix_tensor),
            self.config.materialization_prefix_cache_size,
        )[1]

    def _materialize_segmented_tensor(self, tensor_like: SegmentedTensor) -> torch.Tensor:
        if not tensor_like.compressed_chunks:
            return tensor_like.raw_tail.contiguous()

        prefix_key = tuple(self._cache_key(chunk) for chunk in tensor_like.compressed_chunks)
        prefix_tensor = None
        if self.config.materialization_cache_enabled:
            prefix_tensor = self._prefix_from_cache(prefix_key, tensor_like.compressed_chunks)
        if prefix_tensor is None:
            self._materialization_stats["prefix_cache_misses"] += 1
            if (
                self.config.materialization_cache_enabled
                and len(prefix_key) > 1
                and prefix_key[:-1] in self._prefix_materialization_cache
            ):
                cached_prefix = self._prefix_from_cache(
                    prefix_key[:-1],
                    tensor_like.compressed_chunks[:-1],
                )
                if cached_prefix is not None:
                    tail_chunk = self._decompress_tensor(tensor_like.compressed_chunks[-1])
                    prefix_tensor = torch.cat([cached_prefix, tail_chunk], dim=tensor_like.axis).contiguous()
                    self._materialization_stats["prefix_cache_extensions"] += 1
            if prefix_tensor is None:
                prefix_tensor = torch.cat(
                    [self._decompress_tensor(chunk) for chunk in tensor_like.compressed_chunks],
                    dim=tensor_like.axis,
                ).contiguous()
            if self.config.materialization_cache_enabled:
                prefix_tensor = self._cache_prefix(
                    prefix_key,
                    tensor_like.compressed_chunks,
                    prefix_tensor,
                )

        if tensor_like.raw_tail.shape[tensor_like.axis] == 0:
            return prefix_tensor
        return torch.cat([prefix_tensor, tensor_like.raw_tail], dim=tensor_like.axis).contiguous()

    def _decompress_tensor(
        self,
        tensor_like: torch.Tensor | QuantizedTensor | LowRankTensor | SegmentedTensor,
    ) -> torch.Tensor:
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like
        if isinstance(tensor_like, QuantizedTensor):
            return self._materialize_cached_tensor(tensor_like)
        if isinstance(tensor_like, LowRankTensor):
            return self._materialize_cached_tensor(tensor_like)
        if isinstance(tensor_like, SegmentedTensor):
            return self._materialize_segmented_tensor(tensor_like)
        raise TypeError(f"Unsupported tensor type: {type(tensor_like)!r}")
