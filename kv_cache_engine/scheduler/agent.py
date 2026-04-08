from __future__ import annotations

from kv_cache_engine.config import EdgeConfig, SchedulerConfig
from kv_cache_engine.types import KVCacheState, SchedulePlan


class SchedulerAgent:
    def __init__(self, config: SchedulerConfig, edge_config: EdgeConfig):
        self.config = config
        self.edge_config = edge_config

    def plan(
        self,
        state: KVCacheState,
        current_memory_bytes: int,
        last_latency_ms: float,
        step_index: int,
    ) -> SchedulePlan:
        memory_pressure = (
            current_memory_bytes / self.edge_config.max_memory_bytes
            if self.edge_config.simulate_edge
            else 0.0
        )
        latency_pressure = (
            last_latency_ms / self.config.pressure_latency_ms
            if self.config.pressure_latency_ms > 0
            else 0.0
        )

        target = min(max(state.seq_len, self.config.min_cache_tokens), self.config.max_cache_tokens)
        if self.config.adaptive_window:
            if (
                memory_pressure >= self.config.pressure_memory_utilization
                or latency_pressure >= 1.0
            ):
                target = max(
                    self.config.min_cache_tokens,
                    min(
                        self.config.base_cache_tokens,
                        int(max(state.seq_len, self.config.base_cache_tokens) * self.config.shrink_factor),
                    ),
                )
            elif memory_pressure <= self.config.pressure_memory_utilization * 0.5 and latency_pressure <= 0.7:
                target = min(self.config.max_cache_tokens, int(target * self.config.grow_factor))
            else:
                target = min(self.config.max_cache_tokens, max(self.config.base_cache_tokens, target))

        collect_attentions = (
            step_index % self.config.attention_probe_interval == 0
            or memory_pressure >= self.config.pressure_memory_utilization
        )
        apply_compression = (
            step_index % self.config.compression_interval == 0
            or memory_pressure >= self.config.pressure_memory_utilization
        )
        compression_method = "quantization_int4" if memory_pressure > 0.9 else "quantization"

        return SchedulePlan(
            target_cache_tokens=target,
            collect_attentions=collect_attentions,
            apply_compression=apply_compression,
            compression_method=compression_method,
            memory_pressure=memory_pressure,
            latency_pressure=latency_pressure,
        )
