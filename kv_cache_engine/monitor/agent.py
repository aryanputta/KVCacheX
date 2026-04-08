from __future__ import annotations

import math
from dataclasses import asdict

import torch

from kv_cache_engine.config import EdgeConfig, MonitorConfig
from kv_cache_engine.types import KVCacheState, RunSummary, StepMetrics
from kv_cache_engine.utils import percentile, process_memory_bytes, safe_divide


class MonitorAgent:
    def __init__(self, config: MonitorConfig, edge_config: EdgeConfig):
        self.config = config
        self.edge_config = edge_config
        self.step_metrics: list[StepMetrics] = []
        self.peak_process_memory_bytes = 0
        self.peak_cache_bytes = 0

    def reset(self) -> None:
        self.step_metrics = []
        self.peak_process_memory_bytes = 0
        self.peak_cache_bytes = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def current_memory_bytes(self) -> int:
        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated())
        return process_memory_bytes()

    def record_step(
        self,
        step: int,
        latency_ms: float,
        cache_state: KVCacheState | None,
        nll: float | None = None,
        token_agreement: float | None = None,
        kernel_time_ms: float | None = None,
        bandwidth_bytes: float | None = None,
        metadata: dict | None = None,
    ) -> StepMetrics:
        process_memory = self.current_memory_bytes()
        cache_bytes = cache_state.estimated_bytes() if cache_state is not None else 0
        compression_ratio = cache_state.compression_ratio if cache_state is not None else 1.0
        if self.edge_config.simulate_edge:
            process_memory = max(process_memory, cache_bytes)
        metric = StepMetrics(
            step=step,
            latency_ms=latency_ms,
            process_memory_bytes=process_memory,
            cache_bytes=cache_bytes,
            compression_ratio=compression_ratio,
            nll=nll,
            token_agreement=token_agreement,
            kernel_time_ms=kernel_time_ms,
            bandwidth_bytes=bandwidth_bytes,
            metadata=metadata or {},
        )
        self.step_metrics.append(metric)
        self.peak_process_memory_bytes = max(self.peak_process_memory_bytes, process_memory)
        self.peak_cache_bytes = max(self.peak_cache_bytes, cache_bytes)
        return metric

    def snapshot(self) -> dict[str, float]:
        last_latency_ms = self.step_metrics[-1].latency_ms if self.step_metrics else 0.0
        return {
            "current_memory_bytes": float(self.current_memory_bytes()),
            "last_latency_ms": last_latency_ms,
            "peak_process_memory_bytes": float(self.peak_process_memory_bytes),
            "peak_cache_bytes": float(self.peak_cache_bytes),
        }

    def finalize(
        self,
        workload_name: str,
        mode: str,
        run_kind: str,
        prompt_tokens: int,
        generated_tokens: int,
        output_text: str,
    ) -> RunSummary:
        latencies = [metric.latency_ms for metric in self.step_metrics]
        nlls = [metric.nll for metric in self.step_metrics if metric.nll is not None]
        agreements = [
            metric.token_agreement for metric in self.step_metrics if metric.token_agreement is not None
        ]
        mean_nll = float(sum(nlls) / len(nlls)) if nlls else None
        perplexity = math.exp(mean_nll) if mean_nll is not None else None
        token_agreement = float(sum(agreements) / len(agreements)) if agreements else None
        mean_latency = float(sum(latencies) / len(latencies)) if latencies else 0.0
        summary = RunSummary(
            workload_name=workload_name,
            mode=mode,
            run_kind=run_kind,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            output_text=output_text,
            mean_latency_ms=mean_latency,
            p95_latency_ms=percentile(latencies, 95),
            tokens_per_sec=safe_divide(generated_tokens, sum(latencies) / 1000.0),
            peak_process_memory_bytes=int(self.peak_process_memory_bytes),
            peak_cache_bytes=int(self.peak_cache_bytes),
            mean_nll=mean_nll,
            perplexity=perplexity,
            token_agreement=token_agreement,
            compression_ratio=float(
                sum(metric.compression_ratio for metric in self.step_metrics) / len(self.step_metrics)
            )
            if self.step_metrics
            else 1.0,
            extra_metrics={
                "step_metrics": [asdict(metric) for metric in self.step_metrics],
            },
        )
        return summary
