from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class RuntimeConfig:
    seed: int = 7
    device: str = "auto"
    dtype: str = "auto"
    torch_compile: bool = False
    profile_cuda: bool = True


@dataclass
class ModelConfig:
    name: str = "distilgpt2"
    revision: str = "main"
    trust_remote_code: bool = False
    max_new_tokens: int = 48
    use_attention_outputs: bool = True
    attention_implementation: str = "eager"
    smoke_model_name: str = "distilgpt2"
    full_benchmark_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class WorkloadConfig:
    long_context_targets: list[int] = field(default_factory=lambda: [2048, 4096, 8192])
    synthetic_prompt_tokens: int = 1536
    streaming_chunk_tokens: int = 128
    streaming_steps: int = 8
    multi_turn_turns: int = 6
    benchmark_prompts_path: str = "benchmark/data/real_prompts.json"


@dataclass
class BenchmarkConfig:
    modes: list[str] = field(
        default_factory=lambda: ["no_cache", "standard_cache", "kvcachex"]
    )
    repetitions: int = 1
    warmup_runs: int = 1
    teacher_forced_accuracy: bool = True
    greedy_similarity: bool = True


@dataclass
class CompressionConfig:
    enabled: bool = True
    method: str = "quantization"
    quantization_bits: int = 8
    asymmetric_kv_quantization: bool = False
    key_quantization_scheme: str = "affine_per_channel"
    value_quantization_scheme: str = "affine_per_token"
    materialization_cache_enabled: bool = False
    materialization_chunk_cache_size: int = 128
    materialization_prefix_cache_size: int = 16
    segmented_prefix_enabled: bool = True
    segment_tail_tokens: int = 96
    segment_flush_tokens: int = 48
    low_rank_ratio: float = 0.5
    cluster_ratio: float = 0.5
    cluster_prefix_fraction: float = 0.7
    min_seq_for_low_rank: int = 512
    min_seq_for_clustering: int = 512


@dataclass
class EvictionConfig:
    enabled: bool = True
    target_keep_ratio: float = 0.7
    attention_threshold: float = 0.002
    recent_tokens_to_keep: int = 128
    pin_first_tokens: int = 16
    min_tokens_to_keep: int = 128
    prefill_observation_window: int = 64
    prefill_head_aware_eviction: bool = True
    dynamic_decode_pruning: bool = True
    decode_prune_margin: int = 64
    head_score_weight: float = 0.65
    head_support_weight: float = 0.35
    semantic_model_path: str = "models/token_importance_model.pkl"
    train_if_missing: bool = True
    attention_weight: float = 0.45
    recency_weight: float = 0.2
    semantic_weight: float = 0.35


@dataclass
class SchedulerConfig:
    adaptive_window: bool = True
    base_cache_tokens: int = 384
    max_cache_tokens: int = 384
    min_cache_tokens: int = 128
    pressure_latency_ms: float = 40.0
    pressure_memory_utilization: float = 0.8
    attention_probe_interval: int = 12
    compression_interval: int = 1
    grow_factor: float = 1.2
    shrink_factor: float = 0.8


@dataclass
class MonitorConfig:
    attention_decay: float = 0.92
    accuracy_warning_nll: float = 2.5
    token_agreement_warning: float = 0.8


@dataclass
class EdgeConfig:
    simulate_edge: bool = True
    max_memory_bytes: int = 4 * 1024 * 1024 * 1024


@dataclass
class OutputConfig:
    metrics_csv: str = "results/metrics.csv"
    experiment_logs: str = "results/experiment_logs.json"
    bottleneck_report: str = "analysis/bottleneck_report.md"
    failure_report: str = "analysis/failure_cases.md"
    dashboard_html: str = "dashboard/kvcachex_dashboard.html"


@dataclass
class KVCacheXConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    workloads: WorkloadConfig = field(default_factory=WorkloadConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    eviction: EvictionConfig = field(default_factory=EvictionConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(instance: T, values: dict[str, Any]) -> T:
    for field_info in fields(instance):
        if field_info.name not in values:
            continue
        current_value = getattr(instance, field_info.name)
        new_value = values[field_info.name]
        if is_dataclass(current_value) and isinstance(new_value, dict):
            _update_dataclass(current_value, new_value)
        else:
            setattr(instance, field_info.name, new_value)
    return instance


def load_config(path: str | Path) -> KVCacheXConfig:
    config = KVCacheXConfig()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _update_dataclass(config, raw)
