from __future__ import annotations

from copy import deepcopy

from kv_cache_engine.config import KVCacheXConfig
from kv_cache_engine.utils import write_json

from .runner import BenchmarkRunner


class IterativeOptimizer:
    def __init__(self, config: KVCacheXConfig):
        self.base_config = deepcopy(config)

    @staticmethod
    def _score(metrics_df) -> dict:
        optimized = metrics_df[metrics_df["mode"] == "kvcachex"]
        if optimized.empty:
            return {
                "memory_reduction_pct": 0.0,
                "latency_improvement_pct": 0.0,
                "token_agreement": 0.0,
            }
        return {
            "memory_reduction_pct": float(-optimized["memory_delta_pct_vs_standard"].mean()),
            "latency_improvement_pct": float(-optimized["latency_delta_pct_vs_standard"].mean()),
            "token_agreement": float(optimized["token_agreement_vs_standard"].mean()),
        }

    def _propose(self, config: KVCacheXConfig, score: dict) -> KVCacheXConfig:
        candidate = deepcopy(config)
        if score["memory_reduction_pct"] < 30.0:
            candidate.compression.asymmetric_kv_quantization = True
            candidate.compression.quantization_bits = 4
            candidate.eviction.target_keep_ratio = max(0.45, candidate.eviction.target_keep_ratio - 0.1)
            candidate.scheduler.max_cache_tokens = max(
                candidate.scheduler.min_cache_tokens,
                int(candidate.scheduler.max_cache_tokens * 0.8),
            )
        if score["latency_improvement_pct"] < 20.0:
            candidate.scheduler.base_cache_tokens = max(
                candidate.scheduler.min_cache_tokens,
                int(candidate.scheduler.base_cache_tokens * 0.8),
            )
            candidate.eviction.recent_tokens_to_keep = max(
                32, int(candidate.eviction.recent_tokens_to_keep * 0.75)
            )
        if score["token_agreement"] < 0.85:
            candidate.compression.asymmetric_kv_quantization = False
            candidate.compression.quantization_bits = 8
            candidate.eviction.target_keep_ratio = min(
                0.85, candidate.eviction.target_keep_ratio + 0.1
            )
            candidate.eviction.recent_tokens_to_keep += 32
        return candidate

    def run(
        self,
        iterations: int = 3,
        model_name: str | None = None,
        log_path: str = "results/optimizer_history.json",
    ) -> tuple[KVCacheXConfig, list[dict]]:
        current_config = deepcopy(self.base_config)
        history: list[dict] = []
        best_config = deepcopy(current_config)
        best_score = {"memory_reduction_pct": -1.0, "latency_improvement_pct": -1.0, "token_agreement": 0.0}

        for iteration in range(iterations):
            runner = BenchmarkRunner(current_config)
            metrics_df, _ = runner.run(model_name=model_name)
            score = self._score(metrics_df)
            history.append(
                {
                    "iteration": iteration,
                    "score": score,
                    "config": current_config.to_dict(),
                }
            )
            composite = score["memory_reduction_pct"] + score["latency_improvement_pct"]
            best_composite = best_score["memory_reduction_pct"] + best_score["latency_improvement_pct"]
            if composite > best_composite and score["token_agreement"] >= best_score["token_agreement"]:
                best_config = deepcopy(current_config)
                best_score = score
            current_config = self._propose(current_config, score)

        write_json(log_path, history)
        return best_config, history
