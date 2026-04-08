from __future__ import annotations

import unittest

from kv_cache_engine.compression import CompressionAgent
from kv_cache_engine.config import CompressionConfig, EdgeConfig, EvictionConfig, MonitorConfig, SchedulerConfig
from kv_cache_engine.eviction import EvictionAgent
from kv_cache_engine.monitor import MonitorAgent
from kv_cache_engine.scheduler import SchedulerAgent

from tests.helpers import make_state


class PipelineTests(unittest.TestCase):
    def test_agent_pipeline_runs_end_to_end(self) -> None:
        state = make_state(seq_len=96)
        scheduler = SchedulerAgent(SchedulerConfig(base_cache_tokens=64, max_cache_tokens=64), EdgeConfig())
        plan = scheduler.plan(state, current_memory_bytes=10_000_000, last_latency_ms=50.0, step_index=2)
        evicted = EvictionAgent(EvictionConfig()).prune_state(state, plan.target_cache_tokens)
        compressed = CompressionAgent(CompressionConfig()).compress_state(
            evicted, method_override=plan.compression_method
        )
        restored = CompressionAgent(CompressionConfig()).decompress_state(compressed)
        monitor = MonitorAgent(MonitorConfig(), EdgeConfig())
        monitor.record_step(step=0, latency_ms=12.0, cache_state=compressed)
        summary = monitor.finalize(
            workload_name="pipeline",
            mode="kvcachex",
            run_kind="greedy",
            prompt_tokens=96,
            generated_tokens=1,
            output_text="done",
        )
        self.assertLessEqual(restored.seq_len, state.seq_len)
        self.assertGreater(summary.compression_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
