from __future__ import annotations

import unittest

from kv_cache_engine.config import EdgeConfig, MonitorConfig, SchedulerConfig
from kv_cache_engine.monitor import MonitorAgent
from kv_cache_engine.scheduler import SchedulerAgent

from tests.helpers import make_state


class SchedulerMonitorTests(unittest.TestCase):
    def test_scheduler_shrinks_under_pressure(self) -> None:
        state = make_state(seq_len=512)
        scheduler = SchedulerAgent(
            SchedulerConfig(base_cache_tokens=256, max_cache_tokens=384, min_cache_tokens=128),
            EdgeConfig(simulate_edge=True, max_memory_bytes=1024),
        )
        plan = scheduler.plan(state, current_memory_bytes=2048, last_latency_ms=80.0, step_index=1)
        self.assertLessEqual(plan.target_cache_tokens, 256)

    def test_monitor_collects_summary(self) -> None:
        state = make_state(seq_len=32)
        monitor = MonitorAgent(MonitorConfig(), EdgeConfig())
        monitor.record_step(step=0, latency_ms=10.0, cache_state=state, nll=1.0, token_agreement=1.0)
        summary = monitor.finalize(
            workload_name="unit",
            mode="kvcachex",
            run_kind="teacher_forced",
            prompt_tokens=32,
            generated_tokens=1,
            output_text="ok",
        )
        self.assertGreater(summary.peak_cache_bytes, 0)
        self.assertAlmostEqual(summary.mean_nll, 1.0)


if __name__ == "__main__":
    unittest.main()
