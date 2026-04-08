from __future__ import annotations

import unittest

from kv_cache_engine.compression import CompressionAgent
from kv_cache_engine.config import CompressionConfig, EvictionConfig
from kv_cache_engine.eviction import EvictionAgent

from tests.helpers import make_state


class StressTests(unittest.TestCase):
    def test_large_context_cache_reduction(self) -> None:
        state = make_state(seq_len=2048, heads=4, hidden=16)
        evicted = EvictionAgent(EvictionConfig(target_keep_ratio=0.25, min_tokens_to_keep=256)).prune_state(
            state, target_cache_tokens=512
        )
        compressed = CompressionAgent(CompressionConfig()).compress_state(
            evicted, method_override="hybrid"
        )
        self.assertLessEqual(evicted.seq_len, 512)
        self.assertGreater(compressed.compression_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
