from __future__ import annotations

import unittest

from benchmark.iterative_loop import IterativeOptimizer
from kv_cache_engine.config import KVCacheXConfig


class OptimizerTests(unittest.TestCase):
    def test_optimizer_enables_asymmetric_quantization_under_memory_pressure(self) -> None:
        optimizer = IterativeOptimizer(KVCacheXConfig())
        candidate = optimizer._propose(
            KVCacheXConfig(),
            {
                "memory_reduction_pct": 20.0,
                "latency_improvement_pct": 25.0,
                "token_agreement": 0.95,
            },
        )
        self.assertTrue(candidate.compression.asymmetric_kv_quantization)
        self.assertEqual(candidate.compression.quantization_bits, 4)

    def test_optimizer_reverts_to_safer_quantization_on_accuracy_drop(self) -> None:
        config = KVCacheXConfig()
        config.compression.asymmetric_kv_quantization = True
        config.compression.quantization_bits = 4
        optimizer = IterativeOptimizer(config)
        candidate = optimizer._propose(
            config,
            {
                "memory_reduction_pct": 40.0,
                "latency_improvement_pct": 10.0,
                "token_agreement": 0.8,
            },
        )
        self.assertFalse(candidate.compression.asymmetric_kv_quantization)
        self.assertEqual(candidate.compression.quantization_bits, 8)


if __name__ == "__main__":
    unittest.main()
