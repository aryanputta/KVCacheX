from __future__ import annotations

import unittest

import torch

from kv_cache_engine.compression import CompressionAgent, cluster_state
from kv_cache_engine.compression.low_rank import compress_tensor_low_rank, decompress_low_rank_tensor
from kv_cache_engine.compression.quantization import dequantize_tensor, quantize_tensor
from kv_cache_engine.config import CompressionConfig
from kv_cache_engine.types import LayerKVCache, SegmentedTensor

from tests.helpers import make_state


class CompressionTests(unittest.TestCase):
    def test_int8_quantization_round_trip(self) -> None:
        tensor = torch.randn(1, 2, 16, 8)
        quantized = quantize_tensor(tensor, bits=8)
        restored = dequantize_tensor(quantized)
        self.assertEqual(restored.shape, tensor.shape)
        self.assertLess(float(torch.mean(torch.abs(restored - tensor))), 0.05)

    def test_asymmetric_quantization_round_trip_uses_expected_shapes(self) -> None:
        tensor = torch.randn(1, 2, 12, 8)
        key_quantized = quantize_tensor(tensor, bits=4, scheme="affine_per_channel")
        value_quantized = quantize_tensor(tensor, bits=4, scheme="affine_per_token")
        restored_key = dequantize_tensor(key_quantized)
        restored_value = dequantize_tensor(value_quantized)

        self.assertEqual(key_quantized.scale.shape, (1, 2, 1, 8))
        self.assertEqual(value_quantized.scale.shape, (1, 2, 12, 1))
        self.assertIsNotNone(key_quantized.offset)
        self.assertIsNotNone(value_quantized.offset)
        self.assertLess(float(torch.mean(torch.abs(restored_key - tensor))), 0.2)
        self.assertLess(float(torch.mean(torch.abs(restored_value - tensor))), 0.2)

    def test_low_rank_round_trip_shape(self) -> None:
        tensor = torch.randn(1, 2, 12, 6)
        compressed = compress_tensor_low_rank(tensor, rank_ratio=0.5)
        restored = decompress_low_rank_tensor(compressed)
        self.assertEqual(restored.shape, tensor.shape)

    def test_clustering_reduces_prefix_tokens(self) -> None:
        state = make_state(seq_len=48)
        clustered = cluster_state(state, cluster_ratio=0.5, prefix_fraction=0.5)
        self.assertLess(clustered.seq_len, state.seq_len)

    def test_agent_compress_decompress(self) -> None:
        state = make_state(seq_len=32)
        agent = CompressionAgent(CompressionConfig())
        compressed = agent.compress_state(state, method_override="quantization")
        restored = agent.decompress_state(compressed)
        self.assertTrue(compressed.is_compressed())
        self.assertEqual(restored.seq_len, state.seq_len)

    def test_agent_uses_asymmetric_kv_schemes(self) -> None:
        state = make_state(seq_len=32)
        agent = CompressionAgent(
            CompressionConfig(
                method="quantization",
                quantization_bits=4,
                asymmetric_kv_quantization=True,
                segmented_prefix_enabled=False,
            )
        )
        compressed = agent.compress_state(state, method_override="quantization")
        layer = compressed.layers[0]
        self.assertEqual(layer.key.scheme, "affine_per_channel")
        self.assertEqual(layer.value.scheme, "affine_per_token")
        restored = agent.decompress_state(compressed)
        self.assertLess(
            float(torch.mean(torch.abs(restored.layers[0].key - state.layers[0].key))),
            0.2,
        )
        self.assertLess(
            float(torch.mean(torch.abs(restored.layers[0].value - state.layers[0].value))),
            0.2,
        )

    def test_segmented_prefix_reuses_compressed_prefix(self) -> None:
        state = make_state(seq_len=24)
        config = CompressionConfig(
            method="quantization",
            quantization_bits=8,
            segmented_prefix_enabled=True,
            segment_tail_tokens=4,
            segment_flush_tokens=2,
        )
        agent = CompressionAgent(config)
        compressed = agent.compress_state(state, method_override="quantization")
        layer = compressed.layers[0]
        self.assertIsInstance(layer.key, SegmentedTensor)
        first_chunk_count = len(layer.key.compressed_chunks)
        self.assertEqual(layer.key.raw_tail.shape[2], 4)

        next_key = torch.cat([state.layers[0].key, torch.randn(1, 2, 1, 8)], dim=2)
        next_value = torch.cat([state.layers[0].value, torch.randn(1, 2, 1, 8)], dim=2)
        extended = make_state(seq_len=25)
        extended.layers = [LayerKVCache(key=next_key, value=next_value, compression_method="raw")]
        extended.positions = torch.arange(25, dtype=torch.long)
        extended.token_ids = torch.arange(25, dtype=torch.long)
        extended.cumulative_attention = torch.linspace(0.1, 1.0, steps=25)
        extended.recent_attention = torch.linspace(0.2, 1.0, steps=25)
        extended.embedding_norms = torch.linspace(0.3, 1.3, steps=25)

        recompressed = agent.compress_state(
            extended,
            method_override="quantization",
            previous_state=compressed,
        )
        recompressed_layer = recompressed.layers[0]
        self.assertIsInstance(recompressed_layer.key, SegmentedTensor)
        self.assertGreaterEqual(len(recompressed_layer.key.compressed_chunks), first_chunk_count)
        self.assertEqual(recompressed_layer.key.raw_tail.shape[2], 4)
        restored = agent.decompress_state(recompressed)
        self.assertEqual(restored.seq_len, 25)

    def test_segmented_materialization_hits_prefix_cache_on_repeat(self) -> None:
        state = make_state(seq_len=24)
        config = CompressionConfig(
            method="quantization",
            quantization_bits=8,
            segmented_prefix_enabled=True,
            segment_tail_tokens=4,
            segment_flush_tokens=2,
            materialization_cache_enabled=True,
            materialization_chunk_cache_size=16,
            materialization_prefix_cache_size=4,
        )
        agent = CompressionAgent(config)
        compressed = agent.compress_state(state, method_override="quantization")

        agent.reset_runtime_cache()
        first_restored = agent.decompress_state(compressed)
        first_stats = agent.materialization_stats()
        second_restored = agent.decompress_state(compressed)
        second_stats = agent.materialization_stats()

        self.assertEqual(first_restored.seq_len, second_restored.seq_len)
        self.assertGreater(first_stats["prefix_cache_misses"], 0)
        self.assertGreater(second_stats["prefix_cache_hits"], first_stats["prefix_cache_hits"])

    def test_segmented_materialization_extends_cached_prefix(self) -> None:
        state = make_state(seq_len=10)
        config = CompressionConfig(
            method="quantization",
            quantization_bits=8,
            segmented_prefix_enabled=True,
            segment_tail_tokens=4,
            segment_flush_tokens=2,
            materialization_cache_enabled=True,
            materialization_chunk_cache_size=16,
            materialization_prefix_cache_size=4,
        )
        agent = CompressionAgent(config)
        compressed = agent.compress_state(state, method_override="quantization")
        agent.reset_runtime_cache()
        _ = agent.decompress_state(compressed)

        next_key = torch.cat([state.layers[0].key, torch.randn(1, 2, 1, 8)], dim=2)
        next_value = torch.cat([state.layers[0].value, torch.randn(1, 2, 1, 8)], dim=2)
        extended = make_state(seq_len=11)
        extended.layers = [LayerKVCache(key=next_key, value=next_value, compression_method="raw")]
        extended.positions = torch.arange(11, dtype=torch.long)
        extended.token_ids = torch.arange(11, dtype=torch.long)
        extended.cumulative_attention = torch.linspace(0.1, 1.0, steps=11)
        extended.recent_attention = torch.linspace(0.2, 1.0, steps=11)
        extended.embedding_norms = torch.linspace(0.3, 1.3, steps=11)
        recompressed = agent.compress_state(
            extended,
            method_override="quantization",
            previous_state=compressed,
        )

        restored = agent.decompress_state(recompressed)
        stats = agent.materialization_stats()

        self.assertEqual(restored.seq_len, 11)
        self.assertGreater(stats["prefix_cache_extensions"], 0)


if __name__ == "__main__":
    unittest.main()
