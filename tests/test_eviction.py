from __future__ import annotations

import unittest

import torch

from kv_cache_engine.config import EvictionConfig
from kv_cache_engine.eviction import EvictionAgent

from tests.helpers import make_state


class EvictionTests(unittest.TestCase):
    def test_eviction_keeps_recent_tokens(self) -> None:
        state = make_state(seq_len=40)
        config = EvictionConfig(
            target_keep_ratio=0.5,
            recent_tokens_to_keep=8,
            pin_first_tokens=4,
            min_tokens_to_keep=16,
        )
        agent = EvictionAgent(config)
        pruned = agent.prune_state(state, target_cache_tokens=20)
        self.assertLessEqual(pruned.seq_len, 20)
        expected_recent = torch.arange(32, 40, dtype=torch.long)
        self.assertTrue(set(expected_recent.tolist()).issubset(set(pruned.positions.tolist())))

    def test_prefill_head_aware_eviction_keeps_head_salient_tokens(self) -> None:
        state = make_state(seq_len=48)
        config = EvictionConfig(
            target_keep_ratio=0.33,
            recent_tokens_to_keep=4,
            pin_first_tokens=2,
            min_tokens_to_keep=8,
            prefill_observation_window=8,
        )
        agent = EvictionAgent(config)
        attentions = torch.zeros(1, 2, 48, 48, dtype=torch.float32)
        attentions[0, 0, -8:, 10] = 1.0
        attentions[0, 1, -8:, 20] = 1.0
        pruned = agent.prune_prefill_state(
            state,
            attentions=(attentions,),
            target_cache_tokens=16,
        )
        kept_positions = set(pruned.positions.tolist())
        self.assertLessEqual(pruned.seq_len, 16)
        self.assertIn(10, kept_positions)
        self.assertIn(20, kept_positions)


if __name__ == "__main__":
    unittest.main()
