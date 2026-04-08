from __future__ import annotations

import torch

from kv_cache_engine.types import KVCacheState, LayerKVCache


def make_state(seq_len: int = 32, heads: int = 2, hidden: int = 8) -> KVCacheState:
    key = torch.randn(1, heads, seq_len, hidden)
    value = torch.randn(1, heads, seq_len, hidden)
    layers = [LayerKVCache(key=key, value=value, compression_method="raw")]
    positions = torch.arange(seq_len, dtype=torch.long)
    token_ids = torch.arange(seq_len, dtype=torch.long)
    cumulative = torch.linspace(0.1, 1.0, steps=seq_len)
    recent = torch.linspace(0.2, 1.0, steps=seq_len)
    embedding_norms = torch.linspace(0.3, 1.3, steps=seq_len)
    return KVCacheState(
        layers=layers,
        positions=positions,
        token_ids=token_ids,
        cumulative_attention=cumulative,
        recent_attention=recent,
        embedding_norms=embedding_norms,
    )
