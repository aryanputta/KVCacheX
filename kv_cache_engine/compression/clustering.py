from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from kv_cache_engine.types import KVCacheState, LayerKVCache


def _cluster_features(state: KVCacheState, prefix_count: int) -> np.ndarray:
    first_layer = state.layers[0]
    key = first_layer.key
    value = first_layer.value
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError("Token clustering expects raw tensors.")
    key_features = key[:, :, :prefix_count, :].mean(dim=(0, 1)).detach().cpu().numpy()
    value_features = value[:, :, :prefix_count, :].mean(dim=(0, 1)).detach().cpu().numpy()
    return np.concatenate([key_features, value_features], axis=-1)


def cluster_state(
    state: KVCacheState,
    cluster_ratio: float,
    prefix_fraction: float,
) -> KVCacheState:
    if state.seq_len < 4:
        return state
    prefix_count = max(2, int(state.seq_len * prefix_fraction))
    prefix_count = min(prefix_count, state.seq_len)
    cluster_count = max(1, int(prefix_count * cluster_ratio))
    if cluster_count >= prefix_count:
        return state

    features = _cluster_features(state, prefix_count)
    clustering = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=0,
        batch_size=min(256, prefix_count),
        n_init="auto",
    )
    labels = clustering.fit_predict(features)
    label_tensor = torch.as_tensor(labels, device=state.positions.device, dtype=torch.long)
    active_labels = sorted(torch.unique(label_tensor).tolist())

    representative_indices = []
    for label in active_labels:
        member_indices = torch.nonzero(label_tensor == label, as_tuple=False).flatten()
        representative_indices.append(int(member_indices[0].item()))

    clustered_layers: list[LayerKVCache] = []
    for layer in state.layers:
        if not isinstance(layer.key, torch.Tensor) or not isinstance(layer.value, torch.Tensor):
            raise TypeError("Token clustering expects raw tensor layers.")
        key_clusters = []
        value_clusters = []
        for label in active_labels:
            member_indices = torch.nonzero(label_tensor == label, as_tuple=False).flatten()
            key_clusters.append(layer.key[:, :, member_indices, :].mean(dim=2))
            value_clusters.append(layer.value[:, :, member_indices, :].mean(dim=2))
        clustered_key = torch.stack(key_clusters, dim=2)
        clustered_value = torch.stack(value_clusters, dim=2)
        if prefix_count < state.seq_len:
            clustered_key = torch.cat([clustered_key, layer.key[:, :, prefix_count:, :]], dim=2)
            clustered_value = torch.cat(
                [clustered_value, layer.value[:, :, prefix_count:, :]], dim=2
            )
        clustered_layers.append(
            LayerKVCache(
                key=clustered_key.contiguous(),
                value=clustered_value.contiguous(),
                compression_method="clustered",
            )
        )

    cluster_meta = []
    for label, rep_idx in zip(active_labels, representative_indices):
        member_indices = torch.nonzero(label_tensor == label, as_tuple=False).flatten()
        cluster_meta.append(
            {
                "label": label,
                "members": member_indices.tolist(),
                "representative_index": rep_idx,
            }
        )

    prefix_positions = []
    prefix_token_ids = []
    prefix_cumulative = []
    prefix_recent = []
    prefix_embedding_norms = []
    for label, rep_idx in zip(active_labels, representative_indices):
        member_indices = torch.nonzero(label_tensor == label, as_tuple=False).flatten()
        prefix_positions.append(state.positions.index_select(0, member_indices).float().mean().round())
        prefix_token_ids.append(state.token_ids[rep_idx])
        prefix_cumulative.append(state.cumulative_attention.index_select(0, member_indices).mean())
        prefix_recent.append(state.recent_attention.index_select(0, member_indices).mean())
        prefix_embedding_norms.append(state.embedding_norms.index_select(0, member_indices).mean())

    def _merge_prefix(prefix_values: list[torch.Tensor], tail: torch.Tensor) -> torch.Tensor:
        prefix_tensor = torch.stack(prefix_values).to(device=tail.device, dtype=tail.dtype)
        if prefix_count < state.seq_len:
            return torch.cat([prefix_tensor, tail[prefix_count:]], dim=0)
        return prefix_tensor

    clustered_state = replace(
        state,
        layers=clustered_layers,
        positions=_merge_prefix(prefix_positions, state.positions).to(torch.long),
        token_ids=_merge_prefix(prefix_token_ids, state.token_ids).to(torch.long),
        cumulative_attention=_merge_prefix(prefix_cumulative, state.cumulative_attention),
        recent_attention=_merge_prefix(prefix_recent, state.recent_attention),
        embedding_norms=_merge_prefix(prefix_embedding_norms, state.embedding_norms),
        metadata={**state.metadata, "cluster_meta": cluster_meta},
    )
    return clustered_state
