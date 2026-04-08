from __future__ import annotations

import numpy as np
import torch

from kv_cache_engine.config import EvictionConfig
from kv_cache_engine.types import KVCacheState

from .importance_model import TokenImportancePredictor, build_feature_matrix


def _normalize_tensor(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    max_value = values.abs().max().clamp_min(1e-8)
    return values / max_value


def _unique_topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    if scores.numel() == 0 or k <= 0:
        return torch.zeros(0, dtype=torch.long, device=scores.device)
    topk = min(k, scores.numel())
    return torch.topk(scores, k=topk).indices.to(torch.long)


class EvictionAgent:
    def __init__(self, config: EvictionConfig):
        self.config = config
        self.predictor = TokenImportancePredictor(config.semantic_model_path)
        if self.predictor.exists():
            self.predictor.load()

    def fit_predictor(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.predictor.fit(features, labels)
        self.predictor.save()

    def should_prune_decode(self, state: KVCacheState, target_cache_tokens: int) -> bool:
        if not self.config.enabled or not self.config.dynamic_decode_pruning:
            return False
        return state.seq_len > (target_cache_tokens + self.config.decode_prune_margin)

    def prune_prefill_state(
        self,
        state: KVCacheState,
        attentions: tuple[torch.Tensor, ...] | None,
        target_cache_tokens: int | None = None,
    ) -> KVCacheState:
        if (
            not self.config.enabled
            or not self.config.prefill_head_aware_eviction
            or attentions is None
            or state.seq_len <= self.config.min_tokens_to_keep
        ):
            return state

        target = target_cache_tokens or int(state.seq_len * self.config.target_keep_ratio)
        target = max(self.config.min_tokens_to_keep, min(target, state.seq_len))
        if target >= state.seq_len:
            return state

        observation_window = min(self.config.prefill_observation_window, state.seq_len)
        per_head_scores = []
        for layer_attention in attentions:
            if layer_attention is None:
                continue
            attention_slice = layer_attention[0, :, -observation_window:, :]
            per_head_scores.append(attention_slice.mean(dim=1))
        if not per_head_scores:
            return self.prune_state(state, target_cache_tokens=target)

        flattened_scores = torch.cat(per_head_scores, dim=0)
        aggregate_scores = _normalize_tensor(flattened_scores.mean(dim=0))
        total_heads = max(1, flattened_scores.shape[0])
        topk_per_head = max(1, target // total_heads)
        support = torch.zeros(state.seq_len, device=state.positions.device, dtype=torch.float32)
        candidate_indices = []
        for head_scores in flattened_scores:
            top_indices = _unique_topk_indices(head_scores, topk_per_head)
            candidate_indices.append(top_indices)
            support.index_add_(
                0,
                top_indices,
                torch.ones(top_indices.numel(), device=support.device, dtype=support.dtype),
            )
        support = _normalize_tensor(support)

        semantic_score = aggregate_scores.clone()
        if self.predictor.pipeline is not None:
            semantic_np = self.predictor.predict_scores(build_feature_matrix(state))
            semantic_score = torch.as_tensor(
                semantic_np, device=state.positions.device, dtype=torch.float32
            )

        recency_score = _normalize_tensor(
            torch.linspace(0.0, 1.0, steps=state.seq_len, device=state.positions.device)
        )
        combined_score = (
            self.config.head_score_weight * aggregate_scores
            + self.config.head_support_weight * support
            + self.config.recency_weight * recency_score
            + self.config.semantic_weight * semantic_score
        )
        forced_keep = torch.cat(
            [
                torch.arange(
                    min(self.config.pin_first_tokens, state.seq_len), device=state.positions.device
                ),
                torch.arange(
                    max(0, state.seq_len - self.config.recent_tokens_to_keep),
                    state.seq_len,
                    device=state.positions.device,
                ),
            ]
        ).to(torch.long)
        top_candidates = torch.cat(candidate_indices) if candidate_indices else torch.zeros(
            0, device=state.positions.device, dtype=torch.long
        )
        keep_indices = torch.unique(torch.cat([forced_keep, top_candidates]), sorted=True)

        if keep_indices.numel() < target:
            remaining_mask = torch.ones(state.seq_len, device=state.positions.device, dtype=torch.bool)
            remaining_mask.index_fill_(0, keep_indices, False)
            remaining = torch.nonzero(remaining_mask, as_tuple=False).flatten()
            if remaining.numel():
                extra_budget = min(target - keep_indices.numel(), remaining.numel())
                extra_indices = _unique_topk_indices(
                    combined_score.index_select(0, remaining), extra_budget
                )
                keep_indices = torch.unique(
                    torch.cat([keep_indices, remaining.index_select(0, extra_indices)]), sorted=True
                )
        elif keep_indices.numel() > target:
            forced_mask = torch.zeros(state.seq_len, device=state.positions.device, dtype=torch.bool)
            forced_mask.index_fill_(0, forced_keep, True)
            forced_indices = torch.nonzero(forced_mask, as_tuple=False).flatten()
            candidate_mask = torch.zeros(state.seq_len, device=state.positions.device, dtype=torch.bool)
            candidate_mask.index_fill_(0, keep_indices, True)
            candidate_mask &= ~forced_mask
            remainder = torch.nonzero(candidate_mask, as_tuple=False).flatten()
            budget = max(0, target - forced_indices.numel())
            selected = _unique_topk_indices(combined_score.index_select(0, remainder), budget)
            keep_indices = torch.unique(
                torch.cat([forced_indices, remainder.index_select(0, selected)]), sorted=True
            )

        pruned_state = state.slice(keep_indices)
        pruned_state.metadata["eviction"] = {
            "before_tokens": state.seq_len,
            "after_tokens": pruned_state.seq_len,
            "target_tokens": target,
            "strategy": "prefill_head_aware",
            "observation_window": observation_window,
        }
        pruned_state.metadata["head_support"] = support.index_select(0, keep_indices).detach().cpu().tolist()
        return pruned_state

    def prune_state(
        self,
        state: KVCacheState,
        target_cache_tokens: int | None = None,
    ) -> KVCacheState:
        if not self.config.enabled or state.seq_len <= self.config.min_tokens_to_keep:
            return state

        target = target_cache_tokens or int(state.seq_len * self.config.target_keep_ratio)
        target = max(self.config.min_tokens_to_keep, min(target, state.seq_len))
        if target >= state.seq_len:
            return state

        attention_score = _normalize_tensor(
            (state.cumulative_attention * 0.7) + (state.recent_attention * 0.3)
        )
        recency_score = _normalize_tensor(
            torch.linspace(0.0, 1.0, steps=state.seq_len, device=state.positions.device)
        )
        semantic_score = attention_score.clone()
        if self.predictor.pipeline is not None:
            semantic_np = self.predictor.predict_scores(build_feature_matrix(state))
            semantic_score = torch.as_tensor(
                semantic_np, device=state.positions.device, dtype=torch.float32
            )

        combined_score = (
            self.config.attention_weight * attention_score
            + self.config.recency_weight * recency_score
            + self.config.semantic_weight * semantic_score
        )

        forced_keep = torch.cat(
            [
                torch.arange(
                    min(self.config.pin_first_tokens, state.seq_len), device=state.positions.device
                ),
                torch.arange(
                    max(0, state.seq_len - self.config.recent_tokens_to_keep),
                    state.seq_len,
                    device=state.positions.device,
                ),
            ]
        ).to(torch.long)
        threshold_keep = torch.nonzero(
            attention_score >= self.config.attention_threshold, as_tuple=False
        ).flatten()
        keep_indices = torch.unique(torch.cat([forced_keep, threshold_keep]), sorted=True)

        if keep_indices.numel() < target:
            candidate_indices = torch.tensor(
                [idx for idx in range(state.seq_len) if idx not in set(keep_indices.tolist())],
                device=state.positions.device,
                dtype=torch.long,
            )
            if candidate_indices.numel():
                candidate_scores = combined_score.index_select(0, candidate_indices)
                budget = min(target - keep_indices.numel(), candidate_indices.numel())
                topk = torch.topk(candidate_scores, k=budget).indices
                keep_indices = torch.unique(
                    torch.cat([keep_indices, candidate_indices.index_select(0, topk)]), sorted=True
                )
        elif keep_indices.numel() > target:
            forced_mask = torch.zeros(state.seq_len, device=state.positions.device, dtype=torch.bool)
            forced_mask.index_fill_(0, forced_keep, True)
            forced = torch.nonzero(forced_mask, as_tuple=False).flatten()
            remainder = torch.nonzero(~forced_mask, as_tuple=False).flatten()
            remainder_scores = combined_score.index_select(0, remainder)
            budget = max(0, target - forced.numel())
            selected = torch.topk(remainder_scores, k=min(budget, remainder.numel())).indices
            keep_indices = torch.unique(
                torch.cat([forced, remainder.index_select(0, selected)]), sorted=True
            )

        pruned_state = state.slice(keep_indices)
        pruned_state.metadata["eviction"] = {
            "before_tokens": state.seq_len,
            "after_tokens": pruned_state.seq_len,
            "target_tokens": target,
        }
        return pruned_state
