from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kv_cache_engine.types import KVCacheState


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    scale = float(np.max(np.abs(values)))
    if scale <= 1e-8:
        return np.zeros_like(values)
    return values / scale


def build_feature_matrix(state: KVCacheState) -> np.ndarray:
    positions = state.positions.detach().cpu().numpy().astype(np.float32)
    cumulative = state.cumulative_attention.detach().cpu().numpy().astype(np.float32)
    recent = state.recent_attention.detach().cpu().numpy().astype(np.float32)
    embedding_norms = state.embedding_norms.detach().cpu().numpy().astype(np.float32)
    token_ids = state.token_ids.detach().cpu().numpy().astype(np.int64)

    max_position = float(np.max(positions)) if positions.size else 1.0
    age = max_position - positions
    unique_tokens, token_counts = np.unique(token_ids, return_counts=True)
    token_count_lookup = {token: count for token, count in zip(unique_tokens, token_counts)}
    inverse_frequency = np.asarray(
        [1.0 / token_count_lookup[int(token)] for token in token_ids], dtype=np.float32
    )

    return np.stack(
        [
            _normalize(age),
            _normalize(cumulative),
            _normalize(recent),
            _normalize(embedding_norms),
            _normalize(inverse_frequency),
        ],
        axis=1,
    )


def attention_labels(attention_scores: np.ndarray, keep_ratio: float) -> np.ndarray:
    if attention_scores.size == 0:
        return attention_scores.astype(np.int64)
    threshold = np.quantile(attention_scores, max(0.0, min(1.0, 1.0 - keep_ratio)))
    return (attention_scores >= threshold).astype(np.int64)


class TokenImportancePredictor:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.pipeline: Pipeline | None = None

    def exists(self) -> bool:
        return self.model_path.exists()

    def load(self) -> None:
        with self.model_path.open("rb") as handle:
            self.pipeline = pickle.load(handle)

    def save(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Cannot save an unfitted importance predictor.")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with self.model_path.open("wb") as handle:
            pickle.dump(self.pipeline, handle)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.pipeline = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
            ]
        )
        self.pipeline.fit(features, labels)

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Importance predictor must be loaded or fitted before inference.")
        return self.pipeline.predict_proba(features)[:, 1]

    def predict_state_scores(self, state: KVCacheState) -> np.ndarray:
        features = build_feature_matrix(state)
        return self.predict_scores(features)
