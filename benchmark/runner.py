from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kv_cache_engine.config import KVCacheXConfig, load_config
from kv_cache_engine.utils import ensure_parent, percent_change, write_json
from models import InferenceRunner

from .workloads import WorkloadSample, build_workloads


def _token_agreement(reference: list[int], candidate: list[int]) -> float:
    if not reference:
        return 0.0
    overlap = min(len(reference), len(candidate))
    matches = sum(
        1 for ref_token, cand_token in zip(reference[:overlap], candidate[:overlap]) if ref_token == cand_token
    )
    return matches / len(reference)


class BenchmarkRunner:
    def __init__(self, config: str | Path | KVCacheXConfig = "config.yaml"):
        self.config = load_config(config) if isinstance(config, (str, Path)) else deepcopy(config)
        self.inference_runner = InferenceRunner(self.config)

    def _max_supported_tokens(self, model_name: str | None = None) -> int:
        model, _ = self.inference_runner.model_manager.load(model_name=model_name)
        model_limit = (
            getattr(model.config, "max_position_embeddings", None)
            or getattr(model.config, "n_positions", None)
            or max(self.config.workloads.long_context_targets)
        )
        return max(256, int(model_limit) - int(self.config.model.max_new_tokens) - 8)

    def build_workloads(self, model_name: str | None = None) -> list[WorkloadSample]:
        _, tokenizer = self.inference_runner.model_manager.load(model_name=model_name)
        return build_workloads(
            tokenizer=tokenizer,
            config=self.config.workloads,
            max_supported_tokens=self._max_supported_tokens(model_name=model_name),
        )

    def warmup(self, workloads: list[WorkloadSample], model_name: str | None = None) -> None:
        if not workloads:
            return
        prompt = workloads[0].prompt
        for _ in range(self.config.benchmark.warmup_runs):
            self.inference_runner.run(
                prompt=prompt,
                workload_name="warmup",
                mode="standard_cache",
                max_new_tokens=min(4, self.config.model.max_new_tokens),
                model_name=model_name,
            )

    def train_importance_predictor(
        self,
        workloads: list[WorkloadSample],
        model_name: str | None = None,
    ) -> None:
        predictor = self.inference_runner.eviction_agent.predictor
        if predictor.exists() or not self.config.eviction.train_if_missing:
            if predictor.exists() and predictor.pipeline is None:
                predictor.load()
            return

        feature_batches = []
        label_batches = []
        calibration_steps = min(16, self.config.model.max_new_tokens)
        for workload in workloads[:2]:
            features, labels = self.inference_runner.collect_importance_training_data(
                prompt=workload.prompt,
                max_steps=calibration_steps,
                model_name=model_name,
            )
            if features.size and labels.size:
                feature_batches.append(features)
                label_batches.append(labels)
        if feature_batches and label_batches:
            features = np.concatenate(feature_batches, axis=0)
            labels = np.concatenate(label_batches, axis=0)
            self.inference_runner.eviction_agent.fit_predictor(features, labels)

    @staticmethod
    def _summary_row(
        summary,
        workload: WorkloadSample,
        model_name: str,
        token_agreement_vs_standard: float | None,
    ) -> dict[str, Any]:
        return {
            "workload_name": workload.name,
            "workload_category": workload.category,
            "target_tokens": workload.target_tokens,
            "mode": summary.mode,
            "run_kind": summary.run_kind,
            "model_name": model_name,
            "prompt_tokens": summary.prompt_tokens,
            "generated_tokens": summary.generated_tokens,
            "mean_latency_ms": summary.mean_latency_ms,
            "p95_latency_ms": summary.p95_latency_ms,
            "tokens_per_sec": summary.tokens_per_sec,
            "peak_process_memory_bytes": summary.peak_process_memory_bytes,
            "peak_cache_bytes": summary.peak_cache_bytes,
            "mean_nll": summary.mean_nll,
            "perplexity": summary.perplexity,
            "compression_ratio": summary.compression_ratio,
            "token_agreement_vs_standard": token_agreement_vs_standard,
            "output_text": summary.output_text,
        }

    def run(self, model_name: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
        target_model_name = model_name or self.config.model.name
        workloads = self.build_workloads(model_name=target_model_name)
        self.warmup(workloads, model_name=target_model_name)
        self.train_importance_predictor(workloads, model_name=target_model_name)

        rows: list[dict[str, Any]] = []
        logs: list[dict[str, Any]] = []

        for workload in workloads:
            reference_greedy = self.inference_runner.run(
                prompt=workload.prompt,
                workload_name=workload.name,
                mode="standard_cache",
                max_new_tokens=self.config.model.max_new_tokens,
                model_name=target_model_name,
            )
            reference_eval = self.inference_runner.run(
                prompt=workload.prompt,
                workload_name=workload.name,
                mode="standard_cache",
                max_new_tokens=len(reference_greedy.generated_token_ids),
                forced_tokens=reference_greedy.generated_token_ids,
                model_name=target_model_name,
            )
            rows.append(
                self._summary_row(
                    reference_eval.summary,
                    workload,
                    target_model_name,
                    token_agreement_vs_standard=1.0,
                )
            )
            logs.append(reference_greedy.to_log_dict())
            logs.append(reference_eval.to_log_dict())

            for mode in self.config.benchmark.modes:
                if mode == "standard_cache":
                    continue
                eval_run = self.inference_runner.run(
                    prompt=workload.prompt,
                    workload_name=workload.name,
                    mode=mode,
                    max_new_tokens=len(reference_greedy.generated_token_ids),
                    forced_tokens=reference_greedy.generated_token_ids,
                    model_name=target_model_name,
                )
                greedy_run = self.inference_runner.run(
                    prompt=workload.prompt,
                    workload_name=workload.name,
                    mode=mode,
                    max_new_tokens=self.config.model.max_new_tokens,
                    model_name=target_model_name,
                )
                agreement = _token_agreement(
                    reference_greedy.generated_token_ids, greedy_run.generated_token_ids
                )
                rows.append(
                    self._summary_row(
                        eval_run.summary,
                        workload,
                        target_model_name,
                        token_agreement_vs_standard=agreement,
                    )
                )
                logs.append(eval_run.to_log_dict())
                logs.append(greedy_run.to_log_dict())

        metrics_df = pd.DataFrame(rows)
        for workload_name, workload_rows in metrics_df.groupby("workload_name"):
            baseline = workload_rows[workload_rows["mode"] == "standard_cache"].iloc[0]
            baseline_latency = float(baseline["mean_latency_ms"])
            baseline_memory = float(baseline["peak_cache_bytes"])
            baseline_perplexity = float(baseline["perplexity"])
            workload_mask = metrics_df["workload_name"] == workload_name
            metrics_df.loc[workload_mask, "latency_delta_pct_vs_standard"] = metrics_df.loc[
                workload_mask, "mean_latency_ms"
            ].apply(lambda value: percent_change(baseline_latency, float(value)))
            metrics_df.loc[workload_mask, "memory_delta_pct_vs_standard"] = metrics_df.loc[
                workload_mask, "peak_cache_bytes"
            ].apply(lambda value: percent_change(baseline_memory, float(value)))
            metrics_df.loc[workload_mask, "accuracy_delta_pct_vs_standard"] = metrics_df.loc[
                workload_mask, "perplexity"
            ].apply(lambda value: percent_change(baseline_perplexity, float(value)))

        metrics_path = ensure_parent(self.config.outputs.metrics_csv)
        metrics_df.to_csv(metrics_path, index=False)
        log_payload = {
            "config": self.config.to_dict(),
            "model_name": target_model_name,
            "workloads": [asdict(workload) for workload in workloads],
            "runs": logs,
        }
        write_json(self.config.outputs.experiment_logs, log_payload)
        return metrics_df, log_payload
