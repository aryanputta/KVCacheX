from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kv_cache_engine.config import KVCacheXConfig, load_config
from kv_cache_engine.utils import ensure_parent, read_json


def _build_heatmap_data(log_payload: dict) -> tuple[list[list[float]], list[str], list[str]]:
    for run in log_payload["runs"]:
        if run["mode"] == "kvcachex" and run["run_kind"] == "teacher_forced":
            step_metrics = run["extra_metrics"]["step_metrics"]
            z_values = []
            x_labels = None
            y_labels = []
            for metric in step_metrics:
                positions = metric["metadata"].get("positions", [])
                scores = metric["metadata"].get("importance_scores", [])
                if not positions or not scores:
                    continue
                x_labels = [str(position) for position in positions]
                z_values.append(scores)
                y_labels.append(f"step_{metric['step']}")
            if z_values and x_labels is not None:
                return z_values, x_labels, y_labels
    return [[0.0]], ["0"], ["step_0"]


def build_dashboard(config: str | Path | KVCacheXConfig = "config.yaml") -> str:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    metrics = pd.read_csv(resolved.outputs.metrics_csv)
    logs = read_json(resolved.outputs.experiment_logs)

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Memory vs Latency",
            "Compression Ratio vs Accuracy",
            "Token Importance Heatmap",
            "Tokens per Second",
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"type": "heatmap"}, {"type": "bar"}]],
    )

    for mode, frame in metrics.groupby("mode"):
        figure.add_trace(
            go.Scatter(
                x=frame["peak_cache_bytes"],
                y=frame["mean_latency_ms"],
                mode="markers+text",
                name=f"{mode} memory-latency",
                text=frame["workload_name"],
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=frame["compression_ratio"],
                y=frame["token_agreement_vs_standard"],
                mode="markers+text",
                name=f"{mode} compression-accuracy",
                text=frame["workload_name"],
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Bar(
                x=frame["workload_name"],
                y=frame["tokens_per_sec"],
                name=f"{mode} throughput",
            ),
            row=2,
            col=2,
        )

    z_values, x_labels, y_labels = _build_heatmap_data(logs)
    figure.add_trace(
        go.Heatmap(z=z_values, x=x_labels, y=y_labels, coloraxis="coloraxis"),
        row=2,
        col=1,
    )

    figure.update_layout(
        title="KVCacheX Benchmark Dashboard",
        coloraxis={"colorscale": "Viridis"},
        template="plotly_white",
        height=900,
    )
    output_path = ensure_parent(resolved.outputs.dashboard_html)
    figure.write_html(output_path, include_plotlyjs="cdn")
    return str(output_path)


if __name__ == "__main__":
    build_dashboard("config.yaml")
