from __future__ import annotations

from pathlib import Path

import pandas as pd

from kv_cache_engine.config import KVCacheXConfig, load_config
from kv_cache_engine.utils import ensure_parent, read_json


def _write_markdown(path: str | Path, lines: list[str]) -> None:
    output_path = ensure_parent(path)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def build_bottleneck_report(config: KVCacheXConfig) -> str:
    metrics = pd.read_csv(config.outputs.metrics_csv)
    optimized = metrics[metrics["mode"] == "kvcachex"]
    baseline = metrics[metrics["mode"] == "standard_cache"]

    lines = [
        "# KVCacheX Bottleneck Report",
        "",
        "## Aggregate Outcomes",
        "",
        f"- Mean memory reduction vs standard cache: {-optimized['memory_delta_pct_vs_standard'].mean():.2f}%",
        f"- Mean latency improvement vs standard cache: {-optimized['latency_delta_pct_vs_standard'].mean():.2f}%",
        f"- Mean token agreement vs standard cache: {optimized['token_agreement_vs_standard'].mean():.3f}",
        f"- Mean compression ratio: {optimized['compression_ratio'].mean():.2f}x",
        "",
        "## Workload Breakdown",
        "",
    ]

    for _, row in optimized.sort_values("latency_delta_pct_vs_standard").iterrows():
        lines.extend(
            [
                f"### {row['workload_name']}",
                f"- Category: {row['workload_category']}",
                f"- Peak cache bytes: {int(row['peak_cache_bytes'])}",
                f"- Memory delta vs standard: {-float(row['memory_delta_pct_vs_standard']):.2f}%",
                f"- Latency delta vs standard: {-float(row['latency_delta_pct_vs_standard']):.2f}%",
                f"- Perplexity delta vs standard: {float(row['accuracy_delta_pct_vs_standard']):.2f}%",
                "",
            ]
        )

    worst_latency = optimized.sort_values("latency_delta_pct_vs_standard", ascending=False).head(1)
    worst_accuracy = optimized.sort_values("accuracy_delta_pct_vs_standard", ascending=False).head(1)
    worst_memory = optimized.sort_values("memory_delta_pct_vs_standard", ascending=False).head(1)
    lines.extend(
        [
            "## Dominant Bottlenecks",
            "",
            f"- Largest latency regression risk: {worst_latency.iloc[0]['workload_name']}",
            f"- Largest accuracy drift risk: {worst_accuracy.iloc[0]['workload_name']}",
            f"- Weakest memory win: {worst_memory.iloc[0]['workload_name']}",
            "",
            "## Suggested Next Moves",
            "",
            "- Tighten eviction only where token agreement stays high; otherwise expand the recency budget.",
            "- Use int4 compression only under genuine memory pressure and fall back to int8 elsewhere.",
            "- Probe attention less often when latency spikes originate from monitoring overhead rather than attention itself.",
        ]
    )
    return "\n".join(lines)


def build_failure_report(config: KVCacheXConfig) -> str:
    logs = read_json(config.outputs.experiment_logs)
    failures: list[str] = [
        "# KVCacheX Failure Cases",
        "",
        "## Accuracy and Latency Regressions",
        "",
    ]

    for run in logs["runs"]:
        summary_mode = run["mode"]
        if summary_mode != "kvcachex":
            continue
        step_metrics = run["extra_metrics"]["step_metrics"]
        latency_values = [metric["latency_ms"] for metric in step_metrics]
        mean_latency = sum(latency_values) / len(latency_values) if latency_values else 0.0
        high_latency_steps = [
            metric["step"] for metric in step_metrics if metric["latency_ms"] > mean_latency * 1.5
        ]
        high_nll_steps = [
            metric["step"]
            for metric in step_metrics
            if metric.get("nll") is not None and metric["nll"] > config.monitor.accuracy_warning_nll
        ]
        if high_latency_steps or high_nll_steps or (
            run.get("token_agreement") is not None
            and run["token_agreement"] < config.monitor.token_agreement_warning
        ):
            failures.extend(
                [
                    f"### {run['workload_name']} ({run['run_kind']})",
                    f"- High latency steps: {high_latency_steps or 'none'}",
                    f"- High NLL steps: {high_nll_steps or 'none'}",
                    f"- Token agreement: {run.get('token_agreement', 'n/a')}",
                    f"- Mean latency: {run['mean_latency_ms']:.2f} ms",
                    "",
                ]
            )

    if len(failures) == 4:
        failures.extend(
            [
                "- No major regressions exceeded the configured thresholds in the recorded runs.",
                "",
            ]
        )
    failures.extend(
        [
            "## Interpretation",
            "",
            "- High NLL usually indicates that compression or eviction removed a token the next step relied on.",
            "- Latency spikes often appear when cache compaction and attention probing happen in the same decode step.",
            "- Low token agreement with acceptable perplexity suggests stylistic drift rather than hard factual loss.",
        ]
    )
    return "\n".join(failures)


def generate_reports(config: str | Path | KVCacheXConfig = "config.yaml") -> None:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    bottleneck_report = build_bottleneck_report(resolved)
    failure_report = build_failure_report(resolved)
    _write_markdown(resolved.outputs.bottleneck_report, bottleneck_report.splitlines())
    _write_markdown(resolved.outputs.failure_report, failure_report.splitlines())


if __name__ == "__main__":
    generate_reports("config.yaml")
