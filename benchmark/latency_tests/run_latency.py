from __future__ import annotations

from benchmark.runner import BenchmarkRunner


if __name__ == "__main__":
    runner = BenchmarkRunner("config.yaml")
    metrics_df, _ = runner.run()
    print(
        metrics_df[
            [
                "workload_name",
                "mode",
                "mean_latency_ms",
                "p95_latency_ms",
                "tokens_per_sec",
                "latency_delta_pct_vs_standard",
            ]
        ].to_string(index=False)
    )
