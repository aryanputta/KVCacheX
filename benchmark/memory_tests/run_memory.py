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
                "peak_process_memory_bytes",
                "peak_cache_bytes",
                "compression_ratio",
                "memory_delta_pct_vs_standard",
            ]
        ].to_string(index=False)
    )
