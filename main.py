from __future__ import annotations

import argparse

from analysis.generate_reports import generate_reports
from benchmark import BenchmarkRunner, IterativeOptimizer
from dashboard.build_dashboard import build_dashboard
from kv_cache_engine.config import load_config
from models import InferenceRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KVCacheX benchmark and inference CLI")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run the full benchmark suite.")
    benchmark_parser.add_argument("--model-name", default=None, help="Override the model name.")

    infer_parser = subparsers.add_parser("infer", help="Run a single inference job.")
    infer_parser.add_argument("--prompt", required=True, help="Prompt text to evaluate.")
    infer_parser.add_argument(
        "--mode",
        default="kvcachex",
        choices=["no_cache", "standard_cache", "kvcachex"],
        help="Inference mode.",
    )
    infer_parser.add_argument("--model-name", default=None, help="Override the model name.")
    infer_parser.add_argument("--max-new-tokens", type=int, default=None, help="Decode length.")

    optimize_parser = subparsers.add_parser(
        "optimize", help="Run the iterative benchmark-driven optimizer loop."
    )
    optimize_parser.add_argument("--iterations", type=int, default=3, help="Optimization iterations.")
    optimize_parser.add_argument("--model-name", default=None, help="Override the model name.")

    dashboard_parser = subparsers.add_parser("dashboard", help="Build the HTML dashboard.")
    dashboard_parser.add_argument("--model-name", default=None, help="Unused, reserved for parity.")

    subparsers.add_parser("analyze", help="Generate analysis markdown from saved benchmark results.")
    subparsers.add_parser(
        "train-importance", help="Fit the token importance model from calibration workloads."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "benchmark":
        runner = BenchmarkRunner(config)
        metrics_df, _ = runner.run(model_name=args.model_name)
        generate_reports(config)
        build_dashboard(config)
        print(
            metrics_df[
                [
                    "workload_name",
                    "mode",
                    "mean_latency_ms",
                    "peak_cache_bytes",
                    "compression_ratio",
                    "token_agreement_vs_standard",
                ]
            ].to_string(index=False)
        )
        return

    if args.command == "infer":
        runner = InferenceRunner(config)
        artifacts = runner.run(
            prompt=args.prompt,
            workload_name="ad_hoc",
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            model_name=args.model_name,
        )
        print(artifacts.summary.output_text)
        print(
            f"\nmean_latency_ms={artifacts.summary.mean_latency_ms:.2f} "
            f"tokens_per_sec={artifacts.summary.tokens_per_sec:.2f} "
            f"peak_cache_bytes={artifacts.summary.peak_cache_bytes}"
        )
        return

    if args.command == "optimize":
        optimizer = IterativeOptimizer(config)
        _, history = optimizer.run(iterations=args.iterations, model_name=args.model_name)
        print(history)
        return

    if args.command == "dashboard":
        output_path = build_dashboard(config)
        print(output_path)
        return

    if args.command == "analyze":
        generate_reports(config)
        print(config.outputs.bottleneck_report)
        print(config.outputs.failure_report)
        return

    if args.command == "train-importance":
        runner = BenchmarkRunner(config)
        workloads = runner.build_workloads()
        runner.train_importance_predictor(workloads)
        print(config.eviction.semantic_model_path)
        return


if __name__ == "__main__":
    main()
