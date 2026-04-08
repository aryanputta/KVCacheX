# KVCacheX

KVCacheX is a memory-aware inference optimizer for transformer LLMs that compresses, prunes, and schedules KV cache state during autoregressive decoding. The repo benchmarks three modes side by side:

- `no_cache`: recomputes the full prompt every step.
- `standard_cache`: uses the model's default KV cache.
- `kvcachex`: dynamically compresses and evicts cache entries under memory and latency pressure.

## What It Does

KVCacheX is organized around four independently testable agents:

- `CompressionAgent`: int8/int4 quantization, low-rank factorization, and prefix token clustering.
- `EvictionAgent`: one-shot head-aware post-prefill eviction, recency-aware decode pruning, and ML-assisted token retention.
- `SchedulerAgent`: adaptive cache sizing, attention probing cadence, and pressure-aware compression.
- `MonitorAgent`: latency, memory, throughput, cache size, perplexity, token agreement, and step logs.

The validated default profile is intentionally conservative: quantization plus bounded eviction is enabled by default, while token clustering remains available as an experimental mode for larger GPU-bound long-context studies.
KVCacheX now stores compressed KV state as a segmented cache by default: an older compressed prefix plus a `96`-token raw tail, which avoids re-quantizing the entire cache on every decode step while preserving a hot decode buffer.

The benchmark layer builds long-context, streaming, and multi-turn workloads, calibrates a token-importance model from attention traces, runs comparative evaluations, and writes:

- `results/metrics.csv`
- `results/experiment_logs.json`
- `analysis/bottleneck_report.md`
- `analysis/failure_cases.md`
- `dashboard/kvcachex_dashboard.html`

## Validated Result

The strongest verified configuration in this repo is the default profile in `config.yaml`:

- segmented KV cache with a compressed prefix and `96`-token raw tail
- symmetric `int8` quantization
- one-shot head-aware prefill eviction
- guarded decode pruning under scheduler control

On the saved reduced smoke benchmark artifacts in `results/metrics.csv`, that profile delivers:

- `60.35%` mean KV-cache memory reduction vs `standard_cache`
- `19.87%` mean latency improvement vs `standard_cache`
- `1.000` mean token agreement vs `standard_cache`
- `2.11x` mean cache compression ratio

Validation status:

- `17/17` tests passing via `python -m unittest discover -s tests -v`

Two more advanced systems paths are implemented but intentionally left disabled by default because they did not beat the validated profile on the reduced CPU benchmark:

- asymmetric K/V quantization
- chunk-selective prefix materialization

## Repo Layout

```text
KVCacheX/
├── kv_cache_engine/
│   ├── compression/
│   ├── eviction/
│   ├── scheduler/
│   └── monitor/
├── models/
├── cuda_kernels/
├── benchmark/
├── analysis/
├── dashboard/
├── tests/
├── results/
├── main.py
├── config.yaml
└── requirements.txt
```

## Quickstart

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the full benchmark:

```bash
python3 main.py benchmark
```

Run a single prompt through KVCacheX:

```bash
python3 main.py infer --mode kvcachex --prompt "Summarize the scheduling strategy in one paragraph."
```

Run the iterative improvement loop:

```bash
python3 main.py optimize --iterations 3
```

Generate reports and dashboard from saved results:

```bash
python3 main.py analyze
python3 main.py dashboard
```

## Recommended Models

- Smoke tests: `distilgpt2`
- Full long-context benchmarking: `Qwen/Qwen2.5-0.5B-Instruct`

The default config uses `distilgpt2` so the repo stays runnable on CPU-only machines. For 8k-32k prompt studies, switch `model.name` to the long-context model in `config.yaml`.

## Benchmark Method

1. Run `standard_cache` greedily to obtain a reference continuation.
2. Re-run `standard_cache`, `no_cache`, and `kvcachex` with teacher forcing on the same continuation.
3. Measure latency, throughput, peak cache bytes, perplexity drift, and token agreement.
4. Generate analysis markdown and an HTML dashboard from the saved logs.

## Notes

- CUDA paths are supported automatically when `torch.cuda.is_available()` is true.
- Edge deployment constraints are simulated through `edge.max_memory_bytes`.
- The token-importance model is trained automatically and saved to `models/token_importance_model.pkl`.
