# Chunk-Selective Prefix Materialization Experiment

Date: 2026-04-08

## Goal

Reduce cold-prefix decode cost by reusing previously materialized compressed-prefix work instead of re-dequantizing every unchanged prefix chunk on each decode step.

Implemented behavior:

- bounded dequantized chunk cache
- bounded materialized prefix cache
- incremental prefix extension when a new compressed chunk is appended
- per-run cache reset to avoid cross-run memory growth

## Validation

- Unit and integration tests: `python -m unittest discover -s tests -v`
- Result: `17/17` passing

## Benchmark Setup

- Model: `distilgpt2`
- Workloads:
  - `long_context_512`
  - `streaming_384`
  - `multi_turn_384`
- Decode length: `6`
- Comparison target: `results/seg_tail_96_metrics.csv`

## Runtime Evidence

Materialization stats from the opt-in teacher-forced runs:

- `chunk_cache_hits`: `126`
- `chunk_cache_misses`: `168`
- `prefix_cache_hits`: `15`
- `prefix_cache_misses`: `60`
- `prefix_cache_extensions`: `15`

This confirms that repeated segmented decode steps are reusing prior prefix materialization work instead of fully rebuilding every prefix from scratch.

## Result

Opt-in benchmark artifact:

- `results/materialized_prefix_metrics.csv`
- `results/materialized_prefix_logs.json`

Observed reduced smoke result:

- memory reduction: `60.35%`
- latency improvement: `-0.42%`
- token agreement: `1.000`
- compression ratio: `2.11x`

Compared with the saved segmented baseline, the reduced CPU smoke benchmark regressed in latency despite successful cache reuse. The most likely reason is that the benchmark is short (`6` decode steps) and CPU-bound, so Python-side cache bookkeeping costs outweigh the saved dequantization work.

## Decision

The feature is implemented and production-gated, but it remains disabled by default in `config.yaml` until it is validated on:

1. longer decode horizons,
2. larger long-context workloads, or
3. GPU runs where dequantization cost is a larger share of step latency.

## Recommended Next Step

If we want this line of work to pay off in the benchmarked default path, the next change should reduce Python orchestration overhead too:

1. move prefix reuse into a fused CUDA or Torch kernel path, or
2. materialize only the active prefix slices selected by scheduler or eviction instead of reconstructing the whole kept prefix.
