# KVCacheX Bottleneck Report

## Aggregate Outcomes

- Mean memory reduction vs standard cache: 60.35%
- Mean latency improvement vs standard cache: 19.87%
- Mean token agreement vs standard cache: 1.000
- Mean compression ratio: 2.11x

## Workload Breakdown

### streaming_384
- Category: streaming
- Peak cache bytes: 6196608
- Memory delta vs standard: 56.79%
- Latency delta vs standard: 43.02%
- Perplexity delta vs standard: 0.01%

### long_context_512
- Category: long_context
- Peak cache bytes: 6196608
- Memory delta vs standard: 67.49%
- Latency delta vs standard: 10.77%
- Perplexity delta vs standard: -1.69%

### multi_turn_384
- Category: multi_turn
- Peak cache bytes: 6196608
- Memory delta vs standard: 56.79%
- Latency delta vs standard: 5.82%
- Perplexity delta vs standard: 0.42%

## Dominant Bottlenecks

- Largest latency regression risk: multi_turn_384
- Largest accuracy drift risk: multi_turn_384
- Weakest memory win: streaming_384

## Suggested Next Moves

- Tighten eviction only where token agreement stays high; otherwise expand the recency budget.
- Use int4 compression only under genuine memory pressure and fall back to int8 elsewhere.
- Probe attention less often when latency spikes originate from monitoring overhead rather than attention itself.
