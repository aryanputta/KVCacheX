# KVCacheX Failure Cases

## Accuracy and Latency Regressions

### long_context_512 (teacher_forced)
- High latency steps: [0]
- High NLL steps: none
- Token agreement: 1.0
- Mean latency: 95.76 ms

### long_context_512 (greedy)
- High latency steps: [0]
- High NLL steps: none
- Token agreement: 1.0
- Mean latency: 103.24 ms

### streaming_384 (teacher_forced)
- High latency steps: [0]
- High NLL steps: none
- Token agreement: 1.0
- Mean latency: 62.11 ms

### streaming_384 (greedy)
- High latency steps: [0]
- High NLL steps: none
- Token agreement: 1.0
- Mean latency: 68.16 ms

### multi_turn_384 (teacher_forced)
- High latency steps: [0]
- High NLL steps: none
- Token agreement: 1.0
- Mean latency: 85.74 ms

### multi_turn_384 (greedy)
- High latency steps: [0]
- High NLL steps: none
- Token agreement: 1.0
- Mean latency: 101.88 ms

## Interpretation

- High NLL usually indicates that compression or eviction removed a token the next step relied on.
- Latency spikes often appear when cache compaction and attention probing happen in the same decode step.
- Low token agreement with acceptable perplexity suggests stylistic drift rather than hard factual loss.
