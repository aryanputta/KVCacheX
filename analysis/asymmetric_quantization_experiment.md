# Asymmetric K/V Quantization Experiment

Date: 2026-04-08

## Goal

Evaluate a KIVI-inspired asymmetric quantization path for KVCacheX:

- keys: `affine_per_channel`
- values: `affine_per_token`
- segmented compressed prefix with raw decode tail

## Validation

- Unit and integration tests: `python -m unittest discover -s tests -v`
- Result: `15/15` passing

## Smoke Benchmark Setup

- Model: `distilgpt2`
- Workloads:
  - `long_context_512`
  - `streaming_384`
  - `multi_turn_384`
- Decode length: `6` tokens

## Single-Pass Comparison

| Profile | Memory Reduction | Latency Improvement | Token Agreement | Compression Ratio |
| --- | ---: | ---: | ---: | ---: |
| `sym_int8` | `60.35%` | `-0.06%` | `1.000` | `2.11x` |
| `asym_int8` | `57.90%` | `-6.22%` | `1.000` | `1.97x` |
| `asym_int4` | `66.39%` | `4.39%` | `1.000` | `2.39x` |

Artifacts:

- `results/sym_int8_metrics.csv`
- `results/asym_int8_metrics.csv`
- `results/asym_int4_metrics.csv`

## Two-Run Bakeoff

Average of two reduced benchmark runs:

| Profile | Memory Reduction | Latency Improvement | Token Agreement | Compression Ratio |
| --- | ---: | ---: | ---: | ---: |
| `sym_int8` | `60.35%` | `-0.21%` | `1.000` | `2.11x` |
| `asym_int4` | `66.39%` | `-1.34%` | `1.000` | `2.39x` |

Artifacts:

- `results/sym_int8_repeat1_metrics.csv`
- `results/sym_int8_repeat2_metrics.csv`
- `results/asym_int4_repeat1_metrics.csv`
- `results/asym_int4_repeat2_metrics.csv`

## Conclusion

The asymmetric quantizer is production-ready as an opt-in compression mode, but it is not the best default on the current reduced CPU smoke benchmark:

- `asym_int8` regresses both memory efficiency and latency.
- `asym_int4` improves memory reduction and compression ratio while preserving token agreement, but it does not produce a consistent latency win.
- The previously validated segmented `sym_int8` profile remains the safer default until we benchmark on GPU or reduce cold-prefix reconstruction cost further.

## Recommended Next Step

Keep asymmetric quantization available for memory-constrained tuning and pursue a lower-overhead decode path next:

1. fused dequantize-plus-attention for the compressed prefix, or
2. chunk-selective prefix materialization so decode only reconstructs the active prefix slices.
