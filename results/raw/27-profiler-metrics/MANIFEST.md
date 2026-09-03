# Batch 27 — consolidated profiler metrics, TornadoVM vs hand-written CUDA

Captured 2026-09-03. RTX 4090 (sm_89), driver 565.57.01, CUDA 12.6.85,
JDK 25.0.2, TornadoVM 6.0.0-jdk22plus-cuda, Nsight Compute 2024.3.2,
Nsight Systems 2024.5.1.

Two jobs: capture the one demo that had no counter data and should have
(demo 01), and consolidate every profiler measurement in this repo into a
chart-ready form.

Reading guide, metric definitions and the figures this supports:
[`docs/PROFILER-METRICS.md`](../../../docs/PROFILER-METRICS.md).

## New capture — demo 01, vector add

The simplest kernel in the repo, and it turns out to be one of the most useful
comparisons, because it is cleanly DRAM-bandwidth-bound on both sides.

| Metric | TornadoVM | CUDA | Ratio |
|---|---|---|---|
| `instructions_executed` | 33,554,432 | 8,388,608 | **4.00** |
| `global_load_sectors` | 5,242,880 | 4,194,304 | 1.25 |
| `sectors_per_request_load` | **5.00** | **4.00** | 1.25 |
| `achieved_occupancy` | 55.66% | 88.72% | 0.63 |
| `dram_throughput_pct_peak` | 94.63% | 95.46% | 0.99 |
| `kernel_time_ncu` | 187,840 ns | 181,760 ns | **1.034** |

TornadoVM issues **4x the instructions**, moves 1.25x the sectors and reaches
63% of CUDA's occupancy — and lands **3.4% slower**, because both kernels are
at ~95% of peak DRAM bandwidth. Nothing else can matter while that is true.

This is also a **fourth kernel class** showing the exact 1.25x sector penalty
of [#1065](https://github.com/beehive-lab/TornadoVM/issues/1065), after demos
14, 15 and 16.

## Consolidation

`scripts/build-metrics-csv.py` parses the committed captures from batches 21,
22, 23, 24, 25, 26 and this one, and emits:

| File | Rows | Shape |
|---|---|---|
| `metrics.csv` | 172 | long: `demo, kernel, implementation, metric, unit, value, launches, source, file` |
| `comparison.csv` | 52 | paired: `demo, kernel, metric, unit, tornadovm, cuda, ratio_tornadovm_over_cuda` |

Every row carries the `file` it came from, so any number on a slide can be
traced back to the raw capture.

The script reads only committed files and is deterministic — rerun it after
adding a capture rather than editing the CSVs by hand. Where ncu emitted one
row per launch, values are averaged and `launches` records how many.

## Deliberate choices

- **`ratio_tornadovm_over_cuda` is always `tornadovm / cuda`**, never
  normalised so that "bigger is worse". For throughput and occupancy a ratio
  below 1 is the bad direction. The README says which is which; the CSV does
  not encode it.
- **`kernel_time_ncu` and `kernel_time_nsys` are separate metrics** and must
  not share an axis. ncu serialises launches, flushes caches and locks clocks,
  so its absolute times run several times higher than nsys or wall clock.
- **Demo 16 is Java-side only.** Its CUDA equivalent builds and validates but
  was not profiled; that is a gap, not a finding.
- **Demo 12 has no counters.** Its CUDA side needs a CUTLASS checkout that is
  not vendored here.
- **Demos 04, 05, 13 are excluded from codegen comparison.** Both sides call
  the same vendor library, so counters would measure NVIDIA's kernels rather
  than TornadoVM's code generation.

## Files

| File | Contents |
|---|---|
| `metrics.csv` | consolidated long-format measurements |
| `comparison.csv` | paired TornadoVM-vs-CUDA rows with ratios |
| `ncu-tornado01.csv`, `ncu-cuda01.csv` | demo 01 raw ncu captures |
| `*.stderr.log` | profiler stderr |
