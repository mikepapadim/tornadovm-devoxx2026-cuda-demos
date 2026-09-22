# Batch 34 — Track A wall-clock timings re-measured on TornadoVM 7.0.0

Captured 2026-09-22, immediately after the 7.0.0 migration (batch 33). Every
timed Track A demo re-run on `7.0.0-jdk22plus-cuda` with the **same arguments**
the 6.0.0-era batches used, so the two sets are directly comparable.

## Environment

Identical to batch 33: TornadoVM `7.0.0-jdk22plus-cuda` (commit `65eb834`),
OpenJDK 25.0.2, RTX 4090 driver 610.57.04, CUDA toolkit 12.6.85, CUDA 13 runtime
on `LD_LIBRARY_PATH` for the CUTLASS bridge, Ubuntu 22.04.5.

## Method

Demos 06, 07 and 11 were run **five times** each (`tornado` launcher, `java @argfile`,
plus three repeats) because their headline numbers are speedup ratios. Demos 12, 14,
17 and 18 were run once via the `tornado` launcher, matching how batches 19/28 captured
them. Every demo reports its own steady-state median with the first (JIT) execution
excluded; the ratios below are that demo's own arithmetic, not post-hoc.

Arguments, matching the 6.0.0-era captures:

| Demo | Arguments | 6.0.0 baseline from |
|---|---|---|
| 06-cuda-streams | `8 32768 65536 8 both` | batch 18 |
| 07-cuda-graph-benefit | `4096 6 50 both` | batch 18 |
| 11-integrated-showcase | `6 8 8 20 all` | batch 18 |
| 12-cutlass-fused-epilogue | `1024 1024 1024 20` | batch 19 |
| 14-warp-async-shared | `4096 1024 20` | batch 19 |
| 17-matmul-ladder | `2048 10` | batch 28 (wall-clock table) |
| 18-matmul-ladder-fp16 | `1024 10` | **no comparable baseline** — batch 30 captured `nsys` kernel time, not wall clock |

## Measured (Observed — this machine, these runs)

| Demo | Metric | 6.0.0 | 7.0.0 | Change |
|---|---|---|---|---|
| 06 | sequential median | 2174–2176 µs | **1273–1330 µs** | ~1.7x faster |
| 06 | concurrent median | 936–960 µs | **928–1015 µs** | unchanged |
| 06 | concurrency benefit | ~2.3x | **1.29–1.43x** | ratio shrinks |
| 07 | nograph median | 292–364 µs | **298–309 µs** | tighter, same band |
| 07 | graph median | 36 µs | **34.7–35.8 µs** | ~unchanged |
| 07 | graph replay speedup | 8.08–10.00x | **8.46–8.79x** | tighter |
| 11 | baseline median | 831 µs | **800–839 µs** | unchanged |
| 11 | graph median | 148 µs | **80–81 µs** | ~1.85x faster |
| 11 | graph vs baseline | 5.37–5.61x | **9.88–10.49x** | roughly doubles |
| 11 | combined vs baseline | 5.66–5.69x | **9.94–10.49x** | roughly doubles |
| 11 | concurrent vs baseline | 1.08–1.12x | **0.95–1.02x** | benefit gone |
| 12 | fused / unfused median | 317 / 304 µs (0.96x) | **303 / 294 µs (0.97x)** | ~unchanged |
| 14 | naive / optimised median | 228 / 105 µs (2.17x) | **237 / 97 µs (2.44x)** | slightly better |
| 17 | ladder, wall-clock GFLOP/s | 3783 / 4641 / 9723 / 12670 / 13075 / 14449 | **3835 / 4439 / 10495 / 11955 / 12946 / 14187** | within a few % |
| 18 | ladder FP16, wall-clock | — | 2750 / 3441 / 3933 / 10374 / 10579 / 6608 GFLOP/s | first wall-clock capture |

## Reading of the two ratio changes

Neither is a regression in the feature the demo is about:

- **Demo 06's 2.3x → ~1.35x** is the *baseline* getting faster, not concurrency getting
  slower. The concurrent path is flat (936–960 → 928–1015 µs); the single-stream path
  went 2174 → ~1300 µs. There is simply less serial overhead left to recover.
- **Demo 11's concurrent mode losing its 1.08–1.12x** is the same effect at the point
  where the margin was already inside noise. It was called "launch-overhead-bound and
  only marginally faster" at 6.0.0; at 7.0.0 the margin is gone.
- **Demo 11's graph mode roughly doubling** (148 → 80 µs) is the one clear improvement:
  CUDA-graph replay of a 6-chain JIT+cuBLAS graph is ~1.85x faster than on 6.0.0,
  while demo 07's simpler single-chain graph replay is unchanged.

## Not covered by this batch

`nsys` kernel-time and `ncu` hardware-counter rows (batches 19, 22–28, 30) were **not**
re-captured. Those README rows keep their 6.0.0 provenance and are still labelled as
such. In particular demo 17's kernel-time TFLOP/s ladder and demo 18's kernel-time
table are 6.0.0 numbers; only their wall-clock counterparts are re-measured here.

## Files

| File | What it is |
|---|---|
| `06-streams-{tornado,javaargfile,repeat2..4}.log` | demo 06, five runs |
| `07-graphbenefit-{tornado,javaargfile,repeat2..4}.log` | demo 07, five runs |
| `11-showcase-{tornado,javaargfile,repeat2..4}.log` | demo 11, five runs |
| `12-cutlass-tornado.log` | demo 12 at 1024³, 20 executions per mode |
| `14-warp-tornado.log` | demo 14 at 4096x1024, 20 executions per kernel |
| `17-ladder-tornado.log` | demo 17 at 2048, 10 executions per rung |
| `18-ladder-fp16-tornado.log` | demo 18 at 1024, 10 executions per rung |
