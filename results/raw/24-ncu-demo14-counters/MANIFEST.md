# Batch 24 — demo 14 memory counters, and #1065 in a third kernel class

Captured 2026-09-03. RTX 4090 (sm_89), driver 565.57.01, CUDA 12.6.85,
JDK 25.0.2, TornadoVM 6.0.0-jdk22plus-cuda,
Nsight Compute 2024.3.2. `demos/14-warp-async-shared`, `4096 1024 3`.

Batch 22 measured #1065 on demo 15's three float kernels. This batch profiles
demo 14 — a per-row int8 reduction using `cp.async` + shared memory + warp
shuffle, a completely different kernel shape — and finds the same defect, plus
the mechanism behind demo 14's speedup.

## Result 1 — the naive kernels are identical, on both sides

| metric | TornadoVM `rowSumNaive` | CUDA `rowSumNaive` |
|---|---|---|
| sectors per request (ld) | **32.00** | **32.00** |
| global load sectors | **4,194,304** | **4,194,304** |
| bytes used per sector | **3.12%** | **3.12%** |
| kernel time (under `ncu`) | 120,032 ns | 120,096 ns |
| instructions | 487,040 | 259,456 |

One thread per row, each striding across a whole row, so a warp's 32 lanes land
in 32 different sectors — the worst possible access pattern, and 3.12% is
exactly 1/32: one useful byte per 32-byte sector fetched.

The sector counts are **identical to the byte** and the kernel times match to
0.05%. On the memory path, TornadoVM's generated code and hand-written CUDA are
the same kernel. (TornadoVM issues 1.88x more instructions, but at 32
sectors/request the kernel is entirely memory-bound and it does not show.)

## Result 2 — #1065 again, in the optimised kernels

| metric | TornadoVM `rowSumOptimised` | CUDA `rowSumOptimised` | ratio |
|---|---|---|---|
| sectors per request (ld) | **5.00** | **4.00** | 1.25 |
| global load sectors | **163,840** | **131,072** | **1.250** |
| bank conflicts | 190 | 270 | — |
| kernel time (under `ncu`) | 7,360 ns | 7,040 ns | 1.045 |
| instructions | **1,134,592** | 1,658,880 | 0.68 |

The same 5.00-vs-4.00 sectors per request and the same exact 1.250x sector
ratio as demo 15's float kernels, now in an int8 kernel that reaches memory
through `cp.async` rather than ordinary loads. **The header offset is a
property of the array layout, not of a kernel shape or an element type.**

Two things worth noting here that demo 15 could not show:

- **It costs only 4.5% in time on this kernel** (7,360 vs 7,040 ns) despite the
  full 25% extra sector traffic, because the optimised kernel is no longer
  purely bandwidth-bound — the shuffle and shared-memory reduction dominate.
  Consistent with batch 22's `polynomial` finding: the defect is always present,
  and what it costs depends entirely on whether the kernel is memory-bound.
- **TornadoVM executes 32% fewer instructions than hand-written CUDA here**
  (1,134,592 vs 1,658,880) and is still slightly slower, which is the cleanest
  demonstration in this repo that instruction count is not the thing to optimise
  on this workload — the sector traffic is.

## Result 3 — what the optimisation actually buys

Within TornadoVM, naive → optimised:

| metric | naive | optimised | change |
|---|---|---|---|
| sectors per request | 32.00 | 5.00 | **6.4x fewer** |
| global load sectors | 4,194,304 | 163,840 | **25.6x fewer** |
| bank conflicts | 3,944,216 | 190 | **20,760x fewer** |
| instructions | 487,040 | 1,134,592 | 2.3x **more** |
| kernel time (under `ncu`) | 120,032 ns | 7,360 ns | **16.3x faster** |

The optimised kernel does **2.3x more work in instructions and is 16.3x
faster**, because it moves 25.6x fewer sectors and eliminates essentially all
bank conflicts. That is the whole point of the demo, stated in counters:
`cp.async` + shared memory converts a bandwidth-bound kernel into a
compute-bound one.

DRAM reads are ~4.2 MB for every kernel here (the data is read once regardless).
The entire difference is in the L1/sector path, not in memory traffic.

## Caveats

Kernel times above are measured **under `ncu`** — cold single launches with
flushed caches and no clock boost. They are internally consistent but are not
the numbers to quote: batch 19's `nsys` measurement (26.6x naive → optimised,
`results/raw/19-cutlass-cudnn-warp-demos/14-warp-nsys-kernsum.csv`) remains the
timing result, and the repo README quotes that one.

`smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` reads 0 for both
optimised kernels; that LSU metric does not appear to account for `cp.async`
traffic, so no efficiency claim is made for them from it.

## Files

| File | Contents |
|---|---|
| `ncu-tornado14.csv` | TornadoVM, both kernels |
| `ncu-cuda14.csv` | hand-written CUDA, both kernels |
| `*.stderr.log` | profiler stderr |
