# Step 3 — geometry-controlled 2x2, sm_120

Parallel to [`../../nvidia-meeting/task-B2-geometry-controlled/`](../../nvidia-meeting/task-B2-geometry-controlled/)
(sm_89). Same `GeometryControlled.java` / `.cu`, `n = 16777216` float32,
`out = in * 0.25f + 0.1f`, 2 executions profiled, both sides validated.
`nvcc -arch=sm_120 -O3`, Nsight Compute 2025.3.1.0, CUDA 13.0.88.

> **The doc's Step 3 was not run verbatim.** It writes `tvm_$b.csv` / `cuda_$b.csv`
> into `results/nvidia-meeting/task-B2-geometry-controlled/`, where four files of
> exactly those names are the committed sm_89 baseline. Output is redirected here
> instead. See correction 3 in `../summary.md`.

## The 2x2

Kernel ns is the mean of the two profiled executions.

| run | block | ld sec/req | occupancy | regs/thread | DRAM % peak | kernel ns | instructions |
|---|---|---|---|---|---|---|---|
| TornadoVM @256  | 256  | **5.00** | 78.21% | 16 | 86.00 | 152,656 | 9,437,184 |
| TornadoVM @1024 | 1024 | **5.00** | 54.86% | 16 | 66.50 | 197,680 | 9,437,184 |
| CUDA @256       | 256  | **4.00** | 79.78% | 16 | 86.34 | 145,824 | 9,437,184 |
| CUDA @1024      | 1024 | **4.00** | 55.14% | 16 | 65.73 | 198,896 | 9,437,184 |

Block size and registers per thread are read from `launch__block_size` and
`launch__registers_per_thread`, not from source. Geometry is matched.

## Decomposition

| Effect | Comparison | sm_120 | sm_89 |
|---|---|---|---|
| **Alignment + codegen** (geometry controlled) | TVM@256 / CUDA@256 | **1.047** | 1.075 |
| **Block size** within TornadoVM | @1024 / @256 | **1.295** | 1.131 |
| **Block size** within CUDA | @1024 / @256 | **1.364** | 1.159 |
| **Uncontrolled** | TVM@1024 / CUDA@256 | **1.356** | 1.216 |
| product of the first two | 1.047 x 1.295 | **1.356** | 1.216 |

The two effects still multiply out to the uncontrolled figure exactly. The
structural conclusion of the sm_89 bundle is unchanged on Blackwell.

## What changed, and what did not

**Unchanged — the alignment penalty is real and geometry-independent.** 5.00 vs
4.00 sectors per request at *both* block sizes, the same 1.25x of Step 2,
untouched by launch geometry.

**Unchanged — the block-size penalty is not a TornadoVM property.** Going
256 -> 1024 costs 1.295x on TornadoVM and 1.364x on hand-written CUDA. As on
sm_89, the *hand-written* side is hit slightly harder. Registers stay at 16 in all
four runs, so the drop is the tiling of 1024-thread blocks, not spills.

**Larger — the 256 -> 1024 penalty roughly doubled** (1.13-1.16x on sm_89,
1.30-1.36x here), and DRAM utilisation at block=1024 falls to 66% against 88-91%
on sm_89 while block=256 holds 86%. Whatever the cause, choosing block=1024 costs
more on this host than on the 4090, in both languages.

**Closed — the instruction-count gap.** On sm_89 TornadoVM executed 9,437,184
instructions against hand-written CUDA's 7,864,320 (**1.20x**, 18 vs 15 per warp)
for identical arithmetic. Here both sides execute **9,437,184** — 18 per warp,
identical. TornadoVM's count did not move; nvcc's rose to meet it. This is
`nvcc 13.0.88 -arch=sm_120` against the baseline's `12.6 -arch=sm_89`, so it is a
change in the *reference*, not an improvement in TornadoVM, and toolkit and
architecture move together here — **not attributable to either alone**.

## Caveat, carried over unchanged from the sm_89 bundle

> **1.047 is an `ncu`-condition number and must not be quoted as "TornadoVM is
> 1.047x slower".** Every figure in this 2x2 was taken under Nsight Compute,
> which serialises launches, flushes caches and disallows clock boost. For a
> steady-state figure at matched geometry, quote Step 4 (`../task-A/`) under
> `nsys`: 1.019 / 0.914 / 1.005.

## Files

| File | Contents |
|---|---|
| `tvm_{256,1024}.csv`, `cuda_{256,1024}.csv` | raw Nsight Compute captures |
| `*.err` | JVM deprecation warnings only; no profiler error |
