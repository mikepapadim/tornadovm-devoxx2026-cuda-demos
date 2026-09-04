# Step 4 — kernel-time baseline under nsys (task A), sm_120

Command exactly as the doc specifies, `nvcc -arch=sm_120`, nsys 2025.3.2.474.
Both sides validated: max abs err 0.0000001, 0/4194304 elements out of tol.

## Geometry verified from the trace (not from source), as the doc requires

| side | kernel | grid | block | regs/thread |
|---|---|---|---|---|
| TornadoVM | elementwise / polynomial / stencil | 16384x1x1 | 256x1x1 | 16 |
| CUDA      | elementwise / polynomial / stencil | 16384x1x1 | 256x1x1 | 16 |

Both sides match on grid, block **and** registers per thread. Comparison is valid.

## Result — steady-state median kernel time, first instance excluded

| kernel | TornadoVM (ns) | CUDA (ns) | ratio TVM/CUDA | sm_89 baseline |
|---|---|---|---|---|
| elementwise (mem-bound)  | 38819 | 38114 | **1.019** | 1.31 |
| polynomial  (compute-bound) | 54979 | 60131 | **0.914** | 0.88 |
| stencil     (mem-bound)  | 17761 | 17665 | **1.005** | 1.24 |

## Why the first instance is excluded

`elementwise` instance #1 runs at ~17.9 us, then every later instance at ~38.8 us
— more than 2x slower. This is **not** a warm-up effect and it is **not** a
TornadoVM artifact: the pure-CUDA binary shows the identical shape
(17505 ns then ~38.1 us).

Most consistent explanation: the input buffer is still L2-resident from the H2D
copy on the first pass, and is evicted thereafter by the polynomial and stencil
kernels that run between iterations. Steady state is therefore instances 2..20.

Since both sides show it symmetrically it does not bias the ratio, but taking a
plain 20-instance mean would understate both by ~5% unequally. `nsys stats`'
own Avg column is affected; its Med column is not.

## Reading

The two memory-bound kernels, which cost TornadoVM 31% and 24% on sm_89, are
within 2% and 0.5% of hand-written CUDA on sm_120. The compute-bound polynomial
keeps a TornadoVM advantage (0.91 here vs 0.88 on sm_89) — TornadoVM fully
unrolls the degree-256 loop where nvcc keeps it rolled.

**Caveat, stated because it is load-bearing:** this is not a like-for-like
attribution. sm_120 vs sm_89 changes the GPU, the driver (580.142 vs 565.57.01),
the toolkit (13.0.88 vs 12.6.85), gcc (12.3 vs 11.4) and the OS (23.10 vs 22.04)
all at once. The convergence on memory-bound kernels is real and measured on this
host; *which* of those variables produced it is not established here.
