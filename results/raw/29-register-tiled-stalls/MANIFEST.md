# Batch 29 — why the register-tiled kernel is slower, measured

Demo 17's ladder shows TornadoVM's register-tiled GEMM trailing the
hand-written CUDA running the identical algorithm. This batch finds the cause.

RTX 4090 (sm_89), driver 565.57.01, CUDA 12.6.85, JDK 25.0.2,
TornadoVM 6.0.0-jdk22plus-cuda, Nsight Compute 2024.3.2. n=1024, block 256 both
sides, single profiled launch.

## What it is not

Every conventional explanation is eliminated by measurement:

| metric | TornadoVM | hand-written CUDA | |
|---|---|---|---|
| local memory ld/st sectors | **0** | **0** | no register spilling on either side |
| registers per thread | 72 | 70 | near-identical pressure |
| instructions executed | **43,714,560** | 44,615,680 | TornadoVM issues **fewer** |
| FFMA thread-instructions | **1,073,741,824** | **1,073,741,824** | *exactly* identical arithmetic |
| shared-memory bank conflicts | **602,365** | 622,879 | TornadoVM has **fewer** |
| LSU instructions | **6,332,416** | 6,461,440 | TornadoVM has **fewer** |
| block size | 256 | 256 | held constant |
| achieved occupancy | **32.94%** | 32.08% | TornadoVM is **higher** |
| **kernel time (ncu)** | **192,608 ns** | **83,840 ns** | **2.3x slower** |

Same arithmetic, fewer instructions, fewer conflicts, higher occupancy, no
spill — and 2.3x slower. So the instructions are stalling, not multiplying.

## What it is

| warp stall reason (per issue active) | TornadoVM | CUDA |
|---|---|---|
| **long_scoreboard** (global memory dependency) | **12.25** | **1.14** |
| short_scoreboard (shared memory) | 2.16 | 1.48 |
| barrier | 0.78 | 0.61 |
| mio_throttle | 0.48 | 0.74 |
| wait (fixed latency) | 0.35 | 0.40 |
| **issue_active, % of peak** | **20.61** | **51.49** |

**`long_scoreboard` is 10.7x higher** and the issue rate is 40% of CUDA's.
Every other stall category is comparable. The kernel is waiting on global
memory that the hand-written version has already hidden.

## Interpretation

The kernel stages tiles from global into shared memory, barriers, then does the
FFMA work. nvcc hoists the **next** k-tile's global loads above the current
tile's compute, so the memory latency overlaps with arithmetic. TornadoVM's
generated CUDA keeps them in program order, so each tile's loads are exposed.

This is consistent with everything above: identical arithmetic, fewer
instructions (no prefetch bookkeeping), and a 10x stall on exactly the
dependency that prefetching removes.

## Scope

Not a general code-generation deficit. Demo 17's other rungs, running the same
algorithm on both sides, come out at 1.01-1.02x. It is specific to kernels that
stage through shared memory in a loop, which is the shape every hand-tiled GEMM
takes.

Note the 2.3x here is an **ncu** figure at n=1024, single cold launch. Demo 17's
`nsys` steady-state figure at n=2048 is 1.43x. Both are real; they are different
measurement modes and must not be quoted interchangeably — see
`results/nvidia-meeting/measurement-mode/`.

## Files

| File | Contents |
|---|---|
| `tornadovm-registertiled-generated.cu` | the generated CUDA for the kernel |
| `MANIFEST.md` | this file |
