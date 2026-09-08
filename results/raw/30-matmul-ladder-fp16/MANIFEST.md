# Batch 30 — the FP16 ladder, and why wall clock misreads it

Captured 2026-09-08. RTX 4090 (sm_89), driver 565.57.01, CUDA 12.6.85,
JDK 25.0.2, TornadoVM 6.0.0-jdk22plus-cuda. `demos/18-matmul-ladder-fp16`.

Six implementations of one `C = A * B` in FP16, all validated against a CPU
reference computed over the same FP16-rounded inputs.

## Wall clock and kernel time disagree on the ranking

At `n = 512` the demo's own wall-clock summary reports the tiled rung as
**slower than naive (0.5x)**. At kernel level it is **1.5x faster**. Same run,
opposite conclusion: ~100 us of per-execution host dispatch swamps a kernel that
takes less than that.

This is why `scripts/compare-ladder.sh` exists.

## Kernel time, nsys, n=1024

| rung | GPU avg | GFLOP/s | vs slowest |
|---|---|---|---|
| naive `@Parallel` | 496.6 us | 4,324 | 1.0x |
| `KernelContext` tiled | 337.9 us | 6,355 | 1.5x |
| **`KernelContext` MMA** | 192.3 us | 11,167 | 2.6x |
| CUTLASS `hgemm` | 30.4 us | 70,737 | 16.4x |
| cuBLAS `ampere_fp16_s1688gemm` | 21.5 us | 99,788 | 23.1x |
| cuBLAS `ampere_s1688gemm_fp16` | 21.4 us | 100,250 | 23.2x |

## Counters, ncu (not comparable with the times above)

| kernel | instrs | regs | spill | bank-conf | gmem-stall | issue% |
|---|---|---|---|---|---|---|
| naive | 4,450,304 | 40 | **0** | 61 | **38.89** | 9.30 |
| kcTiled | 2,701,312 | 40 | **0** | 65,161 | 10.40 | 19.50 |
| kcMma | 810,752 | 48 | **0** | 299,463 | 17.77 | 4.57 |
| CUTLASS | 42,720 | 230 | **0** | 12,435 | 0.48 | 11.46 |

Three readings:

- **Nothing spills.** Private accumulators stay in registers on every rung.
- **naive is latency-bound, not compute-bound** — a 38.89 global-memory stall at
  a 9.30% issue rate. Tiling cuts the stall to 10.40 and doubles the issue rate,
  which is the entire point of that rung.
- **kcMma executes 5.5x fewer instructions than kcTiled** because the tensor core
  does the arithmetic, but its 299,463 bank conflicts and 4.57% issue rate say
  the staging around the `mma`, not the `mma`, is the cost.

## Against hand-written CUDA, same algorithms

`cuda-run.log`, same n and rep count:

| rung | TornadoVM | hand-written | ratio |
|---|---|---|---|
| naive | 496.6 us | 439.3 us | 1.13x |
| tiled | 337.9 us | 331.6 us | 1.02x |
| MMA | 192.3 us | 103.4 us | **1.86x** |
| cuBLAS GemmEx FP16 | 21.5 us | 22.4 us | **0.96x** |
| cuBLAS GemmEx FP16->FP32 | 21.4 us | 22.3 us | **0.96x** |

The library rungs at 0.96x are the control: the same cuBLAS kernel on both
sides, so anything near 1.00x confirms the harness. Tiled is at parity. The MMA
rung is 1.86x, the same shape of gap as demo 17's register-tiled rung and
consistent with the diagnosis in `results/raw/29-register-tiled-stalls/` —
these are the kernels that stage through shared memory in a loop.

## The honest gap

`kcMma` reaches 11,167 GFLOP/s against cuBLAS's ~100,000. The Java rung emits
the same `mma.sync.aligned.m16n8k16` the library kernels use, confirmable with
`tornado --printKernel`, so the instruction is not the difference. What the
libraries add is multi-stage pipelining, swizzled shared-memory layouts that
avoid those 299,463 bank conflicts, and a tile shape chosen per problem.

Reaching the tensor core from Java is a few lines. Reaching cuBLAS's throughput
is a different problem, and this ladder measures how much of it the library is
doing.

## Files

| File | Contents |
|---|---|
| `compare-ladder-18.log` | `scripts/compare-ladder.sh 18` output: nsys times and ncu counters |
| `cuda-run.log` | hand-written CUDA side, same shapes |
