# Batch 20 — hand-written CUDA equivalents

Captured 2026-09-02. A CUDA C++ version of every Track A demo, in the same
folder as its Java source, so the two can be read side by side.

Environment: RTX 4090 (sm_89), driver 565.57.01, CUDA toolkit 12.6.85,
cuDNN 9.23, CUTLASS v3.5.1 (external checkout, not vendored).

## Result

**24/24 checks pass** (12 compiles + 12 runs) via `scripts/run-all-cuda.sh`.
Every CUDA program reproduces its Java counterpart's result:

| Demo | Agreement with the Java version |
|---|---|
| 00 | identical output (`out: [1..8]`) |
| 01 | `Result is correct` |
| 02 | identical, all 8 replays correct, same per-replay values |
| 04 | identical, `output[0]=157.0` every iteration |
| 05 | bit-identical: `maxError=4.76837e-07`, `filtered[0]=0.49999997` |
| 06 | both modes correct |
| 07 | all 50 executions correct in both modes |
| 08 | identical validation, max abs err `0.00000`; exactly 1 tensor-core instruction (`cuobjdump -sass \| grep -c HMMA` → 1, matching the Java `--printKernel` count of 1 `mma.sync.aligned`) |
| 11 | all four modes correct + Tensor Core bonus PASSED |
| 12 | identical validation, max abs err `0.00781` |
| 13 | identical validation, max abs err `0.000000` |
| 14 | identical validation, max abs err `0.00000` |

## Measured comparison (Observed — same machine, same session)

Steady-state medians. Raw CUDA is faster in every demo; the pattern is the
interesting part.

| Demo | Metric | TornadoVM | CUDA |
|---|---|---|---|
| 06 | sequential → concurrent | 2174 → 960 µs (2.26x) | 1243 → 571 µs (2.18x) |
| 07 | nograph → graph | 292–364 → 36 µs (8.1–10.0x) | 18.6 → 14.5 µs (1.28x) |
| 11 | baseline | 831 µs | 99.9 µs |
| 11 | concurrent vs baseline | 1.12x | 2.06x |
| 11 | graph vs baseline | 5.61x | 0.93x |
| 11 | combined vs baseline | 5.69x | 2.73x |
| 12 | fused / unfused | 317 / 304 µs | 233 / 241 µs |
| 13 | conv block end to end | 367 µs | 69 µs |
| 14 | naive → optimised | 228 → 105 µs (2.17x) | 64 → 14 µs (4.47x) |

Interpretation, recorded as analysis rather than measurement:

1. The gap is **host-side dispatch overhead**, not kernel quality. Demo 14's
   TornadoVM kernel is 26.6x faster than its naive kernel (batch 19, nsys) while
   its wall-clock ratio is only 2.17x — ~100 µs per execution is spent outside
   the kernel. Demo 08 shows both toolchains emitting exactly one tensor-core
   instruction for the same tile.
2. That explains the mirror-image profiles on demos 07 and 11. CUDA graphs
   remove host dispatch cost: raw CUDA has little to remove (1.28x on demo 07,
   and a net loss of 0.93x on demo 11's small workload), TornadoVM has a lot
   (8–10x). Stream concurrency exposes device parallelism: raw CUDA benefits
   more (2.06x vs 1.12x on demo 11) because TornadoVM's dispatch overhead masks
   it.

## Lines of code (non-comment, non-blank)

| Demo | Java | CUDA | Demo | Java | CUDA |
|---|---|---|---|---|---|
| 00 | 33 | 48 | 08 | 129 | **121** |
| 01 | 42 | 65 | 11 | 266 | 298 |
| 02 | 51 | 78 | 12 | 163 | 187 |
| 04 | 100 | 108 | 13 | 138 | 186 |
| 05 | 77 | 104 | 14 | 175 | **164** |
| 06 | 107 | 130 | | | |
| 07 | 110 | 136 | | | |

The Java is usually shorter but not dramatically so, and on demos 08 and 14 it
is longer. Line count is therefore the weakest form of the argument and the
demo READMEs do not lead with it.

## Correctness traps the CUDA versions have to get right

Recorded because each is a silent-wrong-answer bug, and each is what the Java
API is actually buying:

- **Demo 08** — the MMA fragment register mapping. Each lane holds 8 halves of
  A, 4 of B and 4 floats of C at ISA-fixed positions; one wrong index gives a
  wrong matrix with no error.
- **Demo 13** — `CUDNN_CROSS_CORRELATION` vs `CUDNN_CONVOLUTION`. The latter
  flips the filter and stops matching a naively written reference.
- **Demo 02 / 11** — host buffers must be **pinned** (`cudaMallocHost`) for
  graph capture; pageable memory replays stale data.
- **Demo 11** — capturing work issued across a stream pool needs an explicit
  event fork before and join after, or `cudaStreamEndCapture` misses it.
- **Demo 14** — `cp.async`'s destination is a shared-window address:
  `__cvta_generic_to_shared`, not a generic pointer.
- **Demo 12** — CUTLASS's fused epilogue applies bias via its `C` operand with
  `beta = 1`, so the length-N bias vector must be broadcast to MxN first. The
  Java `cutlassGemmBiasRelu` takes the vector directly.

## Files

| File | What it is |
|---|---|
| `run-all-cuda.log` | `scripts/run-all-cuda.sh` summary — 24/24 |
| `cuda-runs/*.build.log` | per-demo `nvcc` output |
| `cuda-runs/*.run.log` | per-demo program output |
| `nvcc-version.log` | toolkit version at capture time |
