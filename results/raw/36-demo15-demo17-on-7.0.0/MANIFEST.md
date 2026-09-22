# Batch 36 — Demos 15 and 17 re-measured on TornadoVM 7.0.0

Captured 2026-09-22. Replaces the 6.0.0 numbers that the demo 15 and demo 17
READMEs were built on (batches 21, 22, 28), which batch 35 found stale after
upstream #1066 and #1079 shipped in 7.0.0. Same machine: RTX 4090 (sm_89), driver
610.57.04, CUDA 12.6.85, JDK 25.0.2, TornadoVM `7.0.0-jdk22plus-cuda` (65eb834),
Nsight Systems via `/usr/local/cuda-12.6/bin/nsys`, Nsight Compute 2024.3.2.
Each README's own parameters are used, so the numbers replace theirs one-for-one.

## Demo 15 — kernel time, n = 4,194,304, degree 256, 20 executions, 3 runs per side

`nsys` `cuda_gpu_kern_sum` Avg, mean of 3 runs; every run shows 20 instances per
kernel; spread across runs under 0.7% (`15-nsys-kernsum.csv`).

| Kernel | TornadoVM | CUDA | Ratio | 6.0.0 ratio |
|---|---|---|---|---|
| `elementwise` | 10.98 µs | 10.69 µs | CUDA 1.03x | CUDA 1.31x |
| `stencil` | 11.95 µs | 11.63 µs | CUDA 1.03x | CUDA 1.24x |
| `polynomial` | 35.09 µs | 40.27 µs | TornadoVM 1.15x | TornadoVM 1.13x |

### Memory counters are now identical (`15-ncu-tornado.csv`, `15-ncu-cuda.csv`)

| Kernel | ld sectors | st sectors | ld sectors/req | DRAM read |
|---|---|---|---|---|
| `elementwise` | 524,288 both | 524,288 both | 4 both | 16,780,160 vs 16,779,904 B |
| `polynomial` | 524,288 both | 524,288 both | 4 both | 16,784,256 vs 16,780,928 B |
| `stencil` | 1,835,006 both | 524,288 both | 4.67 both | 16,781,056 vs 16,780,288 B |

On 6.0.0 TornadoVM read 655,360 / 1,966,080 load sectors at 5.00 per request
(batch 22). Now every sector count equals hand-written CUDA's exactly.

The generated code did **not** change: `--printKernel` still indexes at `+ 4L`
(`15-printkernel-degree8.log`). #1066 pads the allocation so that `base + 16` is
32-byte aligned; per the PR, kernels are byte-identical and `withBatch` slots are
deliberately not padded (not tested here).

### The residual ~3% (`15-ncu-instructions-*.csv`)

| Kernel | instructions TornadoVM / CUDA | global ld/st instructions | registers |
|---|---|---|---|
| `elementwise` | 2,359,296 / 1,966,080 = 1.20x | equal | 16 / 16 |
| `stencil` | 5,242,880 / 3,670,016 = 1.43x | equal | 16 / 16 |
| `polynomial` | 35,782,656 / 44,826,624 = 0.80x | equal | 16 / 16 |

With memory traffic identical, extra non-memory instructions are the visible
remaining difference on the two memory-bound kernels. That is consistent with the
~3% but **not shown to cause it**; note `docs/ANALYSIS-GUIDE.md` already retracts
one attribution of the 1.20x (to bounds checks).

### Probes (hand-written CUDA only; TornadoVM-independent)

- `ProbeHeaderAlignment` (its own timer): offset 4 floats vs 0 costs **1.29x**
  (elementwise 11.7 → 15.1 µs) and **1.26x** (stencil 12.5 → 15.7 µs). This is what
  6.0.0 paid and 7.0.0 no longer does.
- `ProbeJitSpecialisation` under `nsys`, 3 runs (`15-nsys-probes.csv`): runtime
  `degree` 36.77 µs, compile-time `degree` 32.23 µs → **1.14x**, the same 1.14x in
  every run. TornadoVM's measured win over hand-written CUDA is 1.15x. Compare the
  ratios: the probe kernel is not the demo kernel, so absolute times differ.
- `cuobjdump -sass`: the CUDA `polynomial` has 11 `BRA`; TornadoVM's generated
  version is a straight FMA chain (`15-printkernel-degree8.log`).

Wall clock (`15-tornado-run.log`): steady-state median 1181 µs; validation PASSED
on both sides at max abs err 0.0000001.

## Demo 17 — kernel time, n = 2048, 10 executions

`nsys` Avg per kernel, mean of 3 runs, 10 instances each, spread ≤ 0.4%
(`17-nsys-kernsum.csv`). Hand-written CUDA: 3 runs of `MatMulLadder.cu`, identical
to the unit each time (`17-cuda-3runs.log`).

| Rung | TornadoVM | GFLOP/s | vs naive | hand-written | ratio |
|---|---|---|---|---|---|
| 1. naive | 3404.7 µs | 5,046 | 1.0x | 3383 µs | 1.006x |
| 2. KernelContext tiled | 2597.5 µs | 6,614 | 1.3x | 2593 µs | 1.002x |
| 3. KernelContext register-tiled | 499.8 µs | 34,373 | 6.8x | 489 µs | **1.022x** |
| 4. CUTLASS task | 426.3 µs | 40,300 | 8.0x | — | — |
| 5. cuBLAS sgemm | 314.7 µs | 54,591 | 10.8x | 314 µs | 1.002x |
| 6. cuBLAS sgemm TF32 | 221.1 µs | 77,702 | 15.4x | 223 µs | 0.991x |

Rung 3 and 4 use the medians of all 13 7.0.0 runs collected today (below). The
register micro-tile is now worth **5.2x** on top of tiling (6.0.0: 3.8x).

### Rung 3 and CUTLASS across 13 runs

- Rung 3: median **499.8 µs**; 11 of 13 within 498.6–510.6 µs; two low runs
  (480.3 µs — batch 35's single audit run — and 487.7 µs).
- CUTLASS: median **426.3 µs**; 10 of 13 within 425.1–427.8 µs; three low runs
  (397.8, 405.2, 416.2 µs).

**Correction to batch 35:** it reported rung 3 at ~0.98x of hand-written CUDA from
its single 480.3 µs run. The settled figure is **1.02x** — identical to the fix's own
dev-build measurement in batch 31 (499.4 µs, 1.021x).

### 6.0.0 vs 7.0.0 vs 7.0.0 with the fix disabled, interleaved (`17-version-and-toggle.csv`)

Three rounds of 6.0.0 → 7.0.0 → 7.0.0 `-Dtornado.cuda.batchGlobalLoads=false`,
same session, all runs `All six rungs produced the same, correct result`:

| Arm | rung 3 | CUTLASS | naive |
|---|---|---|---|
| 6.0.0 | 696.0 / 697.8 / 697.8 µs | 396.1 / 397.5 / 396.2 µs | 3434.9–3440.0 µs |
| 7.0.0 | 510.6 / 499.8 / 499.8 µs | 397.8 / 426.3 / 426.3 µs | 3403.2–3407.2 µs |
| 7.0.0, fix off | 659.4 / 678.9 / 688.9 µs | 404.2 / 411.7 / 409.6 µs | 3403.3–3403.7 µs |

- 6.0.0 reproduces batch 28 (696.3 µs) to within 0.2%.
- **The fix can be toggled live on the pinned release.** Off: rung 3 ~679 µs,
  **1.39x** of hand-written CUDA. On (the default): ~500 µs, 1.02x. The off arm does
  not fully return to 6.0.0's 697.8 µs; other 6.0.0 → 7.0.0 changes are in play.
- **The CUTLASS rung is ~7.6% slower on 7.0.0 in most runs** (426.3 vs 396–398 µs),
  though the kernel template is identical (`md5` of the kernel name matches batch 28).
  The naive rung, flat across all runs, rules out clock drift. **Cause not
  investigated.**

Wall clock (batch 34, n = 2048): rung 3 1767 µs on 6.0.0 → 1637 µs on 7.0.0; the
whole ladder spans 3.7x in wall clock against 15.4x in kernel time.

## Files

| File | What it is |
|---|---|
| `15-nsys-kernsum.csv` | demo 15, 3 nsys runs per side |
| `15-ncu-tornado.csv`, `15-ncu-cuda.csv` | sector and DRAM counters |
| `15-ncu-instructions-tornado.csv`, `15-ncu-instructions-cuda.csv` | instruction and register counters |
| `15-probe-alignment.log`, `15-probe-specialisation.log` | probes, own timer |
| `15-nsys-probes.csv` | specialisation probe under nsys, 3 runs |
| `15-printkernel-degree8.log` | generated CUDA on 7.0.0 |
| `15-tornado-run.log`, `15-cuda-run.log` | validation and wall clock |
| `17-nsys-kernsum.csv`, `17-tornado-run{1,2,3}.log` | demo 17, 3 nsys runs with validation |
| `17-cuda-3runs.log` | hand-written CUDA ladder, 3 runs |
| `17-nsys-flag-check.csv` | 6 extra runs: nsys with and without `--sample=none --cpuctxsw=none` (no effect) |
| `17-version-and-toggle.csv` | 6.0.0 / 7.0.0 / 7.0.0 fix-off, interleaved |
