# Batch 45 — Demo 25: the TileContext ladder vs. optimised KernelContext and native CUDA Tile

Captured 2026-09-22. RTX 4090 (sm_89), driver 610.57.04, JDK 25.0.2, TornadoVM 7.0.0
(`sdkman-7.0.0` profile, commit 65eb834), CUDA 13.3 `nvcc`/`tileiras` from the pip wheel
(`nvcc-used.txt`), cuda-tile 1.5.0, Nsight Systems via `/usr/local/cuda-12.6/bin/nsys`.

## Result — kernel time, n = 2048, 10 executions, 3 runs per side, spread ≤ 0.9%

Java (`java-nsys-kernsum.csv`, `java-run{1,2,3}.log`):

| Rung | µs | TFLOP/s |
|---|---|---|
| 1. KernelContext, simple (`kcSimple`) | 1006.0 | 17.1 |
| 2. KernelContext, optimised (`kcOptimised`) | 143.8 | 119.5 |
| 3. TileContext 32×32×32 | 1123.7 | 15.3 |
| 4. TileContext 64×64×64 | 502.5 | 34.2 |
| 5. TileContext 128×128×32 | 529.1 | 32.5 |
| 6. TileContext 128×128×64 | 201.7 | 85.2 |
| 7. TileContext 128×128×64 + `occupancy=2` | 137.4 | 125.0 |
| 8. cuBLAS GemmEx FP16→FP32 (`ampere_s1688gemm_fp16_128x64…`) | 117.3 | 146.5 |

Native, `TileLadder.cu` (`cu-nsys-kernsum.csv`, `cu-run{1,2,3}.log`):

| Rung | µs |
|---|---|
| N1–N4. `native_runtime<32,32,32 / 64,64,64 / 128,128,32 / 128,128,64>` | 1976.6 / 1068.8 / 931.9 / 916.7 |
| N5. + constant `n` + `assume_aligned` (`native_specialised<2048>`) | 327.0 |
| N6. + k-loop as straight-line code (`native_straight_2048`) | 200.8 |
| N7. + `occupancy=2` (`native_hinted_2048`) | 136.2 |
| R. cuBLAS (`cutlass_80_tensorop_s16816gemm_f16_128x256…`) | 115.4 |

The four `native_runtime` instantiations share one name prefix; in each run `nsys` lists them
in the order above (largest first), which is how they are told apart in the CSV.

Every one of the six timed runs ends `All rungs produced the same, correct result`; every
rung reports max abs err 0.0000 against a full CPU reference.

## Findings

1. **KernelContext hand-tuning is worth 7.0×** (1006.0 → 143.8 µs), to 1.23× of cuBLAS. Best
   of 11 validated tilings (`tuning/kernelcontext-tilings.csv`).
2. **The TileContext tile shape is worth 5.6×** (1123.7 → 201.7 µs) with an unchanged kernel
   body; the best of 14 validated shapes is 128×128×64 (`tuning/tilecontext-shapes.csv`).
3. **`occupancy=2` takes it to 137.4 µs**, 1.05× faster than the optimised KernelContext
   kernel and 1.17× of cuBLAS. `occupancy=1` does nothing, `occupancy=4` is 4.5× slower
   (`tuning/tilecontext-launch-hints.csv`, values as printed — the per-hint reports were
   overwritten run to run). `java-printkernel-n256.log` shows the attribute on
   `tile128x128x64Hinted` only.
4. **Idiomatic native CUDA Tile is 1.8–4.5× slower than TileContext**, because TornadoVM's
   generated kernel carries constant extents (`ct::extents{2048, 2048}`), `assume_aligned(…, 16_ic)`
   and a fully unrolled k-loop (`tuning/tornadovm-generated-128x128x64.cu.txt`). Given all
   three, native matches TileContext: 200.8 vs 201.7 µs, and 136.2 vs 137.4 µs with the hint.
   TornadoVM's exact generated kernel compiled natively runs at 200.2 µs
   (`tuning/native-attribution-128x128x64.csv`, `replica`), so the compile and launch path adds
   nothing. Each fact alone does little (aligned only 1024.9 µs, constant only 799.4 µs), and
   `#pragma unroll` does not produce the third (325.8 µs, same as without).

## Suites (`run-all-demos.log`, `run-all-cuda.log`)

- `run-all-demos.sh`: **69 passed, 0 failed, 0 skipped** — 23 demos including 25.
- `run-all-cuda.sh` (CUTLASS 3.5.1): **45 passed, 2 failed** — 25 passes; the failures are
  05 and 24, the toolchain effects recorded in batch 44.

## Tuning bugs found and fixed along the way

- The first generated KernelContext variants all failed validation: a stray string `replace`
  in the generator dropped the buffer selector from every A-tile copy. Fixed before any
  number was taken; `tuning/kernelcontext-tilings.csv` holds only the fixed runs.

## Files

| File | What it is |
|---|---|
| `java-run{1,2,3}.log`, `java-nsys-kernsum.csv` | the Java demo under nsys, n=2048 |
| `cu-run{1,2,3}.log`, `cu-nsys-kernsum.csv`, `cu-build.log`, `nvcc-used.txt` | the native twin under nsys |
| `java-printkernel-n256.log` | generated tile source for all five tile kernels, n=256 |
| `harness-25-*.log`, `run-all-demos.log`, `run-all-cuda.log` | both suites with demo 25 registered |
| `tuning/*.csv` | tile-shape, KernelContext-tiling, launch-hint and native-attribution sweeps |
| `tuning/TileProbe.java`, `KcProbe.java`, `gen.py`, `gen2.py`, `native_variants.cu`, `replica_kernel.inc` | the probes that produced them |
