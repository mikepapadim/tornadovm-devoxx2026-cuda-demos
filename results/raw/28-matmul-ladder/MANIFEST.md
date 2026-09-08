# Batch 28 — the matrix-multiply ladder

Captured 2026-09-08. RTX 4090 (sm_89), driver 565.57.01, CUDA 12.6.85,
JDK 25.0.2, TornadoVM 6.0.0-jdk22plus-cuda, Nsight Systems 2024.5.1.
`demos/17-matmul-ladder`, n = 2048, 10 executions, inputs uploaded once.

The same FP32 `C = A * B` six ways, validated against one CPU reference.

## Kernel time (nsys), TornadoVM side

| Rung | kernel time | GFLOP/s | vs naive |
|---|---|---|---|
| naive `@Parallel` | 3434.6 µs | 5,002 | 1.0x |
| KernelContext tiled | 2613.2 µs | 6,574 | 1.3x |
| KernelContext register-tiled | 696.3 µs | 24,673 | **4.9x** |
| CUTLASS task | 396.3 µs | 43,352 | 8.7x |
| cuBLAS sgemm | 320.1 µs | 53,673 | 10.7x |
| cuBLAS sgemm TF32 | 225.5 µs | 76,173 | **15.2x** |

Tiling alone is worth 1.3x. The register micro-tile is worth **3.8x on top of
it** — the highest-leverage step available to a Java programmer here.

## Against hand-written CUDA, same rungs

| Rung | TornadoVM | hand-written | ratio |
|---|---|---|---|
| naive | 3434.6 µs | 3375 µs | **1.02x** |
| tiled | 2613.2 µs | 2592 µs | **1.01x** |
| register-tiled | 696.3 µs | 488 µs | **1.43x** |
| cuBLAS sgemm | 320.1 µs | 313 µs | 1.02x |
| cuBLAS TF32 | 225.5 µs | 223 µs | 1.01x |

**The generated code is at parity on the simple rungs and diverges only on the
register-blocked one.** That localises the codegen gap to kernels with small
fully-unrollable inner loops over a per-thread accumulator array, rather than
leaving it as a generic slowdown.

The library rungs match to 1-2% because they are the same kernel, which is the
control that makes the rung-3 number believable.

## Kernel names

```
396287 ns  cutlass::Kernel2<DefaultGemmUniversal<float, RowMajor, ...>>
320086 ns  cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>
225536 ns  cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>
```

**cuBLAS dispatches CUTLASS-derived kernels**, and picks `256x128_8x4` for this
shape where TornadoVM's CUTLASS task instantiates one fixed 128x128x32 tile for
every problem. That is why rung 4 trails rung 5, and it is consistent with
`results/nvidia-meeting/cutlass-kalign/`, which excluded alignment as the cause
of the CUTLASS-vs-cuBLAS gap.

## Validation

All six rungs PASSED. Rungs 1-5 at max abs err 0.00000; rung 6 (TF32) at
0.00007 against a deliberately looser 2e-2 tolerance, since TF32 keeps ~10
mantissa bits in the multiply and accumulates in FP32.

## Caveats

Wall-clock output from the program itself is **not** a code-quality measure — it
includes dispatch and a 16 MB copy-out per execution, which compresses the
ladder to a 3.8x spread. Use the kernel times above. One GPU, one shape, one
run; nothing here generalises to a percentage.

## Files

| File | Contents |
|---|---|
| `17-tornado-run.log` | six-rung run with validation |
| `17-tornado-kernsum.csv` | `nsys cuda_gpu_kern_sum` |
| `d17.nsys-rep` | raw trace |
| `17-cuda-run.log` | hand-written CUDA counterpart |
