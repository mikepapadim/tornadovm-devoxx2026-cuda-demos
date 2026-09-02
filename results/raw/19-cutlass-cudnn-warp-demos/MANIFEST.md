# Batch 19 — CUTLASS, cuDNN+JIT, and warp/async/shared demos

Captured 2026-09-02 on the pinned TornadoVM 6.0.0 CUDA SDK
(`6.0.0-jdk22plus-cuda`, JDK 25.0.2, RTX 4090, driver 565.57.01, CUDA 12.6.85).

Three new Track A demos, each run under both supported paths (`tornado` and
`java @argfile`) and each profiled with Nsight Systems 2024.5.1.113.

## Demos

| Demo | What it shows |
|---|---|
| `demos/12-cutlass-fused-epilogue` | CUTLASS fused epilogue (`gemmBiasRelu`) vs. `hgemm` + a separate JIT bias/ReLU pass |
| `demos/13-cudnn-jit-convblock` | cuDNN and JIT kernels alternating in one TaskGraph: JIT scale → cuDNN conv2d → JIT addBias → cuDNN relu |
| `demos/14-warp-async-shared` | `cp.async` + shared memory + warp shuffle from Java, in one kernel |

## Files

| File | What it is |
|---|---|
| `12-cutlass-*.log`, `13-cudnn-*.log`, `14-warp-*.log` | runs under `tornado` and `java @argfile` |
| `*-nsys-kernsum.csv` | `nsys stats --report cuda_gpu_kern_sum` per demo |
| `*-nsys-memsum.csv` | `nsys stats --report cuda_gpu_mem_time_sum` per demo |
| `*.nsys-rep` | raw Nsight Systems traces (open with `nsys-ui`) |
| `14-warp-printkernel.log` | `--printKernel` dump; generated-CUDA evidence for cp.async/shfl/shared |

## Measured (Observed — this machine, this run)

### 12 — CUTLASS fused epilogue

Wall-clock does **not** resolve the effect: fused 317 µs vs. unfused 304 µs
steady-state median at 1024³, i.e. within noise, because each execution is
dominated by the 2 MB D2H copy of C and host-side dispatch. The kernel timeline
is the evidence (512³, 10 executions per mode):

| Kernel | Instances | Avg (ns) |
|---|---|---|
| `cutlass::Kernel2<… LinearCombinationRelu …>` (fused) | 10 | 16547.4 |
| `cutlass::Kernel2<… LinearCombination …>` (unfused GEMM) | 10 | 16105.8 |
| `biasRelu` (unfused JIT epilogue) | 10 | 2124.8 |
| `scale` (JIT, both graphs) | 20 | 1726.4 |

Per execution: fused 16547 ns vs. unfused 16106 + 2125 = 18231 ns — the fused
epilogue is ~1.7 µs (~9%) cheaper and costs one fewer kernel launch. The fusion
is visible in the kernel's own template parameter (`LinearCombinationRelu` vs.
plain `LinearCombination`).

Both modes validate against the sequential Java reference: max abs err
`0.00781`, 0/1048576 cells out of tolerance.

### 13 — cuDNN + JIT conv block

Correctness: **max abs err `0.000000`**, 0/65536 elements out of tolerance, for a
four-stage graph alternating cuDNN library tasks and JIT-compiled Java kernels.

Kernel breakdown (NCHW 4x16x32x32, 16 filters, 10 executions):

| Kernel | Time (%) | Instances | Avg (ns) | Origin |
|---|---|---|---|---|
| `implicit_convolve_sgemm<float, …>` | 61.4 | 10 | 6384.1 | cuDNN conv2d |
| `op_generic_tensor_kernel<(int)1, …>` | 14.6 | 10 | 1513.6 | cuDNN relu |
| `scale` | 12.1 | 10 | 1257.5 | JIT (Java method name) |
| `addBias` | 11.9 | 10 | 1235.2 | JIT (Java method name) |

Four kernels, one per graph stage, ten instances each — and the JIT tasks appear
in an NVIDIA profiler under their Java method names.

### 14 — warp shuffle + cp.async + shared memory

| Measurement | naive | optimised | ratio |
|---|---|---|---|
| wall-clock steady-state median (4096x1024) | 228 µs | 105 µs | 2.17x (2.06–2.25x across runs) |
| **GPU kernel time** (nsys, 10 executions) | 105668.1 ns | 3971.2 ns | **26.6x** |

The wall-clock understates the kernel speedup by ~12x, because each execution
pays a fixed ~100 µs of host-side dispatch and D2H copy that neither kernel can
avoid. Both readings are reported in the demo README; quoting only one would
either undersell the techniques or oversell the demo.

Both kernels validate exactly: max abs err `0.00000`, 0/4096 rows out of
tolerance.

Generated-CUDA evidence from `--printKernel` (`14-warp-printkernel.log`):

| Pattern | Count |
|---|---|
| `cp.async` | 5 |
| `cp.async.commit_group` | 1 |
| `cp.async.wait_group` | 1 |
| `__shfl_down_sync` | 1 |
| `__shared__` | 2 |
| `__syncthreads` | 3 |

## API behaviour established by probing

`KernelContext.asyncCopyToLocal(int[] dst, int dstSlot, <src> src, int srcOffset)`
copies **exactly 4 bytes** per call, and `srcOffset` is in **source-array
elements** — bytes for a `ByteArray`, half-floats for a `HalfFloatArray`. This is
not stated in the API javadoc; it was established here with a dedicated probe
(a `HalfFloatArray` of known values staged at `tid * 2`, which returned elements
`2*tid` and `2*tid + 1`, i.e. element offsets, not byte offsets).

## Bugs found

Two filed upstream with minimal reproducers:

- [beehive-lab/TornadoVM#1063](https://github.com/beehive-lab/TornadoVM/issues/1063)
  — `CuDnn.sdpaForward` launches no kernel and silently returns all zeros. The
  SDK's own `BenchmarkSdpa` prints `Results DO NOT match` with `cudnn=0.0`, and
  `nsys` shows only the JIT `attention` kernel (31 instances) and zero cuDNN
  kernels for the whole process. Deterministic. Demo 13 therefore uses
  `cudnnConv2d`/`cudnnRelu`, which are correct.
- [beehive-lab/TornadoVM#1064](https://github.com/beehive-lab/TornadoVM/issues/1064)
  — CUDA lowering fails with `Node implementing Lowerable not handled:
  NewInstance` when a ternary precedes an allocation
  (`new HalfFloat(v > 0 ? v : 0)`), while `new HalfFloat(Math.max(v, 0))`
  compiles. Deterministic, single-file reproducer.

Observed but **not** filed, for lack of a reliable reproducer: an `@Parallel`
reduction over a `ByteArray` intermittently printed `[Bailout] Running the
sequential implementation` and once produced wrong results (3855/4096 rows)
instead of a clean fallback. It did not reproduce under `--debug` or
`--fullDebug`. Demo 14's baseline was rewritten to use `KernelContext` indexing,
which has been stable across every run since.
