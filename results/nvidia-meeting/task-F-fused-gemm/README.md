# Task F — fused GEMM decision baseline

Unblocked by cloning CUTLASS v3.5.1, which demo 12's hand-written CUDA side
requires and which is not vendored in this repo.

sm_89 / RTX 4090, driver 565.57.01, CUDA 12.6.85, JDK 25.0.2,
TornadoVM 6.0.0-jdk22plus-cuda (SDKMAN release), CUTLASS **v3.5.1**,
Nsight Systems 2024.5.1. fp16 in, fp16 out, row-major, 20 executions,
first execution discarded.

```bash
git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git
export CUTLASS_DIR=$PWD/cutlass
nvcc -arch=sm_89 -O3 -std=c++17 -I$CUTLASS_DIR/include -I$CUTLASS_DIR/tools/util/include \
     -o cuda12 CutlassFusedEpilogue.cu
nsys profile --trace=cuda --sample=none --cpuctxsw=none -o tvm_$S \
  $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . CutlassFusedEpilogue $S $S $S 20
nsys profile --trace=cuda --sample=none --cpuctxsw=none -o cuda_$S ./cuda12 $S $S $S 20
```

Variants compared, all four requested by the task:
1. TornadoVM JIT epilogue (`scale`, `biasRelu`)
2. CUTLASS `libraryTask` **fused** GEMM+bias+ReLU (`LinearCombinationRelu`)
3. **Unfused** library GEMM (`LinearCombination`) + JIT epilogue
4. Hand-written CUDA, same CUTLASS kernels

Raw per-kernel data: `kernel-times.csv`. Fused and unfused CUTLASS kernels are
distinguished by their epilogue template (`LinearCombinationRelu` vs
`LinearCombination`), not by ordering.

## Kernel-only GPU time

### m=n=k=1024 (medium)

| Path | Kernels | Total GPU ns |
|---|---|---|
| TornadoVM **fused** | `scale` 4,316.1 + `LinearCombinationRelu` 30,359.8 | **34,675.9** |
| TornadoVM **unfused** | `scale` 4,316.1 + `LinearCombination` 30,055.8 + `biasRelu` 4,924.9 | **39,296.8** |
| CUDA **fused** | `scaleKernel` 3,504.8 + `LinearCombinationRelu` 30,466.9 | **33,971.7** |
| CUDA **unfused** | `scaleKernel` 3,504.8 + `LinearCombination` 29,734.2 + `biasReluKernel` 4,092.8 | **37,331.8** |

**Fusion saves 11.8% of GPU time on TornadoVM (4,620.9 ns) and one kernel
launch.** The fused epilogue costs only **+1.0%** on the GEMM itself
(30,359.8 vs 30,055.8) against 4,924.9 ns as a separate pass.

### m=n=k=256 (small, latency-oriented)

| Path | Kernels | Total GPU ns |
|---|---|---|
| TornadoVM fused | 1,122.4 + 9,955.1 | **11,077.5** |
| TornadoVM unfused | 1,122.4 + 9,515.2 + 1,172.8 | **11,810.4** |
| CUDA fused | 1,100.8 + 10,092.8 | **11,193.6** |
| CUDA unfused | 1,100.8 + 9,364.6 + 1,313.5 | **11,778.9** |

Fusion saves 6.2% on TornadoVM at this size — less, because the epilogue is a
smaller fraction of a small GEMM.

### m=n=k=4096 (large, throughput-oriented)

| Path | Kernels | Total GPU ns |
|---|---|---|
| TornadoVM fused | `scale` 87,937.4 + `LinearCombinationRelu` 858,509.7 | **946,447.1** |
| TornadoVM unfused | 87,937.4 + `LinearCombination` 856,770.7 + `biasRelu` 82,977.4 | **1,027,685.5** |
| CUDA fused | `scaleKernel` 66,388.2 + 853,759.4 | **920,147.6** |
| CUDA unfused | 66,388.2 + 823,562.6 + `biasReluKernel` 58,897.7 | **948,848.5** |

Fusion saves **7.9%** on TornadoVM, 3.0% on CUDA.

Fusion saving by shape, TornadoVM: 6.2% (256) → 11.8% (1024) → 7.9% (4096). It
peaks at the middle shape; at 4096 the GEMM dominates so the epilogue is a
smaller fraction, and at 256 the epilogue is cheap in absolute terms.

## Splitting by kernel origin — and a launch-geometry confounder

> **Correction.** An earlier revision of this file attributed the JIT-kernel
> ratios below to the task B alignment effect. **That attribution was not
> supported**: the JIT kernels do not share launch geometry between the two
> implementations, so it is not a controlled comparison. Corrected here.

| Kernel | Origin | grid × block, TornadoVM | grid × block, CUDA | same geometry? |
|---|---|---|---|---|
| `LinearCombination{,Relu}` | CUTLASS | (32,32,1) × 128 | (32,32,1) × 128 | **yes** |
| `scale`, `biasRelu` | JIT vs hand-written | 16384 × **1024** | 65536 × **256** | **no** |

Total threads match (16384 × 1024 = 65536 × 256), but the block size does not.

### Valid comparison — the library kernels

Both sides invoke the same CUTLASS kernel, which picks its own launch
configuration, so this *is* controlled:

| Kernel | m=n=k=1024 | m=n=k=4096 | ratio (1024 / 4096) |
|---|---|---|---|
| `LinearCombinationRelu` | 30,359.8 / 30,466.9 | 858,509.7 / 853,759.4 | **1.00 / 1.01** |
| `LinearCombination` | 30,055.8 / 29,734.2 | 856,770.7 / 823,562.6 | **1.01 / 1.04** |

**Library-task integration costs essentially nothing.** TornadoVM reaches the
same CUTLASS kernel at the same performance as a hand-written caller. That is a
clean result and the one worth carrying into the meeting.

### Not a controlled comparison — the JIT kernels

| Kernel | 1024 | 4096 | ratio |
|---|---|---|---|
| `scale` | 4,316.1 / 3,504.8 | 87,937.4 / 66,388.2 | 1.23 / **1.32** |
| `biasRelu` | 4,924.9 / 4,092.8 | 82,977.4 / 58,897.7 | 1.20 / **1.41** |

These ratios **cannot be attributed to code generation or to alignment**, for
two reasons. The block sizes differ (1024 vs 256). And the ratio *grows with
problem size* (1.20 → 1.41), whereas the alignment penalty measured in task B is
bounded at 1.25 by sector arithmetic and does not scale. The growth is the
signature of the occupancy ceiling a 1024-thread block imposes on sm_89 — the
same effect documented for demo 01, where a 1024-thread block tiles once into
1536 threads/SM and caps occupancy at 66.7%.

**What this actually shows is a runtime default, not a compiler defect:**
TornadoVM's default worker grid selects 1024-thread blocks for these elementwise
tasks. Isolating alignment from occupancy here would require re-running the JIT
side with a `GridScheduler` pinned to 256 threads. Not done; recorded as a gap.

## End-to-end wall clock — and an anomaly

Steady-state median wall clock, reported by the demos themselves:

| | TornadoVM | CUDA |
|---|---|---|
| m=n=k=256 fused | 203 µs | — |
| m=n=k=256 unfused | 199 µs | — |
| m=n=k=1024 fused | **342 µs** | 236 µs |
| m=n=k=1024 unfused | **313 µs** | 249 µs |

**On TornadoVM the fused path is slower end-to-end than the unfused path
(342 vs 313 µs), while its GPU time is 11.8% lower.** The hand-written CUDA
side shows the expected ordering (fused 236 < unfused 249 µs).

This inversion is **reported, not explained.** It is not visible in kernel time
and is inconsistent with the kernel-only result, so something in the host path
differs between the two TornadoVM graphs. Attributing it needs a per-execution
API differencing run of both modes, which has not been done. It is exactly the
kind of end-to-end/kernel-only divergence this bundle separates by design, and
no fusion recommendation should be drawn from wall clock alone here.

## Cold vs warm compilation — acceptance criterion 8

From `tvm_profiler_1024.log` (`-Dtornado.profiler=True`, m=n=k=1024). TornadoVM
reports compile time per task graph:

| Phase | First execution (cold) | Steady state (warm) |
|---|---|---|
| `TASK_COMPILE_GRAAL_TIME` | **42,230,354 ns (42.2 ms)** | 0 |
| `TASK_COMPILE_DRIVER_TIME` (NVRTC) | **16,281,649 ns (16.3 ms)** | 0 |
| `TOTAL_TASK_GRAPH_TIME` | 87,100,266 ns (87.1 ms) | 394,381–556,135 ns |
| `TOTAL_KERNEL_TIME` | 464,892 ns | 62,397–91,488 ns |

**Compilation is 58.5 ms of the 87.1 ms cold first execution — 67%.** Graal
front-end work (42.2 ms) is 2.6x the NVRTC/driver back-end cost (16.3 ms).
Steady-state execution is ~0.4 ms, so **cold is ~200x warm**.

This is the price of the JIT specialisation advantage measured in task C, and it
is paid once per task graph. Relevant to the CUDA Tile discussion: any
Tile-based path would have to fit inside a comparable runtime compile budget.

> The profiler perturbs the run it measures — steady-state `TOTAL_TASK_GRAPH_TIME`
> here (~0.4 ms) is above the 0.342 ms wall clock measured without it. Use these
> figures for the cold/warm *split*, not as absolute steady-state timings.

## Files

| File | Contents |
|---|---|
| `kernel-times.csv` | per-kernel instances and mean ns, both implementations, both shapes |
| `tvm_{256,1024}.log`, `cuda_{256,1024}.log` | run output including validation |
| `cuda-cutlass-build.log` | nvcc build of the hand-written side against CUTLASS v3.5.1 |
