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

**Run in progress at the time of writing.** Recorded as pending rather than
estimated.

## The cross-check worth noting

Splitting the totals by kernel origin at m=n=k=1024:

| Kernel | Origin | TornadoVM | CUDA | ratio |
|---|---|---|---|---|
| `LinearCombinationRelu` | CUTLASS library | 30,359.8 | 30,466.9 | **1.00** |
| `LinearCombination` | CUTLASS library | 30,055.8 | 29,734.2 | **1.01** |
| `scale` | JIT / hand-written | 4,316.1 | 3,504.8 | **1.23** |
| `biasRelu` | JIT / hand-written | 4,924.9 | 4,092.8 | **1.20** |

**The library kernels are identical because both sides call the same CUTLASS
kernel** — as they must be. The difference is confined to the two small
elementwise kernels, at **1.20–1.23x**, which is the same ratio the alignment
sweep (task B) attributes to `FloatArray`'s 16-byte payload offset on
bandwidth-bound kernels. Task F therefore corroborates task B on an independent
workload, and localises the gap to TornadoVM-generated code rather than to the
library-task integration.

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

## Gap still open

**Cold vs warm compilation time is not yet separated.** A profiler-enabled run
was captured (`tvm_profiler_1024.log`) but the Graal/driver compile-time split
has not been extracted. Task F's acceptance criterion 8 remains unmet.

## Files

| File | Contents |
|---|---|
| `kernel-times.csv` | per-kernel instances and mean ns, both implementations, both shapes |
| `tvm_{256,1024}.log`, `cuda_{256,1024}.log` | run output including validation |
| `cuda-cutlass-build.log` | nvcc build of the hand-written side against CUTLASS v3.5.1 |
