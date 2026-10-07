# 31 — Tensor-core fragments straight from global memory, and why the launch shape matters

**Concept (read in ~1 minute):** a one-warp-per-16×16-tile FP16 GEMM whose `m16n8k16`
fragments are loaded straight from global memory with
`KernelContext.mmaLoadA/mmaLoadB(HalfFloatArray, row, col, ld)` (TornadoVM PR
[#1195](https://github.com/beehive-lab/TornadoVM/pull/1195)): no shared memory, no fp16
packing, no barrier. The k-loop is three loads and two `mma` calls.

The same kernel body runs with two launch shapes. With no shared memory, the only reuse
of A and B is through the cache, and the launch decides which warps run side by side:

| # | Launch | Warps per group | Output tiles per group |
|---|---|---|---|
| 1 | 1D, 256-thread groups | 8 | 1×8 strip along one row of C |
| 2 | 2D, 128×4 groups | 16 | 4×4 block |

Source: [`MmaGlobalLoads.java`](MmaGlobalLoads.java). Compare with the shared-memory
staging version, demo 18's rung 3 and demo 25's rung 1.

## Build and run

```bash
source ../../scripts/setup-env.sh
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . MmaGlobalLoads.java
tornado --classpath . MmaGlobalLoads 2048 10
```

Arguments: `<n> <executions>`; `n` must be a multiple of 64 (default 2048). The program ends
with `Both launches produced the same, correct result`.

## Kernel time

RTX 5070 Ti (sm_120), TornadoVM `develop` + #1194 + #1195, nsys median, measured with the
harness in
[beehive-lab/tornado-babylon-comparison](https://github.com/beehive-lab/tornado-babylon-comparison)
(variants `kcMmaGlobal`, `kcMmaGlobal2D`, and `kcMma`, the shared-memory version):

| n | shared-memory staging | global loads, 1D launch | global loads, **2D launch** |
|---:|---:|---:|---:|
| 1024 | 221.9 µs | 104.9 µs | **100.0 µs** |
| 2048 | 1,511.9 µs | 841.9 µs | **736.4 µs** |
| 4096 | 20,610.6 µs | 7,988.0 µs | **5,786.1 µs** |
| 8192 | 188,506.3 µs | 203,685.8 µs | **50,940.1 µs** |

- **Loading fragments from global is worth 2.1–3.6×** over staging them through shared
  memory at the same one-warp-per-tile structure (2D launch, n ≤ 4096).
- **The launch shape decides whether it scales.** At n = 8192, B (128 MB of FP16) no longer
  fits in L2. The 1D strip launch loses its reuse and falls to the shared-memory version's
  speed; the 2D launch keeps ~22 TFLOP/s at every size, 3.7× the shared-memory version.

## Requirements

A TornadoVM with PR #1195 (the `HalfFloatArray` overloads of `mmaLoadA`/`mmaLoadB`). The demo
is tagged `requires=mma-global`; `run-all-demos.sh` skips it on profiles that do not set
`TORNADO_HAS_MMA_GLOBAL_LOAD=1`. Tensor cores (sm_80+), CUDA backend.

## Validation

Both launches are checked against a CPU reference over the **whole** output matrix, computed
from the same FP16-rounded random inputs, with a relative tolerance of `1e-4 · √n`.
