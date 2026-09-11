# Track A demos — Java + TornadoVM 6.0.0 CUDA

Presenter-friendly demos, one concept per class, increasing in complexity.
Each demo directory has its own `README.md` with build/run commands, a
`java @argfile` path, expected output, and a stage-failure fallback.

| # | Demo | Concept |
|---|------|---------|
| [00-hello-gpu](00-hello-gpu/) | `Hello.java` | Smallest TornadoVM program: one `@Parallel` task, one `TaskGraph`. |
| [01-first-cuda-kernel](01-first-cuda-kernel/) | `VectorAddKernel.java` | Vector add + `--printKernel` to show the actual generated CUDA source. |
| [02-cuda-runtime-api](02-cuda-runtime-api/) | `CudaGraphReplay.java` | `TornadoExecutionPlan#withCUDAGraph()` — CUDA graph capture/replay from Java, CUDA-only runtime API (not a vendor-library task). |
| [04-cublas-hybrid](04-cublas-hybrid/) | `CuBlasSgemvHybrid.java` | One `TaskGraph`, three stages: JIT `scale` task → cuBLAS `sgemv` library task → JIT `bias` task, all on shared device buffers. |
| [05-cufft-hybrid](05-cufft-hybrid/) | `CuFftLowPassHybrid.java` | One `TaskGraph`, four stages: cuFFT `forward` (R2C) → JIT `lowPass` task → cuFFT `inverse` (C2R) → JIT `normalize` task, a GPU-resident low-pass filter. |
| [06-cuda-streams](06-cuda-streams/) | `CudaStreamsOverlap.java` | `TornadoExecutionPlan#withIntraPlanConcurrency()` — 8 independent pipelines, sequential (1 stream) vs. concurrent (4-stream pool). |
| [07-cuda-graph-benefit](07-cuda-graph-benefit/) | `CudaGraphBenefit.java` | Same 6-stage JIT task-graph run `nograph` vs. `graph` (`withCUDAGraph()`) for 50 executions each — isolates and quantifies the steady-state replay speedup that demo 02's correctness demo doesn't measure. |
| [08-tensor-core-mma](08-tensor-core-mma/) | `TensorCoreMMA.java` | Smallest possible Tensor Core demo: one warp, one `M16N8K16` fp16 tile, exactly one `mma.sync.aligned` instruction (confirmed via `--printKernel`), next to a scalar no-MMA reference kernel with zero `mma.sync` instructions. |
| [11-integrated-showcase](11-integrated-showcase/) | `IntegratedShowcase.java` | Everything at once: JIT kernel + cuBLAS library task (demo 04's shape) × 6 independent chains, run baseline / `withIntraPlanConcurrency()` (demo 06) / `withCUDAGraph()` (demo 07) / both combined (experimental), plus demo 08's Tensor Core `mma.sync` kernel as a bonus stage. |
| [12-cutlass-fused-epilogue](12-cutlass-fused-epilogue/) | `CutlassFusedEpilogue.java` | CUTLASS fused epilogue: `gemmBiasRelu` (GEMM + bias + ReLU in one kernel) vs. `hgemm` + a separate JIT bias/ReLU pass. The fusion is visible in the CUTLASS kernel's own template name (`LinearCombinationRelu`), and the unfused mode shows a second `biasRelu` kernel in the timeline. |
| [13-cudnn-jit-convblock](13-cudnn-jit-convblock/) | `CuDnnConvBlockHybrid.java` | A CNN block alternating vendor and JIT kernels in one graph: JIT `scale` → cuDNN `conv2d` → JIT `addBias` → cuDNN `relu`. Nsight Systems shows all four as separate kernels, the two JIT ones under their own Java method names. |
| [14-warp-async-shared](14-warp-async-shared/) | `WarpAsyncSharedReduce.java` | Three hand-tuned CUDA optimisations written in Java in one kernel: async copy (`cp.async.ca.shared.global`), shared memory (`__shared__`) and warp shuffle (`__shfl_down_sync`) — all three confirmed in the `--printKernel` dump. 26.6x faster than the naive kernel at the kernel level. |
| [15-kernel-time-comparison](15-kernel-time-comparison/) | `KernelTimeComparison.java` | **Kernel time only**, TornadoVM vs hand-written CUDA, measured with `nsys`: three kernels with different bottlenecks, both differences attributed to a specific cause with a standalone probe. Memory-bound gap is the `FloatArray` header offset; compute-bound win is JIT specialisation. |
| [17-matmul-ladder](17-matmul-ladder/) | `MatMulLadder.java` | The same FP32 GEMM six ways — naive `@Parallel`, `KernelContext` tiled, `KernelContext` register-tiled, CUTLASS, cuBLAS, cuBLAS TF32 — validated identically and compared at the kernel level. The register micro-tile buys 3.8x over plain tiling; generated code is within 1-2% of hand-written CUDA on the simple rungs and 1.43x on the register-tiled one. |
| [19-cutile-matmul](19-cutile-matmul/) | `TileMatMul.java` | **CUDA Tile**: the same FP16 GEMM as a naive kernel, a hand-tiled `KernelContext` kernel, and a `TileContext` kernel that says only `tc.mma(a, b, acc)`. Rung 3 compiles through NVIDIA CUDA Tile — `ct::mma`, no inline PTX, tensor cores chosen by `tileiras`. |
| [20-cutile-hybrid](20-cutile-hybrid/) | `TileHybridPipeline.java` | One `TaskGraph`, four stages: `KernelContext` JIT → **CUDA Tile** GEMM → cuBLAS `sgemv` → `@Parallel` JIT, then all four captured into one CUDA graph. A tile task chains and captures like any other task. |
| [21-cutile-flash-attention](21-cutile-flash-attention/) | `TileFlashAttention.java` | Flash attention with online softmax in fifteen lines of Java, ported from NVIDIA's TileGym, against a materialised three-kernel path. 128× `HMMA.16816.F32` in the cubin, no `mma.sync` written by hand. |
| [22-matmul-ladder-fp16-tile](22-matmul-ladder-fp16-tile/) | `MatMulLadderFP16Tile.java` | Demo 18's FP16 ladder with a **`TileContext`** rung inserted between the hand-written `mma.sync` rung and the vendor libraries. At n=1024 the tile rung is 2.8x faster than hand-written MMA and 3.3x slower than cuBLAS, in nsys kernel time. |

## Building and running

```bash
source ../scripts/setup-env.sh      # from a demo directory; or scripts/setup-env.sh from the repo root
cd 00-hello-gpu
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . Hello.java

tornado --classpath . Hello                          # canonical launcher
java @$TORNADOVM_HOME/tornado-argfile -cp . Hello    # reproducibility path
```

No `--enable-preview` anywhere: the pinned `6.0.0-jdk22plus-cuda` SDK is a
non-preview build (`etc/tornado.jdk`: floor 22, preview false), unlike
`6.0.0-jdk21-cuda`, which is JDK-21-only. See the repo README for why that
distinction matters.

**The argfile is not committed.** `tornado --generate-argfile` writes it to
`$TORNADOVM_HOME/tornado-argfile` with absolute, JDK-specific flags
(`-XX:+EnableJVMCI` is required on JDK ≤ 26 and fatal on 27+), so it belongs to
the installed SDK, not to this repo. `scripts/setup-env.sh` regenerates it for
whichever JDK is active.

`bash ../scripts/run-all-demos.sh` compiles and runs all thirteen demos both ways
and exits non-zero on any failure.

### The CUDA Tile demos (19, 20, 21, 22) are separate

They are **not** in `run-all-demos.sh`, because they do not run on the pinned 6.0.0 SDK at
all: `TileContext` does not exist there. They need a TornadoVM built from the cuTile branch,
CUDA Toolkit 13.3+ and driver R580+ (`env/versions.env`, section *CUDA Tile*), and they
compile with `--release 21 --enable-preview` because that branch is a jdk21-dev build.

```bash
TORNADOVM_HOME=<cutile SDK> JAVA_HOME=<jdk21> bash ../scripts/run-cutile-demos.sh
# 12 passed, 0 failed   (4 demos x compile + launcher + java @argfile)
```

## CUDA equivalents

Each demo folder also contains a hand-written CUDA C++ version of the same
program, named after the Java file (`Hello.java` / `Hello.cu`). They exist to be
read side by side, and all thirteen compile and produce the same results:

```bash
bash ../scripts/run-all-cuda.sh   # 13 compiles + 13 runs + 2 probes; CUDA toolkit only, no JDK
```

The three CUDA Tile demos also have `.cu` versions, but they are not in that script either:
they need `nvcc --enable-tile` from CUDA 13.3+, not the system 12.6. Each demo's README has
its own one-line build command, and all three compile and run — they report **kernel time**
with CUDA events, which is the honest companion to the Java demos' wall clock.

Demo 12 needs a CUTLASS checkout (header-only, not vendored):
`git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git`
and `export CUTLASS_DIR=$PWD/cutlass`.

Each demo's README has a **CUDA equivalent** section covering what the CUDA
version has to do by hand, and — where the demos are timed — how the two
compare. The short version: raw CUDA is faster everywhere, the gap is
host-side dispatch overhead rather than kernel quality, and that is exactly
why `withCUDAGraph()` buys TornadoVM 8–10x on demo 07 while buying raw CUDA
only 1.28x. The repo README has the full table.

## Evidence

All twelve demos run on TornadoVM 6.0.0 / JDK 25.0.2 / RTX 4090: **36/36**
checks pass (12 compiles + 12 `tornado` runs + 12 `java @argfile` runs).
Logs for the nine migrated demos: `results/raw/18-tornadovm-6-migration/`.
Logs for demos 12–14, including Nsight Systems kernel summaries:
`results/raw/19-cutlass-cudnn-warp-demos/`.

Demos 12, 13 and 14 each carry an **Nsight Systems section** in their README with
the exact `nsys profile` / `nsys stats` commands and the captured output — for
12 and 14 the profiler, not the wall clock, is what actually shows the effect.

Earlier evidence from the 5.2.1 source-built pin is kept unmodified under
`results/raw/02-hello-kernel/` … `results/raw/17-final-rehearsal/` for
historical comparison. Where a per-demo README cites numbers, it cites the
6.0.0 run and says so.

Nsight Systems traces from the 5.2.1 pass (kernel/memcpy timing, stream
overlap timelines) remain valid as mechanism evidence and are still cited:
`results/raw/06-cuda-streams/`, `results/raw/09-profiling/`. Nsight Compute
hardware-counter metrics remain blocked on this machine — see
`results/failures/08-nsight-compute-permission.md`.

**Simplicity/consistency audit:** `docs/demo-audit-checklist.md`.

**New to this repo?** `docs/profiling-quickstart.md` is a copy-paste runbook:
how to build/run any demo, how to profile with Nsight Systems, and a per-demo
fallback table for presenting live.
