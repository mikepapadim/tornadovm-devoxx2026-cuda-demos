# Track A demos — Java + TornadoVM 7.0.0 CUDA

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
| [15-kernel-time-comparison](15-kernel-time-comparison/) | `KernelTimeComparison.java` | **Kernel time only**, TornadoVM vs hand-written CUDA, measured with `nsys`: three kernels with different bottlenecks. On 7.0.0 the memory-bound kernels are within 3% of hand-written CUDA — the 1.24–1.31x gap this demo traced to the `FloatArray` header on 6.0.0 was fixed upstream — and the compute-bound kernel is 1.15x *faster*, from JIT specialisation. |
| [16-tensor-core-datatypes](16-tensor-core-datatypes/) | `TensorCoreDataTypes.java` | The same tensor-core GEMM through the four operand types demo 08 does not cover — BF16, int8, FP8 e4m3, FP8 e5m2 — each validated against a CPU reference, with the emitted `mma.sync` variant for each. |
| [17-matmul-ladder](17-matmul-ladder/) | `MatMulLadder.java` | The same FP32 GEMM six ways — naive `@Parallel`, `KernelContext` tiled, `KernelContext` register-tiled, CUTLASS, cuBLAS, cuBLAS TF32 — validated identically and compared at the kernel level. The register micro-tile buys 5.2x over plain tiling; on 7.0.0 every JIT-compiled rung is within 2% of hand-written CUDA, including the register-tiled one that was 1.43x on 6.0.0 — and the fix can be toggled off live to show the gap. |
| [18-matmul-ladder-fp16](18-matmul-ladder-fp16/) | `MatMulLadderFP16.java` | Demo 17's climb in FP16, with the rung FP32 cannot have: a `KernelContext` kernel using `ctx.mma` to reach tensor cores directly from Java, alongside CUTLASS `hgemm` and cuBLAS `GemmEx`. |
| [19-cutile-matmul](19-cutile-matmul/) | `TileMatMul.java` | **CUDA Tile**: the same FP16 GEMM as a naive kernel, a hand-tiled `KernelContext` kernel, and a `TileContext` kernel that says only `tc.mma(a, b, acc)`. Rung 3 compiles through NVIDIA CUDA Tile — `ct::mma`, no inline PTX, tensor cores chosen by `tileiras`. |
| [20-cutile-hybrid](20-cutile-hybrid/) | `TileHybridPipeline.java` | One `TaskGraph`, four stages: `KernelContext` JIT → **CUDA Tile** GEMM → cuBLAS `sgemv` → `@Parallel` JIT, then all four captured into one CUDA graph. A tile task chains and captures like any other task. |
| [21-cutile-flash-attention](21-cutile-flash-attention/) | `TileFlashAttention.java` | Flash attention with online softmax in fifteen lines of Java, ported from NVIDIA's TileGym, against a materialised three-kernel path. 128× `HMMA.16816.F32` in the cubin, no `mma.sync` written by hand. |
| [22-matmul-ladder-fp16-tile](22-matmul-ladder-fp16-tile/) | `MatMulLadderFP16Tile.java` | Demo 18's FP16 ladder with a **`TileContext`** rung inserted between the hand-written `mma.sync` rung and the vendor libraries. At n=1024 the tile rung is 2.8x faster than hand-written MMA and 3.3x slower than cuBLAS, in nsys kernel time. |
| [23-cutile-row-scan](23-cutile-row-scan/) | `CuTileRowScan.java` | A per-row prefix sum is one call, `tc.prefixSum`. The row is 1000 wide against a 128-wide tile, so `loadMasked`/`storeMasked` handle the 24-lane ragged tail and a `[1,1]` carry tile rides the loop. The first demo here with a **scan, a masked load/store and a loop-carried reduction**. |
| [24-cutile-histogram](24-cutile-histogram/) | `CuTileHistogram.java` | A value histogram with **`PartitionView.atomicAdd`** — 4096 tile blocks folding into 256 bins, ~1M contended adds. Also the answer to "what replaces a scatter?": a predicate over the whole tile, since CUDA Tile has no gather/scatter. The only demo using the tile atomics. |

## Building and running

```bash
source ../scripts/setup-env.sh      # from a demo directory; or scripts/setup-env.sh from the repo root
cd 00-hello-gpu
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . Hello.java

tornado --classpath . Hello                          # canonical launcher
java @$TORNADOVM_HOME/tornado-argfile -cp . Hello    # reproducibility path
```

No `--enable-preview` anywhere: the pinned `7.0.0-jdk22plus-cuda` SDK is a
non-preview build (`etc/tornado.jdk`: floor 22, preview false), unlike
`7.0.0-jdk21-cuda`, which is JDK-21-only. See the repo README for why that
distinction matters.

Demos 12, 17 and 18 additionally need a **CUDA 13 runtime** on `LD_LIBRARY_PATH`:
7.0.0's `libtornado-cutlass.so` is linked against `libcudart.so.13`.
`scripts/setup-env.sh` handles this and warns if it cannot find one.

**The argfile is not committed.** `tornado --generate-argfile` writes it to
`$TORNADOVM_HOME/tornado-argfile` with absolute, JDK-specific flags
(`-XX:+EnableJVMCI` is required on JDK ≤ 26 and fatal on 27+), so it belongs to
the installed SDK, not to this repo. `scripts/setup-env.sh` regenerates it for
whichever JDK is active.

`bash ../scripts/run-all-demos.sh` compiles and runs every demo both ways and exits
non-zero on any failure.

### The CUDA Tile demos (19-24)

They are in the same runner now. Which SDK is active is one line -- `TORNADO_SDK_PROFILE`
in `env/versions.env` -- and a demo the active SDK cannot run is **skipped**, not failed:

```bash
source ../scripts/setup-env.sh              # default profile `sdkman-7.0.0`: has the tile API
bash ../scripts/run-all-demos.sh
# 66 passed, 0 failed, 0 skipped            (22 demos x compile + launcher + java @argfile)

export TORNADO_SDK_PROFILE=sdkman-6.0.0     # the released SDK: no tile API
source ../scripts/setup-env.sh
bash ../scripts/run-all-demos.sh
# 48 passed, 0 failed, 6 skipped            (19-24 report SKIPPED_REQUIREMENT)
```

The tile demos need CUDA Toolkit 13.3+ and driver R580+ on top of that; see
[`docs/cutile-api.md`](../docs/cutile-api.md) for the userspace toolchain install.

> `--release 21 --enable-preview` is **no longer needed**. Those flags existed because the
> tile demos were pinned to the `feat/cutile` branch, a jdk21-dev build. PR #1083 is merged
> and shipped in 7.0.0, a jdk22plus build, so the tile demos compile and run on the same
> JDK 25 as everything else. `scripts/run-cutile-demos.sh` is kept only for the old
> feature-branch SDK.

Every run passes `-Dtornado.recover.bailout=False`. Without it a tile kernel that fails to
compile falls back to the JVM, computes the right answer and prints `correct` -- the exact
word the runner greps for -- so a wholly blocked GPU path would score as a pass.

## CUDA equivalents

Each demo folder also contains a hand-written CUDA C++ version of the same
program, named after the Java file (`Hello.java` / `Hello.cu`). They exist to be
read side by side, and every one checks its result against a reference (demo 18's
`.cu` used to print timings only; it validates since 3ade5c5).

```bash
bash ../scripts/run-all-cuda.sh   # 22 compiles + 22 runs + 2 probes; CUDA toolkit only, no JDK
```

Results depend on the machine's toolchain — on the sm_89 box it ends 43 passed,
2 failed (demos 05 and 24, both toolchain effects); see "CUDA equivalents" in the repo
README for why.

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

All 22 demos, CUDA Tile demos included, run on the pinned TornadoVM 7.0.0 / JDK 25.0.2 /
RTX 4090: **66/66** checks pass (22 compiles + 22 `tornado` runs + 22 `java @argfile`
runs, nothing skipped) — `results/raw/44-merged-7.0.0-all-demos/run-all-demos.log`.
Before the CUDA Tile demos were merged in, demos 00-18 alone were 48/48
(`results/raw/40-tornadovm-7-migration/`); wall-clock timings for them were re-measured
on 7.0.0 in `results/raw/41-tornadovm-7-timings/`.
Logs for the nine demos migrated to 6.0.0: `results/raw/18-tornadovm-6-migration/`.
Logs for demos 12–14, including Nsight Systems kernel summaries:
`results/raw/19-cutlass-cudnn-warp-demos/`.

Demos 12, 13 and 14 each carry an **Nsight Systems section** in their README with
the exact `nsys profile` / `nsys stats` commands and the captured output — for
12 and 14 the profiler, not the wall clock, is what actually shows the effect.

Earlier evidence from the 5.2.1 source-built pin is kept unmodified under
`results/raw/02-hello-kernel/` … `results/raw/17-final-rehearsal/` for
historical comparison. Where a per-demo README cites numbers, it cites the
5.2.1 or 6.0.0 run it was measured on and says so — those numbers predate the
7.0.0 pin and have not been re-measured.

Nsight Systems traces from the 5.2.1 pass (kernel/memcpy timing, stream
overlap timelines) remain valid as mechanism evidence and are still cited:
`results/raw/06-cuda-streams/`, `results/raw/09-profiling/`. Nsight Compute
hardware-counter metrics remain blocked on this machine — see
`results/failures/08-nsight-compute-permission.md`.

**Simplicity/consistency audit:** `docs/demo-audit-checklist.md`.

**New to this repo?** `docs/profiling-quickstart.md` is a copy-paste runbook:
how to build/run any demo, how to profile with Nsight Systems, and a per-demo
fallback table for presenting live.
