# TornadoVM on NVIDIA CUDA

Java GPU kernels on TornadoVM's **CUDA backend**, pinned to the **TornadoVM 7.0.0**
release, each paired with a hand-written CUDA C++ equivalent and measured against it.
Write kernels in plain Java, drive CUDA graphs and streams from `TornadoExecutionPlan`,
call cuBLAS/cuFFT/cuDNN/CUTLASS without JNI — and write **CUDA Tile** kernels that never
name a thread.

Compiler engineer? Start at [`docs/NVIDIA-BRIEF.md`](docs/NVIDIA-BRIEF.md).

## Quick start

```bash
# JDK 22+ and the TornadoVM 7.0.0 CUDA SDK, from SDKMAN
curl -s "https://get.sdkman.io" | bash && source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk install java 25.0.2-open
sdk install tornadovm 7.0.0-jdk22plus-cuda

# CUDA 13 userspace toolkit: the CUDA Tile path and 7.0.0's CUTLASS bridge need it
pip install --user nvidia-cuda-nvcc 'cuda-tile[tileiras]' nvidia-cuda-cccl

git clone https://github.com/mikepapadim/tornadovm-devoxx2026-cuda-demos
cd tornadovm-devoxx2026-cuda-demos
source scripts/setup-env.sh       # JAVA_HOME, TORNADOVM_HOME, PATH, LD_LIBRARY_PATH, argfile
tornado --devices
bash scripts/run-all-demos.sh     # all 23 demos, both run paths: 69 checks
```

Every demo runs two ways — the `tornado` launcher and plain `java`:

```bash
cd demos/00-hello-gpu
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . Hello.java
tornado --classpath . Hello
java @$TORNADOVM_HOME/tornado-argfile -cp . Hello
```

`setup-env.sh` regenerates `$TORNADOVM_HOME/tornado-argfile` for the active JDK; it is not
committed because its flags are absolute-path and JDK-specific.

### Things that will bite you

- **Use the `jdk22plus` SDK, not `jdk21`.** `7.0.0-jdk21-cuda` is compiled with preview
  features and runs on JDK 21 only.
- **7.0.0's CUTLASS bridge links CUDA 13** (`libcudart.so.13`). On a CUDA 12 box, demos
  that use a CUTLASS task abort at it unless a CUDA 13 runtime is on `LD_LIBRARY_PATH` —
  `setup-env.sh` adds the pip wheel's and warns if it is missing.
- **Run tile demos with `-Dtornado.recover.bailout=False`** (the runner does). Every
  `TileContext` method has a plain-Java fallback, so otherwise a tile kernel that fails
  to compile runs on the host, prints `correct`, and looks like a pass.
- **Compare kernel time, not wall clock.** Host dispatch and copy-out dominate these demos'
  wall clock; `scripts/compare-ladder.sh 17|18|22` and each demo's README give the `nsys`
  commands.

## SDK profiles

One line in `env/versions.env` picks the TornadoVM the demos run against:

```
TORNADO_SDK_PROFILE=sdkman-7.0.0
```

It names a file in [`env/sdk/`](env/sdk/) that pins the SDK and declares what it can do.
A demo tagged `requires=tile` reports `SKIPPED_REQUIREMENT` — not a failure — on a profile
without the tile API.

| Profile | SDK | Tile API |
| --- | --- | --- |
| **`sdkman-7.0.0`** (default) | released, `sdk install tornadovm 7.0.0-jdk22plus-cuda` | yes |
| `develop` | source build of upstream `develop` in `vendor/tornadovm` | yes |

To switch for one shell, **export** first (`VAR=x source …` does not persist):

```bash
export TORNADO_SDK_PROFILE=develop && source scripts/setup-env.sh
```

## Demos

**†** = CUDA Tile. Verified on TornadoVM 7.0.0, JDK 25, RTX 4090 (sm_89): **69/69** — every
demo compiles and passes under both run paths (`results/raw/45-tile-ladder/run-all-demos.log`).

| # | Demo | What it shows |
|---|------|---------------|
| [00](demos/00-hello-gpu/) | `Hello` | Smallest TornadoVM program: one `@Parallel` task, one `TaskGraph` |
| [01](demos/01-first-cuda-kernel/) | `VectorAddKernel` | Vector add, and `--printKernel` to read the generated CUDA |
| [02](demos/02-cuda-runtime-api/) | `CudaGraphReplay` | `withCUDAGraph()`: capture once, replay and validate |
| [04](demos/04-cublas-hybrid/) | `CuBlasSgemvHybrid` | JIT → cuBLAS `sgemv` → JIT in one graph, on shared buffers |
| [05](demos/05-cufft-hybrid/) | `CuFftLowPassHybrid` | cuFFT → JIT low-pass → inverse cuFFT → JIT, GPU-resident |
| [06](demos/06-cuda-streams/) | `CudaStreamsOverlap` | `withIntraPlanConcurrency()`: one stream vs. a stream pool |
| [07](demos/07-cuda-graph-benefit/) | `CudaGraphBenefit` | The same graph with and without CUDA-graph replay |
| [08](demos/08-tensor-core-mma/) | `TensorCoreMMA` | One `mma.sync.aligned.m16n8k16` from Java, confirmed in PTX and counters |
| [11](demos/11-integrated-showcase/) | `IntegratedShowcase` | JIT + cuBLAS × 6 chains, run plain / concurrent / graph / both |
| [12](demos/12-cutlass-fused-epilogue/) | `CutlassFusedEpilogue` | CUTLASS GEMM+bias+ReLU in one kernel vs. GEMM + a JIT pass |
| [13](demos/13-cudnn-jit-convblock/) | `CuDnnConvBlockHybrid` | JIT and cuDNN kernels alternating in one CNN block |
| [14](demos/14-warp-async-shared/) | `WarpAsyncSharedReduce` | `cp.async`, shared memory and warp shuffles written in Java |
| [15](demos/15-kernel-time-comparison/) | `KernelTimeComparison` | **Kernel time only**, TornadoVM vs. hand-written CUDA, every gap root-caused |
| [16](demos/16-tensor-core-datatypes/) | `TensorCoreDataTypes` | BF16, int8, FP8 e4m3/e5m2 tensor-core MMA from Java |
| [17](demos/17-matmul-ladder/) | `MatMulLadder` | FP32 GEMM six ways, naive → register-tiled → CUTLASS → cuBLAS TF32 |
| [18](demos/18-matmul-ladder-fp16/) | `MatMulLadderFP16` | FP16 GEMM six ways, including `ctx.mma` tensor cores from Java |
| [19](demos/19-cutile-matmul/) † | `TileMatMul` | The same GEMM with threads, then with tiles — no thread index anywhere |
| [20](demos/20-cutile-hybrid/) † | `TileHybridPipeline` | A tile task chained with JIT and cuBLAS tasks and captured into one CUDA graph |
| [21](demos/21-cutile-flash-attention/) † | `TileFlashAttention` | Flash attention with online softmax, ported from NVIDIA's TileGym |
| [22](demos/22-matmul-ladder-fp16-tile/) † | `MatMulLadderFP16Tile` | Demo 18's ladder with a `TileContext` rung |
| [23](demos/23-cutile-row-scan/) † | `CuTileRowScan` | A per-row prefix sum in one call, with a ragged tail |
| [24](demos/24-cutile-histogram/) † | `CuTileHistogram` | Tile atomics: `PartitionView.atomicAdd` into 256 bins |
| [25](demos/25-tile-ladder/) † | `TileLadder` | **The TileContext ladder** vs. a fully optimised KernelContext GEMM vs. native CUDA Tile |

Each demo's README has its build/run commands, expected output, profiling recipe and a
fallback for when it misbehaves on stage. [`demos/README.md`](demos/README.md) has the long
descriptions.

## Headline results (TornadoVM 7.0.0)

RTX 4090, driver 610.57.04, JDK 25.0.2. Measurements on this machine, not general claims.

| Demo | Result | Evidence |
|---|---|---|
| 25 | FP16 GEMM, kernel time at n=2048: TileContext **137.4 µs** with the right tile shape and `occupancy=2`, vs. **143.8 µs** for a fully hand-optimised KernelContext kernel and 117.3 µs for cuBLAS. Tile shape alone is worth 5.6× | `results/raw/45-tile-ladder/` |
| 25 | Native CUDA Tile written with a runtime `n` is **up to 4.5× slower** than TileContext. Given the three facts TornadoVM's JIT knows (constant extents, alignment, unrolled k-loop) it matches exactly | `results/raw/45-tile-ladder/` |
| 17 | Every JIT-compiled GEMM rung within **2%** of hand-written CUDA in kernel time, register-tiled included (1.02×) | `results/raw/43-demo15-demo17-on-7.0.0/` |
| 15 | Memory-bound kernels within **3%** of hand-written CUDA, with identical memory-sector counts; compute-bound kernel **1.15× faster** than CUDA, from JIT specialisation | `results/raw/43-demo15-demo17-on-7.0.0/` |
| 07 | CUDA-graph replay: **8.5–8.8×** faster steady state (≈300 → 35 µs) | `results/raw/41-tornadovm-7-timings/` |
| 11 | Graph vs. baseline on a 6-chain JIT + cuBLAS graph: **9.9–10.5×** | `results/raw/41-tornadovm-7-timings/` |
| 06 | Stream pool vs. one stream: **1.29–1.43×** | `results/raw/41-tornadovm-7-timings/` |

The pattern across the repo: **generated kernels are competitive with hand-written CUDA;
the remaining cost is host dispatch**, which is why `withCUDAGraph()` is the
highest-leverage call in the API.

## CUDA equivalents

Every demo in the table above also has a hand-written CUDA C++ version (`Hello.java` next
to `Hello.cu`) that validates its own output, so a fast wrong kernel cannot pass as a win:

```bash
bash scripts/run-all-cuda.sh     # no JDK needed
```

Demo 12 needs a CUTLASS checkout (`git clone --depth 1 --branch v3.5.1
https://github.com/NVIDIA/cutlass.git && export CUTLASS_DIR=$PWD/cutlass`). The CUDA Tile
equivalents build with `nvcc --enable-tile` from the CUDA 13.3 wheel.

On the sm_89 machine the suite ends **45 passed, 2 failed**, both toolchain effects rather
than demo bugs: 05's `.cu` fails when built by the pip-wheel `nvcc`, which ships no cuFFT
(built with a full toolkit's `nvcc` it passes), and 24's `.cu` needs newer CUDA Tile
headers than this machine has (`partition_view::atomic_add`).

## Upstream issues filed

Found while building these demos and reported with minimal reproducers:

| Issue | Summary | Effect here |
|---|---|---|
| [#1063](https://github.com/beehive-lab/TornadoVM/issues/1063) | `CuDnn.sdpaForward` returned an all-zero result without launching a kernel | demo 13 uses `cudnnConv2d`/`cudnnRelu`; do not demo SDPA live |
| [#1064](https://github.com/beehive-lab/TornadoVM/issues/1064) | CUDA lowering crashes when a ternary precedes an allocation | demo 12's `biasRelu` uses `Math.max` |
| [#1065](https://github.com/beehive-lab/TornadoVM/issues/1065) | `FloatArray`'s 16-byte header misaligned warp-coalesced accesses | diagnosed with demo 15; **fixed in 7.0.0** by [#1066](https://github.com/beehive-lab/TornadoVM/pull/1066) |
| [#1067](https://github.com/beehive-lab/TornadoVM/issues/1067) | A `KernelContext` kernel that fails to compile silently returns wrong results via the sequential fallback | found adding demo 16 |
| [#1105](https://github.com/beehive-lab/TornadoVM/issues/1105) | CUDA Tile FP8 arithmetic does not compile | see [`docs/cutile-api.md`](docs/cutile-api.md) |
| [#1107](https://github.com/beehive-lab/TornadoVM/pull/1107) (PR) | `tornado.recover.bailout` should default to `False` | why every tile run here passes the flag |

## Docs

- [`docs/NVIDIA-BRIEF.md`](docs/NVIDIA-BRIEF.md) — start here: lowering path, measurements, ceiling
- [`docs/ANALYSIS-GUIDE.md`](docs/ANALYSIS-GUIDE.md) — how to read the findings, and the ways to get them wrong
- [`docs/cutile-api.md`](docs/cutile-api.md) — the CUDA Tile API from source
- [`docs/compilation-pipeline.md`](docs/compilation-pipeline.md) — the CUDA pipeline class by class
- [`docs/REPRODUCE-ON-ANOTHER-GPU.md`](docs/REPRODUCE-ON-ANOTHER-GPU.md) — rerun the evidence pack on another GPU
- Talks: [`docs/talk-1-hybrid-api.md`](docs/talk-1-hybrid-api.md), [`docs/talk-2-llm-inference.md`](docs/talk-2-llm-inference.md)

Demos 09 and 10 (GPULlama3.java with Quarkus and LangChain4j) are separate Maven projects
and are not part of the 7.0.0 suite; see talk 2.

## Repository layout

- `demos/` — one directory per demo: the Java source, its hand-written `.cu` twin, a README
- `scripts/` — `setup-env.sh`, `run-all-demos.sh`, `run-all-cuda.sh`, `compare-ladder.sh`,
  `verify.sh` (checks deliverables and cited evidence; no GPU needed)
- `env/` — the pinned environment and SDK profiles
- `results/raw/` — immutable raw evidence; `results/failures/` — captured failures
- `STATE.md` — the study ledger, batch by batch

## Scope

**CUDA only.** No OpenCL, Metal, legacy PTX backend or Babylon anywhere in this repo
(`scripts/verify.sh` enforces it).

## License

[Apache License 2.0](LICENSE). The demo sources are original; TornadoVM is separately
licensed and installed from SDKMAN, not vendored.
