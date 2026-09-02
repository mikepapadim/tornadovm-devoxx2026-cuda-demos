# TornadoVM Devoxx 2026 — CUDA Demos

Java + NVIDIA CUDA demos for two Devoxx 2026 sessions, running on the
**TornadoVM 6.0.0 CUDA release**. Write GPU kernels in plain Java, drive
CUDA-runtime behaviour (graph capture/replay, multi-stream concurrency) from
`TornadoExecutionPlan`, and call cuBLAS/cuFFT without writing a line of JNI.

No source build required — TornadoVM 6.0.0 installs from SDKMAN in one command.

## Quick install

```bash
# 1. SDKMAN (skip if you already have it)
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"

# 2. A JDK 22+ and the TornadoVM 6.0.0 CUDA SDK
sdk install java 25.0.2-open
sdk install tornadovm 6.0.0-jdk22plus-cuda
```

That is the whole toolchain. Point the shell at it and confirm the GPU is visible:

```bash
git clone https://github.com/mikepapadim/tornadovm-devoxx2026-cuda-demos
cd tornadovm-devoxx2026-cuda-demos
source scripts/setup-env.sh   # sets JAVA_HOME / TORNADOVM_HOME / PATH, generates the argfile
tornado --devices
```

```
Number of Tornado drivers: 1
Driver: CUDADriver
  Total number of CUDADriver devices  : 1
  Tornado device=0:0  (DEFAULT)
	CUDA --  [NVIDIA CUDA] -- NVIDIA GeForce RTX 4090
```

Run everything — compiles all twelve demos and runs each both via the `tornado`
launcher and via `java @argfile`:

```bash
bash scripts/run-all-demos.sh
```

### Pick the right SDK: `jdk22plus`, not `jdk21`

TornadoVM 6.0.0 ships two CUDA SDKs and they are **not** interchangeable:

| SDKMAN candidate | JDK contract | Use it? |
|---|---|---|
| `6.0.0-jdk22plus-cuda` | floor JDK 22, **no** preview features | **Yes** — what this repo pins |
| `6.0.0-jdk21-cuda` | JDK 21 **only**, compiled `--enable-preview` | No — preview bytecode pins you to exactly JDK 21 |

Picking `jdk21` and running it on anything but JDK 21 fails immediately with
`This TornadoVM SDK was built for JDK 21 with preview features enabled`. The
`jdk22plus` SDK is the reason this repo no longer needs `--enable-preview`
anywhere: not on `javac`, not on the JVM, not in the argfile.

## Running a demo

Two supported paths, both verified for every demo on every commit
(`scripts/run-all-demos.sh`):

```bash
cd demos/00-hello-gpu
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . Hello.java

tornado --classpath . Hello                          # canonical launcher
java @$TORNADOVM_HOME/tornado-argfile -cp . Hello    # explicit, reproducible
```

`tornado --generate-argfile` writes `$TORNADOVM_HOME/tornado-argfile`. It is
generated into the SDK, not committed here, because its flags are absolute-path
and JDK-specific — `-XX:+EnableJVMCI` is required on JDK ≤ 26 and fatal on JDK 27+.
`scripts/setup-env.sh` regenerates it for whichever JDK is active.

## Track A demos — Hybrid API (`demos/`)

Each demo is one self-contained Java file. Every row below runs on
TornadoVM 6.0.0 / JDK 25 / RTX 4090; logs in
`results/raw/18-tornadovm-6-migration/` (demos 00–11) and
`results/raw/19-cutlass-cudnn-warp-demos/` (demos 12–14).

**Demos 12, 13 and 14 each document how to profile them with Nsight Systems**,
with the exact commands and captured output in their README. For 12 and 14 the
profiler, not the wall clock, is what shows the effect at all.

| # | Demo | Concept | Run |
|---|------|---------|-----|
| [00](demos/00-hello-gpu/) | `Hello.java` | Smallest TornadoVM program: one `@Parallel` task, one `TaskGraph` | `tornado --classpath . Hello` |
| [01](demos/01-first-cuda-kernel/) | `VectorAddKernel.java` | Vector add + `--printKernel` to show the generated CUDA source | `tornado --classpath . VectorAddKernel` |
| [02](demos/02-cuda-runtime-api/) | `CudaGraphReplay.java` | `withCUDAGraph()` — CUDA graph capture + 8 validated replays | `tornado --classpath . CudaGraphReplay` |
| [04](demos/04-cublas-hybrid/) | `CuBlasSgemvHybrid.java` | One graph, three stages: JIT `scale` → cuBLAS `sgemv` → JIT `bias` | `tornado --classpath . CuBlasSgemvHybrid` |
| [05](demos/05-cufft-hybrid/) | `CuFftLowPassHybrid.java` | cuFFT R2C → JIT low-pass → cuFFT C2R → JIT normalize, GPU-resident | `tornado --classpath . CuFftLowPassHybrid` |
| [06](demos/06-cuda-streams/) | `CudaStreamsOverlap.java` | `withIntraPlanConcurrency()` — 1 stream vs. a 4-stream pool | `tornado --classpath . CudaStreamsOverlap 8 32768 65536 8 both` |
| [07](demos/07-cuda-graph-benefit/) | `CudaGraphBenefit.java` | Same graph `nograph` vs. `graph`, 50 executions — quantifies replay speedup | `tornado --classpath . CudaGraphBenefit 4096 6 50 both` |
| [08](demos/08-tensor-core-mma/) | `TensorCoreMMA.java` | One warp, one `M16N8K16` fp16 tile, exactly one `mma.sync.aligned` | `tornado --printKernel --classpath . TensorCoreMMA` |
| [11](demos/11-integrated-showcase/) | `IntegratedShowcase.java` | Everything at once: JIT + cuBLAS × 6 chains, baseline/concurrent/graph/combined | `tornado --classpath . IntegratedShowcase 6 8 8 20 all` |
| [12](demos/12-cutlass-fused-epilogue/) | `CutlassFusedEpilogue.java` | CUTLASS fused epilogue (GEMM+bias+ReLU in one kernel) vs. GEMM + a separate JIT pass | `tornado --classpath . CutlassFusedEpilogue` |
| [13](demos/13-cudnn-jit-convblock/) | `CuDnnConvBlockHybrid.java` | CNN block alternating vendor and JIT kernels: JIT scale → cuDNN conv2d → JIT bias → cuDNN relu | `tornado --classpath . CuDnnConvBlockHybrid` |
| [14](demos/14-warp-async-shared/) | `WarpAsyncSharedReduce.java` | `cp.async` + shared memory + `__shfl_down_sync` from Java, verified in the generated CUDA | `tornado --classpath . WarpAsyncSharedReduce` |

Every demo also runs as `java @$TORNADOVM_HOME/tornado-argfile -cp . <MainClass> [args]`.

**JBang: still not verified here** — `which jbang` returns exit 1 on this
machine (re-checked 2026-09-02). Each demo README documents the expected JBang
shape and says explicitly not to run it live.

## Measured on TornadoVM 6.0.0 (Observed — this machine, this run)

RTX 4090, driver 565.57.01, CUDA 12.6.85, JDK 25.0.2. Not general claims.

| Demo | Metric | Result | Evidence |
|---|---|---|---|
| 00, 01, 02 | Correctness | all pass under both `tornado` and `java @argfile` | `results/raw/18-tornadovm-6-migration/` |
| 04-cublas-hybrid | Correctness | 5/5 iterations match the sequential Java reference | `04-cublas-*.log` |
| 05-cufft-hybrid | Correctness | 5/5 iterations, max abs error `4.77e-7` vs. the analytic signal | `05-cufft-*.log` |
| 06-cuda-streams | Concurrency benefit | sequential 2174–2176 µs vs. concurrent 936–960 µs → **~2.3x** | `06-streams-*.log` |
| 07-cuda-graph-benefit | Graph replay speedup | nograph 292–364 µs vs. graph 36 µs → **8.1x / 10.0x** across the two run paths | `07-graphbenefit-*.log` |
| 08-tensor-core-mma | Generated code | exactly **1** `mma.sync.aligned.m16n8k16` PTX instruction; 0 in the scalar reference | `08-mma-printkernel.log` |
| 11-integrated-showcase | Mode comparison vs. baseline | concurrent 1.08–1.12x, graph 5.37–5.61x, combined 5.66–5.69x | `11-showcase-*.log` |
| 12-cutlass-fused-epilogue | Fused vs. unfused epilogue, GPU kernel time | fused 16547 ns vs. unfused 16106 + 2125 = 18231 ns per execution (~9% less GPU time, one fewer kernel) | `19-…/12-cutlass-nsys-kernsum.csv` |
| 13-cudnn-jit-convblock | Correctness of a 4-stage cuDNN+JIT graph | max abs err `0.000000` vs. the CPU reference | `19-…/13-cudnn-tornado.log` |
| 13-cudnn-jit-convblock | Kernel breakdown | cuDNN conv 61.4%, cuDNN relu 14.6%, JIT `scale` 12.1%, JIT `addBias` 11.9% | `19-…/13-cudnn-nsys-kernsum.csv` |
| 14-warp-async-shared | Optimised vs. naive, wall-clock | 2.06x–2.25x across runs | `19-…/14-warp-tornado.log` |
| 14-warp-async-shared | Optimised vs. naive, **GPU kernel time** | 105668 ns vs. 3971 ns → **26.6x** | `19-…/14-warp-nsys-kernsum.csv` |
| 14-warp-async-shared | Generated CUDA | `cp.async.ca.shared.global`, `cp.async.commit_group`, `cp.async.wait_group`, `__shfl_down_sync`, 2x `__shared__` all present | `19-…/14-warp-printkernel.log` |
| all 12 demos | Compile + run, both paths | **36/36 checks pass** | `run-all-demos.log` |

Compared with the previous 5.2.1 source-built pin, demo 07's replay speedup
rose from 6.47–7.02x to 8.08–10.00x, and demo 06's concurrency benefit became
unambiguous (~2.3x). Demo 11's `concurrent` mode remains launch-overhead-bound
and only marginally faster (1.08–1.12x), unchanged in character from 5.2.1.

**Still blocked, system-wide:** Nsight Compute (`ncu`) hardware counters —
`ERR_NVGPUCTRPERM`, `NVreg_RestrictProfilingToAdminUsers=1`, no passwordless
sudo on this machine. Not a TornadoVM limitation.
`results/failures/08-nsight-compute-permission.md`.

## Upstream issues filed

Two reproducible bugs were found in TornadoVM 6.0.0 while building demos 12–14
and reported upstream with minimal test cases:

| Issue | Summary | Effect here |
|---|---|---|
| [beehive-lab/TornadoVM#1063](https://github.com/beehive-lab/TornadoVM/issues/1063) | `CuDnn.sdpaForward` launches no kernel and silently returns an all-zero result — the SDK's own `BenchmarkSdpa` fails, and Nsight Systems confirms zero cuDNN kernels are launched | demo 13 uses `cudnnConv2d`/`cudnnRelu` instead of SDPA; do not demo SDPA live |
| [beehive-lab/TornadoVM#1064](https://github.com/beehive-lab/TornadoVM/issues/1064) | CUDA lowering crashes with `Node implementing Lowerable not handled: NewInstance` when a ternary precedes an allocation; `Math.max` compiles | demo 12's JIT `biasRelu` uses `Math.max`, not `v > 0 ? v : 0` |

One further problem was **observed but not filed**, because it could not be
reduced to a reliable reproducer: an `@Parallel` reduction over a `ByteArray`
(`sum += data.get(...)` into a local) intermittently emitted
`[Bailout] Running the sequential implementation` and, on one run, produced
wrong results (3855/4096 rows) rather than falling back cleanly. It did not
reproduce under `--debug` or `--fullDebug`. Demo 14's baseline was rewritten to
use `KernelContext` indexing instead, which is stable across every run since.
Worth revisiting with a dedicated reproducer before reporting.

## What changed migrating 5.2.1 → 6.0.0

| | Before (5.2.1, source build) | Now (6.0.0, SDKMAN) |
|---|---|---|
| Install | `git clone` + `make BACKEND=cuda` into `vendor/tornadovm` | `sdk install tornadovm 6.0.0-jdk22plus-cuda` |
| JDK | 21 only, `--enable-preview` on `javac` **and** the JVM | any JDK ≥ 22, no preview flags |
| Env var | `TORNADO_SDK` | `TORNADOVM_HOME` |
| Argfile | `demos/tornado.args`, committed with absolute paths baked in | `$TORNADOVM_HOME/tornado-argfile`, generated by `tornado --generate-argfile` |
| JVMCI | platform module as-is | vendored + `--patch-module jdk.internal.vm.ci` (JDK 22–26), fully vendored on JDK 27+ |
| Native access | — | `--enable-native-access=tornado.runtime` |

No demo source needed an API change: all nine compile unmodified against
`tornado-api-6.0.0`.

## Track B — GPULlama3.java (`demos/09`, `demos/10`) — not migrated

The GPULlama3.java demos and every Track B measurement under `results/` were
captured against the previous source-built `5.2.1-jdk21-dev` pin and are
**deliberately left untouched** by this migration. Their historical findings
stand as recorded, but they do not describe the 6.0.0 runtime:

- FP16 and Q8_0 inference **worked** (153–186 tok/s, 1B model); legacy Q4_0 was
  **blocked** upstream (`UnsupportedOperationException`). `docs/quantization-paths.md`.
- Quarkus and LangChain4j integrations **built** but were **blocked at model load**:
  the 5.2.1 CUDA build was JDK 21 `--enable-preview` bytecode while both
  integrations require JDK 23+, and no single JVM satisfied both.
  `docs/quarkus-langchain4j-integration.md`.

Worth noting: 6.0.0's `jdk22plus` SDK removes the `--enable-preview` half of
that blocker, so the JDK-version conflict is no longer structural. Re-testing
Track B on 6.0.0 is open work, not a result — nothing here claims it now works.

## Talks

1. **TornadoVM Hybrid API: Java + NVIDIA CUDA Libraries** — `docs/talk-1-hybrid-api.md`
2. **Java LLM Inference with TornadoVM** — `docs/talk-2-llm-inference.md`

## Repository layout

- `demos/` — Track A demos, one directory and one README each.
- `scripts/setup-env.sh` — sets `JAVA_HOME`/`TORNADOVM_HOME`, generates the argfile.
- `scripts/run-all-demos.sh` — compiles and runs all 12 demos both ways. Needs a GPU.
- `scripts/verify.sh` — validates deliverables and cited evidence paths. No GPU needed.
- `docs/` — talk drafts, runbook, claims ledger, supporting evidence documents.
- `results/raw/` — immutable raw outputs. `results/failures/` — captured failures.
- `env/versions.env` — the pinned environment. `STATE.md` — durable study state.

## Scope

**CUDA only.** No OpenCL, Metal, the legacy PTX backend, Babylon, or another
GPU framework anywhere in this repo (`scripts/verify.sh` enforces this).
