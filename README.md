# TornadoVM on NVIDIA CUDA

Java GPU kernels on the **TornadoVM 6.0.0 CUDA release**, each paired with a
hand-written CUDA C++ equivalent and measured against it. Write GPU kernels in
plain Java, drive CUDA-runtime behaviour (graph capture/replay, multi-stream
concurrency) from `TornadoExecutionPlan`, and call cuBLAS/cuFFT without writing
a line of JNI.

No source build required — TornadoVM 6.0.0 installs from SDKMAN in one command.

## Start here: how does the generated code actually compare?

If you only read one demo, read
**[15 — kernel-time comparison](demos/15-kernel-time-comparison/)**. Every other
timed demo here reports wall-clock, which on TornadoVM is dominated by host-side
dispatch. Demo 15 isolates **kernel time alone** across three kernels with
deliberately different bottlenecks, and root-causes both directions of the
result rather than leaving them as "the compiler is better/worse":

| Kernel | TornadoVM | CUDA | |
|---|---|---|---|
| `elementwise` (memory-bound) | 13.94 µs | 10.62 µs | CUDA 1.31x faster |
| `stencil` (memory-bound) | 14.32 µs | 11.55 µs | CUDA 1.24x faster |
| `polynomial` (compute-bound) | 35.24 µs | 39.93 µs | **TornadoVM 1.13x faster** |

- The memory-bound gap is a **data-layout bug, not code generation**: `FloatArray`'s
  16-byte header misaligns every warp-wide access. Nsight Compute counts **5.00
  sectors per request against hand-written CUDA's 4.00** — identical to the same
  CUDA kernel forced to a 4-float offset. Filed as
  [TornadoVM#1065](https://github.com/beehive-lab/TornadoVM/issues/1065).
- The compute-bound win is **JIT specialisation**, not better arithmetic: `degree`
  is a runtime value, so Graal unrolls on it. Give nvcc the same value as a
  template parameter and it lands at 34.7 µs — equal.

**Controlling for both, the generated arithmetic is equivalent.** Demo 08 makes
the same point at the instruction level: TornadoVM and hand-written CUDA each
execute exactly **1 HMMA instruction and 16 tensor-pipe cycles** for the same
`M16N8K16` tile, by hardware counter.

Reading this as a compiler engineer? **[`docs/NVIDIA-BRIEF.md`](docs/NVIDIA-BRIEF.md)**
is the start-here page: the compilation pipeline, what is measured and how, and
where the remaining gaps are.

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

Each demo is one self-contained Java file, paired with a hand-written CUDA C++
equivalent in the same folder (see **CUDA equivalents** below). Every row below runs on
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
| [16](demos/16-tensor-core-datatypes/) | `TensorCoreDataTypes.java` | **BF16, int8, FP8 e4m3 and FP8 e5m2** MMA from Java — every operand type the backend can emit, each validated and counted | `tornado --classpath . TensorCoreDataTypes` |
| [15](demos/15-kernel-time-comparison/) | `KernelTimeComparison.java` | **Start here.** Kernel time only, TornadoVM vs hand-written CUDA over 3 kernels; both deltas root-caused with `nsys` + Nsight Compute counters | `tornado --classpath . KernelTimeComparison` |

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
| 08-tensor-core-mma | **Hardware counters** | **1** HMMA instruction + **16** tensor-pipe cycles, identical to hand-written CUDA; **0** in the scalar reference | `23-ncu-tensor-core-counters/` |
| 16-tensor-core-datatypes | All 4 remaining operand types | BF16, int8, FP8 e4m3, FP8 e5m2 each **PASSED, max abs err 0.00000** over 256 cells, Java and CUDA alike | `26-tensor-core-datatypes/` |
| 16-tensor-core-datatypes | **Hardware counters** | counters match emitted PTX exactly; **int8 dispatches to IMMA**, BF16 and both FP8 formats to **HMMA** | `26-tensor-core-datatypes/` |
| 15-kernel-time-comparison | **Hardware counters** | every TornadoVM global access costs **5.00 sectors/request** vs CUDA's **4.00** — #1065, measured not inferred | `22-ncu-alignment-counters/` |
| 14-warp-async-shared | **Hardware counters** | naive 32.00 sectors/request (3.12% bytes used) → optimised 5.00; **25.6x fewer sectors, 20,760x fewer bank conflicts, 2.3x _more_ instructions** | `24-ncu-demo14-counters/` |
| 14-warp-async-shared | #1065 in a 3rd kernel class | optimised: TornadoVM 163,840 load sectors vs CUDA's 131,072 — **1.250x**, int8 via `cp.async` | `24-ncu-demo14-counters/` |
| 14-warp-async-shared | **Host dispatch, itemised** | ~**8.3 µs/execution** of CUDA driver overhead (excl. genuine kernel wait); **1,620** transfer calls vs CUDA's **41**, but 94.8% of those are one-time start-up | `25-host-dispatch-breakdown/` |
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

**Previously blocked, now resolved (2026-09-03):** Nsight Compute (`ncu`)
hardware counters. Two independent causes — the `ncu` on `PATH` resolves to
`2026.2.1.0`, which cannot connect to this driver at all (use
`/opt/nvidia/nsight-compute/2024.3.2/ncu`), and `ERR_NVGPUCTRPERM` from
`NVreg_RestrictProfilingToAdminUsers=1`, fixed via `modprobe.d` and a reboot.
Neither was a TornadoVM limitation. Diagnosis and fix:
`results/failures/08-nsight-compute-permission.md`. Counter measurements this
unblocked: `results/raw/22-ncu-alignment-counters/` (#1065) and
`results/raw/23-ncu-tensor-core-counters/` (demo 08 Tensor Core).

## CUDA equivalents

Every Track A demo ships a hand-written CUDA C++ version in the same folder
(`Hello.java` next to `Hello.cu`, and so on), so the Java and the CUDA can be
read side by side. All twelve compile and run, and each produces the same
result as its Java counterpart:

```bash
bash scripts/run-all-cuda.sh          # 13 compiles + 13 runs + 2 probes, no JDK needed
```

Demo 12 additionally needs CUTLASS, which is header-only and not vendored here:

```bash
git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git
export CUTLASS_DIR=$PWD/cutlass
```

### What the comparison actually shows

Steady-state medians, same machine, same session, same workloads. **Raw CUDA is
faster everywhere** — the interesting part is the pattern, not the direction:

| Demo | Metric | TornadoVM | CUDA |
|---|---|---|---|
| 06 | sequential → concurrent | 2174 → 960 µs (**2.26x**) | 1243 → 571 µs (**2.18x**) |
| 07 | nograph → graph | 292–364 → 36 µs (**8.1–10.0x**) | 18.6 → 14.5 µs (**1.28x**) |
| 11 | concurrent vs baseline | 1.12x | **2.06x** |
| 11 | graph vs baseline | **5.61x** | 0.93x |
| 13 | conv block, end to end | 367 µs | 69 µs |
| 14 | naive → optimised | 228 → 105 µs (2.17x) | 64 → 14 µs (4.47x) |

**Demo 15 measures kernel time alone** and finds the picture is different once
host overhead is excluded — see its README for the full analysis:

| Kernel | TornadoVM | CUDA | |
|---|---|---|---|
| `elementwise` (memory-bound) | 13.94 µs | 10.62 µs | CUDA 1.31x faster |
| `stencil` (memory-bound) | 14.32 µs | 11.55 µs | CUDA 1.24x faster |
| `polynomial` (compute-bound) | 35.24 µs | 39.93 µs | **TornadoVM 1.13x faster** |

Both differences were attributed to a specific cause with a standalone probe,
rather than left as "the compiler is better/worse":

- The memory-bound gap is **entirely** TornadoVM's 16-byte `FloatArray` header,
  which misaligns warp-coalesced 128-byte accesses. Nsight Compute measures it
  on the generated kernels directly: every global load and store reports **5.00
  sectors per request against hand-written CUDA's 4.00**, a count identical to
  the same CUDA kernel deliberately run at a 4-float offset. Running that
  offset kernel also reproduces the wall-clock ratio (1.28x / 1.27x vs the
  measured 1.31x / 1.24x). Filed as
  [#1065](https://github.com/beehive-lab/TornadoVM/issues/1065).
- The compute-bound win is **entirely** JIT specialisation: `degree` is a task
  argument, so Graal unrolls the FMA chain on its actual value. Give nvcc the
  same information via a template parameter and it lands at 34.7 µs against
  TornadoVM's 35.24 µs — equal.

Controlling for both, **the generated arithmetic is equivalent**. Two things
also follow from the wall-clock table above, both worth saying out loud rather
than hiding:

1. **The gap is host-side dispatch overhead, not kernel quality.** Demo 14's
   TornadoVM *kernel* is 26.6x faster than its naive kernel; its wall-clock is
   only 2.17x faster, because ~100 µs per execution goes elsewhere. That ~100 µs
   breaks down as ~8.3 µs of per-execution CUDA driver overhead (measured by
   differencing execution counts, so start-up cancels) plus host-side runtime
   work a CUDA-only trace cannot see. For the same 40 launches TornadoVM makes
   **1,620 memory-transfer calls against hand-written CUDA's 41**, though 94.8%
   of those are one-time start-up traffic. Demo 08's MMA kernel is
   confirmed by hardware counter, not just generated code: TornadoVM and
   hand-written CUDA each execute **exactly 1 HMMA instruction and 16
   tensor-pipe cycles**, and the scalar control executes **0**.
2. **That is why `withCUDAGraph()` is the highest-leverage call in the API.**
   CUDA graphs remove host dispatch cost. Raw CUDA has little to remove, so
   graphs buy it 1.28x — and on demo 11's small workload they actually cost it
   slightly (0.93x). TornadoVM has a lot to remove, so graphs buy it 8–10x.
   Conversely, stream concurrency helps raw CUDA more (2.06x vs 1.12x on demo
   11), because TornadoVM's dispatch overhead masks the device parallelism
   that streams expose.

### Lines of code

Non-comment, non-blank lines. The Java is usually shorter, but not dramatically
so — and on two demos it is *longer*:

| Demo | Java | CUDA | | Demo | Java | CUDA |
|---|---|---|---|---|---|---|
| 00 hello | 33 | 48 | | 08 tensor-core | 129 | **121** |
| | | | | 16 datatypes | 309 | **217** |
| 01 vector-add | 42 | 65 | | 11 showcase | 266 | 298 |
| 02 cuda-graph | 51 | 78 | | 12 cutlass | 163 | 187 |
| 04 cublas | 100 | 108 | | 13 cudnn | 138 | 186 |
| 05 cufft | 77 | 104 | | 14 warp/async | 175 | **164** |
| 06 streams | 107 | 130 | | | | |
| 07 graph-benefit | 110 | 136 | | | | |

Line count is the weakest part of the argument, so the demo READMEs lead with
what the CUDA version has to get *right* instead: the MMA fragment register
mapping (demo 08), `CUDNN_CROSS_CORRELATION` vs `CUDNN_CONVOLUTION` (demo 13),
pinned host memory for graph capture (demo 02), the generic→shared address cast
for `cp.async` (demo 14), and CUTLASS's tile shapes and bias-broadcast
convention (demo 12). Each of those is a silent-wrong-answer bug if you miss it.

## Upstream issues filed

Three reproducible bugs were found in TornadoVM 6.0.0 while building demos 12–15
and reported upstream with minimal test cases:

| Issue | Summary | Effect here |
|---|---|---|
| [beehive-lab/TornadoVM#1063](https://github.com/beehive-lab/TornadoVM/issues/1063) | `CuDnn.sdpaForward` launches no kernel and silently returns an all-zero result — the SDK's own `BenchmarkSdpa` fails, and Nsight Systems confirms zero cuDNN kernels are launched | demo 13 uses `cudnnConv2d`/`cudnnRelu` instead of SDPA; do not demo SDPA live |
| [beehive-lab/TornadoVM#1064](https://github.com/beehive-lab/TornadoVM/issues/1064) | CUDA lowering crashes with `Node implementing Lowerable not handled: NewInstance` when a ternary precedes an allocation; `Math.max` compiles | demo 12's JIT `biasRelu` uses `Math.max`, not `v > 0 ? v : 0` |

| [beehive-lab/TornadoVM#1065](https://github.com/beehive-lab/TornadoVM/issues/1065) | `FloatArray`'s 16-byte header misaligns warp-coalesced accesses, costing ~25–30% on bandwidth-bound kernels | quantified in demo 15 and measured with Nsight Compute counters (5.00 vs 4.00 sectors/request) |

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

> **Historical — captured 2026-08-21 on TornadoVM `5.2.1-jdk21-dev`, JDK 21.0.2.**
> Every number in this section predates the 6.0.0 migration (2026-09-02) and
> describes a runtime this repo no longer pins. Do not read it as current.

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
- `scripts/run-all-cuda.sh` — builds and runs the hand-written CUDA equivalents. Needs a GPU and the CUDA toolkit, but no JDK.
- `scripts/verify.sh` — validates deliverables and cited evidence paths. No GPU needed.
- `docs/NVIDIA-BRIEF.md` — start-here page for compiler engineers: lowering path, measurements, ceiling.
- `docs/compilation-pipeline.md` — the CUDA pipeline class by class, and where a second emitter would plug in.
- `docs/` — talk drafts, runbook, claims ledger, supporting evidence documents.
- `results/raw/` — immutable raw outputs. `results/failures/` — captured failures.
- `env/versions.env` — the pinned environment. `STATE.md` — durable study state.

## Scope

**CUDA only.** No OpenCL, Metal, the legacy PTX backend, Babylon, or another
GPU framework anywhere in this repo (`scripts/verify.sh` enforces this).

## License

[Apache License 2.0](LICENSE). The demo sources here are original and do not
copy TornadoVM source; Apache-2.0 matches TornadoVM's own primary license.
TornadoVM itself is separately licensed (Apache-2.0, GPLv2-CE and MIT across
its components) and is installed from SDKMAN, not vendored into this repo.
