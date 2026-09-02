# Profiler quick-start and demo runbook

For a Java developer who has never touched TornadoVM or Nsight before.

The build/run commands in §0–§1 were re-run on 2026-09-02 against the current
pinned environment in `env/versions.env`: TornadoVM `6.0.0-jdk22plus-cuda`
(SDKMAN release), JDK 25.0.2, RTX 4090, driver `565.57.01`. The Nsight
sections (§2 onward) were captured on 2026-08-22 against the earlier 5.2.1
source-built pin (`99549c9862eda8d584e35e99924f9c865501eb3a`, Nsight Systems
`2024.5.1.113`, Nsight Compute `2024.3.2.0`/`2026.2.1.0`); the profiling
mechanics are unchanged, but those specific traces were not re-captured on
6.0.0. Copy-paste them; if one doesn't reproduce, see "If something doesn't
work" at the bottom.

## 0. One-time setup

Install the toolchain once with SDKMAN (see the repo `README.md` for the full
quick-install), then point the shell at it:

```bash
sdk install java 25.0.2-open
sdk install tornadovm 6.0.0-jdk22plus-cuda

cd /path/to/tornadovm-devoxx2026-cuda-demos
source scripts/setup-env.sh
echo "$TORNADOVM_HOME"   # sanity check: must print a path, not empty
tornado --devices        # must list exactly one CUDA device
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
```

Install `6.0.0-jdk22plus-cuda`, **not** `6.0.0-jdk21-cuda`: the latter is
compiled with JDK 21 preview features and runs on JDK 21 only.

Run the `nvidia-smi` check before any timed or profiled run — a busy GPU
(another process already using it) will silently skew every number below.
Every run captured in this repo was taken with the GPU idle (`0 %`, `4 MiB`).

## 1. Run a demo (two supported ways)

Every demo under `demos/` follows the same shape. Using `demos/00-hello-gpu`
as the example (swap in any other demo directory and its main class):

```bash
cd demos/00-hello-gpu
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . Hello.java
```

No `--enable-preview`: the pinned `jdk22plus` SDK is a non-preview build. The
wildcard classpath covers the vendor-library jars some demos need
(`tornado-cublas-6.0.0.jar` for `04-cublas-hybrid`, `tornado-cufft-6.0.0.jar`
for `05-cufft-hybrid`).

**Way 1 — `tornado` launcher (canonical, resolves the CUDA backend automatically):**

```bash
tornado --classpath . Hello
```

**Way 2 — `java @arg-file` (reproducibility form, per `docs/run-conventions.md`):**

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp . Hello
```

The argfile is **not** committed to this repo. `tornado --generate-argfile`
writes it into the installed SDK at `$TORNADOVM_HOME/tornado-argfile`, because
its flags are absolute-path and JDK-specific (`-XX:+EnableJVMCI` is required on
JDK ≤ 26 and fatal on JDK 27+). `scripts/setup-env.sh` regenerates it for
whichever JDK is active, so switching JDK needs no manual step.

To check every demo at once, both ways: `bash scripts/run-all-demos.sh`
(9 compiles + 9 `tornado` runs + 9 `java @argfile` runs; exits non-zero on any
failure).

**JBang: still not verified on this machine.** `which jbang` returns exit 1 as
of 2026-09-02 (same finding as every earlier task). Every demo's `README.md`
documents the untested-but-expected JBang invocation shape (e.g.
`jbang -cp "$TORNADOVM_HOME/share/java/tornado/*"
--java-opts="@$TORNADOVM_HOME/tornado-argfile" Hello.java`) — do not run these
live in front of an audience; they have never been executed on this
environment.

Add `--enableProfiler console` to the `tornado` command on any demo to print
a per-task-graph JSON block confirming `"BACKEND": "CUDA"`,
`"DEVICE": "NVIDIA GeForce RTX 4090"` — the fastest way to prove a demo is
really running on the GPU, no external profiler needed.

## 2. Nsight Systems (`nsys`) — timeline, kernel/API/memcpy timing

This is the profiler to reach for first: no special permissions needed, works
against any demo or against GPULlama3.java as-is.

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
  -o /tmp/nsys-<tag> tornado --classpath . Hello
```

Then parse it into readable CSVs:

```bash
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_kern_exec_sum \
  --format csv --output /tmp/nsys-<tag> /tmp/nsys-<tag>.nsys-rep
```

What each report answers:

| Report | Answers |
|---|---|
| `cuda_gpu_kern_sum` | Which kernel took the most GPU time, how many times it ran, avg/median/min/max duration. |
| `cuda_api_sum` | CPU-side launch overhead — `cuLaunchKernel`/`cuGraphLaunch` call counts and timings, plus one-time `cuCtxCreate_v2` setup cost. |
| `cuda_gpu_mem_time_sum` | H2D vs. D2H memcpy time — spot data-movement-bound demos. |
| `cuda_kern_exec_sum` | Per-launch API/queue/kernel-exec breakdown — separates "issuing the launch" from "waiting" from "actually computing." |
| `cuda_gpu_trace` | Raw per-launch timestamps — needed to compute time-to-first-token / warm-up-vs-steady-state, e.g. `results/raw/12-llm-profiling/PROFILING-SUMMARY.md` §6. |

**Read this first, every time:** `cuCtxCreate_v2` (CUDA context creation) is
consistently 89–99% of `Time (%)` in every trace captured in this repo,
because the trace spans the whole JVM process (JIT warm-up included), not
just the workload. Never read the raw `Time (%)` column as "compute cost"
without checking whether `cuCtxCreate_v2` is sitting at the top of it — see
`results/raw/09-profiling/PROFILING-SUMMARY.md` and
`results/raw/12-llm-profiling/PROFILING-SUMMARY.md` §4 for worked examples.

Verified live on this machine (2026-08-22, `demos/00-hello-gpu`): both
commands above ran cleanly and produced a non-empty `cuda_gpu_kern_sum.csv`
with `addOne` at 1,600 ns — confirming the recipe genuinely captures real
CUDA activity, not a CPU fallback (watch for a red `[Bailout] Running the
sequential implementation` line in the run's own stdout — if that appears,
the trace will contain no CUDA kernel data at all, and `nsys stats` reports
`PROCESSED (EMPTY RESULTS)`, seen firsthand while preparing this guide).

## 3. Nsight Compute (`ncu`) — hardware counters (occupancy, GPU util %, tensor-pipe activity)

**Known blocker on this machine, re-verified 2026-08-22 (fourth confirmation
after tasks 08/09/12):**

```bash
# Default `ncu` on PATH resolves to the newest install (2026.2.1.0) —
# cannot connect to this machine's driver (565.57.01) at all:
ncu --target-processes all -k "regex:^<kernelName>$" --launch-count 1 \
  --metrics gpu__time_duration.sum tornado --classpath . <MainClass>
# ==ERROR== Nsight Compute failed to connect to the CUDA driver (stub libcuda.so[.1] on path?).

# The older, driver-era-matching install connects but is refused by the
# driver's default admin-only counter-access restriction:
/opt/nvidia/nsight-compute/2024.3.2/ncu --target-processes all \
  -k "regex:^<kernelName>$" --launch-count 1 \
  --metrics gpu__time_duration.sum tornado --classpath . <MainClass>
# ==PROF== Connected to process <pid>
# ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
# NVIDIA GPU Performance Counters on the target device 0.
```

Both re-run and reproduced against `demos/08-tensor-core-mma`
(`gemmMMASingleTile`) on 2026-08-22 exactly as documented in
`results/failures/08-nsight-compute-permission.md`. Root cause: the NVIDIA
driver's `NVreg_RestrictProfilingToAdminUsers=1` default; this environment
has no passwordless `sudo`, and changing the kernel module parameter needs a
reboot (out of scope for an autonomous/reversible-action run — see
`results/failures/08-nsight-compute-permission.md` for the full analysis).

**If a future environment has counter access** (root/sudo, or
`NVreg_RestrictProfilingToAdminUsers=0`): use the `2024.3.2` install
directly (not the bare `ncu` on `PATH`), and pick a kernel with real
workload/duration — the trivial `Hello`/`addOne` kernel (8 elements,
~1.6 µs) was observed during this guide's own verification to make
TornadoVM print `[Bailout] Running the sequential implementation` when run
under `ncu`'s instrumentation and produce "No kernels were profiled" instead
of the permission error; `TensorCoreMMA`, `CuBlasSgemvHybrid`, or the FFN
kernels in `results/raw/12-llm-profiling/PROFILING-SUMMARY.md` §8 do not
have this problem and are ready-to-rerun target kernels.

**Until then: `ncu` metrics (occupancy, GPU utilization %, memory throughput
%, instruction mix, tensor-pipe activity) are blocked on this machine for
every demo and for GPULlama3.java** — do not claim any such number live.
`nsys` (§2 above) and `--printKernel` generated-code inspection (§5 below)
are the two working alternatives already used throughout this repo.

## 4. Report locations

| What | Where |
|---|---|
| Per-demo timed-run logs (correctness, `--enableProfiler console` JSON) | `results/raw/<NN-demo>/*.log` — path listed in that demo's own `README.md` |
| Dedicated Nsight Systems pass across demos 04/05/07/08 | `results/raw/09-profiling/PROFILING-SUMMARY.md` + raw `.nsys-rep`/CSVs in the same directory |
| Nsight Systems timeline for CUDA-streams overlap (demo 06) | `results/raw/06-cuda-streams/nsys-{sequential,concurrent}.nsys-rep`, `nsys-timeline-evidence.txt` |
| GPULlama3.java Nsight Systems profiling | `results/raw/12-llm-profiling/PROFILING-SUMMARY.md` |
| Nsight Compute blocker writeup (canonical) | `results/failures/08-nsight-compute-permission.md` |
| `--printKernel` generated CUDA/PTX source | `results/raw/02-hello-kernel/vectoraddkernel-run.log` (demo 01), `results/raw/08-tensor-core-mma/tensorcoremma-printkernel.log` (demo 08) |

## 5. `--printKernel` — generated-code evidence (works everywhere, no profiler needed)

When `ncu` is blocked, this is the fallback for "prove this really compiled
to CUDA/PTX," used throughout this repo (e.g. demo 08's Tensor Core claim
rests on this, not on `ncu`):

```bash
tornado --printKernel --classpath . <MainClass>
```

Prints the actual generated `extern "C" __global__ void ...` CUDA source (or,
for MMA kernels, the inline `mma.sync`/`ldmatrix` PTX asm) to stdout before
`nvcc`/`ptxas` compiles it.

## 6. Per-demo quick reference

| Demo | Build needs | Profiling workload used in this repo | Fallback if the live run fails |
|---|---|---|---|
| [00-hello-gpu](../demos/00-hello-gpu/) | `tornado-api` only | n/a (too small/fast to profile meaningfully) | `results/raw/02-hello-kernel/hello-run*.log`; re-run `tornado --devices` first |
| [01-first-cuda-kernel](../demos/01-first-cuda-kernel/) | `tornado-api` only | `--printKernel` output | `results/raw/02-hello-kernel/vectoraddkernel-run.log` |
| [02-cuda-runtime-api](../demos/02-cuda-runtime-api/) | `tornado-api` only | not separately profiled | `results/raw/03-cuda-runtime-api/cudagraphreplay-run.log` |
| [04-cublas-hybrid](../demos/04-cublas-hybrid/) | `+ tornado-cublas` jar | `nsys profile ... CuBlasSgemvHybrid 512 512 10` | `results/raw/04-cublas-hybrid/cublassgemvhybrid-run.log` |
| [05-cufft-hybrid](../demos/05-cufft-hybrid/) | `+ tornado-cufft` jar | `nsys profile ... CuFftLowPassHybrid 4096 16 20` (do not use `n=65536` — known CPU-fallback bug, see `STATE.md` batch 09) | `results/raw/05-cufft-hybrid/cufftlowpasshybrid-run.log` |
| [06-cuda-streams](../demos/06-cuda-streams/) | `tornado-api` only | `nsys` sequential-only and concurrent-only traces (see report-locations table) | `results/raw/06-cuda-streams/cudastreamsoverlap-run.log` + pre-captured CSVs |
| [07-cuda-graph-benefit](../demos/07-cuda-graph-benefit/) | `tornado-api` only | `nsys profile ... CudaGraphBenefit 4096 6 50 both` | captured logs + demo 02's simpler correctness-only demo |
| [08-tensor-core-mma](../demos/08-tensor-core-mma/) | `tornado-api` only | `--printKernel` (works); `ncu` blocked (see §3) | `results/raw/08-tensor-core-mma/tensorcoremma-printkernel.log` |
| [09-quarkus-langchain4j-gpullama3](../demos/09-quarkus-langchain4j-gpullama3/) | n/a — build-only, blocked | n/a | Do not attempt live; explain from `docs/quarkus-langchain4j-integration.md`'s exact error transcript instead |
| [10-langchain4j-gpullama3](../demos/10-langchain4j-gpullama3/) | n/a — build-only, blocked | n/a | Same as above |

Full detail for every row is in that demo's own `README.md` under
`## If the demo fails on stage` / `## Fallback if the live demo fails` —
this table is a pointer, not a replacement.

## 7. GPULlama3.java quick-start

> **Track B was not migrated to TornadoVM 6.0.0.** This section describes the
> earlier source-built 5.2.1 pin it was captured against and still assumes a
> `vendor/tornadovm` checkout with `setvars.sh`. Do not mix it with the
> SDKMAN-based §0 setup above.

```bash
source vendor/tornadovm/setvars.sh
cd vendor/GPULlama3.java && source ./set_paths
./llama-tornado --gpu --cuda \
  --model "$GPULLAMA3_MODEL_USED" \
  --prompt "What is the capital of France?" --seed 7 --max-tokens 32
```

Profiler JSON (confirms CUDA backend, not a claimed flag):

```bash
./llama-tornado --gpu --cuda --profiler \
  --model "$GPULLAMA3_MODEL_USED" \
  --prompt "What is the capital of France?" --seed 7 --max-tokens 32
```

Exact reproducible `java` invocation (no `@arg-file` is generated by
GPULlama3.java itself):

```bash
./llama-tornado --gpu --cuda --model "$GPULLAMA3_MODEL_USED" \
  --prompt "..." --show-command
```

Nsight Systems trace (capture the exact command from `--show-command` above
into `java-command.txt`, then):

```bash
nsys profile --trace=cuda,nvtx,osrt \
  --output=/tmp/nsys-llm --force-overwrite=true \
  bash -c "$(cat java-command.txt)"
```

Full worked example with parsed CSVs and presenter interpretation:
`results/raw/12-llm-profiling/PROFILING-SUMMARY.md`.

**Build blocker to know about before presenting:** GPULlama3.java's `pom.xml`
pins a different TornadoVM release than this repo's pinned SDK. The working
build command is:

```bash
./mvnw clean install -DskipTests -Dtornadovm.base.version=5.2.1 -Djdk.version.suffix=-jdk21-dev
```

`mvn clean` is required — a non-clean rebuild after only changing `-D`
properties leaves stale `.class` files and silently reproduces the failure.
See `docs/gpullama3-reproduction.md` §2 for the full root-cause writeup.

Quantized-model paths (FP16/Q8_0 work, Q4_0 is blocked upstream, not a local
bug): `docs/quantization-paths.md`. JBang: not verified, same finding as
every demo above.

## 8. Presenter cheat sheet — what to point out live

- **First**, before any timing claim: `--enableProfiler console` JSON
  `"BACKEND": "CUDA"` / `"DEVICE": "..."` — the cheapest, fastest proof a
  demo is really on the GPU.
- **cuBLAS/cuFFT hybrid demos (04/05):** the interesting thing is not "GPU
  is faster" but "one `TaskGraph`, JIT Java tasks and vendor-library calls
  share device buffers with zero host round-trips in between" —
  `--enableProfiler console` shows every stage on the same `DEVICE`.
- **CUDA streams (06):** point at the Nsight Systems timeline, not a
  number — sequential mode uses exactly 1 CUDA stream with back-to-back
  launches; concurrent mode spreads across exactly 4 streams (the default
  COMPUTE-pool size) with genuinely overlapping windows. The *mechanism* is
  the reliable story; the wall-clock speedup is workload/GPU/driver-
  dependent and not guaranteed to reproduce identically live.
- **CUDA graph replay (07):** `cuGraphLaunch` (1 call/execution, replays all
  6 captured stages) vs. `cuLaunchKernel` (6 calls/execution) — fewer CPU-side
  API calls is the mechanism; the multi-x steady-state speedup is this-run
  evidence, not a guarantee.
- **Tensor Core (08):** the claim rests on `--printKernel` showing a literal
  `mma.sync.aligned.m16n8k16...` PTX instruction next to a scalar reference
  kernel from the *same compile* with zero `mma.sync` occurrences — a direct
  code comparison, more convincing live than a blocked hardware-counter
  number would have been anyway.
- **GPULlama3.java:** warn the audience up front about "time to first
  token" (~756 ms on this machine, dominated by one-time JIT compilation of
  ~148 task-graph variants, not slow GPU execution) vs. steady-state decode
  (~6.45 ms/token here) — these have different causes and a live demo will
  visibly pause before the first generated token.
- **Anywhere a `ncu` question comes up:** be upfront that hardware-counter
  metrics (occupancy, GPU utilization %, tensor-pipe activity) are blocked
  on this presentation machine by a driver permission restriction, not a
  TornadoVM limitation — `results/failures/08-nsight-compute-permission.md`
  has the full reproduction if asked.

## If something doesn't work

1. `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader`
   — if it's not `0 %` / near-empty, another process is on the GPU; kill it
   or wait, don't trust numbers from a shared device.
2. `tornado --devices` — must show exactly one CUDA device. If it doesn't,
   the environment is broken, not the demo; fall back to the pre-captured
   logs in `results/raw/` for that demo and narrate from those instead of
   re-running live.
3. Watch stdout for a red `[Bailout] Running the sequential implementation`
   line — this means TornadoVM silently fell back to the CPU. The demo will
   still print a numerically correct result but is not running on the GPU;
   do not claim GPU execution from output correctness alone, always check
   for this line or the profiler JSON's `"BACKEND"` field.
4. Every demo's own `README.md` has a `## If the demo fails on stage` /
   `## Fallback if the live demo fails` section with demo-specific recovery
   steps and the exact pre-captured log to fall back to — §6 above links to
   all of them.
