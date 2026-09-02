# Claims ledger

Every claim that appears in `README.md`, `docs/talk-1-hybrid-api.md`, and
`docs/talk-2-llm-inference.md`, mapped to its evidence. Classification per
`PLAN.md` §6: **Observed** (reproduced locally, log captured), **Source-backed**
(verified in the current checkout, not run), **Documented** (upstream docs
only), **Hypothesis**, **Blocked** (attempted, failed, reason captured).

No claim below is invented. Every row cites a source path, a probe command, or
a result log that actually exists in this repository.

System under test, **Track A demos**: TornadoVM `6.0.0-jdk22plus-cuda`
(SDKMAN release, `env/versions.env`), JDK 25.0.2, CUDA backend only, RTX 4090,
driver 565.57.01, CUDA toolkit 12.6.85. Migration evidence:
`results/raw/18-tornadovm-6-migration/`.

System under test, **Track B (GPULlama3.java) and every claim below marked
5.2.1**: the earlier source-built pin `vendor/tornadovm` `develop` @
`99549c9862eda8d584e35e99924f9c865501eb3a`, JDK 21.0.2. Track B was **not**
migrated to 6.0.0; its claims are recorded as they were observed and are not
restated as current 6.0.0 behaviour.
GPULlama3.java claims additionally pinned at `GPULLAMA3_SHA=bbe42fdc8cd475bb6104cefa42118dd6e068538b`.

## Track A — Hybrid API

| # | Claim | Classification | Evidence |
|---|---|---|---|
| A0 | A Java `@Parallel` task compiles and runs as a real CUDA kernel via `TaskGraph`/`TornadoExecutionPlan`. | Observed | `demos/00-hello-gpu/Hello.java`; `results/raw/02-hello-kernel/hello-run*.log`; `--enableProfiler console` shows `"BACKEND": "CUDA"`. |
| A0 | The generated CUDA source for a JIT kernel is inspectable before `nvcc`/`ptxas` compiles it. | Observed | `tornado --printKernel`; `demos/01-first-cuda-kernel/`; `results/raw/02-hello-kernel/vectoraddkernel-run.log`. |
| A1 | There is no raw `cudaMalloc`/`cudaMemcpy`-style JNI binding exposed to user code; CUDA-runtime *behaviour* is exposed via `TornadoExecutionPlan.withCUDAGraph()` / `.withIntraPlanConcurrency()` / `.withStagedTransfers()`. | Source-backed | `docs/hybrid-api-inventory.md` §6; `TornadoExecutionPlan` Javadoc, "currently realised on the CUDA backend" / no-op on OPENCL/METAL. |
| A1 | `withCUDAGraph()` performs real CUDA-graph capture/replay (`cuStreamBeginCapture`/`cuGraphInstantiate`/`cuGraphLaunch`) from Java, correctness-validated. | Observed | `demos/02-cuda-runtime-api/CudaGraphReplay.java`; `results/raw/03-cuda-runtime-api/cudagraphreplay-run.log`; native source `tornado-drivers/cuda-jni/.../CUDAGraph.cpp`. |
| A2 | Java code can invoke cuBLAS (`sgemv`) as a library task, no handwritten JNI. | Observed | `demos/04-cublas-hybrid/CuBlasSgemvHybrid.java`; `results/raw/04-cublas-hybrid/cublassgemvhybrid-run.log`; upstream `TestCuBlas` 12/12 PASS (`docs/hybrid-api-inventory.md` §3). |
| A3 | Java code can invoke cuFFT (R2C/C2R) as a library task. | Observed | `demos/05-cufft-hybrid/CuFftLowPassHybrid.java`; `results/raw/05-cufft-hybrid/cufftlowpasshybrid-run.log`; upstream `TestCuFft` 8/8 PASS. |
| A4 | A JIT Java kernel and a cuBLAS library task can share device buffers inside one `TaskGraph`, zero host round-trips between stages. | Observed | `CuBlasSgemvHybrid.java` (scale→sgemv→bias, one graph); `--enableProfiler console` shows every stage on the same `DEVICE`. |
| A5 | A JIT Java kernel and a cuFFT library task can share device buffers inside one `TaskGraph`. | Observed | `CuFftLowPassHybrid.java` (forward→lowPass→inverse→normalize, one graph), 5/5 iterations correct vs. analytic reference. |
| A6 | Multi-stream concurrent execution (`withIntraPlanConcurrency()`) is real: distinct CUDA streams, genuinely overlapping kernel windows. | Observed | `demos/06-cuda-streams/CudaStreamsOverlap.java`; Nsight Systems timeline `results/raw/06-cuda-streams/nsys-timeline-evidence.txt` (1 stream sequential vs. 4-stream pool concurrent). |
| A6 | Multi-stream concurrency gives a wall-clock speedup on every workload. | **False as a general claim — do not make it.** Workload-dependent, Observed both directions. | Demo 06 (small heavy-inner-loop kernels): overlap observed. Demo 11 (`IntegratedShowcase`, tiny launch-overhead-bound sgemv chains): concurrent mode ranged 0.78x–1.21x vs. baseline across 3 runs — sometimes *slower*. `STATE.md` batch 15/16; `demos/11-integrated-showcase/README.md`. |
| A7 | CUDA graph replay reduces steady-state per-execution overhead for a repeated multi-stage task-graph. | Observed, this-run numbers only | `demos/07-cuda-graph-benefit/README.md`: 3 runs, size=4096/6 stages/50 executions, steady-state median speedup 6.47x, 6.58x, 7.02x (graph vs. nograph). Not a general TornadoVM claim — quote as "observed on this machine, this workload." |
| A7 | `withCUDAGraph()` and `withIntraPlanConcurrency()` can be combined on one `TornadoExecutionPlan`. | Observed (this repo's own probe — **not documented upstream**, no upstream test combines both) | `demos/11-integrated-showcase/IntegratedShowcase.java` `combined` mode; validated correct across 3 runs; `STATE.md` batch 15. Lands *between* `graph` alone and `baseline`, i.e. concurrency stacked on graph replay did not help further at this workload size — presenter caveat, not a win to claim. |
| A7 | cuBLAS/cuFFT/cuDNN/cuSPARSE/CUTLASS library tasks are all CUDA-graph-capturable (`prepare()` pre-allocates capture-unsafe workspace before capture). | Source-backed + Observed | `TornadoLibraryProvider.prepare()` Javadoc; `testXxxWithCudaGraph` suites pass for all 5 libraries (`docs/hybrid-api-inventory.md` §3, 71/71 provider unit tests). |
| — | cuDNN, cuSPARSE, CUTLASS providers exist and pass their upstream unit-test suites on this pinned SHA/GPU. | Observed | `TestCuDnn` 9/9, `TestCusparse` 11/11, `TestCutlass` 25/25 PASS, `docs/hybrid-api-inventory.md` §2–3. Not built into a repo-original demo (out of A0–A7 scope this batch) — do not claim a live demo exists for these three, only upstream-test evidence. |
| — | Tensor Core (`mma.sync`) codegen from `KernelContext` intrinsics is real, CUDA-only, compute capability 8.0+. | Observed (generated-code evidence) | `demos/08-tensor-core-mma/TensorCoreMMA.java`; `--printKernel` shows literal `mma.sync.aligned.m16n8k16...` PTX next to a zero-`mma.sync` scalar reference from the same compile. `results/raw/08-tensor-core-mma/tensorcoremma-printkernel.log`. Hardware-counter (tensor-pipe activity %) confirmation is **Blocked** — see below. |
| — | Hardware-counter metrics (occupancy, GPU utilization %, memory throughput %, tensor-pipe activity) are obtainable on this machine via Nsight Compute. | **Blocked** | `results/failures/08-nsight-compute-permission.md`: `ERR_NVGPUCTRPERM`, `NVreg_RestrictProfilingToAdminUsers=1`, no passwordless sudo. Re-verified 4x (tasks 08, 09, 12, 14). Never claim an `ncu` number live on this machine. |

## Track B — GPULlama3.java on TornadoVM CUDA

| # | Claim | Classification | Evidence |
|---|---|---|---|
| B0 | GPULlama3.java's documented build (`make` / `mvnw install`) succeeds as-is. | Observed | `results/raw/10-gpullama3/build.log`, `BUILD SUCCESS`. |
| B0 | The documented build's compiled classes link against a *different* TornadoVM release (5.0.0) than this repo's pinned CUDA build (5.2.1-jdk21-dev), causing a runtime `TornadoInternalError` (`writeReplace()`/Serializable mismatch) when run against the pinned SDK. | Observed (failure reproduced) | `docs/gpullama3-reproduction.md` §2; `results/raw/10-gpullama3/inference-run.log`. Workaround: `./mvnw clean install -Dtornadovm.base.version=5.2.1 -Djdk.version.suffix=-jdk21-dev` (build-property override only, source unmodified, per `CLAUDE.md`). |
| B1/B2 | Minimal inference (`--gpu --cuda`, FP16 model) produces correct, coherent output on the CUDA backend. | Observed, 4 reproductions total | `docs/gpullama3-reproduction.md` §3; profiler JSON confirms `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"` per task-graph stage. |
| B2 | Throughput: ~150–165 tok/s, FP16, 1B model, this GPU, 25–64 generated tokens. | Observed, this-run/this-model/this-GPU only | `results/raw/10-gpullama3/inference-run-reverify.log` (153.18 tok/s); `results/raw/11-quantization/fp16-inference-run.log` (164.20 tok/s). Not a general benchmark claim. |
| B3 | TornadoVM execution supports exactly FP16 and Q8_0 weight formats on this pinned SHA; K-quants (Q4_K/Q5_K/Q6_K) dequantize to Q8_0 at load; legacy Q4_0 and F32 are explicitly rejected. | Source-backed (two independent gate checkpoints) | `docs/quantization-paths.md` §1: `AbstractModelLoader.getModelQuantization` (`:41-48`), `LlamaModelLoader.createTornadoVMWeights` (`:109-114`), `ForwardPlanFactory.create` (`Q4_0`/`F32` → explicit `throw`). |
| B3 | FP16 path: working. | Observed, 4 runs | `docs/quantization-paths.md` §2. |
| B3 | Q8_0 path: working, 186.47 tok/s this run, profiler-confirmed CUDA. | Observed, 2 runs | `docs/quantization-paths.md` §3; `results/raw/11-quantization/q8_0-inference-run*.log`. |
| B3 | Q4_0 path: fails deterministically with `UnsupportedOperationException: Unsupported quantization format: 2`, before any GPU work starts. | Observed (blocked, matches source prediction exactly) | `docs/quantization-paths.md` §4; `results/raw/11-quantization/q4_0-inference-run.log`. Upstream marks it "not yet implemented" (`ForwardPlanFactory.java:84`), not a permanent limitation. |
| B3 | Q4_K/Q5_K/Q6_K work via the Q8_0 dequantization path. | Documented (source-backed, **not independently reproduced** — no K-quant GGUF file available on this machine) | `docs/quantization-paths.md` §5. Do not claim this as Observed. |
| B4 | Dominant GPU kernel during FP16 decode is the FFN gate/up projection (`fusedRmsNormFFNGateUp`). | Observed | `results/raw/12-llm-profiling/PROFILING-SUMMARY.md` §5–6 (39.0% of GPU kernel time). |
| B4 | H2D memcpy dominates memory traffic (99.1%), consistent with small per-token transfers rather than bulk weight movement. | Observed | `results/raw/12-llm-profiling/PROFILING-SUMMARY.md`. |
| B4 | Time-to-first-token (~756 ms) is dominated by one-time JIT compilation of ~148 task-graph variants, not slow GPU execution; steady-state decode is ~6.45 ms/token. | Observed, cross-checked within ~1% against the run's own reported tok/s | `results/raw/12-llm-profiling/PROFILING-SUMMARY.md` §6 (min 5.01 ms, max 20.78 ms, avg 6.453 ms/token; cross-check 1000/153.54 ≈ 6.51 ms/token). |
| B4 | `ncu` hardware-counter metrics for GPULlama3.java are blocked on this machine, same root cause as Track A. | Blocked | `results/raw/12-llm-profiling/PROFILING-SUMMARY.md` §7; `results/failures/08-nsight-compute-permission.md`. |
| B5 | Quarkus LangChain4j GPULlama3 extension (`quarkus-langchain4j-gpu-llama3:1.13.0`) builds and links against this repo's pinned CUDA build via a direct dependency override. | Observed | `docs/quarkus-langchain4j-integration.md` §2; `demos/09-quarkus-langchain4j-gpullama3/pom.xml`. |
| B5 | Quarkus integration boots successfully under JDK 25 (CDI/ArC wiring, model registry/cache lookup all resolve) but fails loading GPULlama3.java's own preview-flagged classes (`UnsupportedClassVersionError`, class file version 65.65535 vs. the JVM's expected preview marker). | Observed (failure reproduced, root cause identified) | `docs/quarkus-langchain4j-integration.md` §3; `results/raw/13-quarkus-langchain4j/quarkus-run-jdk25-blocked.log`. Root cause: preview-flagged bytecode is loadable *only* by the exact JDK feature release that produced it (JDK 21) — but the integration's own deployment classes need JDK 23+. No single JVM process satisfies both. |
| B6 | LangChain4j GPULlama3 provider (`dev.langchain4j:langchain4j-gpu-llama3:1.19.0-beta29`) — every published version to date is a `-betaN` release, despite the "officially supported" framing in GPULlama3.java's README. | Source-backed (Maven Central `maven-metadata.xml`, checked live) | `docs/quarkus-langchain4j-integration.md` §1. |
| B6 | LangChain4j standalone integration hits the identical JDK-version-split blocker as B5. | Observed | `docs/quarkus-langchain4j-integration.md` §3; `results/raw/13-quarkus-langchain4j/langchain4j-standalone-jdk25-blocked.log`. |
| B5/B6 | A JDK 25 build of GPULlama3.java (`-jdk25` profile) plus a JDK-25 TornadoVM CUDA backend might resolve the split. | **Hypothesis, not attempted** | `docs/quarkus-langchain4j-integration.md` §4 — building a second JDK-25 TornadoVM CUDA backend from the pinned SHA is a non-trivial rebuild, flagged as follow-up, out of scope this batch. |
| B7 | A presenter-safe end-to-end inference demo with deterministic prompt/seed and a documented fallback exists. | Observed | `docs/profiling-quickstart.md` §7 (exact commands); `docs/demo-runbook.md` (this batch) step-by-step with recovery path to pre-captured logs. |

## Cross-cutting

| Claim | Classification | Evidence |
|---|---|---|
| All 6 Hybrid API library providers (cuBLAS, cuBLASLt, cuFFT, cuDNN, cuSPARSE, CUTLASS) are registered Maven modules, built successfully by `make BACKEND=cuda`. | Observed | `results/raw/00-baseline/tornadovm-build-cuda.log`; `docs/hybrid-api-inventory.md` §2. |
| 71/71 upstream provider unit tests pass on this pinned SHA/GPU across all 6 providers. | Observed | `docs/hybrid-api-inventory.md` §3 (per-suite pass counts). |
| Every demo in `demos/` (00, 01, 02, 04–08, 11) runs correctly via all three invocation shapes it documents as supported (`tornado`, `java @arg-file`; JBang explicitly marked untested). | Observed | Per-demo `README.md`; `docs/demo-audit-checklist.md` (task 16 audit, one live re-verification of demo 06's arg-file path this batch). |
| JBang works on this environment. | **False — do not claim.** | `which jbang` → exit 1, reconfirmed every task 00–17. Documented as unsupported-on-this-environment everywhere, not silently omitted. |
| Nsight Compute (`ncu`) hardware counters are usable on this machine. | **Blocked, not a TornadoVM limitation.** | `results/failures/08-nsight-compute-permission.md`, `NVreg_RestrictProfilingToAdminUsers=1` driver default, no passwordless sudo to change it (out of scope for a reversible autonomous action). |

## No-claim list (explicitly out of scope, never state these)

- No OpenCL/Metal/legacy-PTX/CPU performance numbers anywhere in this repo (hard scope, `CLAUDE.md`).
- No Babylon comparison (hard scope).
- No multi-GPU claim (one NVIDIA GPU only, per `PLAN.md` §2).
- No general "TornadoVM is Nx faster" claim — every speedup number in this repo is scoped to "this run, this workload, this machine" (demos 06, 07, 11; `docs/hybrid-api-inventory.md` §6 explicitly repeats this caveat).
