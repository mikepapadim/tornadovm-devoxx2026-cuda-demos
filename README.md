# TornadoVM Devoxx 2026 — CUDA Demos

Autonomous research, demo engineering, and talk-content study for two Devoxx
sessions built around the **latest TornadoVM CUDA backend**.

Every number and every command below was captured on this pinned
environment (`env/versions.env`): `vendor/tornadovm` `develop` @
`99549c9862eda8d584e35e99924f9c865501eb3a`, one NVIDIA GeForce RTX 4090,
driver 565.57.01, CUDA toolkit 12.6.85, JDK 21.0.2. Full claim → evidence
map: `docs/claims.md`.

## Talks

### 1. TornadoVM Hybrid API: Java + NVIDIA CUDA Libraries

Show Java developers how to write GPU kernels directly in Java, access
CUDA-runtime behaviour (graph capture/replay, multi-stream concurrency)
from `TornadoExecutionPlan`, invoke cuBLAS/cuFFT without handwritten JNI,
and combine Java kernels with vendor-optimized NVIDIA libraries in one
execution graph. Draft: `docs/talk-1-hybrid-api.md`.

### 2. Java LLM Inference with TornadoVM

Show how TornadoVM underpins local LLM inference in Java, using
GPULlama3.java as the case study — quantized data paths (FP16/Q8_0
working, Q4_0 explicitly unimplemented upstream), GPU profiling, and
Quarkus/LangChain4j integration, including where that integration is
currently blocked and why. Draft: `docs/talk-2-llm-inference.md`.

## Scope

**CUDA only.** The study uses the current TornadoVM `develop` tree, pinned
by SHA for every measurement batch. No OpenCL, Metal, the legacy PTX
backend, Babylon, or another GPU framework anywhere in this repo
(`scripts/verify.sh` checks this).

## Track A demos — Hybrid API (`demos/`)

Each demo is a single presenter-friendly Java file. Every row below has been
run on the pinned CUDA backend; raw logs are cited per demo.

| # | Demo | Run | `java @arg-file` |
|---|------|-----|-------------------|
| [00](demos/00-hello-gpu/) | `Hello.java` — one `@Parallel` task, smallest TornadoVM program | `tornado --classpath . Hello` | `java @../tornado.args -cp . Hello` |
| [01](demos/01-first-cuda-kernel/) | `VectorAddKernel.java` — vector add + `--printKernel` generated-CUDA-source proof | `tornado --classpath . VectorAddKernel` | `java @../tornado.args -cp . VectorAddKernel` |
| [02](demos/02-cuda-runtime-api/) | `CudaGraphReplay.java` — `withCUDAGraph()` capture + 8 replays, CUDA-runtime API, not a vendor library | `tornado --classpath . CudaGraphReplay` | `java @../tornado.args -cp . CudaGraphReplay` |
| [04](demos/04-cublas-hybrid/) | `CuBlasSgemvHybrid.java` — JIT `scale` → cuBLAS `sgemv` → JIT `bias`, one graph, shared buffers | `tornado --classpath . CuBlasSgemvHybrid` | `java @../tornado.args -cp . CuBlasSgemvHybrid` |
| [05](demos/05-cufft-hybrid/) | `CuFftLowPassHybrid.java` — cuFFT forward (R2C) → JIT low-pass → cuFFT inverse (C2R) → JIT normalize | `tornado --classpath . CuFftLowPassHybrid` | `java @../tornado.args -cp . CuFftLowPassHybrid` |
| [06](demos/06-cuda-streams/) | `CudaStreamsOverlap.java` — sequential (1 stream) vs. concurrent (4-stream pool) with Nsight Systems overlap evidence | `tornado --classpath . CudaStreamsOverlap 8 32768 65536 8 both` | `java @../tornado.args -cp . CudaStreamsOverlap 8 32768 65536 8 both` |
| [07](demos/07-cuda-graph-benefit/) | `CudaGraphBenefit.java` — `nograph` vs. `graph`, 50 executions, quantified steady-state speedup | `tornado --classpath . CudaGraphBenefit 4096 6 50 both` | `java @../tornado.args -cp . CudaGraphBenefit 4096 6 50 both` |
| [08](demos/08-tensor-core-mma/) | `TensorCoreMMA.java` — one `M16N8K16` fp16 tile, one `mma.sync.aligned` instruction, vs. scalar reference | `tornado --printKernel --classpath . TensorCoreMMA` | `java @../tornado.args -cp . TensorCoreMMA` |
| [11](demos/11-integrated-showcase/) | `IntegratedShowcase.java` — kernel + cuBLAS x 6 chains, baseline/concurrent/graph/combined + Tensor Core bonus stage | `tornado --classpath . IntegratedShowcase 6 8 8 20 all` | `java @../tornado.args -cp . IntegratedShowcase 6 8 8 20 all` |
| [09](demos/09-quarkus-langchain4j-gpullama3/) | Quarkus + LangChain4j GPULlama3 extension — builds, **blocked at runtime** (JDK preview-feature vs. JDK-23+ split, see below) | n/a — do not run live, see `docs/quarkus-langchain4j-integration.md` | n/a |
| [10](demos/10-langchain4j-gpullama3/) | Standalone LangChain4j GPULlama3 provider — same blocker as 09 | n/a — do not run live | n/a |

`demos/tornado.args` is a committed `tornado --generate-argfile` output,
generated against the pinned build in `env/versions.env`. **JBang: not
verified on this machine** — `which jbang` returns exit 1, reconfirmed every
task; every demo README documents the expected-but-untested JBang shape and
says explicitly not to run it live. Full presenter usability audit:
`docs/demo-audit-checklist.md`.

## Measured results (Observed, this machine, this run — not general claims)

| Demo | Metric | Result | Tool | Evidence |
|---|---|---|---|---|
| 04-cublas-hybrid | Correctness | 5/5 iterations correct vs. sequential Java reference | `--enableProfiler console` | `results/raw/04-cublas-hybrid/` |
| 05-cufft-hybrid | Correctness | 5/5 iterations correct vs. analytic low-frequency signal | — | `results/raw/05-cufft-hybrid/` |
| 06-cuda-streams | Stream count / overlap | sequential: 1 CUDA stream, strictly back-to-back; concurrent: 4-stream pool, genuinely overlapping windows | Nsight Systems `2024.5.1.113` (`cuda_gpu_trace`) | `results/raw/06-cuda-streams/nsys-timeline-evidence.txt` |
| 07-cuda-graph-benefit | Steady-state speedup, graph vs. nograph (size=4096, 6 stages, 50 executions) | 6.47x, 6.58x, 7.02x across 3 independent runs | wall-clock, cross-checked | `demos/07-cuda-graph-benefit/README.md` |
| 08-tensor-core-mma | Generated code | 1x `mma.sync.aligned.m16n8k16...` PTX instruction vs. 0x in the scalar reference from the same compile | `--printKernel` | `results/raw/08-tensor-core-mma/tensorcoremma-printkernel.log` |
| 09-profiling | Dominant setup cost | `cuCtxCreate_v2` is 89–99% of `Time (%)` in every trace (whole-JVM-process trace, not compute cost) | Nsight Systems | `results/raw/09-profiling/PROFILING-SUMMARY.md` |
| 11-integrated-showcase | `graph` mode speedup vs. baseline | 4.9x–6.6x across runs | wall-clock | `results/raw/11-integrated-showcase/`, `STATE.md` batch 15/16 |
| 11-integrated-showcase | `concurrent` mode vs. baseline (small launch-overhead-bound workload) | 0.78x–1.21x — **not reliably faster**, workload-dependent | wall-clock | `STATE.md` batch 15 |
| GPULlama3.java FP16 | Throughput, 1B model, 25–64 tokens | 153.18–164.20 tok/s | `--profiler` JSON, confirms `"BACKEND": "CUDA"` | `results/raw/10-gpullama3/`, `results/raw/11-quantization/` |
| GPULlama3.java Q8_0 | Throughput, same model family | 186.47 tok/s | same | `results/raw/11-quantization/q8_0-inference-run.log` |
| GPULlama3.java FP16 decode | Dominant GPU kernel | `fusedRmsNormFFNGateUp` (FFN gate/up), 39.0% of GPU kernel time | Nsight Systems | `results/raw/12-llm-profiling/PROFILING-SUMMARY.md` |
| GPULlama3.java FP16 decode | Memory traffic | 99.1% H2D memcpy | Nsight Systems | `results/raw/12-llm-profiling/PROFILING-SUMMARY.md` |
| GPULlama3.java FP16 decode | Time-to-first-token vs. steady-state | ~756 ms (JIT warm-up, ~148 task-graph variants) vs. ~6.45 ms/token steady-state (cross-checked within ~1% of reported tok/s) | Nsight Systems `cuda_gpu_trace` | `results/raw/12-llm-profiling/PROFILING-SUMMARY.md` §6 |

**Blocked, system-wide, both talks:** Nsight Compute (`ncu`) hardware
counters (occupancy, GPU utilization %, memory throughput %, tensor-pipe
activity) — `ERR_NVGPUCTRPERM`, `NVreg_RestrictProfilingToAdminUsers=1`, no
passwordless sudo on this machine. Re-verified 4x (tasks 08, 09, 12, 14).
Not a TornadoVM limitation. Full writeup: `results/failures/08-nsight-compute-permission.md`.

## Track B — GPULlama3.java on TornadoVM CUDA (Observed unless marked)

- Build: `./mvnw clean install -DskipTests -Dtornadovm.base.version=5.2.1 -Djdk.version.suffix=-jdk21-dev` (a build-property override for a real cross-version incompatibility between GPULlama3.java's documented Maven Central pin and this repo's pinned CUDA SDK — source unmodified). `docs/gpullama3-reproduction.md`.
- Quantization: FP16 and Q8_0 **work**; legacy Q4_0 is **blocked** (`UnsupportedOperationException`, matches source-level dispatch exactly, upstream marks it "not yet implemented"); K-quants are **documented, not independently reproduced** (no test file on this machine). `docs/quantization-paths.md`.
- Quarkus (`quarkus-langchain4j-gpu-llama3:1.13.0`) and LangChain4j (`langchain4j-gpu-llama3:1.19.0-beta29`) integrations both **build** against the pinned CUDA jar but are **blocked at model-load time**: this repo's CUDA build uses JDK 21 `--enable-preview` bytecode (loadable only by JDK 21), while both integration modules require JDK 23+ to load their own classes — no single JVM satisfies both. `docs/quarkus-langchain4j-integration.md`.

## Outputs

- `docs/talk-1-hybrid-api.md` / `docs/talk-2-llm-inference.md` — evidence-backed talk drafts.
- `docs/demo-runbook.md` — exact live-demo sequence, what to say, per-step fallback.
- `docs/claims.md` — every claim → source/probe/result evidence map.
- `docs/hybrid-api-inventory.md`, `docs/gpullama3-reproduction.md`, `docs/quantization-paths.md`, `docs/quarkus-langchain4j-integration.md`, `docs/profiling-quickstart.md`, `docs/demo-audit-checklist.md`, `docs/run-conventions.md` — supporting evidence documents.
- `results/raw/` — immutable raw outputs. `results/failures/` — captured failures and diagnoses.
- `STATE.md` — durable autonomous-study state. `auto/` — autonomous Claude task queue, supervisor, preflight, prompt contracts.
- `scripts/verify.sh` — validates the deliverables above and every cited evidence path exists, without needing a GPU.

## Autonomous execution

```bash
bash auto/gen_tasks.sh
bash auto/preflight.sh
bash auto/supervisor.sh
```

Validate committed evidence at any time, no GPU required:

```bash
bash scripts/verify.sh
```

The loop is derived from the unattended workflow used in the
TornadoVM-vs-Babylon CUDA study: one task per Claude invocation, durable
state, explicit acceptance criteria, retries/timeouts, stall detection, and
a publication guard.

## Publication boundary

Autonomous pushes are restricted to this repository. The agent must not
create or modify issues, PRs, releases, gists, or upstream
TornadoVM/GPULlama3.java repositories.
