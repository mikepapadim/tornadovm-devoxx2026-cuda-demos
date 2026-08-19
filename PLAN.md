# Study Plan

## 1. Objective

Produce a reproducible body of research, working demos, and draft content for the two Devoxx 2026 talks in `README.md`. The repository is a **TornadoVM CUDA study**, not a framework comparison.

Every technical claim must be tied to one of:

1. current upstream source at a recorded SHA;
2. a runnable probe with captured output;
3. a measured demo result with its environment manifest; or
4. an explicitly labelled hypothesis/open question.

## 2. System under test

- TornadoVM: public `develop`, latest at the start of each batch; record `TORNADO_SHA`.
- Backend: `cuda` only.
- NVIDIA GPU: one physical device, recorded with driver/toolkit details.
- CUDA libraries: use the versions actually present and record them.
- Java: record the JDK and TornadoVM runtime configuration.
- GPULlama3.java: use the current reproducible upstream state for LLM work and record its SHA.

Never use the legacy `ptx` backend. Never silently substitute OpenCL/Metal/CPU execution.

## 3. Study tracks

### Track A — Hybrid API fundamentals

Build small, deterministic demos in increasing complexity:

A0. Java GPU kernel / baseline TornadoVM task.
A1. CUDA runtime API access from Java.
A2. Vendor-library call from Java: cuBLAS.
A3. Vendor-library call from Java: cuFFT.
A4. Java kernel + cuBLAS in one execution graph.
A5. Java kernel + cuFFT in one execution graph.
A6. Reusable device buffers / data sharing across tasks.
A7. Error handling, unsupported API behavior, and portability boundaries.

The final live demo should be short enough to rebuild or recover during a conference session.

### Track B — Java LLM inference

B0. Reproduce current GPULlama3.java build/run instructions.
B1. Establish model/tokenizer/runtime prerequisites.
B2. Reproduce a minimal inference path on TornadoVM CUDA.
B3. Probe FP16, Q8, and Q4 paths; only present paths that work on the current code.
B4. Capture kernel/codegen/profiling evidence useful for explaining the optimization story.
B5. Probe Quarkus integration.
B6. Probe LangChain4j integration.
B7. Produce a presenter-safe end-to-end inference demo with a deterministic prompt and fallback path.

## 4. Hybrid API verification

Before writing content, inspect the actual current TornadoVM tree for:

- Hybrid API classes and task-graph composition model;
- CUDA runtime bindings;
- cuBLAS/cuBLASLt providers;
- cuFFT provider;
- cuDNN provider;
- cuSPARSE provider;
- CUTLASS provider;
- buffer sharing and stream semantics;
- build/runtime prerequisites;
- current examples/tests.

Do not assume documentation reflects `develop`. Capture source paths and test/probe commands.

## 5. Measurement contract

For every timed demo:

- record GPU, driver, CUDA toolkit/runtime, JDK, TornadoVM SHA, demo SHA;
- ensure no competing CUDA process is using the GPU;
- warm up before timing;
- separate compilation/setup time from steady-state execution;
- use deterministic inputs where practical;
- validate output before reporting performance;
- retain raw stdout/stderr and machine-readable results;
- never fabricate missing measurements.

Performance is supporting evidence for the talks, not the primary goal. A reliable live demo is more valuable than a fragile benchmark.

## 6. Content contract

Draft content must distinguish:

- **Observed:** reproduced locally and captured.
- **Source-backed:** verified directly in the current repository/source.
- **Documented:** stated by upstream documentation but not independently reproduced.
- **Hypothesis:** plausible but not yet verified.
- **Blocked:** attempted and failed, with the exact reason captured.

No claim enters the final talk drafts without an evidence classification.

## 7. Definition of done

The study is complete when:

- all high-value Hybrid API demos have runnable source and a tested runbook;
- GPULlama3.java current-state reproduction is documented;
- supported quantization paths are verified;
- integration claims have evidence or are explicitly marked blocked;
- both talk drafts have a clear opening, narrative, live-coding sequence, technical explanation, and conclusion;
- every demo has a fallback/recovery path;
- `docs/claims.md` has no unsupported high-confidence claim;
- `scripts/verify.sh` can validate committed evidence without requiring a GPU;
- final rehearsal artifacts are recorded in `results/`.
