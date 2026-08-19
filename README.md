# TornadoVM Devoxx 2026 — CUDA Demos

Autonomous research, demo engineering, and talk-content study for two Devoxx sessions built around the **latest TornadoVM CUDA backend**.

## Talks

### 1. TornadoVM Hybrid API: Java + NVIDIA CUDA Libraries

Show Java developers how to:
- write GPU kernels directly in Java;
- access CUDA runtime APIs from Java;
- invoke cuBLAS and cuFFT without handwritten JNI;
- combine Java kernels with vendor-optimized NVIDIA libraries;
- incrementally accelerate existing Java applications while staying on their preferred JDK.

### 2. Java LLM Inference with TornadoVM

Show how TornadoVM can underpin local LLM inference in Java, using GPULlama3.java as the practical case study, including quantized data paths, GPU optimization, and integration with Quarkus/LangChain4j where reproducible against the current upstream code.

## Scope

**CUDA only.** The study must use the current TornadoVM `develop` tree, pinned by SHA for every measurement batch. Do not build or benchmark OpenCL, Metal, the legacy PTX backend, Babylon, or another GPU framework.

The autonomous agent must verify the actual current API and implementation before making claims. Presence of a class, method, or README statement is not sufficient evidence: runnable probes and captured outputs are preferred.

## Outputs

- `docs/talk-1-hybrid-api.md` — evidence-backed draft content and narrative.
- `docs/talk-2-llm-inference.md` — evidence-backed draft content and narrative.
- `docs/demo-runbook.md` — exact live-demo sequence, setup, timings, checkpoints, and recovery paths.
- `docs/claims.md` — claim → source/code/probe/result evidence map.
- `results/raw/` — immutable raw outputs.
- `results/failures/` — captured failures and diagnoses.
- `STATE.md` — durable autonomous-study state.
- `auto/` — autonomous Claude task queue, supervisor, preflight, and prompt contracts.

## Autonomous execution

```bash
bash auto/gen_tasks.sh
bash auto/preflight.sh
bash auto/supervisor.sh
```

The loop is derived from the unattended workflow used in the TornadoVM-vs-Babylon CUDA study: one task per Claude invocation, durable state, explicit acceptance criteria, retries/timeouts, stall detection, and a publication guard.

## Publication boundary

Autonomous pushes are restricted to this repository. The agent must not create or modify issues, PRs, releases, gists, or upstream TornadoVM/GPULlama3.java repositories.
