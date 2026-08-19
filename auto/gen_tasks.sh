#!/usr/bin/env bash
# Generate the ordered autonomous queue. Safe to re-run: task files are deterministic.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
mkdir -p auto/tasks

w() { local id="$1" title="$2"; shift 2; cat > "auto/tasks/${id}.md" <<EOF
# ${id} — ${title}

$*
EOF
}

w 00 "Pin current TornadoVM CUDA baseline" '## Goal
Clone the current public TornadoVM `develop`, record `TORNADO_SHA`, GPU/driver/CUDA/JDK/tool versions, and verify that `make BACKEND=cuda` builds and exposes a CUDA device.

## Acceptance
`env/versions.env` and a machine manifest exist; the CUDA backend builds; a smoke example runs on the CUDA device; exact SHA is recorded.'

w 01 "Verify current Hybrid API" '## Goal
Inspect the pinned TornadoVM tree and identify the current Hybrid API classes, providers, task-graph composition model, buffer sharing semantics, stream behavior, and tests/examples. Verify documentation against source.

## Acceptance
`docs/hybrid-api-inventory.md` contains source paths and runnable verification commands; unsupported or changed claims are marked.'

w 02 "Build Java CUDA runtime demo" '## Goal
Create a minimal presenter-friendly Java example that demonstrates the current CUDA runtime integration. Provide both `java @arg-file` and a JBang path where technically valid.

## Acceptance
Demo runs on the CUDA backend and has deterministic output, a short README, arg-file, JBang instructions, and captured output.'

w 03 "Build cuBLAS Hybrid API demo" '## Goal
Create a minimal cuBLAS example using the current Hybrid API, then combine a Java-written kernel with a cuBLAS operation where the current API supports it.

## Acceptance
Runnable CUDA-only demo, validation against a Java reference, exact invocation using `java @...`, JBang path if supported, and captured evidence.'

w 04 "Build cuFFT Hybrid API demo" '## Goal
Create and validate a current cuFFT Hybrid API example, then demonstrate composition with a Java kernel where practical.

## Acceptance
Runnable CUDA-only demo with deterministic validation, arg-file/JBang instructions, and captured evidence.'

w 05 "Profile Hybrid API demos with NVIDIA tools" '## Goal
Profile representative Hybrid API demos using Nsight Systems and Nsight Compute when supported by the server/toolkit. Capture CUDA API launches, kernel timings, GPU utilization, occupancy, memory throughput, synchronization, and relevant instruction/tensor-core metrics. Keep profiling runs separate from timed runs.

## Acceptance
Raw profiler artifacts and parsed summaries exist; every reported metric has its exact command/tool version; failures due to performance-counter permissions are documented rather than hidden.'

w 06 "Reproduce current GPULlama3.java" '## Goal
Clone the current GPULlama3.java upstream state, record its SHA, reproduce the current build and a minimal TornadoVM CUDA inference path.

## Acceptance
Current build/run instructions are captured; minimal inference works or a complete reproducible blocker is recorded.'

w 07 "Verify quantized LLM paths" '## Goal
Probe FP16, Q8, and Q4 support in the current GPULlama3.java/TornadoVM CUDA stack. Only present paths that actually work. Capture model/runtime requirements and accuracy/output validation.

## Acceptance
Each path is classified working/documented/blocked with evidence; no unsupported claim enters the talk draft.'

w 08 "Profile GPULlama3 inference" '## Goal
Use NVIDIA profilers on the working LLM path to explain GPU execution: kernel timeline, dominant kernels, memory behavior, occupancy, and tensor-core activity where applicable. Separate compilation/setup from steady-state inference.

## Acceptance
Profiler artifacts, parsed metrics, exact commands, environment manifest, and a presenter-friendly interpretation are recorded.'

w 09 "Probe Quarkus and LangChain4j integration" '## Goal
Test current integration paths against the actual code and dependencies. Build a minimal example only where reproducible.

## Acceptance
Working integrations have runnable demos; blocked integrations contain exact build/runtime errors and are explicitly marked in the claims ledger.'

w 10 "Draft, rehearse, and audit both Devoxx talks" '## Goal
Turn verified evidence into the two talk drafts, demo runbook, speaker notes, recovery paths, and claims ledger. Re-run the final demos and profile representative paths. Ensure all examples document `java @arg-file` and JBang where applicable.

## Acceptance
`docs/talk-1-hybrid-api.md`, `docs/talk-2-llm-inference.md`, `docs/demo-runbook.md`, and `docs/claims.md` are complete and every high-confidence technical claim is evidence-backed.'

printf 'Generated %s tasks in auto/tasks/\n' "$(find auto/tasks -maxdepth 1 -name '*.md' | wc -l)"
