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
Clone current public TornadoVM `develop`, record `TORNADO_SHA`, GPU/driver/CUDA/JDK/tool versions, and verify `make BACKEND=cuda` builds and exposes a CUDA device.

## Acceptance
`env/versions.env` and a machine manifest exist; CUDA backend builds; smoke example runs on CUDA; exact SHA is recorded.'

w 01 "Verify current Hybrid API" '## Goal
Inspect pinned TornadoVM source for current Hybrid API classes, providers, task-graph composition, buffer sharing, streams, graphs, runtime bindings, and tests/examples. Verify docs against source.

## Acceptance
`docs/hybrid-api-inventory.md` contains source paths and runnable verification commands; changed or unsupported claims are marked.'

w 02 "Build Java CUDA hello and kernel demos" '## Goal
Create tiny presenter-friendly Java examples for Java GPU execution and a first CUDA kernel. One concept per class, lightweight comments, deterministic output. Every runnable example gets a short README, `java @run.args`, and a JBang path where technically valid.

Acceptance
Examples run on CUDA and are understandable by a Java developer in about one minute.'

w 03 "Build CUDA Runtime API demo" '## Goal
Create a minimal example of the current CUDA runtime integration from Java. Keep the source short and show the API directly.

Acceptance
CUDA-only demo runs with deterministic validation; `java @run.args` works; JBang path is tested or explicitly marked unsupported.'

w 04 "Build cuBLAS Hybrid API demo" '## Goal
Create a minimal cuBLAS example using the current Hybrid API, then combine a Java-written kernel with cuBLAS where supported. Keep source presenter-friendly.

Acceptance
Runnable CUDA-only demo, Java reference validation, exact arg-file invocation, JBang path if supported, captured evidence.'

w 05 "Build cuFFT Hybrid API demo" '## Goal
Create a minimal cuFFT example and compose it with a Java kernel where practical. Keep the example small enough for live coding.

Acceptance
Runnable CUDA-only demo with deterministic validation, `java @run.args`, JBang path if supported, and captured output.'

w 06 "Build CUDA streams and async demo" '## Goal
Demonstrate multiple CUDA streams, asynchronous operations, overlap, synchronization, and shared device buffers using the current TornadoVM CUDA/Hybrid API. Make the concurrency visible in Nsight Systems.

Acceptance
A simple sequential-vs-multi-stream demonstration exists with exact commands and a profiler trace showing the execution timeline or a documented API limitation.'

w 07 "Build CUDA Graph demo" '## Goal
Investigate and demonstrate CUDA Graph capture/replay or the closest supported current TornadoVM CUDA mechanism. Show why graph launch/replay matters for repeated workloads.

Acceptance
A minimal graph example exists if the current API supports it; otherwise capture a precise source-backed limitation and a useful adjacent demo. Do not invent graph support.'

w 08 "Build MMA / Tensor Core demo" '## Goal
Create a small FP16/matrix workload that can exercise NVIDIA Tensor Core/MMA hardware when the current TornadoVM CUDA code generation and API permit it. Inspect generated code and use Nsight Compute for hardware evidence.

Acceptance
Only call it Tensor Core/MMA accelerated when profiler or generated-code evidence supports that claim. Record relevant instruction/activity metrics and exact GPU/tool versions.'

w 09 "Profile CUDA demos with NVIDIA tools" '## Goal
Profile representative demos with Nsight Systems and Nsight Compute. Capture CUDA API launches, kernel duration, launch overhead, concurrency, GPU utilization, occupancy, memory throughput, synchronization, instruction mix, and tensor/matrix metrics where applicable. Keep profiled runs separate from timed runs.

Acceptance
Raw profiler artifacts and parsed summaries exist; every reported number has its exact command/tool version; performance-counter failures are documented.'

w 10 "Reproduce current GPULlama3.java" '## Goal
Clone current GPULlama3.java upstream state, record SHA, reproduce build and minimal TornadoVM CUDA inference. Document simple run commands with arg-files/JBang where practical.

Acceptance
Current build/run instructions are captured; minimal inference works or a complete reproducible blocker is recorded.'

w 11 "Verify quantized LLM paths" '## Goal
Probe FP16, Q8, and Q4 support in the current GPULlama3.java/TornadoVM CUDA stack. Only present paths that actually work. Capture model/runtime requirements and output validation.

Acceptance
Each path is classified working/documented/blocked with evidence.'

w 12 "Profile GPULlama3 inference" '## Goal
Use NVIDIA profilers on the working LLM path to explain GPU execution: kernel timeline, dominant kernels, memory behavior, occupancy, and tensor-core activity where applicable. Separate compilation/setup from steady-state inference.

Acceptance
Profiler artifacts, parsed metrics, exact commands, environment manifest, and presenter-friendly interpretation are recorded.'

w 13 "Probe Quarkus and LangChain4j integration" '## Goal
Test current integration paths against actual code/dependencies. Build minimal examples only where reproducible.

Acceptance
Working integrations have runnable demos; blocked integrations contain exact errors and are explicitly marked.'

w 14 "Create profiler quick-start and demo runbook" '## Goal
Create a beginner-friendly profiling guide: `java @run.args`, JBang, Nsight Systems, Nsight Compute, report locations, key metrics, and what the presenter should point out. Add recovery/fallback commands for every demo.

Acceptance
A Java developer can run an example and then profile it by copying commands from the docs.'

w 15 "Create integrated CUDA showcase" '## Goal
Build one reliable wow-factor demo combining as many verified capabilities as practical: Java kernel, Hybrid API library call, streams/async, graph replay if supported, and matrix/Tensor Core path if verified.

Acceptance
The integrated demo remains understandable, has a short run command, profiler evidence, and a safe fallback sequence.'

w 16 "Audit demo simplicity and consistency" '## Goal
Review every demo for one-concept-per-class, lightweight comments, deterministic output, simple invocation, `java @arg-file`, and tested JBang where applicable. Remove unnecessary dependencies and code.

Acceptance
Every demo passes a presenter usability checklist and has a clear one-sentence explanation.'

w 17 "Draft, rehearse, and audit both Devoxx talks" '## Goal
Turn verified evidence into the two talk drafts, speaker notes, demo runbook, recovery paths, and claims ledger. Re-run final demos and representative profiler captures. Ensure every technical claim and every README metric is evidence-backed.

Acceptance
Talk drafts, runbook, claims ledger, profiler guide, and final README are complete. The final README contains numbered demos, how-to-run commands, `java @arg-file` examples, JBang examples where valid, and actual NVIDIA profiler numbers with tool/GPU context. Never invent missing metrics.'

printf 'Generated %s tasks in auto/tasks/\n' "$(find auto/tasks -maxdepth 1 -name '*.md' | wc -l)"
