# Track A demos — Java + TornadoVM CUDA

Presenter-friendly demos, one concept per class, increasing in complexity.
Each demo directory has its own `README.md` with build/run commands, a
`java @arg-file` path, expected output, and a stage-failure fallback.

| # | Demo | Concept |
|---|------|---------|
| [00-hello-gpu](00-hello-gpu/) | `Hello.java` | Smallest TornadoVM program: one `@Parallel` task, one `TaskGraph`. |
| [01-first-cuda-kernel](01-first-cuda-kernel/) | `VectorAddKernel.java` | Vector add + `--printKernel` to show the actual generated CUDA source. |
| [02-cuda-runtime-api](02-cuda-runtime-api/) | `CudaGraphReplay.java` | `TornadoExecutionPlan#withCUDAGraph()` — CUDA graph capture/replay from Java, CUDA-only runtime API (not a vendor-library task). |
| [04-cublas-hybrid](04-cublas-hybrid/) | `CuBlasSgemvHybrid.java` | One `TaskGraph`, three stages: JIT `scale` task → cuBLAS `sgemv` library task → JIT `bias` task, all on shared device buffers. |
| [05-cufft-hybrid](05-cufft-hybrid/) | `CuFftLowPassHybrid.java` | One `TaskGraph`, four stages: cuFFT `forward` (R2C) → JIT `lowPass` task → cuFFT `inverse` (C2R) → JIT `normalize` task, a GPU-resident low-pass filter. |
| [06-cuda-streams](06-cuda-streams/) | `CudaStreamsOverlap.java` | `TornadoExecutionPlan#withIntraPlanConcurrency()` — 8 independent pipelines, sequential (1 stream) vs. concurrent (4-stream pool) with Nsight Systems timeline evidence of genuine kernel overlap. |
| [07-cuda-graph-benefit](07-cuda-graph-benefit/) | `CudaGraphBenefit.java` | Same 6-stage JIT task-graph run `nograph` vs. `graph` (`withCUDAGraph()`) for 50 executions each — isolates and quantifies the steady-state replay speedup that demo 02's capture/replay correctness demo didn't measure on its own. |
| [08-tensor-core-mma](08-tensor-core-mma/) | `TensorCoreMMA.java` | Smallest possible Tensor Core demo: one warp, one `M16N8K16` fp16 tile, exactly one `mma.sync.aligned` instruction (confirmed via `--printKernel`), next to a scalar no-MMA reference kernel with zero `mma.sync` instructions. |
| [11-integrated-showcase](11-integrated-showcase/) | `IntegratedShowcase.java` | Everything at once: JIT kernel + cuBLAS library task (demo 04's shape) x 6 independent chains, run baseline / `withIntraPlanConcurrency()` (demo 06) / `withCUDAGraph()` (demo 07) / both combined (experimental — not found anywhere upstream, probed live), plus demo 08's Tensor Core `mma.sync` kernel as a bonus stage. |

`tornado.args` is a `tornado --generate-argfile` output, generated against
the pinned build in `env/versions.env` and committed for the `java @arg-file`
reproducibility path documented in `docs/run-conventions.md`. Regenerate it
if the JDK or TornadoVM build changes.

All demos were run and captured on the pinned CUDA backend
(`vendor/tornadovm` @ `99549c9862eda8d584e35e99924f9c865501eb3a`, RTX 4090).
Raw logs: `results/raw/02-hello-kernel/` (00, 01),
`results/raw/03-cuda-runtime-api/` (02),
`results/raw/04-cublas-hybrid/` (04),
`results/raw/05-cufft-hybrid/` (05),
`results/raw/06-cuda-streams/` (06, incl. Nsight Systems `.nsys-rep` traces),
`results/raw/07-cuda-graph-benefit/` (07),
`results/raw/08-tensor-core-mma/` (08, incl. `--printKernel` generated-code
evidence; Nsight Compute hardware-counter evidence blocked on this machine,
see `results/failures/08-nsight-compute-permission.md`),
`results/raw/11-integrated-showcase/` (11, incl. Nsight Systems `.nsys-rep` +
parsed CSVs).

Dedicated Nsight Systems profiling pass (CUDA API/kernel/memcpy timing,
launch overhead, isolated from the timed runs above) across demos 04, 05, 07,
08: `results/raw/09-profiling/PROFILING-SUMMARY.md` — Nsight Compute
hardware-counter metrics (occupancy, GPU utilization %, memory throughput %,
instruction mix, tensor-pipe activity) remain blocked on this machine, same
root cause as task 08, re-verified 2026-08-21.

**Simplicity/consistency audit:** `docs/demo-audit-checklist.md` reviews every
demo against a presenter-usability checklist (one concept per class,
lightweight comments, deterministic output, simple invocation, `java
@arg-file`, JBang status) and records what was found and fixed.

**New to this repo?** `docs/profiling-quickstart.md` is a beginner-friendly,
copy-paste runbook: how to build/run any demo three ways (`tornado`,
`java @arg-file`, JBang), how to profile any demo with Nsight Systems and
why Nsight Compute is currently blocked here, where every report lives, and
a per-demo fallback/recovery table for presenting live.
