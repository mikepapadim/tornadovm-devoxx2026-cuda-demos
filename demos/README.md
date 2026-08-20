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

`tornado.args` is a `tornado --generate-argfile` output, generated against
the pinned build in `env/versions.env` and committed for the `java @arg-file`
reproducibility path documented in `docs/run-conventions.md`. Regenerate it
if the JDK or TornadoVM build changes.

All demos were run and captured on the pinned CUDA backend
(`vendor/tornadovm` @ `99549c9862eda8d584e35e99924f9c865501eb3a`, RTX 4090).
Raw logs: `results/raw/02-hello-kernel/` (00, 01),
`results/raw/03-cuda-runtime-api/` (02),
`results/raw/04-cublas-hybrid/` (04),
`results/raw/05-cufft-hybrid/` (05).
