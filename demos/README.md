# Track A demos — Java + TornadoVM CUDA

Presenter-friendly demos, one concept per class, increasing in complexity.
Each demo directory has its own `README.md` with build/run commands, a
`java @arg-file` path, expected output, and a stage-failure fallback.

| # | Demo | Concept |
|---|------|---------|
| [00-hello-gpu](00-hello-gpu/) | `Hello.java` | Smallest TornadoVM program: one `@Parallel` task, one `TaskGraph`. |
| [01-first-cuda-kernel](01-first-cuda-kernel/) | `VectorAddKernel.java` | Vector add + `--printKernel` to show the actual generated CUDA source. |

`tornado.args` is a `tornado --generate-argfile` output, generated against
the pinned build in `env/versions.env` and committed for the `java @arg-file`
reproducibility path documented in `docs/run-conventions.md`. Regenerate it
if the JDK or TornadoVM build changes.

All demos were run and captured on the pinned CUDA backend
(`vendor/tornadovm` @ `99549c9862eda8d584e35e99924f9c865501eb3a`, RTX 4090).
Raw logs: `results/raw/02-hello-kernel/`.
