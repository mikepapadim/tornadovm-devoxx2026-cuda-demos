# Track A demos — Java + TornadoVM 6.0.0 CUDA

Presenter-friendly demos, one concept per class, increasing in complexity.
Each demo directory has its own `README.md` with build/run commands, a
`java @argfile` path, expected output, and a stage-failure fallback.

| # | Demo | Concept |
|---|------|---------|
| [00-hello-gpu](00-hello-gpu/) | `Hello.java` | Smallest TornadoVM program: one `@Parallel` task, one `TaskGraph`. |
| [01-first-cuda-kernel](01-first-cuda-kernel/) | `VectorAddKernel.java` | Vector add + `--printKernel` to show the actual generated CUDA source. |
| [02-cuda-runtime-api](02-cuda-runtime-api/) | `CudaGraphReplay.java` | `TornadoExecutionPlan#withCUDAGraph()` — CUDA graph capture/replay from Java, CUDA-only runtime API (not a vendor-library task). |
| [04-cublas-hybrid](04-cublas-hybrid/) | `CuBlasSgemvHybrid.java` | One `TaskGraph`, three stages: JIT `scale` task → cuBLAS `sgemv` library task → JIT `bias` task, all on shared device buffers. |
| [05-cufft-hybrid](05-cufft-hybrid/) | `CuFftLowPassHybrid.java` | One `TaskGraph`, four stages: cuFFT `forward` (R2C) → JIT `lowPass` task → cuFFT `inverse` (C2R) → JIT `normalize` task, a GPU-resident low-pass filter. |
| [06-cuda-streams](06-cuda-streams/) | `CudaStreamsOverlap.java` | `TornadoExecutionPlan#withIntraPlanConcurrency()` — 8 independent pipelines, sequential (1 stream) vs. concurrent (4-stream pool). |
| [07-cuda-graph-benefit](07-cuda-graph-benefit/) | `CudaGraphBenefit.java` | Same 6-stage JIT task-graph run `nograph` vs. `graph` (`withCUDAGraph()`) for 50 executions each — isolates and quantifies the steady-state replay speedup that demo 02's correctness demo doesn't measure. |
| [08-tensor-core-mma](08-tensor-core-mma/) | `TensorCoreMMA.java` | Smallest possible Tensor Core demo: one warp, one `M16N8K16` fp16 tile, exactly one `mma.sync.aligned` instruction (confirmed via `--printKernel`), next to a scalar no-MMA reference kernel with zero `mma.sync` instructions. |
| [11-integrated-showcase](11-integrated-showcase/) | `IntegratedShowcase.java` | Everything at once: JIT kernel + cuBLAS library task (demo 04's shape) × 6 independent chains, run baseline / `withIntraPlanConcurrency()` (demo 06) / `withCUDAGraph()` (demo 07) / both combined (experimental), plus demo 08's Tensor Core `mma.sync` kernel as a bonus stage. |

## Building and running

```bash
source ../scripts/setup-env.sh      # from a demo directory; or scripts/setup-env.sh from the repo root
cd 00-hello-gpu
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . Hello.java

tornado --classpath . Hello                          # canonical launcher
java @$TORNADOVM_HOME/tornado-argfile -cp . Hello    # reproducibility path
```

No `--enable-preview` anywhere: the pinned `6.0.0-jdk22plus-cuda` SDK is a
non-preview build (`etc/tornado.jdk`: floor 22, preview false), unlike
`6.0.0-jdk21-cuda`, which is JDK-21-only. See the repo README for why that
distinction matters.

**The argfile is not committed.** `tornado --generate-argfile` writes it to
`$TORNADOVM_HOME/tornado-argfile` with absolute, JDK-specific flags
(`-XX:+EnableJVMCI` is required on JDK ≤ 26 and fatal on 27+), so it belongs to
the installed SDK, not to this repo. `scripts/setup-env.sh` regenerates it for
whichever JDK is active.

`bash ../scripts/run-all-demos.sh` compiles and runs all nine demos both ways
and exits non-zero on any failure.

## Evidence

All nine demos were re-run on TornadoVM 6.0.0 / JDK 25.0.2 / RTX 4090:
27/27 checks pass (9 compiles + 9 `tornado` runs + 9 `java @argfile` runs).
Logs: `results/raw/18-tornadovm-6-migration/`, summary in
`results/raw/18-tornadovm-6-migration/run-all-demos.log`.

Earlier evidence from the 5.2.1 source-built pin is kept unmodified under
`results/raw/02-hello-kernel/` … `results/raw/17-final-rehearsal/` for
historical comparison. Where a per-demo README cites numbers, it cites the
6.0.0 run and says so.

Nsight Systems traces from the 5.2.1 pass (kernel/memcpy timing, stream
overlap timelines) remain valid as mechanism evidence and are still cited:
`results/raw/06-cuda-streams/`, `results/raw/09-profiling/`. Nsight Compute
hardware-counter metrics remain blocked on this machine — see
`results/failures/08-nsight-compute-permission.md`.

**Simplicity/consistency audit:** `docs/demo-audit-checklist.md`.

**New to this repo?** `docs/profiling-quickstart.md` is a copy-paste runbook:
how to build/run any demo, how to profile with Nsight Systems, and a per-demo
fallback table for presenting live.
