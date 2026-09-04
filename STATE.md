# Autonomous Study State

## Batch 00 — TornadoVM CUDA baseline pinned (2026-08-19)

Task `auto/tasks/00.md` is done. What was measured, all Observed:

- Cloned `beehive-lab/TornadoVM` `develop` at SHA `99549c9862eda8d584e35e99924f9c865501eb3a` into `vendor/tornadovm` (gitignored — upstream checkout, not committed in place; pinned via `env/versions.env`). `develop` and `master` pointed at the same commit at clone time.
- Machine: RTX 4090 (24564 MiB, compute cap 8.9), driver 565.57.01, CUDA toolkit (nvcc/ptxas) 12.6.85, JDK 21.0.2 (sdkman), Maven 3.6.3, gcc 11.4.0, Ubuntu 22.04.5, kernel 6.8.0-58-generic. Full detail in `env/versions.env` and `results/raw/00-baseline/MANIFEST.md`.
- `make BACKEND=cuda` in `vendor/tornadovm` builds successfully (all Maven modules `SUCCESS`, installer reports `Backend : CUDA`, `Commit : 99549c9`). Log: `results/raw/00-baseline/tornadovm-build-cuda.log`.
- `tornado --devices` (after `source vendor/tornadovm/setvars.sh`) shows exactly one CUDA driver, one device (`0:0`), `NVIDIA GeForce RTX 4090`. Log: `results/raw/00-baseline/tornado-devices.log`.
- Smoke example `uk.ac.manchester.tornado.examples.VectorAddInt` (size 256) run via `tornado --enableProfiler console`: 10/10 iterations report `Result is correct`; profiler JSON confirms `BACKEND=CUDA`, `DEVICE=NVIDIA GeForce RTX 4090`, `DEVICE_ID=0:0` for every run. Log: `results/raw/00-baseline/vectoradd-int-cuda-run.log`.

Acceptance criteria for task 00 (env/versions.env + manifest exist; CUDA backend builds; smoke example runs on CUDA; exact SHA recorded) are all met and verified by the commands above.

## Batch 01 — Hybrid API inventory verified (2026-08-19)

Task `auto/tasks/01.md` is done. SHA re-checked against `env/versions.env` before reuse: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, checkout clean. Deliverable: `docs/hybrid-api-inventory.md`. What was measured, all Observed unless marked otherwise:

- Read `HYBRID_API_GUIDE.md` and `docs/source/hybrid-api.rst` in the pinned tree, then verified every structural claim against source: `TaskGraph.libraryTask` has exactly 20 overloads (grep-counted), the SPI classes (`LibraryTaskDescriptor`, `TornadoLibraryProvider`, `LibraryContext`, `LibraryInvocation`, `TornadoNativeStreamSupport`) exist at the documented paths, `persistOnDevice`/`consumeFromDevice`/`withCUDAGraph` exist, and `tornado.py` auto-adds all 5 provider modules to `--add-modules`.
- Counted factory methods per provider class directly in source and cross-checked against the guide's tables (§3 of `docs/hybrid-api-inventory.md`). **Found and recorded two doc/source gaps**: `CuBlas.java` has 7 factories vs. 5 documented (missing `cublasGemmExFP16FP32`, `cublasGemmExBF16`); `Cutlass.java` has 10 vs. 4 documented (missing `cutlassBgemm`, 4 more fused-activation variants, and `cutlassHgemmBatched`). CuBlasLt, cuFFT, cuDNN, cuSPARSE tables matched source exactly.
- Ran the guide's quick-start example as-is: `tornado -m tornado.cublas/uk.ac.manchester.tornado.cublas.tests.TestCuBlasSgemvWithTornadoVMTasksPOST` — 5/5 PASS, JIT-pre → cuBLAS SGEMV → JIT-post matches sequential reference. Log: `results/raw/01-hybrid-api/cublas-sgemv-post.log`.
- Ran all 6 hybrid-API unit-test suites in `tornado-unittests` (`tornado-test -V ...`): cuBLAS 12/12, cuBLASLt 6/6, cuFFT 8/8, cuDNN 9/9, cuSPARSE 11/11, CUTLASS 25/25 — **71/71 PASS, 0 unsupported**, confirming all five native providers (including cuDNN/libcudnn9 and CUTLASS/CMake-fetched headers) are actually functional on this build, not silently degraded. Logs: `results/raw/01-hybrid-api/unittest-*.log`.
- Checked the guide's claim that cuTENSOR lives on an upstream branch `hybrid-cutensor`: `git ls-remote --heads https://github.com/beehive-lab/TornadoVM.git | grep -i tensor` returned **no results** — that branch does not currently exist on the public remote. Flagged as an unverifiable/stale doc claim in the inventory; cuTENSOR itself is confirmed absent from this build (no `tornado-cutensor` module).
- Confirmed no public raw CUDA-runtime Java API (`cudaMalloc`/`cudaMemcpy`) exists outside library tasks — that track is explicitly deferred to `auto/tasks/03.md`.

Acceptance for task 01 (`docs/hybrid-api-inventory.md` contains source paths and runnable verification commands; changed/unsupported claims marked) is met. Note: `docs/hybrid-api-inventory.md` as committed was finalized by a concurrent iteration on the same task (see operational note below) and additionally covers `FrequencyFilterExample` (mixed JIT+cuFFT in one graph) and `MatrixVectorRowMajorWithCuBlas` (JIT-kernel vs. cuBLAS comparison) with their own passing logs — reviewed and consistent with the findings above; the doc/source factory-count gaps and cuTENSOR-branch check from this invocation are additive evidence not contradicted by it.

### Operational note — concurrent supervisor iterations observed

While running this task, `ps` showed **two separate `claude -p` processes** (PIDs 543845 and 544254, both launched by `auto/logs/supervisor.log` for `next=01`, ~40s apart) executing simultaneously against this same non-worktree working directory, alongside a third interactive session (PID 442317). `supervisor.log` shows two consecutive `git pull --rebase` failures ("cannot pull with rebase: You have unstaged changes") before each launch, and the supervisor appears to proceed to spawn a new iteration even after that failure rather than aborting or waiting. Extra untracked files not produced by this invocation (`frequency-filter-example.log`, `matrix-vector-cublas.log`, `test-cudnn.log` (0 bytes), `test-cusparse.log`) appeared in `results/raw/01-hybrid-api/` during this run, consistent with a second concurrent instance also working task 01 in the same directory. This invocation only staged/committed files it authored itself (listed above); the other instance's files were left untouched, uncommitted. **This is a supervisor-script hazard, not a task-01 problem** — the loop-runner should ensure only one iteration runs at a time (e.g. a lockfile) before the next batch, otherwise two instances can race on `STATE.md`, task-completion markers, and commits.

## Batch 02 — Hello GPU + first CUDA kernel demos (2026-08-19)

Task `auto/tasks/02.md` is done. SHA re-checked against `env/versions.env` before starting and re-verified after: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, checkout clean; no concurrent-instance artifacts found under `demos/` or `results/raw/02-hello-kernel/` before this run (see operational note in batch 01 — checked `ps`/`git status` again, no other `claude -p` processes were racing this specific task's files). Deliverable: `demos/00-hello-gpu/`, `demos/01-first-cuda-kernel/`, `demos/README.md`, `demos/tornado.args`. All Observed:

- `demos/00-hello-gpu/Hello.java`: smallest TornadoVM program (one `@Parallel` loop adding 1 to an `IntArray`, one `TaskGraph`, one `TornadoExecutionPlan`), written fresh for this repo (API shape modeled on `vendor/tornadovm/tornado-examples/.../arrays/ArrayAddInt.java`, not copied). Compiles with `javac --release 21 --enable-preview` against the pinned `tornado-api-5.2.1-jdk21-dev.jar` (preview features required — `tornado-api` classes on this build use JDK 21 preview features, discovered via a compile-time error, not assumed). Runs correctly both via `tornado --classpath . Hello` and via `java @tornado.args -cp . Hello`. Logs: `results/raw/02-hello-kernel/hello-run.log`, `hello-run-javaargfile.log`. Profiler run (`--enableProfiler console`) confirms `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"`: `results/raw/02-hello-kernel/hello-run-profiler.log`.
- `demos/01-first-cuda-kernel/VectorAddKernel.java`: element-wise float vector add (size 1024, CLI-configurable), same build/run pattern. Ran with `tornado --enableProfiler console --printKernel --classpath . VectorAddKernel 1024`: captured the actual generated `extern "C" __global__ void vectorAdd(...)` CUDA source TornadoVM's JIT produced from the `@Parallel` loop, `Result is correct`, profiler confirms CUDA backend/RTX 4090. Also ran via `java @tornado.args -cp . VectorAddKernel 1024` — correct, though the profiler is silent in that path (`Enable the profiler with: -Dtornado.profiler=True`, `Total time: 0 ns`) since `--enableProfiler` is a `tornado.py`-only flag, not part of the generated argfile — noted in the demo's README as an observed difference between the two run paths, not a bug. Log: `results/raw/02-hello-kernel/vectoraddkernel-run.log`, `vectoraddkernel-run-javaargfile.log`.
- Both demos re-verified end-to-end from a clean state (deleted `.class` files, re-sourced `vendor/tornadovm/setvars.sh`, rebuilt, reran) exactly per their README instructions before finalizing, to confirm the committed instructions are actually reproducible and not just "worked once."
- `demos/tornado.args`: committed copy of `tornado --generate-argfile` output against the pinned build, used for the `java @arg-file <MainClass>` reproducibility path required by `docs/run-conventions.md`. Contains machine-specific absolute paths (consistent with how `env/versions.env` records absolute paths) — README notes to regenerate if the JDK/TornadoVM build changes.
- JBang path: **not verified**, recorded as such rather than guessed. `which jbang` → command not found on this machine (checked 2026-08-19). Both demo READMEs show the documented-but-unverified JBang invocation shape and explicitly say not to run it live until tested, per `docs/run-conventions.md`'s requirement not to claim `jbang` works without testing on the pinned environment.
- Discovered while running `tornado --generate-argfile` without first sourcing `vendor/tornadovm/setvars.sh`: the `tornado` binary on the default shell `PATH` resolves to an unrelated sdkman-managed build (`5.0.0-jdk21-cuda`, not the pinned `5.2.1-jdk21-dev` @ `99549c9`). This is expected/correct behavior (PATH precedence, not a repo bug) but is a real footgun for future invocations — **always `source vendor/tornadovm/setvars.sh` (or verify `tornado --version` prints commit `99549c9`) in the same shell/command as any `tornado`/`java` invocation**, since the Bash tool does not persist shell state (env vars) across separate tool calls, only cwd.

Acceptance for task 02 ("Examples run on CUDA and are understandable by a Java developer in about one minute") is met: both examples ran correctly on the CUDA backend with captured logs, and both READMEs explain the concept in a short paragraph plus a fenced code sample, aimed at a ~1-minute read.

## Batch 03 — CUDA runtime API demo (2026-08-19)

Task `auto/tasks/03.md` is done. SHA re-checked before and after: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, checkout clean. Checked `ps` for concurrent `claude -p` instances per the batch-01 operational note: only this invocation's own process pair was present, no race. Deliverable: `demos/02-cuda-runtime-api/`. All Observed:

- Batch 01 flagged Track A1 ("CUDA runtime API access from Java, outside library tasks") as an open question — no `cudaMalloc`/`cudaMemcpy`-style binding exists in the pinned tree. Searched `tornado-api`/`tornado-runtime` source this batch and found the actual A1 surface: `TornadoExecutionPlan.withCUDAGraph()`, `.withIntraPlanConcurrency()`, and `.withStagedTransfers()` — all three documented in the class's own javadoc as "currently realised on the CUDA backend" (CUDA graph capture/replay, multi-stream execution, and pinned-memory staging respectively; no-op on OpenCL/Metal). `tornado-unittests/.../streams/TestCUDAStreams.java` independently confirms these are real, exercised code paths (capture/replay, intra-plan concurrency, staged transfers), not dead API.
- `demos/02-cuda-runtime-api/CudaGraphReplay.java`: an `axpy` task-graph executed under `plan.withCUDAGraph()` for 8 replays, mutating `x`/`y` before each replay and validating `result` against a CPU-computed expected value after each one (a stale/incorrect capture would return the first replay's output). Compiles with the same `javac --release 21 --enable-preview` pattern as demos 00/01. Ran via both `tornado --enableProfiler console --classpath . CudaGraphReplay 8192 8` and `java @../tornado.args -cp . CudaGraphReplay 8192 8` — **8/8 replays correct** in both paths. Logs: `results/raw/03-cuda-runtime-api/cudagraphreplay-run.log`, `cudagraphreplay-run-javaargfile.log`.
- Profiler evidence in the captured log confirms `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"` on the first (capture) execution, and shows `TOTAL_TASK_GRAPH_TIME` collapsing from ~95ms (JIT compile + graph capture) to tens-of-microseconds on replays 1–7 — an observable, presenter-visible effect of CUDA-graph replay vs. re-dispatch, not just a correctness claim.
- Re-verified from a clean state (deleted `.class` files, rebuilt, reran via `tornado --classpath .`) before finalizing — reproducible, not a one-off.
- JBang: **not verified** — `which jbang` → command not found on this machine (checked 2026-08-19, same finding as demos 00/01). README shows the documented-but-untested shape and explicitly says not to run it live.
- `docs/hybrid-api-inventory.md` §6 updated to resolve its open A1 question with these findings (source paths + demo pointer), rather than leaving it as an unconfirmed hypothesis.

Acceptance for task 03 ("CUDA-only demo runs with deterministic validation; `java @run.args` works; JBang path is tested or explicitly marked unsupported") is met: deterministic per-replay validation against a CPU reference, both the `tornado` launcher and `java @../tornado.args` (this repo's established arg-file convention, per `docs/run-conventions.md` and demos 00/01) run correctly, and JBang is explicitly marked unverified with a reason.

## Batch 04 — cuBLAS Hybrid API demo (2026-08-19)

Task `auto/tasks/04.md` is done. SHA re-checked before and after: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, checkout clean. Checked `ps` for concurrent `claude -p` instances per the batch-01 operational note: only this invocation's own process pair (`timeout` wrapper + `claude`) was present, no race. Deliverable: `demos/04-cublas-hybrid/`. All Observed:

- `demos/04-cublas-hybrid/CuBlasSgemvHybrid.java`: one `TaskGraph` mixing a JIT `scale` task (`@Parallel` loop, matrix *= 2.0), the `CuBlas.cublasSgemv` library task (`y = alpha*op(A)*x + beta*y`, `CUBLAS_OP_T` for row-major storage per `CuBlas.java`'s javadoc), and a JIT `bias` task (`@Parallel` loop, output += 1.0) — modeled on the shape of `vendor/tornadovm/tornado-cublas/.../tests/TestCuBlasSgemvWithTornadoVMTasksPOST.java` (read in full this batch) but written fresh for this repo with deterministic (non-random) matrix/vector fill and CLI-configurable `<m> <n> <iterations>`.
- Compiles with `javac --release 21 --enable-preview` against `tornado-api-5.2.1-jdk21-dev.jar` + `tornado-cublas-5.2.1-jdk21-dev.jar` (both present in `$TORNADOVM_HOME/share/java/tornado/` on this pinned build). Ran via both `tornado --enableProfiler console --classpath . CuBlasSgemvHybrid 8 8 5` and `java @../tornado.args -cp . CuBlasSgemvHybrid 8 8 5` — **5/5 iterations correct** in both paths, output validated each iteration against a sequential Java scale→matvec→bias reference. Logs: `results/raw/04-cublas-hybrid/cublassgemvhybrid-run.log`, `cublassgemvhybrid-run-javaargfile.log`.
- Profiler evidence confirms all three stages (`cublasHybrid.scale`, `cublasHybrid.sgemv`, `cublasHybrid.bias`) report `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"` — the JIT tasks and the cuBLAS call genuinely execute on the same CUDA device inside one task graph, not a fallback. First iteration's `TOTAL_TASK_GRAPH_TIME` (~171ms) is dominated by Graal+driver JIT compilation of the two Java tasks; iterations 1-4 drop to ~350-500 microseconds once compiled code is reused — a presenter-visible effect, captured in the log.
- Found and fixed a bug during development, not left in the committed demo: an early draft reset `matrix`/`vector` between iterations by reassigning the local variable to a new `FloatArray`, which does nothing because the `TaskGraph` captured the original array object by reference at `.task()`/`.libraryTask()` build time. Fixed by mutating the existing arrays' contents in place (`fillDeterministicMatrix`/`fillDeterministicVector`) before each `execute()` call, matching the pattern in the upstream test this demo is modeled on.
- Re-verified from a clean state (deleted `.class` files, rebuilt, reran via `tornado --classpath .`) before finalizing.
- JBang: **not verified** — `which jbang` → command not found (checked 2026-08-19, same finding as demos 00-03). README shows the documented-but-untested shape and explicitly says not to run it live.
- `docs/hybrid-api-inventory.md` §3 and §6 updated: A2 (cuBLAS call) and A4 (Java-kernel+cuBLAS in one graph) are now backed by this repo's own demo, not just upstream tests/examples; A3/A5-A7 remain open for later tasks.

Acceptance for task 04 ("Runnable CUDA-only demo, Java reference validation, exact arg-file invocation, JBang path if supported, captured evidence") is met: the demo runs correctly on CUDA only, validates every iteration against a Java sequential reference, the exact `java @../tornado.args` invocation is documented and tested, JBang is explicitly marked unverified with a reason (not silently omitted), and evidence is captured under `results/raw/04-cublas-hybrid/`.

## Batch 05 — cuFFT Hybrid API demo (2026-08-20)

Task `auto/tasks/05.md` is done. SHA re-checked before and after: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, checkout clean. Checked `ps` for concurrent `claude -p` instances per the batch-01 operational note: only this invocation's own process pair was present, no race. Deliverable: `demos/05-cufft-hybrid/`. All Observed:

- Read `CuFft.java` (all 8 factory methods: 1D/2D C2C forward/inverse, R2C forward, C2R inverse, double-precision Z2Z) and two source-backed references in full before writing the demo: `tornado-cufft/.../tests/FrequencyFilterExample.java` (upstream's own mixed JIT+cuFFT low-pass example) and `tornado-unittests/.../cufft/TestCuFft.java` (7 tests incl. `testRoundTripWithCudaGraph`, validated against a Java DFT reference).
- Ran the upstream `FrequencyFilterExample` as a sanity probe before writing the repo demo: `tornado -m tornado.cufft/uk.ac.manchester.tornado.cufft.tests.FrequencyFilterExample 4096 16` → `Result is correct`, max error 1.67e-6. Log: `results/raw/05-cufft-hybrid/upstream-frequencyfilterexample.log`.
- `demos/05-cufft-hybrid/CuFftLowPassHybrid.java`: one `TaskGraph` mixing `CuFft.cufftForwardR2C` (library task) → JIT `lowPass` task (zeroes bins ≥ cutoff) → `CuFft.cufftInverseC2R` (library task) → JIT `normalize` task (divides by n, undoing cuFFT's unnormalized scaling) — modeled on `FrequencyFilterExample`'s shape but written fresh with CLI-configurable `<n> <cutoff> <iterations>` and per-iteration profiler timing, matching demo 04's pattern. Input is a deterministic sum of two low-frequency tones (kept) + one high-frequency tone (removed), validated each iteration against the exact analytic low-frequency signal (closed-form, not a numeric DFT).
- **Found and fixed a real compile-time bug, not left in the committed demo**: naming the fourth task's method `normalize` fails TornadoVM's sketch phase with `[ERROR] Java method name corresponds to an OpenCL Token. Change the Java method's name: normalize` — the `TornadoSketcher` rejects that identifier as a reserved OpenCL token even on a CUDA-only build (the check is shared across backends). Confirmed this is why both `FrequencyFilterExample` and `TestCuFft` independently use `scaleBy` for the same operation. Fixed by renaming to `scaleBy`; documented as a gotcha in the demo README so future demo authors don't hit it blind.
- Compiles with `javac --release 21 --enable-preview` against `tornado-api-5.2.1-jdk21-dev.jar` + `tornado-cufft-5.2.1-jdk21-dev.jar`. Ran via both `tornado --enableProfiler console --classpath . CuFftLowPassHybrid 4096 16 5` and `java @../tornado.args -cp . CuFftLowPassHybrid 4096 16 5` — **5/5 iterations correct** in both paths (maxError ~4.77e-7 each iteration). Logs: `results/raw/05-cufft-hybrid/cufftlowpasshybrid-run.log`, `cufftlowpasshybrid-run-javaargfile.log`.
- Profiler evidence confirms all four stages (`cufftHybrid.forward`, `cufftHybrid.lowPass`, `cufftHybrid.inverse`, `cufftHybrid.normalize`) report `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"`. First iteration's `TOTAL_TASK_GRAPH_TIME` (~86ms, dominated by `TOTAL_GRAAL_COMPILE_TIME` ~48ms + `TOTAL_DRIVER_COMPILE_TIME` ~11ms for the two JIT tasks) drops to ~380-460 microseconds on iterations 1-4 once compiled code is reused — same compile-then-reuse effect documented for demos 02/04.
- Re-verified from a clean state (deleted `.class` files, rebuilt, reran via `tornado --classpath .`) before finalizing — reproducible, not a one-off.
- JBang: **not verified** — `which jbang` → exit 1 (checked 2026-08-20, same finding as demos 00-04). README shows the documented-but-untested shape and explicitly says not to run it live.
- `docs/hybrid-api-inventory.md` §3 and §6 updated: A3 (cuFFT call) and A5 (Java-kernel+cuFFT in one graph) are now backed by this repo's own demo, not just upstream tests/examples; A6/A7 remain open for later tasks. `demos/README.md` updated with the new demo row and log path.

Acceptance for task 05 ("Runnable CUDA-only demo, deterministic validation, exact `java @run.args` invocation, JBang path if supported, captured evidence") is met: the demo runs correctly on CUDA only, validates every iteration against a closed-form analytic reference, the `java @../tornado.args` invocation is documented and tested, JBang is explicitly marked unverified with a reason, and evidence is captured under `results/raw/05-cufft-hybrid/`.

## Batch 06 — CUDA streams and async overlap demo (2026-08-20)

Task `auto/tasks/06.md` is done. SHA re-checked before and after: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, checkout clean. Checked `ps` for concurrent `claude -p` instances per the batch-01 operational note: only this invocation's own process pair was present, no race. Deliverable: `demos/06-cuda-streams/`. All Observed:

- Read `TestCUDAStreams.java` (correctness suite) and `TestStreamsPerformance.java` (wall-clock + Nsight-profiling guidance, incl. its documented `nsys profile --trace=cuda,nvtx` invocation) in full before writing the demo. Confirmed the actual A1 CUDA-streams surface is `TornadoExecutionPlan#withIntraPlanConcurrency()`: routes DAG-independent operations to H2D / COMPUTE-pool (default size 4) / D2H role streams, ordered by device events; documented "currently realised on the CUDA backend" in the class javadoc, same family as `withCUDAGraph()` (demo 02) and `withStagedTransfers()`.
- `demos/06-cuda-streams/CudaStreamsOverlap.java`: one `TaskGraph` with 8 independent H2D→compute→D2H pipelines (small-grid, heavy-inner-loop kernel per `TestStreamsPerformance`'s documented condition for kernels to actually co-reside rather than merely being issued on different streams), run 8 times each in **sequential** (no `withIntraPlanConcurrency()`) and **concurrent** (`withIntraPlanConcurrency()`) mode from the same JVM — modeled on `testManyIndependentUnitsMultiStream`/`testSmallKernelConcurrency` but written fresh for this repo with CLI-configurable `<units> <unitSize> <innerIterations> <executions> <mode>`. Every execution in both modes validated against a closed-form CPU reference; median wall-clock reported (printed, not asserted, matching upstream's own non-gating treatment of this number).
- Compiles with `javac --release 21 --enable-preview` against `tornado-api-5.2.1-jdk21-dev.jar` only (no vendor-library jar needed — this is CUDA-runtime, not a `.libraryTask`). Ran via both `tornado --classpath . CudaStreamsOverlap 8 32768 65536 8 both` and `java @../tornado.args -cp . CudaStreamsOverlap 8 32768 65536 8 both` — **all 16 executions (8 sequential + 8 concurrent) correct** in both paths. This run: sequential steady-state median ~2.2ms, concurrent steady-state median ~0.9ms (this-run-only number, explicitly not gated/guaranteed in the README). Logs: `results/raw/06-cuda-streams/cudastreamsoverlap-run.log`, `cudastreamsoverlap-run-javaargfile.log`.
- **Nsight Systems timeline evidence** (no GUI available on this machine, so verified textually from the `cuda_gpu_trace` CSV report instead of a screenshot): captured separate `nsys profile --trace=cuda` traces for `sequential`-only and `concurrent`-only runs (`nsys-sequential.nsys-rep`, `nsys-concurrent.nsys-rep`), extracted via `nsys stats --report cuda_gpu_trace --format csv`. **Sequential**: all 64 `computeSmall` kernel launches land on exactly 1 CUDA stream (id 14); steady-state launches are strictly back-to-back (each start ≈ previous end, no overlap). **Concurrent**: the 64 launches spread across exactly 4 CUDA streams (ids 16–19), matching the documented default COMPUTE-pool size; steady-state launches show genuine overlapping execution windows (e.g. stream 17 starts before stream 16's kernel ends) — real concurrent GPU execution, not just distinct stream IDs issued serially. Raw evidence: `results/raw/06-cuda-streams/nsys-{sequential,concurrent}.nsys-rep`, `*_cuda_gpu_trace.csv`, `nsys-timeline-evidence.txt`.
- Re-verified from a clean state (deleted `.class` files, rebuilt, reran via `tornado --classpath .`) before finalizing — reproducible, not a one-off.
- JBang: **not verified** — `which jbang` → exit 1 (checked 2026-08-20, same finding as demos 00-05). README shows the documented-but-untested shape and explicitly says not to run it live.
- `docs/hybrid-api-inventory.md` §6 (A1 entry) and `demos/README.md` updated with this demo's pointer and Nsight findings.

Acceptance for task 06 ("A simple sequential-vs-multi-stream demonstration exists with exact commands and a profiler trace showing the execution timeline or a documented API limitation") is met: the demo runs correctly on CUDA only in both modes with a Java/CPU reference validating every execution, exact `tornado`/`java @../tornado.args` commands are documented and tested, and a captured Nsight Systems profiler trace (both raw `.nsys-rep` and extracted CSV/text evidence, since no GUI is available on this machine) shows the execution timeline: 1-stream/no-overlap for sequential vs. 4-stream/genuine-overlap for concurrent.

## Batch 07 — CUDA Graph replay benefit quantified (2026-08-20)

Task `auto/tasks/07.md` is done. SHA re-checked before and after: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, checkout clean. Checked `ps` for concurrent `claude -p` instances per the batch-01 operational note: `ps aux` showed one `timeout` wrapper + one `claude` process, i.e. this invocation's own process pair (the supervisor launches `claude -p ...` under `timeout`), not a second racing instance. Deliverable: `demos/07-cuda-graph-benefit/`. All Observed:

- Task 07 asks to "demonstrate CUDA Graph capture/replay ... and show why graph launch/replay matters for repeated workloads." Batch 03's `demos/02-cuda-runtime-api/CudaGraphReplay.java` already demonstrated `withCUDAGraph()` capture/replay correctness, but its only timing evidence conflated first-execution JIT-compile time with graph capture, so it did not isolate or quantify the actual replay benefit. This batch closes that gap with a dedicated benefit-isolating demo rather than re-doing task 03's correctness demo.
- Read `CUDAGraph.cpp` (`vendor/tornadovm/tornado-drivers/cuda-jni/src/main/cpp/source/CUDAGraph.cpp`) in full: confirmed the JNI layer is real CUDA driver stream-capture (`cuStreamBeginCapture`/`cuStreamEndCapture`/`cuGraphInstantiate`/`cuGraphExecUpdate`/`cuGraphLaunch`/`cuGraphExecDestroy`/`cuGraphDestroy`), version-guarded for CUDA 11.x vs 12+ `cuGraphInstantiate` signatures. Also read `tornado-cublas/.../tests/TestCuBlasSgemvWithTasksCudaGraph.java`, an upstream example combining `withCUDAGraph()` with a cuBLAS library task (handle created before capture starts, since native handle creation allocates device memory which is illegal mid-capture) — confirms the mechanism generalizes beyond pure-JIT graphs, noted in the inventory but not independently re-demoed (out of scope for this task).
- `demos/07-cuda-graph-benefit/CudaGraphBenefit.java`: a 6-stage chain of small elementwise JIT tasks (each stage's output feeds the next, `EVERY_EXECUTION` transfers) run for 50 executions in two modes from the same JVM — `nograph` (plain `TornadoExecutionPlan`) and `graph` (`withCUDAGraph()`). Both modes discard execution 0 (JIT compile / capture) and report steady-state median wall-clock; every execution in both modes validated against a closed-form CPU reference, with the input array mutated before every `execute()` call so a stale replay would be caught. Written fresh for this repo, modeled conceptually on demo 06's sequential-vs-concurrent A/B pattern (same median/steady-state-exclude-first idiom) but applied to `withCUDAGraph()` instead of `withIntraPlanConcurrency()`.
- Compiles with `javac --release 21 --enable-preview` against `tornado-api-5.2.1-jdk21-dev.jar` only. Ran via `tornado --classpath . CudaGraphBenefit 4096 6 50 both`, `java @../tornado.args -cp . CudaGraphBenefit 4096 6 50 both`, and `tornado --enableProfiler console --classpath . CudaGraphBenefit 4096 6 5 both` — **all executions correct in both modes, all three runs**. Steady-state median (excl. first execution): 233.6us→36.1us (6.47x) plain, 238.3us→36.2us (6.58x) argfile, 1008.8us→143.7us (7.02x) under profiler (extra per-execution console-dump overhead in both modes, same direction). Re-verified from a clean rebuild (`rm *.class`, rebuild, rerun 20 executions) before finalizing: 344.2us→40.1us (8.59x), same qualitative result, confirming reproducibility rather than a one-off number. Profiler evidence confirms every stage in every execution reports `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"` in both modes. Logs: `results/raw/07-cuda-graph-benefit/cudagraphbenefit-run.log`, `cudagraphbenefit-run-javaargfile.log`, `cudagraphbenefit-profiler.log`.
- The demo's README explicitly labels the specific multiplier as this-run/this-GPU evidence, not a general TornadoVM performance guarantee (same caveat convention as demo 06), and gives a fallback (demo 02's simpler correctness-only demo) if a live re-run doesn't reproduce a clean speedup.
- JBang: **not verified** — `which jbang` → exit 1 (checked 2026-08-20, same finding as demos 00-06). README shows the documented-but-untested shape and explicitly says not to run it live.
- `docs/hybrid-api-inventory.md` §6 (A1 entry) and `demos/README.md` updated with this demo's pointer and headline numbers.

Acceptance for task 07 ("A minimal graph example exists if the current API supports it; otherwise capture a precise source-backed limitation and a useful adjacent demo. Do not invent graph support.") is met: `withCUDAGraph()` graph capture/replay is real and supported on this build (confirmed at the JNI/CUDA-driver-API level, not just the Java surface), a minimal example exists (this demo, plus demo 02's prior correctness-focused example), and it demonstrates — with captured, reproducible, multi-run evidence rather than an invented number — why graph replay matters for repeated workloads: a consistent multi-x steady-state speedup over plain repeated `execute()` for the same task-graph, correctness-validated every execution.

## Batch 08 — Tensor Core / MMA demo (2026-08-20)

Task `auto/tasks/08.md` is done. SHA re-checked before and after: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, checkout clean. Checked `ps` for concurrent `claude -p` instances per the batch-01 operational note: only this invocation's own process pair was present, no race. Deliverable: `demos/08-tensor-core-mma/`. All Observed unless marked otherwise:

- Found the actual Tensor Core / MMA surface by source search (not assumed): `KernelContext` exposes `mmaFragment`/`mmaLoadA`/`mmaLoadB`/`mmaLoadBSwizzled`/`mma`/`mmaBF16`/`mmaInt8`/`mmaFP8E4M3`/`mmaFP8E5M2`/`mmaStore` plus `cp.async` helpers; `CUDAGraphBuilderPlugins.java` lowers these to dedicated Graal nodes (`CUDAMMALoadANode` etc., all under `tornado-drivers/cuda/.../graal/nodes/`) that the CUDA backend LIR emits as real inline PTX asm. The pinned tree already ships a full worked example, `tornado-examples/.../compute/MatrixMultiplicationMMA.java` (six GEMM variants: tiled-fp16-no-MMA baseline, MMA, MMA+cp.async, MMA+bf16, MMA with `[N,K]` weight layout, MMA+swizzled shared memory), read in full before writing this task's own demo.
- Ran the upstream `MatrixMultiplicationMMA` example as a sanity probe at `512 512 512`: all six variants **PASSED** validation (`results/raw/08-tensor-core-mma/upstream-mma-512.log`). `--printKernel` on the same run (`mma-printkernel-512.log`) shows 80 occurrences of real `mma.sync`/`ldmatrix.sync.aligned` asm across the four MMA kernels — confirmed this is genuine PTX-level Tensor Core codegen, not a simulated/software path, before building anything new.
- `demos/08-tensor-core-mma/TensorCoreMMA.java`: written fresh for this repo, deliberately the smallest possible MMA workload rather than a trimmed copy of the upstream benchmark (that kernel's multi-warp/swizzled tiling math was judged too intricate to safely abbreviate without risking a subtle correctness bug) — one warp (32 threads), one `M16N8K16` fp16 tile (`C[16,8]=A[16,16]*B[16,8]`), exactly one `mma.sync` call, next to a scalar (no-MMA, one-thread-per-output-element) reference kernel computing the identical tile from `KernelContext.globalIdx/globalIdy`. Both validated against the same closed-form CPU reference (exact match, max abs err `0.00000`, since the deterministic bounded fp16 inputs and 16-term dot product are exactly representable).
- Compiles with `javac --release 21 --enable-preview` against `tornado-api-5.2.1-jdk21-dev.jar` only (no vendor-library jar — this is CUDA-backend codegen, not a `.libraryTask`). Ran via `tornado --classpath .`, `java @../tornado.args -cp .`, and `tornado --enableProfiler console --classpath .` — all three: both kernels validate exactly, profiler confirms `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"` for both tasks. Re-verified from a clean rebuild (`rm *.class`, rebuild, rerun) before finalizing. Logs: `results/raw/08-tensor-core-mma/tensorcoremma-run.log`, `tensorcoremma-run-javaargfile.log`.
- **Generated-code evidence** (`--printKernel`, `results/raw/08-tensor-core-mma/tensorcoremma-printkernel.log`): `gemmMMASingleTile` contains exactly one `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` plus its two `ldmatrix.sync.aligned` operand loads; `gemmScalarFp16`, compiled from the *same run*, contains zero occurrences of `mma.sync` anywhere in its body — a direct source-backed comparison from one invocation.
- **Nsight Compute hardware-counter evidence: blocked**, recorded per the evidence-rules contract rather than skipped silently. Root causes found and documented in full (commands, exact errors, environment) in `results/failures/08-nsight-compute-permission.md`: (1) the `ncu` on `PATH` resolves to the newest installed Nsight Compute, `2026.2.1.0`, which cannot connect to this machine's driver (`565.57.01`) at all (`Nsight Compute failed to connect to the CUDA driver`) — reproduced on a trivial standalone CUDA C program, confirming it's an environment/version-mismatch issue, not TornadoVM-specific; (2) the older, driver-matching install (`/opt/nvidia/nsight-compute/2024.3.2/ncu`) connects fine but is refused with `ERR_NVGPUCTRPERM` (GPU performance-counter access is admin-only on this driver by default); (3) no passwordless `sudo` is available (`sudo -n true` → password required) in this unattended run, and modifying the NVIDIA kernel module's `NVreg_RestrictProfilingToAdminUsers` parameter system-wide (needs a `modprobe.d` change + reboot) was judged out of scope for an autonomous, reversible-action-only task. Per task 08's own disjunctive acceptance criterion ("profiler **or** generated-code evidence"), the generated-code evidence above is sufficient on its own — no Tensor-Core performance number is claimed anywhere in this batch's output.
- `demos/README.md` and `docs/hybrid-api-inventory.md` updated with this demo's pointer and findings (a new bullet in §6, since Tensor Core/MMA codegen isn't one of the A0-A7 hybrid-API library-provider items — it's CUDA-backend `KernelContext` intrinsic lowering).
- JBang: **not verified** — `which jbang` → exit 1 (checked 2026-08-20, same finding as demos 00-07).

Acceptance for task 08 ("Only call it Tensor Core/MMA accelerated when profiler or generated-code evidence supports that claim. Record relevant instruction/activity metrics and exact GPU/tool versions.") is met: the Tensor-Core-accelerated claim is backed by direct generated-code evidence (exact `mma.sync`/`ldmatrix` PTX asm, contrasted with a zero-`mma.sync` scalar kernel from the same compile), not an unverified inference; the Nsight Compute profiler path was genuinely attempted, found blocked for a documented and verified reason (not silently skipped or faked), and is fully reproducible for a future invocation with GPU counter access; exact GPU (RTX 4090, driver 565.57.01) and tool versions (CUDA 12.6.85, Nsight Compute 2024.3.2.0/2026.2.1.0) are recorded.

## Batch 09 — Profile CUDA demos with NVIDIA tools (2026-08-21)

Task `auto/tasks/09.md` is done. SHA re-checked before and after: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, checkout clean. Checked `ps` for concurrent `claude -p` instances per the batch-01 operational note: only this invocation's own `timeout`+`claude` process pair (PIDs 656869/656870) plus the pre-existing long-running interactive session (PID 442317, present since batch 01) were found — no race. `nvidia-smi` confirmed the GPU was idle (`0 %`, `4 MiB` used) before every profiled run. Deliverable: `results/raw/09-profiling/`. All Observed unless marked otherwise:

- Rebuilt demos 04 (`CuBlasSgemvHybrid`), 05 (`CuFftLowPassHybrid`), 07 (`CudaGraphBenefit`), 08 (`TensorCoreMMA`) fresh (`rm *.class`, `javac --release 21 --enable-preview`) against the jars each demo's own README documents, then profiled each with `nsys profile --trace=cuda,nvtx,osrt` using workload sizes chosen for a representative trace (`512 512 10`, `4096 16 20`, `4096 6 50 both`, and the default single-tile MMA workload respectively) — deliberately separate runs/sizes from each task's own committed timed-run evidence, per the task's "keep profiled runs separate from timed runs" instruction. All four profiled runs validated correct (10/10, 20/20, 300/300 stage-executions, both MMA/scalar kernels).
- **Found and worked around a real correctness bug while picking a profiling size for demo 05**: `CuFftLowPassHybrid 65536 16 10` triggered TornadoVM's `[Bailout] Running the sequential implementation` fallback (silently ran on CPU, not CUDA) and every iteration reported `WRONG` (maxError 1.48, expected ~0). Not investigated further (out of scope for task 09 — a JIT-compile bailout at larger array sizes, distinct from task 05's tested/validated `4096`), but recorded here rather than silently discarded so a future task on cuFFT scaling limits has a starting point. Re-ran at the demo's own tested size (`4096`) with more iterations (`20`) instead — correct.
- Extracted `nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_kern_exec_sum --format csv` for all four `.nsys-rep` traces. Parsed and analyzed in `results/raw/09-profiling/PROFILING-SUMMARY.md`: per-kernel GPU execution duration (e.g. cuBLAS SGEMV kernel avg 2,556.7 ns vs. its JIT `scale`/`bias` neighbors 2,198.5 ns/1,132.8 ns; cuFFT R2C/C2R kernels ~2.9-3.0 us vs. JIT `lowPass`/`scaleBy` ~1.1-1.2 us), CUDA API call counts/timings for launch overhead (e.g. task 07: `cuGraphLaunch` 50 calls avg 11,070.9 ns each replaying 6 captured stages vs. `cuLaunchKernel` 306 calls avg 2,026.1 ns each, the CPU-API-side counterpart to task 07's already-measured GPU-side steady-state speedup), and H2D/D2H memcpy timing. One-time `cuCtxCreate_v2` cost (89-99% of each trace's "Time %" column, since traces capture whole-JVM start to finish) explicitly separated out from steady-state kernel/API activity so it isn't misread as compute cost.
- Cross-stream concurrency evidence was **not** re-captured here — task 06 already produced a dedicated sequential-vs-concurrent A/B design with full Nsight Systems timeline evidence (`results/raw/06-cuda-streams/`); `PROFILING-SUMMARY.md` references it rather than duplicating raw evidence, consistent with the "do not edit or delete existing raw evidence" rule.
- **Nsight Compute (`ncu`) hardware-counter metrics (GPU utilization %, occupancy, memory throughput %, instruction mix, tensor-pipe activity): re-verified BLOCKED**, same root cause as `results/failures/08-nsight-compute-permission.md` (originally recorded 2026-08-20) — re-ran both commands today rather than assuming the prior finding still held: (1) default `ncu` on `PATH` (now confirmed `2026.2.1.0`, build `38283040`) against `demos/08-tensor-core-mma` → `Nsight Compute failed to connect to the CUDA driver`; (2) driver-era-matching `/opt/nvidia/nsight-compute/2024.3.2/ncu` (`2024.3.2.0`, build `34861637`) → connects, then `ERR_NVGPUCTRPERM`; `sudo -n true` still requires a password in this unattended run. Logs: `results/raw/09-profiling/ncu-2026.2.1.0-connect-error.log`, `ncu-2024.3.2.0-nvgpuctrperm-error.log`. No GPU utilization/occupancy/throughput-%/instruction-mix/tensor-pipe number is claimed anywhere in this batch's output.
- Exact tool versions recorded for every number reported: `nsys --version` → `2024.5.1.113-245134619542v0`; both `ncu` versions above. `results/raw/09-profiling/PROFILING-SUMMARY.md` documents the exact `nsys profile`/`nsys stats` commands next to every reported figure.
- `demos/README.md` updated with a pointer to `results/raw/09-profiling/PROFILING-SUMMARY.md` and the re-confirmed Nsight Compute blocker.
- Compiled `.class` files from this batch's rebuilds were deleted before committing (not previously tracked in the repo; consistent with the established pattern of not committing build artifacts).

Acceptance for task 09 ("Raw profiler artifacts and parsed summaries exist; every reported number has its exact command/tool version; performance-counter failures are documented") is met: raw `.nsys-rep` traces + stdout logs + CSV summary reports for all four profiled demos are committed under `results/raw/09-profiling/`, `PROFILING-SUMMARY.md` parses them into a readable summary with the exact command and tool version next to every reported number, and the Nsight Compute performance-counter failure is documented with exact commands/errors/environment (re-verified today, not assumed from the prior batch).

## Batch 10 — GPULlama3.java current-state reproduction (2026-08-21)

Task `auto/tasks/10.md` is done. SHA re-checked before starting: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, checkout clean. Checked `ps` for concurrent `claude -p` instances per the batch-01 operational note: only this invocation's own `timeout`+`claude` process pair plus the long-running interactive session (PID 442317) were present, no race. `nvidia-smi` confirmed the GPU was idle before profiled runs. Deliverable: `vendor/GPULlama3.java` (new upstream checkout, gitignored, pinned by SHA), `docs/gpullama3-reproduction.md`, `env/versions.env` extended with `GPULLAMA3_*` vars. All Observed unless marked otherwise:

- Cloned `beehive-lab/GPULlama3.java` `main` at SHA `bbe42fdc8cd475bb6104cefa42118dd6e068538b` (2026-08-15) into `vendor/GPULlama3.java`. Documented build (`make` == `mvn install -DskipTests`) succeeds as-is (`results/raw/10-gpullama3/build.log`, `BUILD SUCCESS`).
- **Found and worked around a real, reproducible cross-repo version blocker**: `pom.xml` pins `tornado-api`/`tornado-runtime` to `5.0.0-jdk21` from Maven Central, but `llama-tornado` runs the built jar against this repo's pinned TornadoVM SDK (`5.2.1-jdk21-dev` @ `99549c9`) via `--module-path $TORNADOVM_HOME/share/java/tornado`. Running the documented build against the pinned SDK fails **every time** with `TornadoInternalError: ... has no writeReplace(): this task lambda was compiled against a tornado-api release whose TornadoFunctions.TaskN interfaces were not Serializable` (`results/raw/10-gpullama3/inference-run.log`) — a genuine compile-time/runtime TornadoVM-version incompatibility, not a local misconfiguration. Root cause and full analysis in `docs/gpullama3-reproduction.md` §2.
- **Workaround (no source modification, build-property override only, per CLAUDE.md's "pin SHA and use scripts/patches" rule)**: `./mvnw clean install -DskipTests -Dtornadovm.base.version=5.2.1 -Djdk.version.suffix=-jdk21-dev`, resolving `tornado-api`/`tornado-runtime` `5.2.1-jdk21-dev` from the local `.m2` (installed there by this repo's own task-00 `make BACKEND=cuda`, i.e. the exact pinned-SHA artifacts). **`mvn clean` is required** — a non-clean `install` after only changing `-D` properties leaves stale `.class` files (compiler-plugin only checks source mtimes) and silently reproduces the identical failure; discovered and documented as a footgun (`results/raw/10-gpullama3/inference-run-pinned-tornado-api.log` still fails, `build-pinned-tornado-api-clean.log` + `inference-run-pinned-clean.log` succeed).
- **Minimal inference works** on the CUDA backend: `./llama-tornado --gpu --cuda --model .../beehive-llama-3.2-1b-instruct-fp16.gguf --prompt "..."` produces correct, coherent output (e.g. "The capital of France is Paris."), reproduced across 3 separate runs with different prompts/seeds/token-counts, ~150-155 tok/s (no profiler) / ~92 tok/s (`--profiler`, extra JSON-dump overhead, same direction as prior demos' profiler-overhead notes) for the 1B FP16 model on the RTX 4090 — this-run/this-model/this-GPU numbers only, not a general claim. Model file: pre-existing on this machine at `/home/michalis/test_install/GPULlama3.java/beehive-llama-3.2-1b-instruct-fp16.gguf` (outside the repo, not downloaded by this task, path recorded in `env/versions.env`), the exact model named in upstream's own quickstart.
- **CUDA-execution confirmed by profiler JSON, not just a claimed flag**: `--profiler` run shows every TornadoVM stage reporting `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"` (`results/raw/10-gpullama3/inference-run-profiler.log`).
- `--show-command` output captured (`results/raw/10-gpullama3/show-command.log`) as the exact reproducible `java` invocation, since GPULlama3.java doesn't generate its own `@arg-file`.
- JBang: **not verified** — `which jbang` → exit 1 (same finding as demos 00-09, checked again 2026-08-21).
- Documentation nuance recorded, not reproduced as a blocker: README states "GCC/G++ 13+" as a prerequisite; this repo's pinned env has GCC 11.4.0 and already built TornadoVM's native components successfully with it in task 00 — the GCC requirement applies to building the TornadoVM SDK itself, not GPULlama3.java's own (pure-Java) Maven build.
- Also found while picking a jar for the run: `llama-tornado`'s jar-selection glob picks the lexicographically-*last* `gpu-llama3-*.jar` in `target/`, which sorts the un-suffixed `-jdk21.jar` (5.0.0-linked) *after* `-jdk21-dev.jar` (5.2.1-linked) — a silent wrong-jar footgun if both are ever present in `target/` simultaneously. Documented in `docs/gpullama3-reproduction.md`; worked around this run by deleting the stray default-build jar before testing (build-artifact cleanup, not a source change).

Acceptance for task 10 ("Current build/run instructions are captured; minimal inference works or a complete reproducible blocker is recorded") is met: build/run instructions are captured including the real cross-repo TornadoVM-version blocker and its exact, reproducible workaround (not silently sidestepped), and minimal inference is demonstrated working on the CUDA backend with profiler-confirmed evidence across three separate runs.

## Batch 11 — Quantized LLM paths verified (2026-08-21)

Task `auto/tasks/11.md` is done. SHA re-checked before starting: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, checkout clean; `GPULLAMA3_SHA` unchanged (`bbe42fdc8cd475bb6104cefa42118dd6e068538b`, checkout untouched apart from a stray untracked `profiler-log.json` build artifact from task 10, deleted — `vendor/GPULlama3.java` is gitignored, no source modification). Checked `ps`: only this invocation's own `timeout`+`claude` process pair plus the pre-existing long-running interactive session (PID 442317) were present, no race. `nvidia-smi` confirmed the GPU idle (0%, 4 MiB) before every run. Deliverable: `docs/quantization-paths.md`. All Observed unless marked otherwise:

- Read the actual dispatch source before running anything (source-backed, §1 of the doc): TornadoVM weight loading/forward-pass execution gate on `GGMLType` at two independent checkpoints — `AbstractModelLoader.getModelQuantization` (GGUF `general.file_type` int → string, `default` throws `UnsupportedOperationException` for unmapped types incl. `2`=legacy Q4_0), `LlamaModelLoader.createTornadoVMWeights` (re-validates `ggmlType != F16 && != Q8_0` throws), and `ForwardPlanFactory.create` (`case Q4_0 -> throw ... "Q4_0 plans not yet implemented"`, `case F32 -> throw ...`). Confirmed by directory structure: dedicated per-model-family TornadoVM plan classes exist under `tornadovm/plan/components/{fp16,q8_0}/` only — no `q4_0`/`q4_k` directory anywhere in the tree.
- Found all three test models needed on this machine without downloading anything: FP16 (`/home/michalis/test_install/GPULlama3.java/beehive-llama-3.2-1b-instruct-fp16.gguf`, same file task 10 used), Q8_0 and Q4_0 (`/home/michalis/llama3.java-tornadovm/Llama-3.2-1B-Instruct-{Q8_0,Q4_0}.gguf`) — same model family (Llama-3.2-1B-Instruct) across all three for direct comparability. Recorded in `env/versions.env`.
- **FP16: working.** Re-ran `./llama-tornado --gpu --cuda --model .../beehive-llama-3.2-1b-instruct-fp16.gguf --prompt "Explain what a GPU kernel is..." --max-tokens 64 --seed 42` — coherent, correct completion, 164.20 tok/s (this run/this GPU). 4th reproduction of this path total (3x in task 10 + this run). Log: `results/raw/11-quantization/fp16-inference-run.log`.
- **Q8_0: working (new).** Same prompt/model-size against the Q8_0 file — coherent, correct completion (identical sentence to the FP16 run, as expected for a high-fidelity quantization of the same model), 186.47 tok/s. Re-ran with `--profiler` and a different prompt/seed: every TornadoVM stage reports `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"` in the JSON dump (91.27 tok/s under profiler — same profiler-overhead direction noted in every prior demo/task). Logs: `results/raw/11-quantization/q8_0-inference-run.log`, `q8_0-inference-run-profiler.log`.
- **Q4_0: blocked, matches source prediction exactly.** Same command against the Q4_0 file fails deterministically, exit 1, before any GPU/TornadoVM work starts (fails at GGUF-metadata-parse time): `UnsupportedOperationException: Unsupported quantization format: 2 (as int)` at `AbstractModelLoader.getModelQuantization:48` — the `general.file_type=2` legacy-Q4_0 tag falls through the `default` branch identified by source-reading beforehand, reached even earlier than `ForwardPlanFactory`'s own `Q4_0` throw would be. Two independent source-level guards agree; the real jar against the real GGUF file confirms it end-to-end, not just static reading. Classified blocked-not-a-bug: upstream's own message ("not yet implemented") indicates planned future work. Log: `results/raw/11-quantization/q4_0-inference-run.log`.
- **Q4_K/Q5_K/Q6_K (K-quants): documented, not independently reproduced.** Source (`effectiveGpuWeightType`) shows these are accepted and transparently dequantized to Q8_0 at load time, i.e. should work via the same path as Q8_0 above — but no K-quant GGUF file was found on this machine (checked `/home/michalis` depth 5 across every model directory used by prior tasks). Correctly classified as source-backed/documented rather than claimed working, per `PLAN.md` §6's evidence-classification contract.
- **F32: out of this task's requested scope** (task asked for FP16/Q8/Q4 only) — noted in the summary table as source-backed-blocked only (`ForwardPlanFactory.java`'s `F32 -> throw` branch, same pattern as Q4_0), not independently run.
- `docs/gpullama3-reproduction.md` §7 added, cross-linking to `docs/quantization-paths.md`. `env/versions.env` extended with `GPULLAMA3_MODEL_Q8_0`/`GPULLAMA3_MODEL_Q4_0`.

Acceptance for task 11 ("Each path is classified working/documented/blocked with evidence") is met: FP16 and Q8_0 are classified working with reproduced inference runs and profiler-confirmed CUDA execution; Q4_0 is classified blocked with an observed, source-predicted, exact-match failure; K-quants and F32 are classified documented/blocked-source-only with the reason no test file was available or the check was out of scope, rather than silently omitted or invented.

## Batch 12 — Profile GPULlama3.java inference with Nsight Systems (2026-08-21)

Task `auto/tasks/12.md` is done. SHA re-checked before starting: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, `GPULLAMA3_SHA` unchanged (`bbe42fdc8cd475bb6104cefa42118dd6e068538b`), both checkouts clean. Checked `ps`: only this invocation's own `timeout`+`claude` process pair was present, no concurrent race. `nvidia-smi` confirmed the GPU idle (0%, 4 MiB) before every run. Deliverable: `results/raw/12-llm-profiling/PROFILING-SUMMARY.md`, `docs/gpullama3-reproduction.md` §8. All Observed unless marked otherwise:

- Reused task 10's `--show-command` mechanism to extract the exact `java ...` invocation for the FP16 model (same file as tasks 10/11), then ran it directly (bypassing `llama-tornado`'s Python wrapper, which has no profiler-prefix option) both bare (sanity check) and under `nsys profile --trace=cuda,nvtx,osrt`. Correct, coherent output confirmed in both runs (162.62 tok/s bare, 153.54 tok/s under `nsys`, ~6% profiler overhead — same direction as every prior task's profiler-overhead note).
- Parsed `nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_kern_exec_sum,cuda_gpu_trace` into `results/raw/12-llm-profiling/`: 5 CSVs + logs, analyzed in `PROFILING-SUMMARY.md`.
- **Dominant kernel**: `fusedRmsNormFFNGateUp` (FFN gate/up projection), 39.0% of total GPU kernel time (87.15 ms of 223.4 ms across 1040 launches), almost 2× the next-largest kernel (`matrixVectorGenericWithResidual`, 22.9%). Instance-count ratios (1040/65=16, 2080/65=32) matched against source (`AbstractTransformerLayerTaskGraphs.java:43-45`, one task-graph per `config.numberOfLayers()`) rather than asserted from outside knowledge — confirms 16 per-layer instances/forward-pass for this model.
- **Memory behavior**: H2D memcpy dominates (99.1% of memcpy time, 10129 calls), D2H is exactly 65 calls (one per forward pass — the sampled token read back). Asymmetry explained as expected autoregressive-decode behavior (small per-step H2D activation/KV updates vs. a single scalar D2H per step), not investigated as an anomaly.
- **Warm-up vs. steady-state, separated using per-launch timestamps** (`cuda_gpu_trace` CSV, `matrixVectorGeneric` as a once-per-forward-pass marker): first GPU kernel launches ~42.3 ms after trace start (CUDA context/module setup); prompt-processing forward pass completes at ~755.6 ms (dominated by one-time JIT compilation of ~148 task-graph variants, per `cuModuleLoadDataEx`'s 148 calls); steady-state decode averages 6.45 ms/token (tokens 3-64, n=62) — cross-checked against this same run's own reported 153.54 tok/s (1000/153.54 = 6.51 ms/token) and agreeing within ~1%, i.e. two independent measurement methods (Nsight Systems kernel timestamps vs. the app's own end-to-end timer) agree. Confirmed the dominant kernel's own execution duration does **not** grow during warm-up (81.0 µs avg first-pass-of-16 vs. 83.8 µs avg all-later — stable from the first launch), so the ~756 ms "time to first token" is JIT/setup latency sitting before GPU execution, not slower early kernels — an important presenter distinction, not assumed without the per-launch timestamp check.
- **Nsight Compute occupancy/utilization/tensor-core metrics: re-verified BLOCKED a third time**, now specifically against the running LLM `java` process (not just the earlier synthetic/demo binaries from tasks 08/09): `/opt/nvidia/nsight-compute/2024.3.2/ncu --target-processes all -k "regex:^fusedRmsNormFFNGateUp$" --launch-count 1 --metrics gpu__time_duration.sum` against the exact java command connects to the process (`==PROF== Connected to process 683680`) then fails with the same `ERR_NVGPUCTRPERM` as `results/failures/08-nsight-compute-permission.md` — confirms the block is a system-wide driver-permission restriction (`NVreg_RestrictProfilingToAdminUsers=1`), not something specific to the earlier synthetic demo binaries. Log: `results/raw/12-llm-profiling/ncu-2024.3.2.0-llm-attempt.log`. No occupancy/utilization/throughput-%/tensor-core number is claimed for the LLM path.
- `docs/gpullama3-reproduction.md` §8 added, cross-linking to `results/raw/12-llm-profiling/PROFILING-SUMMARY.md`.
- Cleaned up the intermediate `nsys stats`-generated `.sqlite` file before committing (not committed by task 09 either — raw `.nsys-rep` + parsed CSVs are the durable evidence, consistent with the established pattern).

Acceptance for task 12 ("Profiler artifacts, parsed metrics, exact commands, environment manifest, and presenter-friendly interpretation are recorded") is met: `.nsys-rep` trace + 5 parsed CSV reports + run logs are committed under `results/raw/12-llm-profiling/`, `PROFILING-SUMMARY.md` documents the exact command/tool version next to every reported number, kernel timeline/dominant-kernel/memory-behavior/warm-up-vs-steady-state are all covered with presenter-friendly interpretation, and the Nsight Compute occupancy/tensor-core blocker is re-verified against the actual LLM binary (not assumed stale from a different binary) and documented with the exact command/error.

## Batch 13 — Quarkus and LangChain4j integration probed, both blocked on a JDK-version split (2026-08-21)

Task `auto/tasks/13.md` is done. SHA re-checked before starting: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, `GPULLAMA3_SHA` unchanged (`bbe42fdc8cd475bb6104cefa42118dd6e068538b`), both checkouts clean. Deliverable: `docs/quarkus-langchain4j-integration.md`, `demos/09-quarkus-langchain4j-gpullama3/`, `demos/10-langchain4j-gpullama3/`. All Observed unless marked otherwise:

- Checked Maven Central live (`maven-metadata.xml`) rather than trusting the GPULlama3.java README: `io.quarkiverse.langchain4j:quarkus-langchain4j-gpu-llama3` is a real, current, non-beta release (`1.13.0`, 2026-08-18); `dev.langchain4j:langchain4j-gpu-llama3` has **never had a non-beta release** (latest `1.19.0-beta29`, 2026-08-14), despite the README's "officially supported ... since v1.7.1" framing. `quarkus-langchain4j-gpu-llama3:1.13.0`'s own parent POM pins `gpu-llama3.version=1.0.0-jdk21` — the same release line this repo's pinned SHA builds as `gpu-llama3-1.0.0-jdk21-dev`; `langchain4j-gpu-llama3:1.19.0-beta29` instead pins the much older `gpu-llama3:0.4.0-jdk25`.
- Built two minimal demos (`demos/09-.../`, a Quarkus `@RegisterAiService` app; `demos/10-.../`, a standalone `GPULlama3ChatModel` program using the README's own documented builder pattern), each depending on the real Maven Central integration artifact above, with `io.github.beehive-lab:gpu-llama3` **overridden to this repo's pinned `1.0.0-jdk21-dev` build** via a direct dependency declaration (Maven "nearest wins" mediation beats the transitive pin) — same override technique as task 10's tornado-api pin, applied here to a third-party integration POM instead of GPULlama3.java's own.
- **Both integration modules require JDK 23+ just to load their own deployment/runtime classes** (class file versions 67 and 69 respectively) — `mvn package` fails immediately under this repo's pinned JDK 21.0.2 with `UnsupportedClassVersionError`. Log: `results/raw/13-quarkus-langchain4j/quarkus-build-jdk21-fail.log`.
- Switching only the build/run JVM to JDK 25 (available via sdkman on this machine, `env/versions.env: JDK25_SDKMAN_PATH`, **not** this repo's pinned runtime) let both demos build and, for the Quarkus one, boot: CDI/ArC wiring, `@RegisterAiService` proxy generation, config resolution, and `GPULlama3ModelRegistry`'s local-cache lookup (pre-existing, pre-cached model at `/home/michalis/.langchain4j/models/...`, not downloaded by this task) all worked correctly — confirmed by the boot log listing `Installed features: [cdi, langchain4j, langchain4j-gpu-llama3, qute, ...]`. Log: `results/raw/13-quarkus-langchain4j/quarkus-build-jdk25-ok.log`.
- Both then fail identically, at the moment GPULlama3.java's `ModelLoader` class is actually loaded, with `UnsupportedClassVersionError: ... ModelLoader (class file version 65.65535) was compiled with preview features that are unsupported. This version of the Java Runtime only recognizes preview features for class file version 69.65535`. **Root cause, confirmed by inspecting the raw class file bytes** (`unzip -p ... | xxd`): this repo's pinned `gpu-llama3-1.0.0-jdk21-dev` was compiled `--enable-preview` under JDK 21 (major=65, minor=0xFFFF preview marker); by JVM spec, preview-flagged bytecode is loadable **only** by the exact JDK feature release that produced it — never an older *or newer* JDK. Since both current integration modules need JDK 23+/25 just to load, and this repo's pinned GPULlama3.java build is preview-locked to exactly JDK 21, **no single JVM process can run both together** — this is the actual blocker, not a missing feature or a broken build. Logs: `results/raw/13-quarkus-langchain4j/quarkus-run-jdk25-blocked.log`, `results/raw/13-quarkus-langchain4j/langchain4j-standalone-jdk25-blocked.log`.
- Recorded but not chased further: running either demo under the pinned JDK 21.0.2 directly (no JDK switch) exits immediately with status 1 and **zero** stdout/stderr, confirmed via `strace -f` to be a clean JVM-orchestrated shutdown (an early `SIGSEGV`/`rt_sigreturn` pair is benign JVM-internal CPU-feature-probing noise, not a crash) rather than a native crash — cause not identified, doesn't change the conclusion since JDK 21 can't get past `mvn package` for the Quarkus deployment module anyway.
- Hypothesis, not verified this batch: `quarkus-langchain4j-gpu-llama3:1.13.0`'s `jdk25` Maven profile pins `gpu-llama3:1.0.0-jdk25`, suggesting the actually-supported combination is GPULlama3.java `-jdk25` + a JDK-25-built TornadoVM CUDA backend throughout. This repo's pinned TornadoVM CUDA build (task 00) is JDK-21-only; building a second JDK-25 CUDA backend from the same pinned SHA to test this is a non-trivial rebuild, not attempted.
- Used pre-existing, read-only local checkouts (`/home/michalis/jcon/quarkus-langchain4j`, `/home/michalis/jcon/gpullama3-quarkus-langchain4j-demo`) only as API/config-property reference material to design the demos above (they predate the versions probed and use `999-SNAPSHOT`/`gpu-llama3:0.2.2` — not the basis of any reported result).

Acceptance for task 13 ("Working integrations have runnable demos; blocked integrations contain exact errors and are explicitly marked") is met: both integrations have runnable, buildable demo source in `demos/09-.../` and `demos/10-.../`; both are explicitly marked blocked with the exact reproducing commands, exact errors, and a source-verified root cause (not a guess) in `docs/quarkus-langchain4j-integration.md`.

### Next invocation

Start with `auto/tasks/14.md` if it exists (check `auto/tasks/` — task numbering beyond 13 was not confirmed this batch). Before starting: re-check `git status` clean, re-verify `vendor/tornadovm` HEAD still matches `env/versions.env`, and check `ps`/`auto/logs/supervisor.log` for concurrent `claude -p` instances per the operational note in batch 01 (this remains an open supervisor-script hazard, not yet fixed at the tooling level). If GPU performance-counter access becomes available in a future environment (root/sudo, or `NVreg_RestrictProfilingToAdminUsers=0`), `results/raw/09-profiling/PROFILING-SUMMARY.md`, `results/raw/12-llm-profiling/PROFILING-SUMMARY.md`, and `results/failures/08-nsight-compute-permission.md` all have ready-to-rerun `ncu` commands to retroactively add hardware-counter evidence without redoing anything else — task 12's version targets `fusedRmsNormFFNGateUp`/`matrixVectorGeneric`/`processHeadsFlashAttention`/`fusedQKVMatmulX` specifically. Also open: `CuFftLowPassHybrid` silently bails out to a CPU sequential fallback (wrong-looking output, not a crash) at `n=65536` — not investigated, noted above for whichever future task touches cuFFT sizing/scaling limits. For any future GPULlama3.java task: remember to `source vendor/tornadovm/setvars.sh` then `source vendor/GPULlama3.java/set_paths` (not sdkman's `tornado`) and always rebuild with `mvn clean install -Dtornadovm.base.version=5.2.1 -Djdk.version.suffix=-jdk21-dev` (never a bare `make`/`mvn install`, and never skip `clean`) to stay linked against this repo's pinned TornadoVM SDK — see `docs/gpullama3-reproduction.md`. If a future task needs Q4_K/Q5_K/Q6_K coverage to close the one open "documented, not observed" gap from batch 11, no such file was found on this machine as of 2026-08-21 — would need to be sourced/converted first before it could be tested. Task 12 only profiled the FP16 path (task 11's only fully-verified-working path at the time); profiling Q8_0 with the same method would be a straightforward follow-up if a future task wants it, not yet done. Batch 13's JDK-version-split blocker (§ above) affects only the Quarkus/LangChain4j integration paths, not any other demo or the core LLM inference path (tasks 10-12), which remain fully working under the pinned JDK 21 CUDA build.

## Batch 14 — Profiler quick-start and demo runbook (2026-08-22)

Task `auto/tasks/14.md` is done. SHA re-checked before starting: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, `GPULLAMA3_SHA` unchanged, both checkouts clean. Checked `ps` for concurrent `claude -p` instances: only this invocation's own `timeout`+`claude` process pair (PIDs 720598/720599) was present, no race. `nvidia-smi` confirmed the GPU idle (0%, 4 MiB) before every command tested. Deliverable: `docs/profiling-quickstart.md`, `demos/README.md` updated with a pointer. All Observed unless marked otherwise:

- Rather than write the guide from memory of prior batches' READMEs, every command in it was re-run live on this machine before being committed to the doc (not assumed from STATE.md's summaries): `demos/00-hello-gpu` built and run via both `tornado --classpath .` and `java @../tornado.args -cp .` (both correct); a fresh `nsys profile --trace=cuda,nvtx,osrt` capture against it, parsed with `nsys stats --report cuda_gpu_kern_sum,cuda_api_sum` into a non-empty CSV (`addOne`, 1 instance, 1,600 ns) — confirmed real CUDA activity, not a CPU fallback; `demos/01-first-cuda-kernel`'s `--printKernel` command re-run separately, confirmed `extern "C" __global__ void vectorAdd(...)` + `Result is correct`.
- **Found a new, previously undocumented interaction while verifying the `ncu` recipe**: running `/opt/nvidia/nsight-compute/2024.3.2/ncu` against `demos/00-hello-gpu`'s tiny 8-element `Hello`/`addOne` kernel does **not** reproduce the expected `ERR_NVGPUCTRPERM` — instead TornadoVM prints `[Bailout] Running the sequential implementation` under `ncu`'s instrumentation and `ncu` reports "No kernels were profiled" (no CUDA activity to deny counter access to). Re-tested against `demos/08-tensor-core-mma` (`gemmMMASingleTile`) instead — reproduced the documented `ERR_NVGPUCTRPERM` exactly as `results/failures/08-nsight-compute-permission.md` describes. Root cause of the `Hello`-specific bailout not investigated further (out of scope for a documentation task; the workload is too trivial to be a real profiling target anyway) — recorded as a "pick a real-size kernel, not the trivial Hello demo" caveat in the new guide's §3, not silently omitted.
- The guide covers, with commands verified against this environment rather than invented: one-time setup + GPU-idle check; all three run paths (`tornado`, `java @arg-file`, JBang — JBang re-confirmed not installed, `which jbang` exit 1 as of 2026-08-22, same finding as every prior task); Nsight Systems recipe with a table mapping each `nsys stats` report type to what it answers, plus the `cuCtxCreate_v2`-dominates-`Time(%)` reading caveat already established in tasks 09/12; Nsight Compute recipe with the reconfirmed two-stage blocker (`2026.2.1.0` can't connect to the driver at all; `2024.3.2.0` connects but hits `ERR_NVGPUCTRPERM`) and what to do if counter access ever becomes available; a report-locations table; `--printKernel` as the no-permissions-needed generated-code fallback; a per-demo quick-reference table (build needs, profiling workload used elsewhere in this repo, fallback pointer) covering all 10 demo directories including the two blocked Quarkus/LangChain4j integration demos (09/10, correctly marked "do not attempt live"); a GPULlama3.java quick-start section (run/profile/build-blocker, cross-linking `docs/gpullama3-reproduction.md` and `results/raw/12-llm-profiling/PROFILING-SUMMARY.md`); a presenter cheat-sheet distilling the "what to point out" findings already recorded per-demo in tasks 04-08/10-12 into one page; and a final "if something doesn't work" recovery checklist (GPU-busy check, `tornado --devices`, watch for the `[Bailout]` line, per-demo README fallback sections).
- Did **not** duplicate the per-demo `## If the demo fails on stage` / `## Fallback if the live demo fails` sections that already exist in every one of demos 00–08's own `README.md` (task 14's "add recovery/fallback commands for every demo" acceptance was already satisfied incrementally by tasks 00-08, confirmed by grepping every demo README for that heading before writing this guide) — instead the new guide's §6 table points at each one, and §8's cheat sheet adds the higher-level "what to say," which was the actual gap task 14 asked to close.
- No new number, API, or version was invented anywhere in the guide; every command was either re-run this batch or is a direct copy of a command already verified and committed in a prior task's evidence (cited inline).
- Compiled `.class` files from this batch's verification builds (`demos/00-hello-gpu`, `demos/01-first-cuda-kernel`, `demos/08-tensor-core-mma`) and the `/tmp/qs-profiling-check` scratch `.nsys-rep`/CSV were deleted before committing — not durable evidence, purely a doc-accuracy check, consistent with not committing build artifacts.

Acceptance for task 14 ("A Java developer can run an example and then profile it by copying commands from the docs") is met: every command shape in `docs/profiling-quickstart.md` — build, three run paths, `nsys profile`/`nsys stats`, `ncu` (and its documented blocker), `--printKernel` — was executed verbatim on this machine this batch and produced the output the doc claims, not written from memory or inference.

### Next invocation

Start with `auto/tasks/15.md`. Before starting: re-check `git status` clean, re-verify `vendor/tornadovm` HEAD still matches `env/versions.env`, and check `ps` for concurrent `claude -p` instances per the batch-01 operational note. Task 15 ("Create integrated CUDA showcase") can draw directly on demos 04/05/06/07/08's already-verified individual mechanisms (cuBLAS/cuFFT hybrid tasks, stream concurrency, graph replay, Tensor Core MMA) rather than re-deriving any of them from source. The new `docs/profiling-quickstart.md` (this batch) is the reference to link from that showcase demo's own README for its profiling section, rather than re-explaining `nsys`/`ncu` usage inline. The `Hello`-kernel-under-`ncu` bailout quirk noted above is a minor, low-priority open thread (not a blocker for anything) if a future task wants to explain it.

## Batch 15 — Integrated CUDA showcase (2026-08-22)

Task `auto/tasks/15.md` is done. SHA re-checked before starting: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, checkout clean. Checked `ps` for concurrent `claude -p` instances: only this invocation's own `timeout`+`claude` process pair was present, no race. `nvidia-smi` confirmed the GPU idle (0%, 4 MiB) before every command. Deliverable: `demos/11-integrated-showcase/` (`IntegratedShowcase.java`, `README.md`), `demos/README.md` updated with a pointer. All Observed unless marked otherwise:

- Combined, in one `TaskGraph`/one JVM program: JIT `@Parallel` Java kernels (`scale`/`bias`), a cuBLAS `.libraryTask` (`cublasSgemv`, demo 04's pipeline shape) replicated across 6 independent chains (demo 06's N-independent-unit shape), run in four modes — `baseline` (no graph/concurrency), `concurrent` (`withIntraPlanConcurrency()`, demo 06's mechanism), `graph` (`withCUDAGraph()`, demo 07's mechanism), and an **experimental `combined` mode stacking both on the same plan** — plus a Tensor Core `mma.sync` single-tile bonus stage reusing demo 08's kernel verbatim (kept as its own `TaskGraph` since its `WorkerGrid`/datatypes are unrelated to the sgemv chains, not artificially merged).
- **Before writing the combined mode, grepped the pinned upstream tree** (`tornado-unittests/`, `tornado-cublas/.../tests/`) for any test combining `withCUDAGraph()` and `withIntraPlanConcurrency()` on one plan — found none. Found `tornado-cublas/.../tests/TestCuBlasSgemvWithTasksCudaGraph.java` instead, an upstream test confirming cuBLAS-library-task-inside-CUDA-graph-capture is itself a real, verified pattern (modeled `graph` mode's chain shape on it). The `combined` mode is therefore new, repo-original probing, not a documented feature.
- **The combined mode works and validates correctly**, reproduced across 3 independent runs (plain run, full-verbose rerun, and a run under `nsys` instrumentation) — every execution in every mode, every run, validated exactly against a closed-form CPU reference. Classified **Observed**: `plan.withCUDAGraph(); plan.withIntraPlanConcurrency();` called sequentially on the same `TornadoExecutionPlan` (both mutate the shared `tornadoExecutor` state; their return values are typed wrappers not needed for further chaining) is a working combination on this pinned build, RTX 4090, CUDA backend.
- **Found and fixed a real bug during evidence-gathering, not left in committed source**: the steady-state median was computed from the raw-nanosecond sorted array without the `/1000` division applied at the individual per-execution print sites, so an early run printed a summary median (e.g. "712349 us" for baseline) that was actually still in nanoseconds — 1000x too large and inconsistent with the individual per-execution numbers directly above it. Caught by cross-checking the summary line against a full (not truncated-to-3-lines) per-execution printout before trusting any number; fixed (`steadyState[...] / 1000`), rebuilt, and every number in the committed README was captured only after the fix, from a fresh full-verbose run.
- **Honest performance finding, not smoothed over**: `graph` mode gives a large, consistent speedup (4.9x-6.6x vs. baseline across runs) — expected, matches demo 07. `concurrent` mode is *not reliably faster* than baseline for this workload's size (one run 1.21x speedup, another run 0.81x — i.e. slower): unlike demo 06's deliberately small-grid/heavy-inner-loop kernels (sized specifically so a unit doesn't saturate the SMs alone, enabling genuine overlap), this demo's 8x8 sgemv chains are tiny and launch-overhead-bound, leaving little idle SM time for streams to fill. `combined` mode consistently lands *between* `graph` alone and `baseline` in both runs — stacking concurrency on top of an already-replayed graph did not help further at this size. Recorded as a presenter caveat (which mechanism to trust for a live speedup claim vs. which to present as "here's what we measured, it's workload-dependent") rather than omitted or rounded to a cleaner story.
- Verified three run paths (`tornado --classpath .`, `java @../tornado.args`, `tornado --enableProfiler console`) — profiler JSON confirms `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"` for every task across all 6 units.
- Captured a dedicated Nsight Systems trace (`nsys profile --trace=cuda,nvtx,osrt`, all-modes run) and parsed `cuda_gpu_kern_sum`/`cuda_api_sum`/`cuda_gpu_mem_time_sum` into CSVs: confirms real GPU kernels for every stage in one trace — cuBLAS's `gemvx::kernel` (38.4% of GPU kernel time), `bias` (31.3%), `scale` (30.0%), and `gemmMMASingleTile` (0.3%, 1 instance, the Tensor Core kernel present in the same trace as the sgemv pipeline). Nsight Compute hardware-counter evidence not attempted — same system-wide `ERR_NVGPUCTRPERM` block already documented in `results/failures/08-nsight-compute-permission.md`, re-verified applicable (same driver) rather than re-probed from scratch; generated-code evidence for the MMA kernel already exists in demo 08 and is cross-linked, not duplicated.
- Logs/CSVs: `results/raw/11-integrated-showcase/` (`showcase-run.log`, `showcase-run-fullverbose.log`, `showcase-run-javaargfile.log`, `showcase-profiler-console.log`, `showcase-nsys-run.log`, `showcase-nsys.nsys-rep`, 3 parsed CSVs). Compiled `.class` artifact deleted before committing (not committed anywhere else in the repo, confirmed by checking `.gitignore` and sibling demo directories).

Acceptance for task 15 ("The integrated demo remains understandable, has a short run command, profiler evidence, and a safe fallback sequence") is met: one short canonical run command (`tornado --classpath . IntegratedShowcase 6 8 8 20 all`) exercises all five mechanisms understandably (one labelled mode/stage at a time, not interleaved); Nsight Systems trace + parsed CSVs + profiler-console JSON are captured as profiler evidence; the README's "Fallback if the live demo fails" section gives an explicit cheapest-to-riskiest recovery order (`graph` -> `baseline` -> `mma` bonus -> `concurrent`/`combined`) plus device-check and pre-captured-log fallbacks, consistent with every prior demo's fallback contract.

### Next invocation

Start with `auto/tasks/16.md` if it exists (check `auto/tasks/` — task numbering beyond 15 was not confirmed this batch). Before starting: re-check `git status` clean, re-verify `vendor/tornadovm` HEAD still matches `env/versions.env`, and check `ps`/`nvidia-smi` for concurrent instances or GPU activity per the batch-01 operational note. Open, not chased this batch: `combined` mode's lack of extra benefit over `graph` alone was only checked at one workload size (6 units, 8x8 sgemv); a future task could retest at a larger per-unit size (demo 06's heavier-kernel sizing) to see whether the concurrency benefit becomes visible once a single stream's replay is not already launch-bound. Nsight Compute hardware-counter access remains blocked system-wide (`ERR_NVGPUCTRPERM`); the ready-to-rerun commands are already collected in `results/failures/08-nsight-compute-permission.md` if a future environment has counter access.

## Batch 16 — Demo simplicity/consistency audit (2026-08-22)

Task `auto/tasks/16.md` is done. SHA re-checked before starting: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, `GPULLAMA3_SHA` unchanged, both checkouts clean. Checked `ps`: only this invocation's own `claude -p` process pair present, no concurrent race. `nvidia-smi` confirmed the GPU idle (0%, 4 MiB) before every run. Deliverable: `docs/demo-audit-checklist.md`, `demos/README.md` updated with a pointer. All Observed unless marked otherwise:

- Reviewed all 9 Track-A demo source files (`00`, `01`, `02`, `04`–`08`, `11`) against a fixed checklist (one concept per class, lightweight comments, deterministic output, simple invocation, `java @arg-file`, JBang) by direct source inspection (`grep`/`wc`), not from memory of prior batches' summaries: every file has exactly one top-level class declaration (`grep -nE 'class\s+\w+|interface\s+\w+'` across all 9 files, 1 match each), comment-line ratios stay well under code volume (7-44 lines of comments in 50-342-line files), and every demo's `README.md` opens with a "Concept (read in ~1 minute)" one-paragraph explanation immediately after the title (confirmed by reading the first 8 lines of each README, not assumed from the top-level table).
- **Found and fixed 2 genuinely unused imports**, verified by checking each import's class-name usage count in its own file (exactly 1 = only the import line itself, i.e. dead): `demos/08-tensor-core-mma/TensorCoreMMA.java`'s `ImmutableTaskGraph` import, and `demos/11-integrated-showcase/IntegratedShowcase.java`'s `WorkerGrid2D` import. Removed both, then **rebuilt and re-ran both demos end-to-end** to confirm no breakage: `javac --release 21 --enable-preview ...` succeeded for both, `tornado --classpath . TensorCoreMMA` (both scalar and `mma.sync` validations PASSED) and `tornado --classpath . IntegratedShowcase 6 8 8 20 all` (all four modes + bonus MMA stage validated correct, same speedup ordering as task 15: graph 4.59x, combined 2.73x, concurrent 0.78x vs. baseline this run) both still ran correctly after the edit. Compiled `.class` artifacts deleted before committing (not committed anywhere in the repo). Logs: `results/raw/16-demo-audit/08-tensor-core-mma-post-cleanup-run.log`, `results/raw/16-demo-audit/11-integrated-showcase-post-cleanup-run.log`.
- No other unused imports found across the remaining 7 demo files (checked every `import` line's class-name usage count).
- No stray nested/helper classes anywhere; no demo build files (`pom.xml`/`build.gradle`) exist under the plain-Java Track-A demos — each is a single `.java` file with no extra dependency to prune.
- **Spot-checked the `java @arg-file` path live** on a demo not re-verified since task 14 (which covered 00/01/08): `java @../tornado.args -cp . CudaStreamsOverlap 8 32768 65536 8 both` in `demos/06-cuda-streams/` → exit 0, both sequential and concurrent modes report "All executions correct" — confirms the reproducibility path works, not just documented. Log: `results/raw/16-demo-audit/06-cuda-streams-argfile-spotcheck.log`.
- JBang re-confirmed not installed (`which jbang` exit 1, same finding as every prior task 00-14) — already explicitly documented as unsupported-on-this-environment in `docs/run-conventions.md`/`docs/profiling-quickstart.md`, not silently omitted from any demo README.
- Reviewed `demos/09-quarkus-langchain4j-gpullama3/` and `demos/10-langchain4j-gpullama3/` (Maven-project integration demos, blocked per task 13, structurally different from the single-file Track-A shape) separately: both are small (13-30 line Java files, one class with one job each; 70-90 line `pom.xml`s declaring only the integration artifact + pinned override, no unrelated dependency) — confirmed minimal, not re-litigating their already-documented JDK-version-split blocker.
- No number, API, or status was invented; the two import removals were verified dead by direct grep before deletion and the resulting demos were rebuilt/re-run (not just recompiled-and-assumed) before being reported as unbroken.

Acceptance for task 16 ("Every demo passes a presenter usability checklist and has a clear one-sentence explanation") is met: `docs/demo-audit-checklist.md` records a per-demo checklist table (all pass) for all 9 Track-A demos plus a separate minimality review for the 2 integration demos, one real cleanup (2 unused imports) was found and fixed with a verified rebuild/re-run rather than left as a paper finding, and every demo's README already carries a one-paragraph "Concept" explanation confirmed present by direct inspection.

### Next invocation

Start with `auto/tasks/17.md` if it exists (check `auto/tasks/` — task numbering beyond 16 was not confirmed this batch). Before starting: re-check `git status` clean, re-verify `vendor/tornadovm` HEAD still matches `env/versions.env`, and check `ps`/`nvidia-smi` for concurrent instances or GPU activity per the batch-01 operational note. Nothing new was left blocked or half-finished by this batch — the only prior open threads remain the same as batch 15's next-invocation note (combined-mode workload-size retest, Nsight Compute `ERR_NVGPUCTRPERM` system-wide block).

## Batch 17 — Talk drafts, runbook, claims ledger, final README, verify script, rehearsal (2026-08-22)

Task `auto/tasks/17.md` is done. SHA re-checked before starting: `vendor/tornadovm` HEAD still `99549c9862eda8d584e35e99924f9c865501eb3a`, `GPULLAMA3_SHA` unchanged, both checkouts clean. `ps` showed only this invocation's own `claude -p` process pair, no concurrent race. `nvidia-smi` confirmed the GPU idle (0%, 4 MiB) before every rehearsal command. This batch compiled the prior 16 batches' evidence into the final deliverables `README.md` named as outputs, rather than gathering new source evidence — all claims below cite existing `results/raw/`/`docs/*.md` evidence from tasks 00-16, cross-checked, not re-derived from memory. All Observed unless marked otherwise:

- **`docs/claims.md`** (new): every claim in the two talk drafts and the README mapped to its evidence classification (Observed/Source-backed/Documented/Hypothesis/Blocked per `PLAN.md` §6) and exact source/log path. Includes an explicit "No-claim list" (no OpenCL/Metal/PTX/Babylon numbers, no multi-GPU claim, no general "TornadoVM is Nx faster" claim — every speedup is scoped "this run, this workload, this machine").
- **`docs/demo-runbook.md`** (new): exact live-demo sequence for both talks — what to type, what should appear, what to say, per-step fallback to a named `results/raw/` log, a time-budget note (which demos to cut first if short on time), and a "global recovery rules" section (nvidia-smi/tornado --devices checks, watch for the `[Bailout]` CPU-fallback line, never restate a this-run number as a general claim).
- **`docs/talk-1-hybrid-api.md`** and **`docs/talk-2-llm-inference.md`** (new): both have opening, narrative arc, live-coding sequence, a "Technical explanation" section, and a conclusion, per `PLAN.md` §7's definition of done. Both explicitly carry forward the honest negative findings from prior batches (concurrency not reliably faster on launch-overhead-bound workloads; K-quants documented-not-reproduced; Quarkus/LangChain4j JDK preview-feature version-split blocker) rather than smoothing them into a cleaner but false story.
- **`scripts/verify.sh`** (new): validates committed evidence with **no GPU required** — checks the 8 required deliverable files exist, `TORNADO_SHA`/`GPULLAMA3_SHA` are recorded in `env/versions.env`, every `results/raw|failures/...` path cited in `docs/claims.md` resolves on disk (18 paths, all pass), every demo linked from `demos/README.md` has its own `README.md`, and — the hard-scope check — no `results/` evidence subtree exists for a non-CUDA backend and `TORNADO_BACKEND=cuda` is pinned. First version of this check naively grepped for the words "OpenCL"/"Metal"/"Babylon" anywhere in docs, which false-positived on `CLAUDE.md`/`README.md`'s own exclusion statements and API-doc quotes ("no-op on OPENCL/METAL"); rewritten to check for *captured evidence*, not word mentions. `bash scripts/verify.sh` exits 0, 13/13 checks pass.
- **`README.md`** rewritten: numbered demo table (00-11, matching `demos/README.md`) with canonical `tornado` command + `java @arg-file` command per row, a "Measured results" table with actual Nsight Systems/wall-clock numbers and tool/GPU context for every timed claim (matches `docs/claims.md` exactly), explicit JBang-not-verified statement, and pointers to every other deliverable.
- **Final rehearsal, this batch, fresh commands (not copied from prior logs)**: re-compiled and re-ran the entire Talk 1 demo sequence (00, 01, 02, 04, 05, 06, 07, 08, 11) end-to-end via the exact commands now written into `docs/demo-runbook.md`, plus the Talk 2 opening + `--profiler` GPULlama3.java FP16 inference commands. Every demo exit 0, every correctness check passed. Fresh numbers landed inside (07: 6.61x graph speedup, prior range 6.47x-7.02x; 11: graph 5.94x/concurrent 1.09x/combined 3.58x vs. baseline, same ordering as batches 15/16; GPULlama3.java FP16: 155.40 tok/s, prior range 153.18-164.20 tok/s; profiler run: 3,996 task-graph stages confirmed `"BACKEND": "CUDA"`) the ranges already recorded in prior batches — confirms the runbook as written is presenter-ready today, not just on the day each demo was first built. Compiled `.class` artifacts deleted after the rehearsal (`find demos -name '*.class' -delete`), not committed. Logs + summary: `results/raw/17-final-rehearsal/` (13 files incl. `REHEARSAL-SUMMARY.md`).
- No number, API, or claim was invented — every figure in the four new docs and the rewritten README traces to an existing `results/raw/` log (16 of them pre-dating this batch, cross-checked, not re-typed from memory) or to a fresh rehearsal log captured this batch.

Acceptance for task 17 ("Talk drafts, runbook, claims ledger, profiler guide, and final README are complete. The final README contains numbered demos, how-to-run commands, `java @arg-file` examples, JBang examples where valid, and actual NVIDIA profiler numbers with tool/GPU context. Never invent missing metrics.") is met: all four new docs exist and are complete per `PLAN.md` §7's definition of done; the profiler guide (`docs/profiling-quickstart.md`) already existed from task 14 and is cross-linked, not duplicated; the final `README.md` has the numbered demo table, `java @arg-file` examples for every demo that has one, an explicit JBang-not-verified statement (no invented "valid" JBang example exists to include), and a measured-results table citing Nsight Systems/wall-clock numbers with tool and GPU context for every entry; `scripts/verify.sh` validates the whole deliverable set without a GPU and passes 13/13; final rehearsal artifacts are recorded under `results/raw/17-final-rehearsal/`.

### Next invocation

Check `auto/tasks/` for a task numbered 18 or higher — none was confirmed to exist as of this batch (only 00-17.md were present). If none exists, the queued task list is exhausted; re-verify that before assuming more work is queued. If resuming this study's open threads instead: the combined-mode workload-size retest (batch 15) and the Nsight Compute `ERR_NVGPUCTRPERM` system-wide block (re-verify only if a future environment might have counter access — the ready-to-rerun commands are already collected in `results/failures/08-nsight-compute-permission.md`) remain open, plus `docs/quarkus-langchain4j-integration.md` §4's untested JDK-25-throughout hypothesis. Before starting anything: re-check `git status` clean, re-verify `vendor/tornadovm` HEAD still matches `env/versions.env`, and check `ps`/`nvidia-smi` for concurrent instances or GPU activity per the batch-01 operational note.

## Batch 18 — Migrated Track A to the TornadoVM 6.0.0 CUDA release (2026-09-02)

Modernization batch, not a study task: moved the Track A demos off the
source-built `5.2.1-jdk21-dev` pin (`99549c9862eda8d584e35e99924f9c865501eb3a`)
and onto the released TornadoVM **6.0.0** CUDA SDK installed from SDKMAN.
Deliverables: rewritten `README.md` (SDKMAN quick install), `env/versions.env`,
`demos/README.md` + all 9 demo READMEs, `scripts/setup-env.sh`,
`scripts/run-all-demos.sh`, updated `scripts/verify.sh`, `CLAUDE.md`, and the
operational docs. Evidence: `results/raw/18-tornadovm-6-migration/`
(+ `MANIFEST.md`). All Observed:

- **SDK selection matters and is now pinned by name.** 6.0.0 ships two CUDA
  SDKs. `6.0.0-jdk21-cuda` is compiled `--enable-preview` and its launcher
  refuses any JVM but JDK 21 (`etc/tornado.jdk`: `floor=21, preview=true`).
  `6.0.0-jdk22plus-cuda` is the non-preview build (`floor=22, preview=false`).
  This repo pins the latter (`TORNADO_SDKMAN_CANDIDATE` in `env/versions.env`,
  enforced by `scripts/verify.sh`). Installed both to confirm the difference.
- **`TORNADO_SDK` was renamed to `TORNADOVM_HOME`.** 6.0.0's `tornado.py`
  resolves the SDK from `TORNADOVM_HOME`; a stale `TORNADO_SDK` silently
  selects a different SDK and produces a misleading "built for JDK 21" error.
  `scripts/setup-env.sh` unsets the old name.
- **The arg-file moved into the SDK.** `tornado --generate-argfile` now writes
  `$TORNADOVM_HOME/tornado-argfile` and prints that as the documented usage.
  The committed, machine-specific `demos/tornado.args` (absolute paths to
  `/home/michalis/...`) was **deleted**; `verify.sh` now fails if one
  reappears. New 6.0.0 flags in it: `-Djdk.internal.vm.ci.enabled=true`,
  `--patch-module jdk.internal.vm.ci=<sdk>/share/java/jvmci/jvmci-21.0.2.jar`
  (JDK 22–26 vendoring), `--enable-native-access=tornado.runtime`. Gone:
  `--enable-preview`.
- **No demo source needed an API change.** All nine Track A demos compile
  unmodified against `tornado-api-6.0.0` under JDK 25 with no preview flags.
  The migration is entirely packaging, JDK level, and launch flags.
- **All nine demos run correctly both ways**: `tornado --classpath .` and
  `java @$TORNADOVM_HOME/tornado-argfile -cp .`. 27/27 checks pass via the new
  `scripts/run-all-demos.sh` (`run-all-demos.log`).
- **Two behavioural changes vs. 5.2.1, both in demo 11** (n=5 runs): the
  experimental `combined` mode (`withCUDAGraph()` + `withIntraPlanConcurrency()`)
  no longer costs more than `graph` alone — 5.55–5.71× vs. graph's 5.37–5.87×,
  where 5.2.1 measured 3.91× vs. 6.56× and 2.95× vs. 4.90×; and `concurrent`
  mode never went below break-even (1.08–1.13× across all five runs, vs. 5.2.1's
  0.81×–1.21× spread). Mechanism not investigated — recorded as Observed only.
- Demo 07's ratio rose to 8.08–10.00× (from 6.47–7.02×), but its `graph`
  steady-state median is ~36 µs on **both** pins — the `nograph` baseline is
  what moved, so this is dispatch-overhead variance, not a 6.0.0 optimisation.
  Demo 06 (~2.3× concurrency benefit) and demo 08 (exactly one
  `mma.sync.aligned` instruction) are unchanged.

### Deliberately out of scope

- **Track B (GPULlama3.java, `demos/09`, `demos/10`) was not migrated.** Its
  pins, docs, and every `results/` measurement stay on 5.2.1 and are labelled
  as such throughout. Note for a future batch: 6.0.0's `jdk22plus` SDK is a
  non-preview build, which removes the preview-bytecode half of the
  JDK-21-vs-JDK-23+ blocker recorded in batch 13 — that blocker is no longer
  structural, but nothing was rebuilt or re-tested, so it is **not** a claim
  that the integrations now work.
- **Nsight Systems traces were not re-captured.** The 5.2.1 timelines remain
  cited as mechanism evidence only. Nsight Compute is still blocked
  (`ERR_NVGPUCTRPERM`), re-confirmed unchanged.
- `results/raw/00-baseline/` … `17-final-rehearsal/` are left byte-for-byte
  unmodified. Where a doc cites them it now says which pin they came from.
- JBang still not installed on this machine (`which jbang` → exit 1).

## Batch 19 — CUTLASS, cuDNN+JIT and warp/async/shared demos (2026-09-02)

Three new Track A demos on the pinned TornadoVM 6.0.0 CUDA SDK, each with an
Nsight Systems section in its README (commands + captured output). Evidence:
`results/raw/19-cutlass-cudnn-warp-demos/` (+ `MANIFEST.md`). All Observed:

- **`demos/12-cutlass-fused-epilogue`** — CUTLASS fused epilogue
  (`cutlassGemmBiasRelu`, one kernel) vs. `cutlassHgemm` + a separate JIT
  bias/ReLU pass, both as one TaskGraph mixing JIT and library tasks. Both
  validate (max abs err `0.00781`). **Wall-clock cannot show the effect** —
  fused 317 µs vs. unfused 304 µs at 1024³, dominated by the 2 MB D2H copy — so
  the README makes the nsys kernel table the evidence: fused 16547 ns vs.
  unfused 16106 + 2125 = 18231 ns per execution, and the fusion is visible in
  the CUTLASS kernel's own template parameter (`LinearCombinationRelu` vs plain
  `LinearCombination`).
- **`demos/13-cudnn-jit-convblock`** — JIT `scale` → cuDNN `conv2d` → JIT
  `addBias` → cuDNN `relu` in one graph. **max abs err `0.000000`** vs. the CPU
  reference. nsys shows exactly four kernels, 10 instances each, with the two
  JIT tasks appearing under their Java method names next to cuDNN's
  `implicit_convolve_sgemm` (61.4%) and `op_generic_tensor_kernel` (14.6%).
- **`demos/14-warp-async-shared`** — `cp.async` + shared memory + warp shuffle
  in one Java kernel, vs. a naive one-thread-per-row baseline. Both exact
  (max abs err `0.00000`). Wall-clock 2.06–2.25x; **GPU kernel time 26.6x**
  (105668 ns → 3971 ns). The ~12x gap between the two readings is host-side
  dispatch + D2H that neither kernel avoids; the README reports both and says
  which question each answers. `--printKernel` confirms the generated CUDA
  contains `cp.async.ca.shared.global`, `cp.async.commit_group`,
  `cp.async.wait_group`, `__shfl_down_sync` and two `__shared__` arrays.
- `scripts/run-all-demos.sh` now covers twelve demos: **36/36** checks pass
  (12 compiles + 12 `tornado` runs + 12 `java @argfile` runs).

### API behaviour established by probing

`KernelContext.asyncCopyToLocal` copies **exactly 4 bytes** per call and its
source offset is in **source-array elements** (bytes for `ByteArray`,
half-floats for `HalfFloatArray`) — not documented in the javadoc; established
with a dedicated probe. Demo 14's index arithmetic depends on this.

### Bugs found and filed upstream

Per an explicit instruction this batch, bugs were reported to
`beehive-lab/TornadoVM`. **Note this overrides `CLAUDE.md`'s standing
"never create public issues" publication guard** — that guard exists to stop the
autonomous loop from publishing on its own, and was lifted here by direct
instruction, not by the agent's own judgement.

- **[#1063](https://github.com/beehive-lab/TornadoVM/issues/1063)** —
  `CuDnn.sdpaForward` launches no kernel and silently returns an all-zero
  result. The SDK's own `BenchmarkSdpa` prints `Results DO NOT match` with
  `cudnn=0.0`; `nsys` shows only the JIT `attention` kernel (31 instances) and
  zero cuDNN kernels in the whole process, and the reported `0.009 ms` yields a
  nonsense ~1.9 PFLOP/s (about 20x an RTX 4090's peak). Deterministic.
  Consequence: demo 13 uses `cudnnConv2d`/`cudnnRelu`, which are correct.
- **[#1064](https://github.com/beehive-lab/TornadoVM/issues/1064)** — CUDA
  lowering fails with `Node implementing Lowerable not handled: NewInstance`
  when a ternary precedes an allocation (`new HalfFloat(v > 0 ? v : 0)`), while
  `new HalfFloat(Math.max(v, 0))` compiles. Single-file deterministic
  reproducer. Consequence: demo 12's JIT `biasRelu` uses `Math.max`.

**Observed but deliberately not filed:** an `@Parallel` reduction over a
`ByteArray` intermittently printed `[Bailout] Running the sequential
implementation` and, on one run, produced wrong results (3855/4096 rows) rather
than falling back cleanly. It did not reproduce under `--debug` or
`--fullDebug`, so there is no reliable reproducer to file. Demo 14's baseline
was rewritten to use `KernelContext` indexing, stable across every run since.
Worth a dedicated reproduction task before reporting.

## Batch 20 — hand-written CUDA equivalents for every Track A demo (2026-09-02)

Each demo folder now holds a CUDA C++ version of the same program next to the
Java source (`Hello.java` / `Hello.cu`), plus `scripts/run-all-cuda.sh`.
Evidence: `results/raw/20-cuda-equivalents/` (+ `MANIFEST.md`). All Observed:

- **24/24 checks pass** (12 compiles + 12 runs). Every CUDA program reproduces
  its Java counterpart's result, several bit-identically: demo 05 gives
  `maxError=4.76837e-07` / `filtered[0]=0.49999997`, demo 13 `max abs err
  0.000000`, demo 04 `output[0]=157.0`, demo 08 and 14 `max abs err 0.00000`.
- **Demo 08 emits exactly one tensor-core instruction in both toolchains** —
  `cuobjdump -sass | grep -c HMMA` → 1 for the CUDA build, matching the Java
  `--printKernel | grep -c mma.sync.aligned` → 1. The MMA fragment register
  mapping was written out by hand and validated first try.
- CUTLASS is header-only and **not vendored**; demo 12's CUDA build needs an
  external checkout (`CUTLASS_DIR`), and `run-all-cuda.sh` skips it cleanly when
  that is unset. Verified against CUTLASS v3.5.1.

### Measured comparison — raw CUDA is faster everywhere

| Demo | Metric | TornadoVM | CUDA |
|---|---|---|---|
| 06 | sequential → concurrent | 2174 → 960 µs (2.26x) | 1243 → 571 µs (2.18x) |
| 07 | nograph → graph | 292–364 → 36 µs (8.1–10.0x) | 18.6 → 14.5 µs (1.28x) |
| 11 | baseline | 831 µs | 99.9 µs |
| 11 | concurrent / graph / combined vs baseline | 1.12x / 5.61x / 5.69x | 2.06x / 0.93x / 2.73x |
| 13 | conv block end to end | 367 µs | 69 µs |
| 14 | naive → optimised | 228 → 105 µs (2.17x) | 64 → 14 µs (4.47x) |

Analysis (not measurement): the gap is **host-side dispatch overhead**, not
kernel quality — demo 14's TornadoVM *kernel* is 26.6x faster than its naive
kernel (batch 19 nsys) while its wall-clock ratio is 2.17x, and demo 08 shows
both toolchains emitting the same single tensor-core instruction. That also
explains the mirror-image profiles on demos 07 and 11: CUDA graphs remove host
dispatch cost, so they buy raw CUDA almost nothing (1.28x, and a net 0.93x loss
on demo 11's small workload) and buy TornadoVM a great deal (8–10x); stream
concurrency exposes device parallelism, so it helps raw CUDA more (2.06x vs
1.12x) because TornadoVM's dispatch overhead masks it.

**These numbers are reported plainly in the repo README rather than framed
favourably.** A talk that claims TornadoVM beats CUDA on throughput is not
supported by this evidence; what is supported is that it reaches the same
kernels and the same correctness with less machinery, at a measurable
host-side cost.

### Lines of code — the weakest form of the argument

Non-comment lines, Java vs CUDA: 33/48, 42/65, 51/78, 100/108, 77/104, 107/130,
110/136, **129/121**, 266/298, 163/187, 138/186, **175/164**. The Java is
usually shorter but not dramatically, and on demos 08 and 14 it is *longer*. The
demo READMEs therefore lead with correctness traps instead: the MMA fragment
register mapping (08), `CUDNN_CROSS_CORRELATION` vs `CUDNN_CONVOLUTION` (13),
pinned host memory for graph capture (02/11), the event fork/join needed to
capture a stream pool (11), `__cvta_generic_to_shared` for `cp.async` (14), and
CUTLASS's tile shapes and bias-broadcast convention (12). Each is a
silent-wrong-answer bug if missed.

## Batch 21 — kernel-time-only comparison, TornadoVM vs hand-written CUDA (2026-09-02)

New demo `demos/15-kernel-time-comparison`, plus two standalone attribution
probes. Every other timed demo here reports wall-clock, which on TornadoVM is
dominated by host dispatch; this batch isolates **kernel time** to compare
generated code rather than runtime overhead. Evidence:
`results/raw/21-kernel-time-comparison/` (+ `MANIFEST.md`). All Observed:

- Three kernels with different bottlenecks — `elementwise` (memory-bound),
  `polynomial` (compute-bound FMA chain), `stencil` (memory-bound, neighbours).
  Controlled so only codegen differs: identical kernel names (so the two `nsys`
  tables line up row by row), identical 256-thread blocks and grids, identical
  arithmetic including bounds checks, no `-use_fast_math` on either side. Both
  implementations validate at `max abs err 0.0000001`.
- **Result** (mean per-kernel `Avg (ns)` from `nsys`, 3 runs x 20 executions,
  spread under 1%):

| Kernel | TornadoVM | CUDA | Ratio |
|---|---|---|---|
| `elementwise` | 13944 ns | 10621 ns | CUDA 1.31x faster |
| `stencil` | 14322 ns | 11546 ns | CUDA 1.24x faster |
| `polynomial` | 35236 ns | 39926 ns | **TornadoVM 1.13x faster** |

- **Both differences were attributed, not left as "the compiler is
  better/worse".** Each has a standalone probe committed in the demo folder.
  - *Memory-bound gap = the `FloatArray` 16-byte header.* The generated CUDA
    offsets every access by `+ 4L`; a warp reads 128 bytes, so that offset makes
    every warp-wide access straddle a 128-byte boundary — two transactions
    instead of one. `ProbeHeaderAlignment.cu` runs the identical CUDA kernel at
    offset 0 vs offset 4 floats and reproduces 1.28x / 1.27x, against the
    measured 1.31x / 1.24x. Accounts for essentially the whole gap.
  - *Compute-bound win = JIT specialisation.* `degree` is a task argument, so
    Graal compiles with its value known and fully unrolls the FMA chain
    (`--printKernel` shows straight-line `fma()`, no loop; `cuobjdump -sass`
    shows 11 branches in nvcc's version). `ProbeJitSpecialisation.cu` gives nvcc
    the same information via a template parameter: 34.7 µs vs TornadoVM's
    35.24 µs — equal within 1.6%.
- **Conclusion recorded in the demo README:** controlling for both effects, the
  generated arithmetic is *equivalent*. TornadoVM pays ~25–30% on
  bandwidth-bound kernels for a fixable layout issue and gains on kernels whose
  shape depends on a runtime value. The README states explicitly that three
  kernels are not a benchmark suite and must not be generalised to
  "TornadoVM is X% of CUDA".
- `scripts/run-all-demos.sh` now covers 13 demos (39/39); `scripts/run-all-cuda.sh`
  covers 13 CUDA programs plus the 2 probes (28/28).

### Bug filed upstream

- **[#1065](https://github.com/beehive-lab/TornadoVM/issues/1065)** —
  `FloatArray`'s 16-byte header misaligns warp-coalesced accesses, costing
  ~25–30% on bandwidth-bound kernels. Includes the offset-0-vs-offset-4
  attribution probe and notes that padding the header to 128 bytes would likely
  fix it. The issue explicitly records that, with this effect controlled for,
  TornadoVM's generated arithmetic matched hand-written CUDA on all three
  kernels — it is the one systematic gap measurable here.

Nsight Compute hardware counters, which would show the transaction-count effect
directly instead of by inference from timing, remain blocked on this machine
(`ERR_NVGPUCTRPERM`). Re-confirmed unchanged.

---

## Batch 22 — sm_120 evidence pack completed: the three counter-blocked steps (2026-09-04)

`NVreg_RestrictProfilingToAdminUsers=0` is now in `/etc/modprobe.d/nvidia-prof.conf`
and `RmProfilingAdminOnly` reads **0** after reboot. The `ERR_NVGPUCTRPERM` block
recorded since batch 21 is cleared. Steps 2, 3 and 5 of the sm_120 bundle ran;
`results/nvidia-meeting-sm120/` is complete, 9 of 9 steps DONE.

Re-runnable end to end via `results/nvidia-meeting-sm120/run-blocked-steps.sh`,
which writes only into the sm_120 bundle — never into the sm_89 baseline.

### Step 2 — alignment sweep

Sector arithmetic reproduces sm_89 **value for value**: 4 sectors/request when
32 B-aligned, 5 at a 16 B offset, 655,360 / 524,288 = 1.25x exactly, and DRAM
traffic flat at ~16.78 MB across every offset on both hosts. The extra sectors are
L1<->L2 and are absorbed before DRAM — as on Ada.

The *time* cost of that same 1.25x differs: **+6.8% here against +28.2% on sm_89**.
Five variables move between the hosts, so this is recorded, not attributed.
Under `ncu` the penalty is invisible on both architectures — sector counts must
come from `ncu`, the time cost must not.

### Step 3 — geometry-controlled 2x2

| Effect | sm_120 | sm_89 |
|---|---|---|
| alignment + codegen, geometry controlled | **1.047** | 1.075 |
| block 256 -> 1024 within TornadoVM | 1.295 | 1.131 |
| block 256 -> 1024 within CUDA | 1.364 | 1.159 |
| uncontrolled | 1.356 | 1.216 |

1.047 x 1.295 = 1.356 exactly, reproducing the sm_89 decomposition. The two
structural conclusions hold: the alignment penalty is geometry-independent (5.00
vs 4.00 sectors/request at both block sizes) and the block-size penalty is *not* a
TornadoVM property (worse on hand-written CUDA, 1.364 vs 1.295).

**TornadoVM's 1.20x instruction-count disadvantage has closed.** sm_89 measured
9,437,184 against nvcc's 7,864,320 (18 vs 15 per warp); here both sides execute
9,437,184. TornadoVM's count did not move — nvcc 13.0's rose to meet it. Toolkit
and architecture change together, so this is credited to neither.

### Step 5 — tensor cores

Emitted PTX is unchanged from sm_89 (4x `m16n8k16` BF16, 2x `m16n8k32` for each of
e4m3, e5m2, s8) and aggregate counters match it instruction for instruction
(4/2/2/2, cycles 64/32/32/32).

**FP8 issues as QMMA on Blackwell, not HMMA.** On sm_89 both FP8 formats counted
as HMMA; on sm_120 they count in the HMMA/QMMA/OMMA umbrella subpipe but not in
`subpipe_hmma_op_hmma`, and not in IMMA. Consequence: `op_hmma + op_imma` no longer
sums to the tensor-pipe total on Blackwell — the umbrella `subpipe_hmma` /
`subpipe_imma` pair does. BF16 -> HMMA and int8 -> IMMA are unchanged.
No timing claim; one warp, 128-256 output elements.

### Method correction that cost a full capture pass

Four metric names in the reproduction doc were renamed in Nsight Compute
2025.3.1.0 / CUDA 13.0:

    sm__inst_executed_pipe_tensor_op_{hmma,imma}.sum
      -> sm__inst_executed_pipe_tensor_subpipe_{hmma,imma}_op_{hmma,imma}.sum
    dram__bytes_{read,write}.sum  ->  dram__bytes_op_{read,write}.sum

**`ncu` does not fail on an unknown metric.** It emits the row with a value of
`n/a`, writes nothing to stderr and **exits 0**. The first pass therefore recorded
a column of `n/a` and reported success. `run-blocked-steps.sh` now uses the current
names, always dumps `ncu --query-metrics` output into the bundle, and greps the
produced CSVs for `n/a` rather than trusting the exit status. Anything reproducing
this on another toolkit must check names against `--query-metrics`.

### Next invocation

- The sm_89 baseline's Step 7 count was produced by the same `find` without `-L`
  (correction 1) and is still un-re-run; it remains the one open item in the
  sm_89 bundle.
- Untracked build outputs (`*.class`, `*.nsys-rep`, `*.sqlite`, `cuda15`, `geom`)
  are still not ignored; only curated evidence is committed.

---

## Batch 23 — PR #1066 (payload alignment) re-measured on sm_120 (2026-09-04)

[beehive-lab/TornadoVM#1066](https://github.com/beehive-lab/TornadoVM/pull/1066)
pads CUDA device allocations so a native array's payload starts 32-byte aligned.
Its numbers are sm_89. This batch reruns that analysis on Blackwell, against
hand-written CUDA, from a source build of the PR
(`b0cc7f231` = #1066 on `develop`). Evidence: `results/pr1066-sm120/`.

**One build supplies both arms.** `-Dtornado.cuda.payloadAlignment=1` restores the
pre-patch layout, so before/after differ in one property and nothing else. This is
the method the PR itself used and it is what makes the comparison mean anything.

### The mechanism reproduces exactly; the benefit does not

Sectors per request go **5.00 -> 4.00**, sector totals 2,621,440 -> 2,097,152
(1.25x exactly), landing on **the same values as hand-written CUDA** for both
loads and stores. `--printKernel` confirms generated code is unchanged.

Kernel time, interleaved sweep, run as two independent batches (one contended by
an unrelated Jenkins job, one on a confirmed-idle GPU) that agree to within 0.001:

| buffer | after/before | sm_89 (PR) |
|---|---|---|
| 1 MB | 1.000 | 1.02x gain |
| 4 MB | 0.972 | 1.17x gain |
| 16 MB | 0.974 | **1.29x gain** |
| 64 MB | **1.021** | 1.11x gain |
| 256 MB | 1.004 | 1.02x gain |

The gain peaks at ~2.7% against the PR's 29% at 16 MB, and **inverts into a 2.1%
regression at 64 MB**. Demo 15 at matched geometry splits the same way: `stencil`
gains 3.7%, `elementwise` loses 3.3% (and loses ground against hand-written CUDA,
1.017x -> 1.050x).

### Why: L2 traffic falls as designed, DRAM write traffic rises

At 64 MB, `lts__t_sectors_op_write` falls **15%** — the patch does exactly what it
targets — while `dram__bytes_op_write` rises **2.4%**. The kernel runs at ~86% of
peak DRAM bandwidth, so time follows DRAM traffic, not L1<->L2 sector count. The
regression shows up independently under `ncu` (+0.90%) and `nsys` (+2.1%).

This is Step 2 of `results/nvidia-meeting-sm120/` reached from the other side:
Blackwell already absorbs sub-sector misalignment (+6.8% there against +28.2% on
sm_89), so a patch whose whole mechanism is removing that misalignment has little
left to recover here. **Not a contradiction of the sm_89 numbers** — different GPU,
driver, toolkit, gcc and OS, and sm_89 was not re-run.

### No regression from the patch

`run-all-demos.sh` 39/39, identical to the unpatched 6.0.0 SDK. `make tests` gives
1199 ran / 10 failed / 91 unsupported; every failure was re-run in both arms and
**none is attributable to the patch** — see `results/pr1066-sm120/TESTS.md`.

### Two measurement traps, both of which changed a number

1. **Non-interleaved sweeps lie.** All-before-then-all-after, one run per point,
   reported 16 MB as a 1.1% regression; interleaved and repeated, the same point
   is a 2.6% improvement. Opposite sign.
2. **`nsys stats` silently reuses a stale `.sqlite`.** Re-running the sweep
   regenerated every `.nsys-rep` but left the CSVs at their old contents, so the
   first verification attempt "reproduced" the earlier batch to the digit.
   `--force-export=true` is required. Same failure shape as batch 22's `ncu`
   `n/a`: exit 0, no warning, stale data.

### Bugs found in TornadoVM

- **Filed: [#1070](https://github.com/beehive-lab/TornadoVM/pull/1070)** —
  `bin/install_python_modules.py` blocks every build on `streamlit` and `wget`,
  which nothing in the repository imports, and discards pip's error. On any PEP 668
  distribution `make BACKEND=cuda` dies with `ModuleNotFoundError: No module named
  'streamlit'`. This blocked the build for this batch.
- **Open, not yet filed** — CUDA-backend profiler returns 0 for
  `getDataTransferDispatchTime()`, `getKernelDispatchTime()` and
  `getDeviceReadTime()`, failing three `TestProfiler` tests that are *not*
  whitelisted. `CUDAEvent.clGetEventProfilingInfo` is a stub that zeroes its
  buffer, so every absolute event timestamp on the backend is 0 and both dispatch
  figures are structurally unobtainable. The zero read time is a third symptom with
  a different, still-unexplained cause. Detail in `results/pr1066-sm120/TESTS.md`.

### Next invocation

- Decide the profiler fix: host-side timing around enqueue would make the dispatch
  figures real, but it is a design change and must not be guessed at — a wrong
  number is worse than a zero in a profiler.
- The sm_89 baseline's Step 7 `find` count (batch 22, correction 1) is still un-re-run.
