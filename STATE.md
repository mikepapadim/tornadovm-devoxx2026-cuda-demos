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

### Next invocation

Start with `auto/tasks/02.md` (build Java CUDA hello and kernel demos). Before starting, verify no other `auto/tasks/02*` marker races are in flight (see operational note above), re-check `git status` is clean, and re-verify the pinned SHA in `env/versions.env` still matches `vendor/tornadovm` HEAD.
