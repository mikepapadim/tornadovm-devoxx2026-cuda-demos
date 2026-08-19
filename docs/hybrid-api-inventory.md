# Hybrid API Inventory (TornadoVM CUDA, pinned SHA)

Pinned source: `beehive-lab/TornadoVM` `develop` @ `99549c9862eda8d584e35e99924f9c865501eb3a` (see `env/versions.env`). All paths below are relative to `vendor/tornadovm`. Evidence classification per `PLAN.md` §6: **Observed** (reproduced locally, log captured), **Source-backed** (verified directly in this checkout, not run), **Hypothesis**, **Blocked**.

Do not trust upstream documentation for this API — it is unreleased/in-flux on `develop`. Everything below was verified against source and/or a runnable probe, not against docs.

## 1. Core composition model — Source-backed + Observed

The Hybrid API lets native-library calls ride the same `TaskGraph`/`TornadoExecutionPlan` pipeline as JIT-compiled `@Parallel`/`KernelContext` Java kernels, as **library tasks**.

- `TaskGraph.libraryTask(String id, LibraryTaskN<...> code, T1 arg1, ..., TN argN)` — overloads for 1–20 arguments.
  Source: `tornado-api/src/main/java/uk/ac/manchester/tornado/api/TaskGraph.java:737-895`.
- `LibraryTaskDescriptor` — built by library-binding factory methods (`.withLibrary().withFunction().withParameters().withAccess().withTuning()`), consumed by the runtime.
  Source: `tornado-api/src/main/java/uk/ac/manchester/tornado/api/common/LibraryTaskDescriptor.java`.
- `TornadoLibraryProvider` — the SPI (`java.util.ServiceLoader`) implemented by each library module (`provides ... with ...` in each module's `module-info.java`). One provider = one library. `createContext(device, executionPlanId)` builds one native handle per (device, execution plan) bound to the plan's CUDA stream; `prepare()` runs before every launch region (and before CUDA-graph capture) for capture-safe workspace allocation; `dispatch()` executes with args already resolved to device pointers; `destroyContext()` releases native resources.
  Source: `tornado-runtime/src/main/java/uk/ac/manchester/tornado/runtime/library/spi/TornadoLibraryProvider.java`.
- `LibraryTask.java`, `LibraryInvocation.java`, and dispatch wiring in `TornadoTaskGraph.java` / `TornadoVMInterpreter.java` (`tornado-runtime/src/main/java/uk/ac/manchester/tornado/runtime/tasks/`, `.../interpreter/`) connect graph construction to provider dispatch.
- Library tasks and JIT `.task()` calls can be **interleaved in one `TaskGraph`**, sharing device buffers with no host round-trips, and are CUDA-graph capturable. Directly demonstrated (see §3) rather than just claimed in the module doc comments.

**Verification command (backend-agnostic, construction-only):**
```
tornado-test -V uk.ac.manchester.tornado.unittests.tasks.TestLibraryTaskArity
```
Source: `tornado-unittests/src/main/java/uk/ac/manchester/tornado/unittests/tasks/TestLibraryTaskArity.java`. Not run this batch (arity/construction check only, no device dispatch) — Source-backed.

## 2. Providers present in the pinned tree

| Provider | Module | Provider class | Library id | Ops (factory methods) |
|---|---|---|---|---|
| cuBLAS | `tornado-cublas` | `cublas/provider/CuBlasLibraryProvider.java` | `nvidia/cublas` | `cublasSgemv`, `cublasSgemm`, `cublasSgemmTF32`, `cublasSgemmStridedBatched`, `cublasGemmExFP16`, `cublasGemmExFP16FP32`, `cublasGemmExBF16` (`cublas/CuBlas.java`) |
| cuBLASLt | `tornado-cublas` | `cublas/provider/CuBlasLtLibraryProvider.java` | (cuBLASLt, same module) | `ltMatmulFP32`, `ltMatmulFP16`, `ltMatmulFP8`, `ltMatmulBiasFP16`, `ltMatmulGeluBiasFP16` (`cublas/CuBlasLt.java`) |
| cuFFT | `tornado-cufft` | `cufft/provider/CuFftLibraryProvider.java` | `nvidia/cufft` | `cufftForwardC2C`, `cufftInverseC2C`, `cufftForwardR2C`, `cufftInverseC2R` (+ batched/2D/double-precision variants exercised by tests) (`cufft/CuFft.java`) |
| cuDNN | `tornado-cudnn` | `cudnn/provider/CuDnnLibraryProvider.java` | (cuDNN) | `cudnnSoftmax`, `cudnnRelu`, `cudnnSigmoid`, `cudnnTanh`, `cudnnMaxPool2d`, `sdpaForward`, `cudnnConv2d` (`cudnn/CuDnn.java`) |
| cuSPARSE | `tornado-cusparse` | `cusparse/provider/CusparseLibraryProvider.java` | (cuSPARSE) | `cusparseSpMV`, `cusparseSpMM` (CSR) (`cusparse/Cusparse.java`) |
| CUTLASS | `tornado-cutlass` | `cutlass/provider/CutlassLibraryProvider.java` | `nvidia/cutlass` | `cutlassSgemm`, `cutlassHgemm`, `cutlassBgemm`, `cutlassHgemmBatched`, fused `cutlassGemmBias{Relu,Gelu,Silu,Sigmoid,Tanh,HardSwish}` (`cutlass/Cutlass.java`) |

All six are registered as top-level Maven modules in `pom.xml` and were built by `make BACKEND=cuda` in task 00 (`results/raw/00-baseline/tornadovm-build-cuda.log`). CUTLASS is present as a full library-task module pair (JNI + Java factory + provider), matching the plan recorded in `CUTLASS-plan.md` at repo root of the checkout — that file is a design/implementation plan checked into the pinned tree, not upstream documentation; treated as Source-backed only where the described classes actually exist (verified above), not for any unbuilt/aspirational item in it.

## 3. Observed: providers actually dispatch and validate correctly on this GPU

Run this batch, `vendor/tornadovm` @ pinned SHA, `source vendor/tornadovm/setvars.sh`, RTX 4090, CUDA backend only. Raw logs: `results/raw/01-hybrid-api/`.

| Probe | Command | Result | Log |
|---|---|---|---|
| Hybrid JIT+cuFFT pipeline (mixed `.task()` + `.libraryTask()` in **one** graph — R2C → JIT low-pass → C2R → JIT normalize) | `tornado -m tornado.cufft/uk.ac.manchester.tornado.cufft.tests.FrequencyFilterExample` | `Result is correct`, max error 1.67e-6 | `frequency-filter-example.log` |
| Hybrid JIT-kernel vs cuBLAS SGEMV (separate graphs, same run) | `tornado -m tornado.cublas/uk.ac.manchester.tornado.cublas.tests.MatrixVectorRowMajorWithCuBlas 512 128 32` | all 3 variants (`@Parallel`, `KernelContext`, cuBLAS) `OK` vs sequential Java reference | `matrix-vector-cublas.log` |
| cuBLAS unit tests (incl. `testSharedBufferAcrossTaskGraphs`, `testMixedTasksWithCudaGraph`) | `tornado-test -V uk.ac.manchester.tornado.unittests.cublas.TestCuBlas` | 12/12 PASS | `test-cublas.log` |
| cuBLASLt unit tests | `tornado-test -V uk.ac.manchester.tornado.unittests.cublas.TestCuBlasLt` | 6/6 PASS | `test-cublaslt.log` |
| cuFFT unit tests (incl. `testRoundTripWithCudaGraph`) | `tornado-test -V uk.ac.manchester.tornado.unittests.cufft.TestCuFft` | 8/8 PASS | `test-cufft.log` |
| cuDNN unit tests (incl. `testConv2dWithCudaGraph`, SDPA) | `tornado-test -V uk.ac.manchester.tornado.unittests.cudnn.TestCuDnn` | 9/9 PASS | `test-cudnn.log` |
| cuSPARSE unit tests (incl. `testSpMVWithCudaGraph`, `testSpMVWithJitPreAndPost`) | `tornado-test -V uk.ac.manchester.tornado.unittests.cusparse.TestCusparse` | 11/11 PASS | `test-cusparse.log` |
| CUTLASS unit tests (incl. `testSharedBufferAcrossTaskGraphs`, `testGemmWithCudaGraph`, fused epilogues, mixed precision) | `tornado-test -V uk.ac.manchester.tornado.unittests.cutlass.TestCutlass` | 25/25 PASS | `test-cutlass.log` |
| Device sanity | `tornado --devices` | one CUDA device, `NVIDIA GeForce RTX 4090` | `tornado-devices.log` |

Total: **71/71** provider unit tests pass, plus 2 standalone example runs, all on the CUDA backend, all correctness-validated against a Java/sequential reference by the test/example itself. This directly satisfies PLAN.md Track A items A2–A6 (cuBLAS call, cuFFT call, Java-kernel+cuBLAS in one graph family, Java-kernel+cuFFT in one graph, buffer/stream reuse across tasks) at the unit-test and example level; no standalone `A2`/`A3`/`A4`/`A5`-labelled demo file exists yet under a demos/ directory of this repo — these are TornadoVM upstream tests/examples in the pinned checkout, not yet adapted into this repo's own demo set (later Track A tasks).

## 4. Task-graph composition patterns confirmed by source

- **Mixed graph, one execution plan**: `.task()` and `.libraryTask()` calls chained on the same `TaskGraph` builder, snapshotted once, executed via one `TornadoExecutionPlan` — e.g. `tornado-cufft/src/main/java/uk/ac/manchester/tornado/cufft/tests/FrequencyFilterExample.java:78-84` (`transferToDevice` → `libraryTask("forward", CuFft::cufftForwardR2C, ...)` → `task("lowPass", ...)` → `libraryTask("inverse", CuFft::cufftInverseC2R, ...)` → `task("normalize", ...)` → `transferToHost`). Ran successfully in §3.
- **Separate graphs, compared at runtime**: `tornado-cublas/src/main/java/uk/ac/manchester/tornado/cublas/tests/MatrixVectorRowMajorWithCuBlas.java` builds three independent `TaskGraph`s (naive `@Parallel`, `KernelContext`, cuBLAS `libraryTask`) and benchmarks each via its own `TornadoExecutionPlan` — this is the pattern for **comparing** a JIT kernel against a library task, not for fusing them.
- **Shared buffers across task graphs**: `testSharedBufferAcrossTaskGraphs` exists for both cuBLAS (`TestCuBlas.java`) and CUTLASS (`TestCutlass.java`) — Source-backed + Observed (both suites passed in §3); not read line-by-line this batch, so the exact reuse mechanism (same `TornadoXPUDevice`/buffer handle across plans) is Source-backed by name/pass-result only, not by detailed code reading.
- **CUDA-graph capture with library tasks**: `TornadoLibraryProvider.prepare()` is documented (Javadoc) to run before CUDA-graph capture starts on first execution specifically so providers can pre-allocate capture-unsafe memory (cuFFT plans, cuDNN workspaces). `testXxxWithCudaGraph` tests exist and pass for cuBLAS, cuFFT, cuDNN, cuSPARSE, CUTLASS (§3) — Observed that capture+library-task combination works end-to-end; the capture mechanics themselves (`prepare()` idempotency, stream binding) are Source-backed from the interface Javadoc, not independently traced through the interpreter this batch.

## 5. Buffer/stream sharing semantics — Source-backed

- One native library handle/context per **(device, execution plan)**, created by `TornadoLibraryProvider.createContext(device, executionPlanId)` and bound to that plan's CUDA stream via `TornadoNativeStreamSupport` (referenced in `CuBlasLibraryProvider.java` class Javadoc and imports).
- Device buffers are TornadoVM-managed (`XPUBuffer`), not separately allocated by the library provider; the interpreter resolves `LibraryTaskDescriptor` parameters to device pointers before calling `dispatch()`. Exact buffer-header layout details (referenced in `CUTLASS-plan.md` as `XPUBuffer.toBuffer() + ARRAY_HEADER(24)`) were not independently re-derived from `XPUBuffer` source this batch — flagged as Source-backed via the plan doc, not re-verified against the buffer class itself.
- `Access[]` on `LibraryTaskDescriptor` (`READ_ONLY`/`WRITE_ONLY`/`READ_WRITE`) drives TornadoVM's data-transfer/consistency tracking for library-task arguments exactly as it does for JIT task arguments — confirmed by usage pattern in every `CuBlas.java`/`CuFft.java` factory method (e.g. `readOnlyExcept()` helper switching an output to `READ_WRITE` when `beta != 0`).

## 6. Not yet probed / open questions for later tasks

- **A1 (CUDA runtime API access from Java, outside library tasks)**: not covered by this inventory; no dedicated "raw CUDA runtime from Java" class found under the cuBLAS/cuFFT/etc. modules during this pass. Needs its own targeted search in a later task — Hypothesis that this lives in `tornado-drivers/cuda` (device/stream plumbing) rather than a user-facing API, unconfirmed.
- **A7 (error handling / unsupported API behavior / portability boundaries)**: `TornadoVMCUDANotSupported` import appears in `TestCusparse.java` and `TestCutlass.java`, implying some tests are skipped/guarded on non-CUDA backends — not read in detail this batch. Source-backed (name only) — needs follow-up.
- **cuDNN provider list above (`cudnnSoftmax` etc.) and CUTLASS fused-epilogue list** are the factory methods present in the Java binding class; whether every one has a corresponding passing unit test was confirmed only in aggregate (suite pass counts in §3), not method-by-method.
- No repo-local (this repository, not the vendored TornadoVM checkout) Hybrid API demo files exist yet — Track A demos (A0–A7) as *this study's own* runnable artifacts are future tasks; today's evidence is entirely against TornadoVM's own upstream tests/examples at the pinned SHA.

## 7. Reproduction

```bash
cd vendor/tornadovm && source setvars.sh
tornado --devices
tornado -m tornado.cufft/uk.ac.manchester.tornado.cufft.tests.FrequencyFilterExample
tornado -m tornado.cublas/uk.ac.manchester.tornado.cublas.tests.MatrixVectorRowMajorWithCuBlas 512 128 32
tornado-test -V uk.ac.manchester.tornado.unittests.cublas.TestCuBlas
tornado-test -V uk.ac.manchester.tornado.unittests.cublas.TestCuBlasLt
tornado-test -V uk.ac.manchester.tornado.unittests.cufft.TestCuFft
tornado-test -V uk.ac.manchester.tornado.unittests.cudnn.TestCuDnn
tornado-test -V uk.ac.manchester.tornado.unittests.cusparse.TestCusparse
tornado-test -V uk.ac.manchester.tornado.unittests.cutlass.TestCutlass
```
