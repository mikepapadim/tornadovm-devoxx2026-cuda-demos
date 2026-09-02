# 04 — Java kernel + cuBLAS in one execution graph

**Concept (read in ~1 minute):** the Hybrid API lets one `TaskGraph` mix
JIT-compiled Java tasks (`.task(...)`) and vendor-library calls
(`.libraryTask(...)`) that all operate on the same TornadoVM-managed device
buffers, in one execution plan. This demo chains three stages on the GPU
without any host round-trip in between:

1. `scale` — a plain `@Parallel` Java loop, JIT-compiled to CUDA, multiplies
   every matrix element by 2.0.
2. `sgemv` — `CuBlas.cublasSgemv` (`nvidia/cublas`), computing
   `y = alpha * op(A) * x + beta * y` on the *same* device buffer `scale`
   just wrote.
3. `bias` — another `@Parallel` Java loop adds 1.0 to every element of the
   cuBLAS output, still on-device.

cuBLAS assumes column-major storage; the matrix here is built and validated
row-major (like all the other demos), so the graph passes `CUBLAS_OP_T` for
the SGEMV operation — see the javadoc on `CuBlas.cublasSgemv` in
`vendor/tornadovm/tornado-cublas/.../CuBlas.java`. Source-backed pattern
this demo is modeled on:
`vendor/tornadovm/tornado-cublas/.../tests/TestCuBlasSgemvWithTornadoVMTasksPOST.java`
(same pre-task/cuBLAS/post-task shape, verified in `docs/hybrid-api-inventory.md` §3).

Source: [`CuBlasSgemvHybrid.java`](CuBlasSgemvHybrid.java).

Every iteration is validated against a plain sequential Java reference that
performs the identical scale → matvec → bias pipeline on the CPU, using the
same deterministic (non-random) matrix/vector fill.

## Build

```bash
source scripts/setup-env.sh   # from repo root; pins the SDK in env/versions.env
cd demos/04-cublas-hybrid
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  -d . CuBlasSgemvHybrid.java
```

## Run

Canonical, with profiler output showing all three stages (`scale`, `sgemv`,
`bias`) executing on the CUDA backend:

```bash
tornado --enableProfiler console --classpath . CuBlasSgemvHybrid 8 8 5
```

Reproducibility form (`java @arg-file`):

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp . CuBlasSgemvHybrid 8 8 5
```

Arguments: `<m> <n> <iterations>` (defaults `8 8 5` if omitted). `m` is the
number of matrix rows/output elements, `n` the number of columns/input
vector elements.

## Expected output

```
iteration 0: correct output[0]=157.0 expected[0]=157.0
  total task-graph time: 171519337 ns
iteration 1: correct output[0]=157.0 expected[0]=157.0
  total task-graph time: 496022 ns
...
All iterations correct
```

With `--enableProfiler console`, each iteration's JSON block shows
`cublasHybrid.scale`, `cublasHybrid.sgemv`, and `cublasHybrid.bias` — each
with `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"` — confirming
the JIT tasks and the cuBLAS library task all execute on the CUDA device
inside a single task graph. The first iteration's `TOTAL_TASK_GRAPH_TIME` is
dominated by JIT compilation of the two Java tasks (Graal + driver compile,
tens of ms); later iterations drop to sub-millisecond since the compiled
code is reused.

Re-verified on TornadoVM 6.0.0 / JDK 25 — 5/5 iterations correct under both
run paths: `results/raw/18-tornadovm-6-migration/04-cublas-tornado.log`,
`04-cublas-javaargfile.log`.

Earlier 5.2.1 logs:
`results/raw/04-cublas-hybrid/cublassgemvhybrid-run.log`,
`cublassgemvhybrid-run-javaargfile.log`.

## JBang

Not verified: `jbang` is not installed on this machine (`which jbang` →
command not found, checked 2026-08-19, same finding as demos `00`–`03`). The
shape would match those demos' documented-but-unverified pattern — do not
run it live until tested on the pinned environment:

```bash
jbang --version
jbang -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  --java-opts="@$TORNADOVM_HOME/tornado-argfile" \
  CuBlasSgemvHybrid.java
```

## If the demo fails on stage

- If any iteration prints `WRONG`, fall back to the captured log at
  `results/raw/04-cublas-hybrid/cublassgemvhybrid-run.log` and walk through
  the JSON profiler block explaining the three-stage graph instead.
- Re-run `tornado --devices` first — if it does not show exactly one CUDA
  device, the environment, not the demo, is broken.
- `NoClassDefFoundError` for `uk.ac.manchester.tornado.cublas.*` usually
  means the classpath is missing `tornado-cublas-6.0.0.jar` — add
  it alongside `tornado-api-6.0.0.jar` (see Build above).
