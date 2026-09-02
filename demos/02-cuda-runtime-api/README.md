# 02 — CUDA Runtime API access from Java

**Concept (read in ~1 minute):** the previous two demos show a Java kernel
running on the GPU. This one shows a different thing: TornadoVM's Java API
also exposes **CUDA runtime behaviour directly**, not just kernel dispatch.
`TornadoExecutionPlan#withCUDAGraph()` captures the whole task-graph
(H2D copy → kernel launch → D2H copy) as a single **CUDA graph** the first
time `execute()` runs, then **replays** that graph on every call after that
instead of re-issuing each CUDA runtime call individually — this is
`cudaGraphCreate`/`cudaStreamBeginCapture` + `cudaGraphLaunch`, reached from
plain Java. Per the javadoc on `TornadoExecutionPlan`, `withCUDAGraph()`
(along with `withIntraPlanConcurrency()` for CUDA streams and
`withStagedTransfers()` for pinned-memory staging) is "currently realised on
the CUDA backend" — this is CUDA-runtime API surface, distinct from the
vendor-library (cuBLAS/cuFFT/cuDNN/cuSPARSE/CUTLASS) hybrid-API tasks
covered in `docs/hybrid-api-inventory.md` and the `04` demo.

Source: [`CudaGraphReplay.java`](CudaGraphReplay.java).

The demo runs an `axpy` (`result = alpha*x + y`) task-graph 8 times under
graph capture/replay, mutating the inputs before every replay and validating
the output against a CPU-computed expected value after every replay — proof
the captured graph re-reads the new input values on each launch rather than
replaying stale data.

## Build

```bash
source scripts/setup-env.sh   # from repo root; pins the SDK in env/versions.env
cd demos/02-cuda-runtime-api
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  -d . CudaGraphReplay.java
```

## Run

Canonical, with profiler output showing the CUDA backend and per-replay
task-graph time collapsing after the first (capture) execution:

```bash
tornado --enableProfiler console --classpath . CudaGraphReplay 8192 8
```

Reproducibility form (`java @arg-file`):

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp . CudaGraphReplay 8192 8
```

Arguments: `<array size> <number of replays>` (defaults `8192 8` if omitted).

## Expected output

```
replay 0 (x=1.0, y=2.0): correct, result[0]=2.5 expected=2.5
replay 1 (x=2.0, y=3.0): correct, result[0]=4.0 expected=4.0
...
replay 7 (x=8.0, y=9.0): correct, result[0]=13.0 expected=13.0
All replays correct
```

With `--enableProfiler console`, the first execution's profiler block shows
`"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"` and a
`TOTAL_TASK_GRAPH_TIME` in the tens-of-milliseconds range (JIT compile +
graph capture); every subsequent replay's block shows only
`TOTAL_TASK_GRAPH_TIME`, typically well under 100 microseconds — the
observable effect of replaying a captured graph instead of re-dispatching.
Re-verified on TornadoVM 6.0.0 / JDK 25 — all 8 replays correct under both
run paths: `results/raw/18-tornadovm-6-migration/02-cudagraph-tornado.log`,
`02-cudagraph-javaargfile.log`.

Earlier 5.2.1 logs: `results/raw/03-cuda-runtime-api/cudagraphreplay-run.log`,
`cudagraphreplay-run-javaargfile.log`.

## CUDA equivalent

[`CudaGraphReplay.cu`](CudaGraphReplay.cu) is the same demo written directly in CUDA C++, for side-by-side comparison.

```bash
nvcc -arch=sm_89 -o cuda_graph_replay CudaGraphReplay.cu && ./cuda_graph_replay
```

`plan.withCUDAGraph()` is one method call. This is what it stands for:

```c
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
cudaMemcpyAsync(...); axpy<<<...,stream>>>(...); cudaMemcpyAsync(...);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
...
cudaGraphLaunch(graphExec, stream);   // per replay
```

Plus a requirement that is easy to miss: the host buffers must be **pinned**
(`cudaMallocHost`), because the captured copies reference them by address. Use
pageable memory and the capture either fails or silently replays stale data —
which is exactly the failure this demo's per-replay validation is designed to
catch.

Output is identical to the Java version, all 8 replays correct.

`bash scripts/run-all-cuda.sh` builds and runs the CUDA equivalent of every demo (needs only the CUDA toolkit, no JDK).

## JBang

Not verified: `jbang` is not installed on this machine (`which jbang` →
command not found, checked 2026-08-19, same as `../00-hello-gpu/README.md`
and `../01-first-cuda-kernel/README.md`). The shape would match those demos'
documented-but-unverified pattern — do not run it live until tested on the
pinned environment:

```bash
jbang --version
jbang -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  --java-opts="@$TORNADOVM_HOME/tornado-argfile" \
  CudaGraphReplay.java
```

## If the demo fails on stage

- If `withCUDAGraph()` throws or replays report `WRONG`, fall back to the
  captured log at `results/raw/03-cuda-runtime-api/cudagraphreplay-run.log`
  and explain the concept from the printed CUDA-graph timing collapse there.
- Re-run `tornado --devices` first — if it does not show exactly one CUDA
  device, the environment, not the demo, is broken.
- This is CUDA-only behavior by design (the API is a no-op/unsupported on
  other backends per `TornadoExecutionPlan`'s javadoc) — do not attempt to
  "fix" a failure by switching backend.
