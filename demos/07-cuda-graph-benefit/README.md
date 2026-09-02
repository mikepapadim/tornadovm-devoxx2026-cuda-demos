# 07 — Why CUDA Graph replay matters for repeated workloads

**Concept (read in ~1 minute):** demo [02](../02-cuda-runtime-api/) showed that
`TornadoExecutionPlan#withCUDAGraph()` captures a task-graph once and replays
it on every later `execute()` — real `cuStreamBeginCapture` /
`cuStreamEndCapture` / `cuGraphInstantiate` / `cuGraphLaunch` calls underneath
(`vendor/tornadovm/tornado-drivers/cuda-jni/.../CUDAGraph.cpp`). That demo's
timing evidence mixed first-execution JIT-compile time into the "before"
number, so it didn't isolate what graph replay actually buys you.

This demo isolates it: the **same** 6-stage chain of small elementwise JIT
tasks (each stage's output feeds the next, host↔device transfers every
execution) runs for 50 executions in two modes from the same JVM —
`nograph` (plain `TornadoExecutionPlan`, every execution re-issues each
stage's H2D copy / kernel launch / D2H copy as separate CUDA runtime calls)
and `graph` (`withCUDAGraph()`, every execution after the first is a single
`cuGraphLaunch` replay). Both modes discard execution 0 (JIT compile +, for
graph mode, capture) and report the median wall-clock of the rest. Every
execution in both modes is validated against a closed-form CPU reference —
the input array is mutated before every execute() call, so a stale replay
would be caught.

Source: [`CudaGraphBenefit.java`](CudaGraphBenefit.java).

## Build

```bash
source scripts/setup-env.sh   # from repo root; pins the SDK in env/versions.env
cd demos/07-cuda-graph-benefit
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  -d . CudaGraphBenefit.java
```

## Run

```bash
tornado --classpath . CudaGraphBenefit 4096 6 50 both
```

Reproducibility form (`java @arg-file`):

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp . CudaGraphBenefit 4096 6 50 both
```

Arguments: `<array size> <chain stages> <executions> <mode>` where mode is
`nograph`, `graph`, or `both` (default `4096 6 50 both`).

With `--enableProfiler console` (`tornado --enableProfiler console --classpath . CudaGraphBenefit 4096 6 5 both`),
every stage in every execution reports `"BACKEND": "CUDA"`,
`"DEVICE": "NVIDIA GeForce RTX 4090"` — this is genuinely CUDA-backend work
in both modes, not a fallback.

## Expected output (shape)

```
=== NOGRAPH (plain execute) ===
execution 0: correct, wall=<large, JIT compile> us
execution 1: correct, wall=<...> us
...
NOGRAPH (plain execute) steady-state median wall-clock (excl. first execution, n=49): <N1> us
NOGRAPH (plain execute): All executions correct

=== GRAPH (withCUDAGraph) ===
execution 0: correct, wall=<large, JIT compile + graph capture> us
...
GRAPH (withCUDAGraph) steady-state median wall-clock (excl. first execution, n=49): <N2> us
GRAPH (withCUDAGraph): All executions correct

steady-state median: nograph=<N1> us, graph=<N2> us, speedup=<N1/N2>x
```

## What was actually measured (this run, this GPU — not a general guarantee)

Three independent runs on the pinned TornadoVM 6.0.0 CUDA SDK
(`6.0.0-jdk22plus-cuda`, JDK 25.0.2, RTX 4090), all with size=4096,
stages=6, and all executions correct in both modes:

| Run | nograph steady-state median | graph steady-state median | speedup |
|---|---|---|---|
| `tornado --classpath .` | 364.2 us | 36.4 us | 10.00x |
| `java @$TORNADOVM_HOME/tornado-argfile` | 292.4 us | 36.2 us | 8.08x |
| `--enableProfiler console` (5 executions, extra console-dump overhead in both modes) | 1070.9 us | 121.4 us | 8.82x |

The same three runs on the previous 5.2.1 source-built pin gave 6.47x, 6.58x
and 7.02x. The `graph` steady-state median is essentially unchanged (~36 us on
both); what moved is the `nograph` path, so read this as run-to-run variance in
per-execution dispatch overhead, not as a measured 6.0.0 optimisation.

The consistent direction across all three runs (graph replay several times
faster, steady state, same validated correctness) is the presenter-visible
point: for a task-graph made of several small dispatches, CUDA graph replay
collapses per-execution CPU-side dispatch overhead (host-to-device copy +
kernel launch + device-to-host copy, issued as one `cuGraphLaunch` instead of
one CUDA runtime call per stage) — this is the concrete "why it matters for
repeated workloads" that demo 02's capture/replay correctness demo did not
by itself quantify. The exact multiplier is workload/GPU/driver-dependent
(this-run-only number, same caveat as demo 06's stream-overlap numbers) —
don't quote a specific "Nx" figure in the talk as a general TornadoVM claim,
quote it as "observed on this machine, this workload."

Captured logs (TornadoVM 6.0.0):
`results/raw/18-tornadovm-6-migration/07-graphbenefit-tornado.log`,
`07-graphbenefit-javaargfile.log`, `07-graphbenefit-profiler.log`.
Earlier 5.2.1 logs are kept unmodified in
`results/raw/07-cuda-graph-benefit/`.

## Fallback if the live demo fails

If a live re-run doesn't reproduce a clean speedup (e.g. a noisy machine),
fall back to the captured logs above and demo 02's simpler capture/replay
correctness demo, which makes the qualitative point (graph replay exists and
works) without depending on a live timing number.

## CUDA equivalent

[`CudaGraphBenefit.cu`](CudaGraphBenefit.cu) is the same demo written directly in CUDA C++, for side-by-side comparison.

```bash
nvcc -arch=sm_89 -o cuda_graph_benefit CudaGraphBenefit.cu && ./cuda_graph_benefit 4096 6 50 both
```

The same experiment against raw CUDA graphs — and the most interesting
comparison in the repo, because the two do **not** agree:

| | TornadoVM | CUDA |
|---|---|---|
| nograph steady-state median | 292–364 µs | 18.6 µs |
| graph steady-state median | 36 µs | 14.5 µs |
| **speedup from graphs** | **8.1x–10.0x** | **1.28x** |

Read that carefully before quoting it. CUDA graphs remove *host-side dispatch
overhead*. Raw CUDA barely has any for a 6-kernel chain, so graphs buy it
almost nothing (1.28x). TornadoVM has a great deal of it, so graphs buy it a
lot (8–10x) — and even then land at 36 µs, still ~2.5x the handwritten 14.5 µs.

The honest framing for a talk: `withCUDAGraph()` is not making TornadoVM faster
than CUDA, it is removing most of the interpreter's own per-execution cost. It
is the single highest-leverage flag in the API precisely because that cost
exists.

`bash scripts/run-all-cuda.sh` builds and runs the CUDA equivalent of every demo (needs only the CUDA toolkit, no JDK).

## JBang

Not verified: `jbang` is not installed on this machine (`which jbang` → exit
1, checked 2026-08-20, same finding as demos 00–06). The shape would match
those demos' documented-but-unverified pattern — do not run it live until
tested on the pinned environment:

```bash
jbang CudaGraphBenefit.java 4096 6 50 both
```
