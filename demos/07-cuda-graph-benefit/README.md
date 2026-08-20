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
source vendor/tornadovm/setvars.sh   # from repo root
cd demos/07-cuda-graph-benefit
javac --release 21 --enable-preview \
  -cp "$TORNADOVM_HOME/share/java/tornado/tornado-api-5.2.1-jdk21-dev.jar" \
  -d . CudaGraphBenefit.java
```

## Run

```bash
tornado --classpath . CudaGraphBenefit 4096 6 50 both
```

Reproducibility form (`java @arg-file`):

```bash
java @../tornado.args -cp . CudaGraphBenefit 4096 6 50 both
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

Three independent runs on the pinned build (`vendor/tornadovm` @
`99549c9862eda8d584e35e99924f9c865501eb3a`, RTX 4090), all with
size=4096, stages=6, and all executions correct in both modes:

| Run | nograph steady-state median | graph steady-state median | speedup |
|---|---|---|---|
| `tornado --classpath .` | 233.6 us | 36.1 us | 6.47x |
| `java @../tornado.args` | 238.3 us | 36.2 us | 6.58x |
| `--enableProfiler console` (5 executions, extra console-dump overhead in both modes) | 1008.8 us | 143.7 us | 7.02x |

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

Captured logs:
`results/raw/07-cuda-graph-benefit/cudagraphbenefit-run.log`,
`cudagraphbenefit-run-javaargfile.log`, `cudagraphbenefit-profiler.log`.

## Fallback if the live demo fails

If a live re-run doesn't reproduce a clean speedup (e.g. a noisy machine),
fall back to the captured logs above and demo 02's simpler capture/replay
correctness demo, which makes the qualitative point (graph replay exists and
works) without depending on a live timing number.

## JBang

Not verified: `jbang` is not installed on this machine (`which jbang` → exit
1, checked 2026-08-20, same finding as demos 00–06). The shape would match
those demos' documented-but-unverified pattern — do not run it live until
tested on the pinned environment:

```bash
jbang CudaGraphBenefit.java 4096 6 50 both
```
