# 06 — CUDA streams and async overlap

**Concept (read in ~1 minute):** `TornadoExecutionPlan#withIntraPlanConcurrency()`
routes DAG-independent operations within one execution plan onto separate CUDA
streams (H2D / a COMPUTE-pool / D2H "role" streams) instead of TornadoVM's
single default stream, and orders them with device-side events instead of host
sync points. It is documented on `TornadoExecutionPlan`'s own javadoc as
"currently realised on the CUDA backend" — the same CUDA-runtime-behaviour
family as `withCUDAGraph()` (demo `02`) and `withStagedTransfers()`, not a
vendor-library (`.libraryTask`) feature.

This demo builds **8 independent H2D → compute → D2H pipelines in one
`TaskGraph`** — same shape as upstream's own correctness/perf coverage for
this feature:
`vendor/tornadovm/tornado-unittests/.../streams/TestCUDAStreams.java#testManyIndependentUnitsMultiStream`
and
`vendor/tornadovm/tornado-unittests/.../streams/TestStreamsPerformance.java#testSmallKernelConcurrency`
— and runs it two ways from the same JVM:

- **sequential** — no `withIntraPlanConcurrency()`: every unit serialised on
  TornadoVM's one default stream.
- **concurrent** — `withIntraPlanConcurrency()`: units round-robin across the
  CUDA-stream pool and can genuinely co-reside on the GPU.

Each unit's kernel is deliberately **small-grid + heavy-inner-loop** (does not
saturate the SMs by itself) — `TestStreamsPerformance`'s documented condition
for kernels to actually overlap across streams rather than merely being issued
on different streams that still serialise on a full GPU.

Source: [`CudaStreamsOverlap.java`](CudaStreamsOverlap.java).

Every execution, in both modes, is validated against a closed-form CPU
reference per unit. Median wall-clock time per mode is reported (**printed,
not asserted** — overlap speedup is workload/GPU/driver-dependent, matching
how upstream's own `TestStreamsPerformance` treats it: evidence, not a
pass/fail gate).

## Build

```bash
source scripts/setup-env.sh   # from repo root; pins the SDK in env/versions.env
cd demos/06-cuda-streams
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  -d . CudaStreamsOverlap.java
```

## Run

Canonical (both modes back-to-back, one JVM):

```bash
tornado --classpath . CudaStreamsOverlap 8 32768 65536 8 both
```

Reproducibility form (`java @arg-file`):

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp . CudaStreamsOverlap 8 32768 65536 8 both
```

Arguments: `<units> <unitSize> <innerIterations> <executions> <mode>`
(defaults `8 32768 65536 8 both`). `mode` is `sequential`, `concurrent`, or
`both`. Use `sequential`/`concurrent` alone to get a clean, single-mode Nsight
Systems trace (see below).

## Expected output

```
=== SEQUENTIAL (single stream) ===
execution 0: correct, wall=121434 us
execution 1: correct, wall=2292 us
...
SEQUENTIAL (single stream) median wall-clock (excl. first execution): 2169 us
SEQUENTIAL (single stream): All executions correct

=== CONCURRENT (withIntraPlanConcurrency) ===
execution 0: correct, wall=37721 us
execution 1: correct, wall=1289 us
...
CONCURRENT (withIntraPlanConcurrency) median wall-clock (excl. first execution): 916 us
CONCURRENT (withIntraPlanConcurrency): All executions correct
```

Execution 0 in both modes is dominated by JIT compilation (Graal + driver) of
the 8 unit kernels — same first-execution effect documented for demos `02`,
`04`, `05`. Steady-state (executions 1+) is where the streams effect shows:
this machine (RTX 4090) measured ~2.2ms sequential vs. ~0.9ms concurrent
median steady-state wall-clock for this workload — **Observed, this run
only**; re-run to get current numbers, they are not gated.

On TornadoVM 6.0.0 the same workload measured 2174 µs sequential vs. 960 µs
concurrent (`tornado`) and 2176 µs vs. 936 µs (`java @argfile`) — a ~2.3×
concurrency benefit, unchanged in size from the 5.2.1 pin. Captured logs
(6.0.0): `results/raw/18-tornadovm-6-migration/06-streams-tornado.log`,
`06-streams-javaargfile.log`. Earlier 5.2.1 logs, including the Nsight Systems
traces below, are unmodified in `results/raw/06-cuda-streams/`
(`cudastreamsoverlap-run.log`, `cudastreamsoverlap-run-javaargfile.log`).

## Nsight Systems timeline evidence

```bash
source scripts/setup-env.sh   # from repo root; pins the SDK in env/versions.env
nsys profile --trace=cuda -o nsys-sequential \
  tornado --classpath . CudaStreamsOverlap 8 32768 65536 8 sequential
nsys profile --trace=cuda -o nsys-concurrent \
  tornado --classpath . CudaStreamsOverlap 8 32768 65536 8 concurrent
nsys stats --report cuda_gpu_trace --format csv --output . nsys-sequential.nsys-rep
nsys stats --report cuda_gpu_trace --format csv --output . nsys-concurrent.nsys-rep
```

This machine has no GUI available for the Nsight Systems viewer, so the
timeline claim is verified textually from the `cuda_gpu_trace` CSV report
instead of a screenshot — same evidence, read with `awk`/`sort` instead of the
GUI. Captured `.nsys-rep` files, the extracted CSVs, and a plain-text summary
are in `results/raw/06-cuda-streams/` (`nsys-sequential.nsys-rep`,
`nsys-concurrent.nsys-rep`, `*_cuda_gpu_trace.csv`,
`nsys-timeline-evidence.txt`). Findings, this run:

- **Sequential**: all 64 `computeSmall` kernel launches (8 units × 8
  executions) land on **1 CUDA stream** (stream id 14). The last 8 launches
  (steady state) are strictly back-to-back — each kernel's start time equals
  (or is a few hundred ns after) the previous kernel's end time. No overlap.
- **Concurrent**: the 64 kernel launches spread across **4 CUDA streams**
  (ids 16–19) — matching `TestCUDAStreams`'s documented default COMPUTE-pool
  size of 4. In the steady-state round, e.g. stream 17 starts at
  `879597239 ns`, *before* stream 16's kernel (started `879531703 ns`) ends
  at `879747833 ns`; stream 18 starts at `879693016 ns`, before stream 17
  ends. Genuine overlapping execution windows across streams, not just
  distinct stream IDs issued serially.

To open the raw `.nsys-rep` files in the Nsight Systems GUI on a machine that
has one: `nsys-ui results/raw/06-cuda-streams/nsys-concurrent.nsys-rep`.

## CUDA equivalent

[`CudaStreamsOverlap.cu`](CudaStreamsOverlap.cu) is the same demo written directly in CUDA C++, for side-by-side comparison.

```bash
nvcc -arch=sm_89 -o cuda_streams CudaStreamsOverlap.cu && ./cuda_streams 8 32768 65536 8 both
```

`plan.withIntraPlanConcurrency()` becomes: create a pool of four streams,
round-robin the independent units across them, issue every copy as
`cudaMemcpyAsync` on the unit's stream, and synchronise all four at the end.

**Measured on the same machine, same workload:**

| | TornadoVM | CUDA |
|---|---|---|
| sequential (1 stream) | 2174 µs | 1243 µs |
| concurrent (4 streams) | 960 µs | 571 µs |
| **speedup** | **2.26x** | **2.18x** |

The *ratio* is the same — the concurrency win is real and TornadoVM captures
essentially all of it. The absolute difference (~1.7x) is TornadoVM's host-side
dispatch overhead, which is the honest cost of the abstraction here.

`bash scripts/run-all-cuda.sh` builds and runs the CUDA equivalent of every demo (needs only the CUDA toolkit, no JDK).

## JBang

Not verified: `jbang` is not installed on this machine (`which jbang` →
command not found, checked 2026-08-20, same finding as demos `00`–`05`). Do
not run it live until tested on the pinned environment:

```bash
jbang --version
jbang -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  --java-opts="@$TORNADOVM_HOME/tornado-argfile" \
  CudaStreamsOverlap.java
```

## If the demo fails on stage

- If any execution prints `WRONG`, fall back to the captured log at
  `results/raw/06-cuda-streams/cudastreamsoverlap-run.log` and walk through
  the printed per-mode wall-clock numbers and the pre-captured Nsight CSV
  evidence instead of re-running live.
- Re-run `tornado --devices` first — if it does not show exactly one CUDA
  device, the environment, not the demo, is broken.
- If `nsys` is not on `PATH` (it lives at `/usr/local/cuda-12.6/bin/nsys` on
  this machine and was found there directly, without sourcing
  `scripts/setup-env.sh`, when checked 2026-08-20), skip the live capture and open the
  pre-recorded `.nsys-rep`/CSV files from `results/raw/06-cuda-streams/`
  instead.
- Speedup is not guaranteed on a different GPU/driver: if concurrent mode is
  not faster live, fall back to the stream-count/overlap evidence in the CSVs
  (the mechanism, not the wall-clock number, is the reliable part of the
  story).
