# 11 — Integrated CUDA showcase: everything at once

**Concept (read in ~1 minute):** demos `04`–`08` each verified one Hybrid API
mechanism in isolation. This demo puts them all in one program, on the same
GPU, in one JVM run:

1. **JIT Java kernel** — `scale`/`bias`, plain `@Parallel` loops compiled by
   TornadoVM's Graal backend to CUDA (every demo).
2. **Hybrid API vendor-library call** — `CuBlas.cublasSgemv` inside the same
   `TaskGraph` as the JIT tasks, no host round-trip between stages (demo `04`'s
   shape).
3. **Streams/async** — the pipeline above is replicated into **6 independent
   chains in one `TaskGraph`**, run with
   `TornadoExecutionPlan#withIntraPlanConcurrency()` so the independent chains
   can be routed to separate CUDA streams (demo `06`'s mechanism, applied to a
   cuBLAS-bearing graph instead of pure-JIT units).
4. **Graph replay** — the same 6-chain graph captured once with
   `TornadoExecutionPlan#withCUDAGraph()` and replayed on every later
   `execute()` (demo `07`'s mechanism). Combining a cuBLAS library task with
   JIT tasks inside one CUDA-graph capture is itself an **upstream-verified
   pattern**, not something invented for this demo — see
   `vendor/tornadovm/tornado-cublas/.../tests/TestCuBlasSgemvWithTasksCudaGraph.java`.
5. **Combined (experimental)** — `withCUDAGraph()` **and**
   `withIntraPlanConcurrency()` stacked on the *same* plan. This combination
   does not appear anywhere in the pinned upstream tree (checked
   `tornado-unittests/` and `tornado-cublas/.../tests/` before writing this
   demo — no test exercises both together). This demo probes it live rather
   than assuming it works or is unsupported.
6. **Tensor Core / MMA** — demo `08`'s single-warp `M16N8K16` `mma.sync` tile
   kernel, reused verbatim as a final "capability inventory" stage. Kept as
   its own `TaskGraph`/`TornadoExecutionPlan` rather than forced into the
   sgemv pipeline above: its `WorkerGrid` (one warp) and datatypes (fp16 in /
   f32 out) are unrelated to the sgemv chain's shapes, and merging them would
   be an artificial combination, not a real one.

Source: [`IntegratedShowcase.java`](IntegratedShowcase.java).

Every mode validates its output against a closed-form CPU reference, every
execution, before any wall-clock number is printed or reported — same rigor
as demos `04`/`06`/`07`/`08`.

## Build

```bash
source scripts/setup-env.sh   # from repo root; pins the SDK in env/versions.env
cd demos/11-integrated-showcase
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  -d . IntegratedShowcase.java
```

## Run

Canonical (all four sgemv-chain modes + the Tensor Core bonus, one JVM):

```bash
tornado --classpath . IntegratedShowcase 6 8 8 20 all
```

Reproducibility form (`java @arg-file`):

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp . IntegratedShowcase 6 8 8 20 all
```

Arguments: `<units> <m> <n> <executions> <mode>` (defaults `6 8 8 20 all`).
`mode` is `baseline`, `concurrent`, `graph`, `combined`, `mma`, or `all`. Use
a single mode to get a clean, single-mechanism Nsight Systems trace.

## Expected output

```
=== BASELINE (single stream, no graph) ===
execution 0: correct, wall=<large, JIT compile of 18 tasks> us
execution 1: correct, wall=1170 us
...
BASELINE (single stream, no graph) steady-state median wall-clock (n=19): 831 us

=== CONCURRENT (withIntraPlanConcurrency) ===
...
CONCURRENT (withIntraPlanConcurrency) steady-state median wall-clock (n=19): 740 us

=== GRAPH (withCUDAGraph) ===
...
GRAPH (withCUDAGraph) steady-state median wall-clock (n=19): 148 us

=== COMBINED-EXPERIMENTAL (withCUDAGraph + withIntraPlanConcurrency) ===
...
COMBINED-EXPERIMENTAL (withCUDAGraph + withIntraPlanConcurrency) steady-state median wall-clock (n=19): 146 us

=== Summary (steady-state median us, this run/this GPU) ===
baseline   : 831.0
concurrent : 740.0 (1.12x vs baseline)
graph      : 148.0 (5.61x vs baseline)
combined   : 146.0 (5.69x vs baseline)

=== BONUS: Tensor Core mma.sync single-tile GEMM (demo 08 kernel, reused) ===
Tensor Core tile validation: PASSED (max abs err 0.00000, 0/128 cells out of tol)
```

## What was actually measured (Observed)

Pinned SDK: TornadoVM `6.0.0-jdk22plus-cuda` (SDKMAN), JDK 25.0.2,
RTX 4090, driver `565.57.01`, `nvcc`/`ptxas` 12.6.85. Five independent runs,
logs in `results/raw/18-tornadovm-6-migration/` (`11-showcase-tornado.log`,
`11-showcase-javaargfile.log`, `11-showcase-repeat3..5.log`). Numbers from the
earlier 5.2.1 source-built pin are kept for comparison and labelled as such;
their logs are unmodified in `results/raw/11-integrated-showcase/`.

- Ran both supported ways on 6.0.0 — `tornado --classpath .` and
  `java @$TORNADOVM_HOME/tornado-argfile` — all correct in every mode, every
  run. The 5.2.1 pass additionally ran `--enableProfiler console` and
  confirmed `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"` for
  every task (`scale`, `sgemv`, `bias`) across all 6 units
  (`results/raw/11-integrated-showcase/showcase-profiler-console.log`).
- **The experimental combined mode (`withCUDAGraph()` +
  `withIntraPlanConcurrency()` on the same plan) works and validates
  correctly**, across 3 independent runs (`showcase-run.log`,
  `showcase-run-fullverbose.log`, `showcase-nsys-run.log`) — this is new,
  repo-original evidence, not copied from an upstream test (none combines
  both). Classified **Observed**, not just source-backed.
- **First-execution cost tracks each mode's setup work**, consistent with
  demos `06`/`07`'s findings: ~190ms baseline / concurrent (JIT compile of 18
  tasks: 6×scale/sgemv/bias), ~28ms graph (capture, no JIT needed —
  the graph modes still pay stream/graph-object setup but skip Java-side
  per-unit dispatch bookkeeping the first-time JIT compile dominates in
  baseline/concurrent).
- **Steady-state numbers on 6.0.0** (n=19, excl. first execution, 6 units ×
  8×8 sgemv each), across five runs: baseline 811–851 µs, concurrent
  1.08–1.13×, graph 5.37–5.87× (141–153 µs), combined 5.55–5.71×
  (144–149 µs). **Re-run to run; not gated.**
- **Changed vs. 5.2.1 — `combined` mode caught up with `graph`.** On the
  5.2.1 pin, stacking `withIntraPlanConcurrency()` on top of `withCUDAGraph()`
  cost more than `graph` alone in every run (3.91× vs 6.56×; 2.95× vs 4.90×).
  On 6.0.0 the two are indistinguishable within run-to-run noise (combined
  5.55–5.71× vs graph 5.37–5.87×) — the extra stream-routing bookkeeping no
  longer shows up as a penalty on an already-replayed graph. Observed, five
  runs; the mechanism behind the change was not investigated.
- **Honest, presenter-relevant caveat found by measuring, not assumed**:
  `concurrent` mode is *not reliably faster* than `baseline` for this
  workload. On 5.2.1 one run showed a 1.21× speedup and another a 0.81×
  slowdown; on 6.0.0 all five runs landed in a narrow 1.08–1.13× band — better
  than break-even every time, but still a rounding error next to `graph` mode.
  Unlike demo `06`'s heavier small-grid/heavy-inner-loop kernels
  (chosen specifically so a single unit does not saturate the SMs, per
  `TestStreamsPerformance`'s documented condition for genuine overlap), this
  demo's per-unit work (an 8×8 `scale` + `sgemv` + `bias`) is tiny and
  launch-overhead-bound, not compute-bound — there is little idle SM time
  for stream overlap to fill, and the extra stream-routing bookkeeping can
  cost more than it saves at this size. `graph` mode's speedup, by contrast,
  is large and consistent across every run (5.4×–5.9× on 6.0.0, 4.9×–6.6× on
  5.2.1) because it removes
  the dominant cost directly (per-execution host-side CUDA-call dispatch),
  matching demo `07`'s finding. `combined` mode is consistently *between*
  `graph` alone and `baseline` — i.e., stacking `withIntraPlanConcurrency()`
  on top of an already-replayed graph did not help further at this size and
  cost more than `graph` alone on 5.2.1; on 6.0.0 that penalty is gone and
  `combined` simply matches `graph`. Either way the acceptance criterion here
  is "does it work," not "is it optimal." A future task with a
  larger/heavier per-unit workload (demo `06`'s sizing) could test whether
  `combined` mode's concurrency benefit becomes a *gain* once units are large
  enough to saturate a single stream.
- **Tensor Core bonus stage**: validates exactly (max abs err `0.00000`),
  reusing demo `08`'s kernel unmodified — see demo `08`'s own README for the
  `--printKernel` generated-`mma.sync`-asm evidence; not re-captured here to
  avoid duplicating that evidence.
- **A real bug was found and fixed while building this demo, not left in the
  committed source**: the steady-state median was first computed directly
  from the raw-nanosecond sorted array without dividing by 1000, so it was
  printed labelled "us" but was actually still in nanoseconds (1000× too
  large — e.g. an apparent "712349 us" baseline median that did not match any
  of the individual per-execution numbers printed above it, all in the
  500–1300 µs range). Caught by comparing the summary line against the full
  per-execution log rather than trusting the first output, fixed
  (`steadyState[...] / 1000`), and re-verified against a full non-truncated
  per-execution printout before any number above was recorded.

## Nsight Systems evidence

```bash
source scripts/setup-env.sh   # from repo root; pins the SDK in env/versions.env
nsys profile --trace=cuda,nvtx,osrt -o showcase-nsys \
  tornado --classpath . IntegratedShowcase 6 8 8 20 all
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  --format csv --output . showcase-nsys.nsys-rep
```

Captured: `results/raw/11-integrated-showcase/showcase-nsys.nsys-rep` +
`showcase-nsys_cuda_gpu_kern_sum.csv`, `showcase-nsys_cuda_api_sum.csv`,
`showcase-nsys_cuda_gpu_mem_time_sum.csv`, `showcase-nsys-run.log`. Findings:

- `cuda_gpu_kern_sum` shows real GPU kernels for every stage: cuBLAS's
  `gemvx::kernel<...>` (38.4% of GPU kernel time, 240 instances), `bias`
  (31.3%, 240 instances), `scale` (30.0%, 240 instances), and
  `gemmMMASingleTile` (0.3%, 1 instance) — the Tensor Core kernel is
  confirmed present in the same trace as the sgemv pipeline, not asserted
  separately.
- `cuda_api_sum`: `cuGraphLaunch` appears 40 times (the `graph` and
  `combined` modes' 20 executions each, first execution is capture not
  replay so 19+19=38 replays + this count includes some capture-adjacent
  calls; not decomposed further — the presence of `cuGraphLaunch` itself is
  the relevant evidence that graph replay is genuinely exercised, matching
  demo `02`/`07`'s CUDA-graph API evidence). `cuCtxCreate_v2` again dominates
  raw `Time (%)` (one-time context setup, same reading caveat as every prior
  demo's Nsight Systems section — see `docs/profiling-quickstart.md`).
- `cuda_gpu_mem_time_sum`: H2D (64.6%, 988 copies) vs. D2H (35.4%, 241
  copies) — expected, since each chain writes back only the small `output`
  vector but reads both `matrix` and `vector` every execution.

Nsight Compute hardware-counter evidence: not attempted for this demo.
Same system-wide `ERR_NVGPUCTRPERM` block documented in
`results/failures/08-nsight-compute-permission.md` and re-verified in tasks
`08`/`12` applies here too (same driver, same permission model); the
generated-code (`--printKernel`, demo `08`) and Nsight Systems evidence above
already satisfy this task's "profiler evidence" acceptance criterion.

## Fallback if the live demo fails

- If any execution prints `WRONG`, fall back to the captured logs in
  `results/raw/11-integrated-showcase/` (`showcase-run.log`,
  `showcase-run-fullverbose.log`) and walk through the per-mode summary
  instead of re-running live.
- Re-run `tornado --devices` first — if it does not show exactly one CUDA
  device, the environment, not the demo, is broken.
- If the `combined` mode specifically throws (it did not on this machine,
  but it is explicitly experimental — see above), that is not a failure of
  the rest of the demo: run a single mode directly, e.g.
  `tornado --classpath . IntegratedShowcase 6 8 8 20 baseline`, then
  `... graph`, to present the other three mechanisms independently, and use
  the captured log's `COMBINED-EXPERIMENTAL` section to show the result was
  already captured rather than re-attempting live.
- Recovery order for a live conference demo, cheapest/most-reliable first:
  `graph` mode (smallest, fastest, most consistent speedup) → `baseline` →
  `mma` bonus (self-contained, does not depend on cuBLAS) → `concurrent`/
  `combined` (present as "here's what we measured," not as a guaranteed live
  speedup, given the run-to-run variance documented above).
- If `nsys` is not on `PATH` (found directly at `/usr/local/cuda-12.6/bin/nsys`
  on this machine, without sourcing `scripts/setup-env.sh`), skip the live capture and
  open the pre-recorded `.nsys-rep`/CSV files from
  `results/raw/11-integrated-showcase/` instead.

## CUDA equivalent

[`IntegratedShowcase.cu`](IntegratedShowcase.cu) is the same demo written directly in CUDA C++, for side-by-side comparison.

```bash
nvcc -arch=sm_89 -lcublas -o integrated_showcase IntegratedShowcase.cu && ./integrated_showcase 6 8 8 20 all
```

All four modes by hand. In Java they differ by one method call on the same
plan; here each is different host code — a stream pool to build and tear down,
a capture/instantiate/launch cycle, pinned buffers so captured copies stay
valid, and `cublasSetStream` calls to keep cuBLAS bound to the right stream.

The `combined` mode is the fiddly one: capturing work issued across a stream
pool needs an explicit fork (record an event on the capture stream, have every
pool stream wait on it) and a matching join before `cudaStreamEndCapture`.
`withCUDAGraph().withIntraPlanConcurrency()` is two method calls.

**Measured, same machine, 6 chains, 20 executions:**

| Mode | TornadoVM | vs its baseline | CUDA | vs its baseline |
|---|---|---|---|---|
| baseline | 831 µs | — | 99.9 µs | — |
| concurrent | 740 µs | 1.12x | 48.4 µs | **2.06x** |
| graph | 148 µs | **5.61x** | 107.9 µs | 0.93x |
| combined | 146 µs | 5.69x | 36.7 µs | 2.73x |

The two profiles are almost mirror images, and for a consistent reason: graphs
help whoever has host overhead to remove (TornadoVM), streams help whoever is
already close to the hardware (CUDA). On this small workload, graph capture
actually costs raw CUDA slightly more than it saves (0.93x).

`bash scripts/run-all-cuda.sh` builds and runs the CUDA equivalent of every demo (needs only the CUDA toolkit, no JDK).

## JBang

Not verified: `jbang` is not installed on this machine (`which jbang` → exit
1, checked 2026-08-22, same finding as every prior demo). Do not run it live
until tested on the pinned environment:

```bash
jbang -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  --java-opts="@$TORNADOVM_HOME/tornado-argfile" \
  IntegratedShowcase.java
```
