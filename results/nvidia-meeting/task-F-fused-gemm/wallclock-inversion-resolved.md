# Resolving the fused/unfused wall-clock inversion

Iterations 2–3 recorded an unexplained result: at m=n=k=1024 TornadoVM's **fused**
path was *slower* end-to-end (342 µs) than the **unfused** path (313 µs), while
its GPU time was 11.8% *lower*. Hand-written CUDA showed the expected ordering.

**Resolved: it is an execution-ordering artefact of the demo harness, not a
property of fusion.** No fusion conclusion should be drawn from demo 12's
wall-clock output.

## Step 1 — is it real, or noise?

Three further runs at m=n=k=1024, same binary, same host:

| run | fused | unfused | ratio |
|---|---|---|---|
| 1 | 317 µs | 295 µs | 0.93 |
| 2 | 334 µs | 313 µs | 0.94 |
| 3 | 311 µs | 307 µs | 0.99 |

Consistently in the same direction, but the spread within each column (311–334,
295–313) is ~7%, so the effect is small and close to run-to-run noise.

## Step 2 — where does the time go?

`CutlassFusedEpilogue` builds **two `TornadoExecutionPlan`s in one JVM and runs
the fused one first**. Each excludes its own first execution, but that policy
does not remove *process-level* start-up.

Splitting the `nsys` trace at the first `biasRelu` kernel (the boundary between
the two phases):

| | fused phase (first) | unfused phase (second) |
|---|---|---|
| GPU timeline span | **49.07 ms** | **6.22 ms** |
| `cuCtxCreate_v2` | **1 call, 110.02 ms** | 0 |
| `cuStreamSynchronize` | **1,596 calls, 4.82 ms** | 60 calls, 2.36 ms |
| `cuMemcpyDtoHAsync_v2` | **788 calls, 3.99 ms** | 0 |
| `cuMemcpyHtoDAsync_v2` | **796 calls, 1.00 ms** | 38 calls, 0.07 ms |
| `cuMemHostRegister_v2` | **12 calls, 1.38 ms** | 0 |
| `cuMemAlloc_v2` | **17 calls, 0.66 ms** | 0 |
| `cuEventCreate` | **2,117 calls** | 97 calls |

**The fused phase absorbs essentially the entire process start-up** — context
creation, host-memory registration, device allocation, and the 512-each-way
256-byte start-up transfers characterised in
`results/raw/25-host-dispatch-breakdown/`. The unfused phase inherits a fully
warmed runtime and issues only 38 launches and 38 copies.

Whichever plan runs first pays for the process. Fusion is not what is being
measured.

## Consequence

- **Demo 12's wall-clock numbers cannot be used to compare fused against
  unfused.** Its kernel-time numbers can, and they show fusion saving 6.2–11.8%
  of GPU time depending on shape.
- **Discarding the first execution per plan is not sufficient** when several
  plans run sequentially in one JVM. Isolating them needs one plan per process,
  or a process-level warm-up before the first measured plan.
- This does not affect any other result in this bundle. Tasks B, B2, E and G use
  either counters or execution-count differencing, both of which cancel
  process-level start-up by construction.

Raw: `../task-F-fused-gemm/`, trace `tvm_1024.nsys-rep` in the scratch capture
set; the phase split is reproducible with the query recorded here.
