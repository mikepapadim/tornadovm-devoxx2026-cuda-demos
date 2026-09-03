# Batch 25 — demo 14's host-side dispatch cost, itemised

Captured 2026-09-03. RTX 4090 (sm_89), driver 565.57.01, CUDA 12.6.85,
JDK 25.0.2, TornadoVM 6.0.0-jdk22plus-cuda, Nsight Systems 2024.5.1.

> **Correction.** The first version of this manifest claimed the ~100 µs of
> per-execution dispatch cost was ~19 256-byte device-to-host copies per
> execution. **That was wrong.** It divided a *fixed startup* cost by the
> execution count. The 256-byte transfers are constant at 512 each way and
> complete before the first kernel launches. The corrected analysis is below;
> the real per-execution CUDA API cost is **~8.3 µs**, not 100 µs.

## Method — difference two execution counts

Counting calls in a single trace cannot separate startup from per-execution
cost. Profiling the *same* workload at two execution counts and differencing
can: anything constant cancels, and the slope is the true per-execution cost.

`DispatchOverhead.java` (in this directory) is the minimal reproducer — one
task graph, one trivial kernel, one input, one output, executed N times in one
`TornadoExecutionPlan`:

```bash
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . DispatchOverhead.java
for e in 10 40 100; do
  nsys profile --trace=cuda --sample=none --cpuctxsw=none -o e$e \
    $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . DispatchOverhead 1024 $e
  nsys stats --report cuda_api_sum --format csv e$e.nsys-rep
done
```

Use `$JAVA_HOME/bin/java` directly — under `nsys` the `tornado` launcher
resolves a different JDK and dies with `UnsupportedClassVersionError`.

## Result 1 — the 256-byte transfers are startup, not per-execution

| | n=1024, 10 exec | n=1024, 40 exec | n=1048576, 10 exec |
|---|---|---|---|
| 256-byte HtoD | **512** | **512** | **512** |
| 256-byte DtoH | **512** | **512** | **512** |
| 24-byte HtoD | 10 | 40 | 10 |
| payload DtoH | 10 | 40 | 10 |

**Constant at 512 each way — they scale with neither data size nor execution
count.** They are also unaffected by `-Dtornado.max.events`
(32768 / 1024 / 256 all give 512; `repro-transfer-scaling.txt`), which rules out
the wait-list row sizing of issue #1028's open item 3, despite 512 x 256 B being
exactly 128 KB.

Timeline confirms it: in the 100-execution trace, **all 1,024 of them complete
before the first kernel launch**, spread over the first 198 ms of a 262 ms
start-up. They cost ~4.8 ms of host API time, one time, inside a start-up that
issue #1028 measures as dominated by NVRTC compilation and `cuMemHostRegister`.

**Not a per-execution defect, and small against the start-up it sits in.**

## Result 2 — the true per-execution cost, minimal reproducer

Differencing 10 → 100 executions. Call counts are exactly linear:

| CUDA API | calls @10 | @40 | @100 | per execution | ns per execution |
|---|---|---|---|---|---|
| `cuStreamSynchronize` | 1054 | 1144 | 1324 | **3.00** | 2,751 |
| `cuLaunchKernel` | 10 | 40 | 100 | **1.00** | 2,256 |
| `cuMemcpyDtoHAsync_v2` | 522 | 552 | 612 | **1.00** | 1,535 |
| `cuMemcpyHtoDAsync_v2` | 524 | 554 | 614 | **1.00** | 1,232 |
| `cuEventCreate` | 1568 | 1658 | 1838 | **3.00** | 1,127 |
| `cuCtxSetCurrent` | 1070 | 1160 | 1340 | **3.00** | 211 |
| `cuEventRecord` | 1568 | 1658 | 1838 | **3.00** | 179 |
| `cuEventDestroy_v2` | 1568 | 1658 | 1838 | **3.00** | 128 |
| `cuStreamIsCapturing` | 1034 | 1064 | 1124 | **1.00** | 36 |

**~9.5 µs of CUDA API time per execution**, of which the single H2D is the
24-byte kernel argument stack frame — one upload per launch even though it never
changes.

## Result 3 — demo 14, same method

Differencing 5 → 25 executions per kernel (2 task graphs, so 40 extra
executions):

| CUDA API | per execution | ns per execution |
|---|---|---|
| `cuStreamSynchronize` | 3.00 | 54,446 |
| `cuLaunchKernel` | 1.00 | 2,414 |
| `cuMemcpyDtoHAsync_v2` | 1.00 | 2,407 |
| `cuMemcpyHtoDAsync_v2` | 1.00 | 1,654 |
| `cuEventCreate` | 3.00 | 1,212 |
| `cuEventRecord` | 3.00 | 382 |
| `cuCtxSetCurrent` | 3.00 | 249 |
| `cuStreamIsCapturing` | 1.00 | 29 |

The 54,446 ns in `cuStreamSynchronize` is **not overhead** — it is the host
blocking on the kernel. Demo 14's two kernels average
(105,643 + 3,941) / 2 = 54,792 ns, which matches it to within 0.6%.

**Excluding that genuine device wait, per-execution CUDA API overhead is
~8.3 µs**, not the ~100 µs previously claimed here.

## What this does and does not explain

Demo 14's optimised path is ~118 µs of wall-clock per execution against a
3,941 ns kernel. The CUDA driver API accounts for roughly 8–12 µs of the
difference. **The remainder is host-side runtime work that does not appear in a
CUDA-only trace at all**, and attributing it needs JFR rather than `nsys` — the
approach taken upstream in
[TornadoVM#1028](https://github.com/beehive-lab/TornadoVM/issues/1028). No
attribution for it is claimed here.

## Relationship to upstream

Everything measured in Results 2 and 3 as *per-execution* is already diagnosed
in [#1028](https://github.com/beehive-lab/TornadoVM/issues/1028):

- the per-launch 24-byte stack-frame upload is that issue's finding **#1**
- the 3 `cuStreamSynchronize` per execution is its finding **#2**

Both are addressed by PR
[#1022](https://github.com/beehive-lab/TornadoVM/pull/1022), which was **still
open and unmerged** when TornadoVM 6.0.0 was released on 2026-09-02. So this
batch is not a new defect; it is independent confirmation, on a released SDK
rather than on `develop`, that the per-execution dispatch sequence #1028
describes ships in 6.0.0.

## Files

| File | Contents |
|---|---|
| `DispatchOverhead.java` | minimal reproducer |
| `repro-transfer-scaling.txt` | 256-byte counts vs n, executions, `max.events` |
| `repro-api-executions-{10,40,100}.csv` | reproducer API sums |
| `demo14-api-executions-{5,25}.csv` | demo 14 API sums |
| `warp14.nsys-rep`, `cuda14.nsys-rep` | raw traces (20 executions) |
| `warp14-*.csv`, `cuda14-*.csv` | API, kernel and transfer reports |
| `transfer-size-histogram.txt` | per-size transfer counts, both implementations |

## Still true from the original analysis

The comparison against hand-written CUDA stands: for the same 40 launches
TornadoVM makes 1,620 transfer calls against CUDA's 41, and 94.8% of those are
the 256-byte start-up transfers. Hand-written CUDA does 1 H2D and 40 D2H, with
no events and no stream synchronisation. And the kernel measurements are
unaffected — in this steady-state trace TornadoVM's optimised kernel is
**1.08x faster** than the hand-written one (3,941 vs 4,251 ns).
