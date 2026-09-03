# Batch 25 — where demo 14's ~100 µs of host-side dispatch actually goes

Captured 2026-09-03. RTX 4090 (sm_89), driver 565.57.01, CUDA 12.6.85,
JDK 25.0.2, TornadoVM 6.0.0-jdk22plus-cuda, Nsight Systems 2024.5.1.
`demos/14-warp-async-shared`, `4096 1024 20` — 20 executions per kernel,
40 kernel launches total.

Demo 14's README says the optimised kernel is 26.6x faster on the GPU while
wall-clock improves only 2.17x, because "~100 µs per execution goes elsewhere".
That number was never broken down. This batch breaks it down, by tracing the
CUDA driver API rather than the kernels.

## Method

`nsys profile --trace=cuda` on both implementations, then
`nsys stats --report cuda_api_sum` (host-side API call time) alongside
`cuda_gpu_kern_sum` and the memcpy reports. Transfer-size histograms come from
the exported SQLite (`transfer-size-histogram.txt`).

Invoke `java` directly, not the `tornado` launcher — under `nsys` the launcher
resolves a different JDK and fails with `UnsupportedClassVersionError`:

```bash
nsys profile --trace=cuda -o warp14 \
  $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . \
  WarpAsyncSharedReduce 4096 1024 20
```

## Result 1 — the kernels are not the problem

| kernel | TornadoVM | CUDA | |
|---|---|---|---|
| `rowSumNaive` | 105,643 ns | 105,865 ns | equivalent |
| `rowSumOptimised` | **3,941 ns** | 4,251 ns | TornadoVM **1.08x faster** |

In steady state over 20 executions, TornadoVM's optimised kernel is *marginally
faster* than the hand-written CUDA one, despite carrying the #1065 sector
penalty (batch 24). Under Nsight Compute's cold single-launch conditions the
ordering reverses (7,360 vs 7,040 ns). Both gaps are small; the honest reading
is that the two kernels are equivalent and the sector penalty is not the
dominant term for this kernel.

**So the entire wall-clock gap is on the host side.**

## Result 2 — 1,620 memory-transfer calls versus 41

Same 40 kernel launches, same work:

| | TornadoVM | hand-written CUDA |
|---|---|---|
| H2D transfer calls | 812 | **1** |
| D2H transfer calls | 808 | **40** |
| total transfer calls | **1,620** (40.5 per execution) | **41** (1.03 per execution) |
| `cuStreamSynchronize` | 1,656 (41.4 per execution) | 0 |
| event create/record/destroy | 6,392 (159.8 per execution) | 0 |
| kernel launches | 40 | 40 |

**39.5x more transfer calls for the same work.**

## Result 3 — 95% of those transfers are 256 bytes

Transfer-size histogram, TornadoVM:

```
HtoD        256 bytes  x 768        <- 
HtoD         24 bytes  x 40
HtoD    4194320 bytes  x 2          <- the actual int8 input
HtoD      16400 bytes  x 2
DtoH        256 bytes  x 768        <- 
DtoH      16400 bytes  x 40         <- the actual output, 1 per execution
```

and hand-written CUDA, for the identical workload:

```
HtoD    4194304 bytes  x 1
DtoH      16384 bytes  x 40
```

**1,536 of TornadoVM's 1,620 transfers (94.8%) are exactly 256 bytes** — 768 in
each direction, roughly 19 round-trips of 256 bytes per execution, for a task
graph with one kernel and one real output. The payload they carry is ~4.9 KB per
execution against the 16,400-byte real output.

## Result 4 — that is precisely the missing ~100 µs

`cuMemcpyDtoHAsync_v2`, host-side API time:

```
Total 3,999,593 ns over 808 calls, avg 4,950 ns
3,999,593 ns / 40 executions = 99,990 ns = 100.0 us per execution
```

**100.0 µs per execution, against the ~100 µs demo 14's README attributes to
dispatch.** The match is exact.

And it is per-call overhead, not bandwidth — the same copies on the device side:

| | host-side API | device-side |
|---|---|---|
| D2H, average per call | **4,950 ns** | **872.7 ns** |

Each async D2H costs ~5 µs of host time to *issue* and under 1 µs to actually
perform. Issuing ~20 of them per execution is the cost.

Two supporting terms, on top of the above (these overlap in wall-clock and must
not be summed with it): `cuStreamSynchronize` 1,656 calls / 3.95 ms, and 6,392
event create/record/destroy calls / 1.38 ms — about 160 event operations per
execution.

`cuLaunchKernel` is *not* the problem: 40 calls, 156,173 ns total, median
2,362 ns.

## What this means

The gap between demo 14's 26.6x kernel speedup and its 2.17x wall-clock is
**not** kernel quality, and not one large data transfer. It is roughly 40 small
CUDA API round-trips per execution — ~19 of them 256-byte device-to-host copies
that appear to be internal control or status traffic — each costing ~5 µs of
host time to issue.

This is also why `withCUDAGraph()` is worth 8–10x on TornadoVM and only 1.28x on
raw CUDA: graph capture removes exactly this per-execution API traffic.

**Candidate for a fourth upstream issue**, distinct from #1063/#1064/#1065: the
256-byte control transfers scale with executions rather than with data, so they
are pure per-dispatch overhead. Not yet filed — it needs a minimal reproducer
and a read of what those buffers carry before it is worth reporting.

## Files

| File | Contents |
|---|---|
| `warp14.nsys-rep`, `cuda14.nsys-rep` | raw Nsight Systems traces |
| `warp14-cuda_api_sum.csv`, `cuda14-…` | host-side CUDA API call time |
| `*-cuda_gpu_kern_sum.csv` | kernel times |
| `*-cuda_gpu_mem_time_sum.csv`, `*-mem_size_sum.csv` | transfer time and volume |
| `transfer-size-histogram.txt` | per-size transfer counts, both implementations |
