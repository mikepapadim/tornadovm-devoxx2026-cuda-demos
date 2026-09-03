# Task E — runtime and graph composition

Closes the two gaps left open in iteration 1: CUDA Graph per-execution cost by
differencing, and whether device buffers are reused across graph nodes.

sm_89 / RTX 4090, driver 565.57.01, CUDA 12.6.85, JDK 25.0.2,
TornadoVM 6.0.0-jdk22plus-cuda (SDKMAN release). Nsight Systems 2024.5.1.

## E1 — `JIT → libraryTask → JIT` buffer reuse (demo 13)

Graph: `scale` (JIT) → `cudnnConv2d` (libraryTask) → `addBias` (JIT) →
`cudnnRelu` (libraryTask). `n=4 c=16 h=32 w=32 k=16`, 20 executions.
Validation **PASSED**, max abs err 0.000000, 0/65536 elements out of tolerance.

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none -o d13 \
  $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . \
  CuDnnConvBlockHybrid 4 16 32 32 16 20
```

Per-kernel GPU time (`d13-kernsum.csv`), 20 instances each:

| Kernel | Avg (ns) |
|---|---|
| cuDNN conv (implicit GEMM) | 6,399.9 |
| cuDNN relu | 1,516.8 |
| `scale` (JIT) | 1,259.2 |
| `addBias` (JIT) | 1,229.0 |

Transfer histogram (`d13-transfer-histogram.csv`):

| Direction | Bytes | Count |
|---|---|---|
| HtoD | 262,160 | **1** |
| HtoD | 9,232 | 1 |
| HtoD | 80 | 1 |
| HtoD | 24 | 40 |
| HtoD | 256 | 512 |
| DtoH | 262,160 | **20** |
| DtoH | 256 | 512 |

**Device buffers are reused across the JIT ↔ libraryTask boundary. There is no
unintended host round trip.** The 262,160-byte input crosses once
(`FIRST_EXECUTION`); the output crosses once per execution
(`EVERY_EXECUTION`). The intermediate tensors between the four nodes — `scaled`,
`conv`, `biased` — never appear in the histogram, i.e. they stay resident.

Two secondary observations:

- The 24-byte transfers number **40 = 2 JIT tasks × 20 executions**: the
  kernel-argument stack frame re-uploaded per launch
  ([#1028](https://github.com/beehive-lab/TornadoVM/issues/1028) finding 1). The
  two `libraryTask` nodes do not incur one.
- The 256-byte transfers (512 each way) are the fixed start-up traffic
  characterised in `results/raw/25-host-dispatch-breakdown/`; they are constant
  and complete before the first kernel.

## E2 — CUDA Graph replay, per-execution cost by differencing (demo 07)

Six-stage chain, `size=4096 stages=6`. Profiled at 10 and 100 executions in each
mode; deltas divided by 90 so fixed start-up cancels.

```bash
for m in nograph graph; do for e in 10 100; do
  nsys profile --trace=cuda --sample=none --cpuctxsw=none -o d07_${m}_$e \
    $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . \
    CudaGraphBenefit 4096 6 $e $m
done; done
```

| CUDA API | nograph calls/exec | nograph ns/exec | graph calls/exec | graph ns/exec |
|---|---|---|---|---|
| `cuLaunchKernel` | **6.00** | 11,361 | **0.00** | 51 |
| `cuMemcpyHtoDAsync_v2` | **7.00** | 10,600 | **0.00** | −121 |
| `cuEventCreate` | 14.00 | 3,995 | 1.00 | 300 |
| `cuEventRecord` | 14.00 | 2,438 | 1.00 | 44 |
| `cuEventDestroy_v2` | 14.00 | 1,635 | 1.00 | 227 |
| `cuCtxSetCurrent` | 14.00 | 1,004 | 1.00 | 85 |
| `cuMemcpyDtoHAsync_v2` | 1.00 | 1,001 | 0.00 | −102 |
| `cuStreamIsCapturing` | 1.00 | 82 | 0.00 | — |
| **`cuGraphLaunch`** | — | — | **1.00** | **5,108** |
| `cuStreamSynchronize` | 3.00 | 3,669 | 3.00 | 18,388 |

**Host-side dispatch cost, excluding `cuStreamSynchronize`** (which is largely
genuine device wait, not overhead):

| | nograph | graph | |
|---|---|---|---|
| host API ns per execution | **33,774** | **6,283** | **5.4x lower** |

**Graph capture removes the entire per-execution dispatch sequence.** Six
`cuLaunchKernel` and seven `cuMemcpyHtoDAsync` per execution become one
`cuGraphLaunch` and zero copies; 42 event calls become 3. The seven H2D copies
in `nograph` are the per-launch 24-byte kernel-argument stack frames — the same
#1028 finding 1 — and graph replay eliminates them because the arguments are
baked into the captured graph.

This is the mechanism behind the wall-clock graph speedup reported in earlier
batches for TornadoVM (8–10x) versus raw CUDA (1.28x): TornadoVM has a large
per-dispatch cost to remove, hand-written CUDA has very little.

> `cuStreamSynchronize` totals more in graph mode (18,388 vs 3,669 ns/execution).
> It is not overhead: with dispatch batched into one replay, the host blocks
> once on the whole chain instead of interleaving with launch work. Treating it
> as a regression would be a misreading, which is why it is excluded from the
> dispatch comparison above and reported separately.

## Files

| File | Contents |
|---|---|
| `d13.nsys-rep`, `d13-run.log` | `JIT → libraryTask → JIT` trace and validation |
| `d13-kernsum.csv` | per-kernel GPU time, 4 nodes × 20 executions |
| `d13-transfer-histogram.csv` | transfer sizes and counts — the buffer-reuse evidence |
| `api_{nograph,graph}_{10,100}.csv` | CUDA API sums used for the differencing |
| `d07_{nograph,graph}_100.nsys-rep` | raw traces at 100 executions |
