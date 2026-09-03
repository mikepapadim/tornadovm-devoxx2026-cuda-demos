# 16 — Every tensor-core operand type, from Java

**Concept (read in ~1 minute):** demo 08 proves TornadoVM emits one fp16
`mma.sync` from Java. That is the easy one. This demo runs the **same GEMM**
through the four *other* operand types the CUDA backend can emit, and validates
each against a CPU reference:

| Type | Shape | PTX instruction the compiler emits | Tensor pipe |
|---|---|---|---|
| **BF16** | `M16N8K16` | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | HMMA |
| **int8** | `M16N8K32` | `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` | **IMMA** |
| **FP8 e4m3** | `M16N8K32` | `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` | HMMA |
| **FP8 e5m2** | `M16N8K32` | `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32` | HMMA |

Together with demo 08's fp16, that is **every operand combination the 6.0.0 CUDA
backend can generate** — the emitter carries exactly five, and this demo plus
demo 08 exercise all five on real hardware.

Source: [`TensorCoreDataTypes.java`](TensorCoreDataTypes.java).

## Build

```bash
source scripts/setup-env.sh   # from the repo root
cd demos/16-tensor-core-datatypes
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . TensorCoreDataTypes.java
```

> `source scripts/setup-env.sh` must be run **from the repo root**. Run it from
> anywhere else and it silently leaves `JAVA_HOME` pointing at SDKMAN's
> `current` (JDK 21 here), and every command below fails with
> `UnsupportedClassVersionError`.

## Run

```bash
tornado --classpath . TensorCoreDataTypes
java @$TORNADOVM_HOME/tornado-argfile -cp . TensorCoreDataTypes
```

No arguments. The problem size is fixed at `C[16x16] = A[16x32] * B[32x16]`,
one warp — deliberately the smallest size that exercises both `k16` and `k32`
shapes.

### Expected output

```
Tensor-core operand types from Java: C[16x16] = A[16x32] * B[32x16], one warp
  each result validated against a CPU reference over the same stored values

  BF16     m16n8k16  f32.bf16.bf16.f32  PASSED (max abs err 0.00000, 0/256 cells out of tol)
  int8     m16n8k32  s32.s8.s8.s32      PASSED (max abs err 0.00000, 0/256 cells out of tol)
  FP8 e4m3 m16n8k32  f32.e4m3.e4m3.f32  PASSED (max abs err 0.00000, 0/256 cells out of tol)
  FP8 e5m2 m16n8k32  f32.e5m2.e5m2.f32  PASSED (max abs err 0.00000, 0/256 cells out of tol)

All four operand types produced correct results
```

`max abs err 0.00000` is not luck. Every input is a small multiple of 0.5 in
`[-2, 2]` (small integers for int8), which is exactly representable in bf16,
e4m3 **and** e5m2 alike, and the products and sums stay inside f32's exact
range. The CPU reference is computed from the **same stored values** — each
input is round-tripped through its storage format first — so the comparison
measures the kernel, not the quantisation. Any mismatch here is a real
mismatch.

## The two critical examples

### 1. `mmaBF16` — the compute call is the only thing that changes

BF16 shares fp16's tile shape *and* fragment layout, so `mmaLoadA`, `mmaLoadB`
and `mmaStore` are bit-type-agnostic. Only the compute call interprets the
bits:

```java
HalfFloat[] fragA  = ctx.mmaLoadA(aTile, K16);          // same as fp16
HalfFloat[] fragB0 = ctx.mmaLoadB(bTile0, K16);         // same as fp16
fragC0 = ctx.mmaBF16(fragA, fragB0, fragC0, MMAShape.M16N8K16);   // <-- only difference
```

Substituting `ctx.mma(...)` there gives you demo 08's fp16 kernel, unchanged
otherwise. **The numeric format is selected by the instruction, not by the data
layout.**

### 2. FP8 e4m3 vs e5m2 — identical staging, different instruction

The two FP8 kernels in this file are byte-for-byte identical except for one
call. Both stage the same packed bytes through `stageFP8(...)`:

```java
byte[] fragA  = ctx.mmaLoadAFP8(aTile, K32);
byte[] fragB0 = ctx.mmaLoadBFP8(bTile0, K32);

fragC0 = ctx.mmaFP8E4M3(fragA, fragB0, fragC0, MMAShape.M16N8K32);   // e4m3
fragC0 = ctx.mmaFP8E5M2(fragA, fragB0, fragC0, MMAShape.M16N8K32);   // e5m2
```

Same bytes in shared memory, same fragment loads, same store — and two
different tensor-core instructions out. That is the clearest demonstration in
this repo that the Java API is selecting a specific PTX instruction rather than
describing a computation and hoping.

### Packing, which is the part that is easy to get wrong

Both `k32` paths stage **four int8/fp8 values per int32 word**, and `B` is
staged as two 8-column panels because `n8` means each `mma` produces 8 columns:

```java
int packed = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
```

Get the lane order wrong and it still runs and still produces plausible
numbers — it is a silent-wrong-answer bug, which is why this demo validates
every cell rather than spot-checking.

## Generated-code evidence (`--printKernel`)

```bash
tornado --printKernel --classpath . TensorCoreDataTypes
```

All four instructions appear verbatim in the emitted CUDA, as inline PTX:

```
      4 mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
      2 mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
      2 mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32
      2 mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
     10 ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
      5 ldmatrix.sync.aligned.m8n8.x4.shared.b16
```

BF16 shows 4 because `K=32` needs two `k16` steps across two 8-column panels;
the `k32` types need one step, so 2 each.

## Hardware-counter evidence (Nsight Compute)

```bash
/opt/nvidia/nsight-compute/2024.3.2/ncu --csv --target-processes all \
  --metrics sm__inst_executed_pipe_tensor_op_hmma.sum,sm__inst_executed_pipe_tensor_op_imma.sum,\
sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.sum \
  java @$TORNADOVM_HOME/tornado-argfile -cp . TensorCoreDataTypes
```

| Kernel | tensor inst | HMMA | IMMA | tensor-pipe cycles |
|---|---|---|---|---|
| `gemmBF16` | 4 | **4** | 0 | 64 |
| `gemmInt8` | 2 | 0 | **2** | 32 |
| `gemmFP8E4M3` | 2 | **2** | 0 | 32 |
| `gemmFP8E5M2` | 2 | **2** | 0 | 32 |

**The counters match the emitted PTX exactly**, instruction for instruction.
They also show something the source cannot: **int8 dispatches to the IMMA pipe
while BF16 and both FP8 formats go to HMMA.** On Ada, FP8 is counted as a
half-precision tensor operation.

Use the `2024.3.2` install by absolute path, not the `ncu` on `PATH` — see
[`results/failures/08-nsight-compute-permission.md`](../../results/failures/08-nsight-compute-permission.md).

Captured evidence: `results/raw/26-tensor-core-datatypes/`.

## CUDA equivalent

[`TensorCoreDataTypes.cu`](TensorCoreDataTypes.cu) writes the same four
instructions by hand, same problem shape, same test values, same reference.

```bash
nvcc -arch=sm_89 -o tc_datatypes TensorCoreDataTypes.cu && ./tc_datatypes
```

It produces the identical validation table. **This is one of the two demos in
this repo where the Java is *longer* than the CUDA** — 309 lines against 217 —
because the Java version stages through shared memory and `ldmatrix` while the
hand-written version loads each lane's fragment registers directly. That is a
fair trade to show, not something to hide: the CUDA is shorter here precisely
because it skips an abstraction the Java version is using.

What the CUDA version has to get right instead is the **per-lane fragment
register mapping**, and it is unforgiving. Writing this file, the first version
failed with exactly **128 of 256 cells wrong** — every cell in columns 8–15.
The cause is the `n8` in `m16n8k32`: **one `mma` produces only 8 columns**, so
`N = 16` needs two panels with two accumulators. The kernel compiled, ran, threw
no error, and returned confidently wrong numbers for half the output.

That is the whole argument for the Java API in one bug. `ctx.mmaFP8E4M3(...)`
cannot be given the wrong fragment layout, because the layout is the
compiler's job. Both `mmaStore` calls in the Java kernel make the two panels
explicit and impossible to forget:

```java
ctx.mmaStore(fragC0, c, 0, 0, N);   // columns 0..7
ctx.mmaStore(fragC1, c, 0, 8, N);   // columns 8..15
```

## What this demo does *not* claim

- **No speedup number.** 256 output elements on one warp is far too small for a
  wall-clock comparison to mean anything. This demo is about *which instruction
  is emitted and whether the answer is right*, not about throughput.
- **Nothing above `M16N8K32`.** `MMAShape` has exactly two entries and every
  operand combination is `.row.col`. There is no `wgmma` (sm_90) or `tcgen05`
  (sm_100) path in the 6.0.0 emitter, so everything here is an Ada result. See
  [`docs/compilation-pipeline.md`](../../docs/compilation-pipeline.md).
- **Not a GEMM you should copy for production.** One warp, one tile, no
  double-buffering, no `cp.async`. Demo 14 shows the async-copy machinery; this
  one deliberately keeps the staging readable so the MMA calls are the visible
  part.

## Related

- **Demo 08** — the fp16 `mma.sync`, with the scalar control that emits zero
- **Demo 14** — `cp.async` + shared memory + warp shuffle, the other inline-PTX path
- **`docs/compilation-pipeline.md`** — how `ctx.mmaFP8E4M3(...)` becomes that PTX string
