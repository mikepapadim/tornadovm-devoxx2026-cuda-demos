# Batch 26 — every tensor-core operand type, exercised and counted

Captured 2026-09-03. RTX 4090 (sm_89), driver 565.57.01, CUDA 12.6.85,
JDK 25.0.2, TornadoVM 6.0.0-jdk22plus-cuda, Nsight Compute 2024.3.2.
`demos/16-tensor-core-datatypes`.

Demo 08 exercised fp16 only. The 6.0.0 emitter carries exactly five operand
combinations (`CUDALIRStmt$MMAComputeStmt` + `MMAOperand`, see
`docs/compilation-pipeline.md`). This batch exercises the other four on
hardware, so all five are now covered by a running, validated demo.

## Method

`C[16x16] = A[16x32] * B[32x16]`, one warp, four kernels differing only in the
compute call. Every input is a small multiple of 0.5 in `[-2, 2]` (small
integers for int8) — exactly representable in bf16, e4m3 and e5m2 alike — and
the CPU reference is computed from the **same stored values**, each input
round-tripped through its storage format first. So the comparison measures the
kernel rather than the quantisation, and the tolerance can be tight.

## Result 1 — all four validate exactly

```
BF16     m16n8k16  f32.bf16.bf16.f32  PASSED (max abs err 0.00000, 0/256 cells out of tol)
int8     m16n8k32  s32.s8.s8.s32      PASSED (max abs err 0.00000, 0/256 cells out of tol)
FP8 e4m3 m16n8k32  f32.e4m3.e4m3.f32  PASSED (max abs err 0.00000, 0/256 cells out of tol)
FP8 e5m2 m16n8k32  f32.e5m2.e5m2.f32  PASSED (max abs err 0.00000, 0/256 cells out of tol)
```

Zero error on every one of 256 cells, for all four types.

## Result 2 — the emitted PTX (`--printKernel`)

```
      4 mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
      2 mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
      2 mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32
      2 mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
     10 ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
      5 ldmatrix.sync.aligned.m8n8.x4.shared.b16
```

BF16 shows 4 because `K=32` needs two `k16` steps over two 8-column panels; the
`k32` types need one step, hence 2 each. `16-datatypes-printkernel.log`.

## Result 3 — hardware counters match the PTX exactly

| Kernel | `..._pipe_tensor.sum` | HMMA | IMMA | tensor-pipe cycles |
|---|---|---|---|---|
| `gemmBF16` | 4 | **4** | 0 | 64 |
| `gemmInt8` | 2 | 0 | **2** | 32 |
| `gemmFP8E4M3` | 2 | **2** | 0 | 32 |
| `gemmFP8E5M2` | 2 | **2** | 0 | 32 |

Instruction for instruction against the emitted PTX. The counters also show
what the source cannot: **int8 dispatches to the IMMA pipe, while BF16 and both
FP8 formats dispatch to HMMA** — on Ada, FP8 is counted as a half-precision
tensor operation. `16-datatypes-ncu.csv`.

## What is not claimed

No speedup number: 256 output elements on one warp is far too small for
wall-clock to mean anything, and none is reported. Nothing above `M16N8K32` is
reachable in the 6.0.0 emitter, so these are Ada results only.

## Files

| File | Contents |
|---|---|
| `16-datatypes-run.log` | validation output, all four types |
| `16-datatypes-printkernel.log` | full generated CUDA for all four kernels |
| `ptx-instruction-counts.txt` | `mma.sync` / `ldmatrix` tallies from that log |
| `16-datatypes-ncu.csv` | tensor-pipe counters per kernel |
