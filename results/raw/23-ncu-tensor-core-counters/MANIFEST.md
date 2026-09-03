# Batch 23 — Tensor Core hardware counters (unblocks batch 08)

Captured 2026-09-03. RTX 4090 (sm_89), driver 565.57.01, CUDA 12.6.85,
JDK 25.0.2, TornadoVM 6.0.0-jdk22plus-cuda,
**Nsight Compute 2024.3.2** (`/opt/nvidia/nsight-compute/2024.3.2/ncu`).

`results/failures/08-nsight-compute-permission.md` recorded this measurement as
blocked and listed it as the "next action if a future invocation has GPU counter
access". Counter access is now available (see batch 22 for how) and this is that
measurement, run on both implementations.

Demo 08 computes one tile, `C[16x8] = A[16x16] * B[16x8]`, fp16 in / f32 out,
twice: `gemmScalarFp16` (plain arithmetic) and `gemmMMASingleTile`
(`ctx.mma(...)`, one warp). Both validate at `max abs err 0.00000`.

## Result — the Java kernel issues exactly one HMMA, the scalar one issues zero

| | | `sm__inst_executed_pipe_tensor_op_hmma.sum` | `sm__pipe_tensor_op_hmma_cycles_active.sum` | `smsp__inst_executed.sum` |
|---|---|---|---|---|
| **TornadoVM** | `gemmScalarFp16` | **0** | **0** | 628 |
| **TornadoVM** | `gemmMMASingleTile` | **1** | **16** | 183 |
| **CUDA** | `gemmScalarFp16` | **0** | **0** | 400 |
| **CUDA** | `gemmMMASingleTile` | **1** | **16** | 40 |

The tensor-core work is **identical on both sides**: one HMMA instruction, 16
active tensor-pipe cycles. This is hardware-counter confirmation of what
`--printKernel` already showed at the source level in batch 08 — a real
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, not an inference from
timing — and it now also confirms the negative control, that the scalar kernel
touches the tensor pipe zero times.

Within TornadoVM alone, the MMA kernel executes **3.4x fewer instructions**
than its scalar counterpart (183 vs 628) for the same output tile.

## The honest delta — scalar instructions around the MMA

TornadoVM's MMA kernel issues 183 instructions against hand-written CUDA's 40,
for the same single HMMA. The tensor-core instruction is identical; the
surrounding index arithmetic, bounds checks and fragment staging are where the
generated code is heavier. On a one-warp, one-tile kernel that overhead is the
whole kernel — at 3,936 ns vs 1,984 ns, it is also 128 bytes of output, so
neither number is a throughput claim.

**This is the interesting number for a compiler audience**, and it is a better
question than "is it using tensor cores" (it is): the gap is in scalar
setup code, not in tensor-core selection or fragment layout.

## Files

| File | Contents |
|---|---|
| `ncu-tornado08.csv` | TornadoVM, both kernels |
| `ncu-cuda08.csv` | hand-written CUDA, both kernels |
| `*.stderr.log` | profiler stderr |

Source-level evidence from batch 08 remains at
`results/raw/08-tensor-core-mma/tensorcoremma-printkernel.log`.
