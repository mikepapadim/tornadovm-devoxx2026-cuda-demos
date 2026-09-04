# Step 5 — tensor-core path, sm_120

**No timing claim.** 128-256 output elements on one warp is far too small to time;
this step validates code generation and instruction dispatch only.
Nsight Compute 2025.3.1.0, CUDA 13.0.88, demo 16 `TensorCoreDataTypes`.

## Emitted PTX — identical to sm_89

From `--printKernel` (`d16-printkernel.log`, counted in `mma-shapes.txt`):

```
4  mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
2  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
2  mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32
2  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
```

Same shapes, same counts as the sm_89 baseline. TornadoVM's codegen did not change
for Blackwell.

## Counters

| Kernel | tensor inst | `subpipe_hmma` | `subpipe_hmma_op_hmma` | `subpipe_imma` | `subpipe_imma_op_imma` | cycles active | sm_89 (HMMA / IMMA) |
|---|---|---|---|---|---|---|---|
| `gemmBF16`    | 4 | 4 | **4** | 0 | 0 | 64 | 4 / 0 |
| `gemmInt8`    | 2 | 0 | 0 | 2 | **2** | 32 | 0 / 2 |
| `gemmFP8E4M3` | 2 | 2 | **0** | 0 | 0 | 32 | 2 / 0 |
| `gemmFP8E5M2` | 2 | 2 | **0** | 0 | 0 | 32 | 2 / 0 |

Aggregate instruction counts and cycles-active reproduce the sm_89 baseline
exactly (4/2/2/2 and 64/32/32/32), and match the emitted PTX instruction for
instruction.

## Finding — FP8 is a distinct op on Blackwell, not HMMA

On sm_89, both FP8 formats counted as **HMMA**. On sm_120 they count in the
`subpipe_hmma` umbrella — which the metric description spells out as
"HMMA/QMMA/OMMA ops" — but **not** in `subpipe_hmma_op_hmma`, and not in IMMA at
all. The FP8 `mma` issues as **QMMA**, a separate op from HMMA, where Ada folded
it into the HMMA counter.

Consequence for anyone reproducing this: `op_hmma + op_imma` no longer sums to
the tensor-pipe total on Blackwell. The umbrella `subpipe_hmma` /
`subpipe_imma` pair does. int8 -> IMMA and BF16 -> HMMA are unchanged.

## Correction to the reproduction doc — silently empty metrics

`sm__inst_executed_pipe_tensor_op_hmma.sum` and `..._op_imma.sum`, which the doc
specifies, were **renamed** in this profiler generation to
`sm__inst_executed_pipe_tensor_subpipe_hmma_op_hmma.sum` and
`..._subpipe_imma_op_imma.sum`. The same applies to `dram__bytes_read.sum` /
`dram__bytes_write.sum` in Step 2, now `dram__bytes_op_read.sum` /
`dram__bytes_op_write.sum`.

**Nsight Compute does not error on the old names.** It emits the row with a
Metric Value of `n/a` and exits 0, and nothing appears on stderr. A capture script
that checks only the exit status records a column of `n/a` and reports success —
which is what the first pass here did. Verify the names against
`ncu --query-metrics` (`available-tensor-metrics.txt`) rather than trusting the
exit code.

## Files

| File | Contents |
|---|---|
| `d16-printkernel.log`, `mma-shapes.txt` | emitted PTX and its `mma` histogram |
| `d16-ncu.csv` | first capture; `op_hmma`/`op_imma` columns are `n/a` (old metric names) |
| `d16-ncu-subpipe.csv` | re-capture, `subpipe_*_op_*` names |
| `d16-ncu-subpipe-umbrella.csv` | re-capture, umbrella `subpipe_hmma` / `subpipe_imma` |
| `available-tensor-metrics.txt` | `ncu --query-metrics` tensor entries on this host |
