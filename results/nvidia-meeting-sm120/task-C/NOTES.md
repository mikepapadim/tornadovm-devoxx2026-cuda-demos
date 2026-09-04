# Step 8 — SASS (task C), sm_120

Needs no performance counters: `cuobjdump` reads the cubin TornadoVM already
writes to `$TORNADOVM_HOME/var/cuda-codecache/device-0-0/` by default.
28 cubins present after the full suite run.

## Native Blackwell codegen confirmed — no PTX fallback

```
64-bit ELF: type=ET_EXEC, ABI=8, sm=120, toolkit=13.0, flags=0x6007802
    Tool Command Line Arguments: -arch sm_120
    CUDA Virtual SM: sm_120
    code for sm_120
```

The "PTX fallback instead of cubin" hypothesis in the doc's *Where bugs are most
likely* section does **not** occur here. A real sm_120 cubin is produced.

## `polynomial`, degree 256 — TornadoVM vs nvcc

| | instructions | FFMA | FADD | BRA |
|---|---|---|---|---|
| TornadoVM JIT (sm_120) | **288** | 255 | 1 | **1** |
| sm_89 baseline         | 288 | 256 | — | 1 |
| nvcc `-arch=sm_120 -O3` | 96 | 29 | 0 | **11** |
| nvcc, sm_89 baseline    | — | — | — | 13 |

TornadoVM's total instruction count and branch count reproduce the sm_89 baseline
exactly. The kernel is fully unrolled: the polynomial degree is a compile-time
constant at JIT time, so 256 multiply-adds are emitted straight-line with a single
branch.

nvcc cannot do this — degree arrives as a runtime scalar (`int` parameter), so the
loop stays rolled with 11 branches. This is the mechanism behind the Step 4
polynomial ratio of 0.914: TornadoVM's specialisation, not a codegen quality
difference.

## The FFMA 255-vs-256 difference is instruction selection, not semantics

The baseline records 256 FFMA. This run has 255 FFMA **plus 1 FADD** — 256
multiply-add steps either way. CUDA 13.0's ptxas emits the final Horner step as
FADD where 12.6 emitted FFMA. Total instruction count is identical at 288, so
nothing is added or removed.

Same effect on the nvcc side: 11 branches here vs 13 in the baseline. Different
ptxas version, slightly different unroll decision. Neither is a TornadoVM change.

## Full opcode histogram, TornadoVM `polynomial`

```
255 FFMA    15 NOP     2 LDCU    2 LDC     2 IMAD    2 IADD    2 EXIT
  1 ST       1 S2UR     1 S2R     1 MOV     1 LD      1 ISETP   1 FADD   1 BRA
```

Artifacts: `tvm-polynomial-sass.txt`, `nvcc-cuda15-sass.txt`.
