# 08 — Tensor Core / MMA: smallest possible mma.sync demo

**Concept (read in ~1 minute):** the CUDA backend exposes NVIDIA Tensor Core
`mma.sync` instructions through `KernelContext` intrinsics —
`mmaFragment`/`mmaLoadA`/`mmaLoadB`/`mma`/`mmaStore` — that TornadoVM's
Graal-based JIT lowers directly to `ldmatrix.sync.aligned` /
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` inline PTX asm in the
generated CUDA source (confirmed by direct source inspection of the CUDA
backend's Graal nodes/plugins, e.g. `CUDAMMALoadANode.java`,
`CUDAGraphBuilderPlugins.java`, and by `--printKernel` output, not assumed
from the API surface alone). `MMAShape.M16N8K16` is the fp16/bf16 tile
shape, requiring compute capability 8.0+ (Ampere and later; this machine's
RTX 4090 is 8.9).

This demo is deliberately the **smallest possible** MMA workload: one warp
(32 threads), one `M16N8K16` tile (`C[16,8] = A[16,16] * B[16,8]`, fp16
inputs, f32 accumulate), exactly **one** `mma.sync` instruction — next to a
scalar (no-MMA, one-thread-per-output-element) reference kernel computing
the identical tile. Both are validated against the same closed-form CPU
reference. It exists to *show the mechanism* clearly; for a realistic-scale,
multi-warp, tiled GEMM (with cp.async, bf16, swizzled-shared-memory, and
`[N,K]`-layout variants, plus a fp16-tiled-no-MMA baseline for comparison)
already shipped in the pinned TornadoVM tree, see
[Realistic-scale reference](#realistic-scale-reference-upstream-example)
below.

Source: [`TensorCoreMMA.java`](TensorCoreMMA.java).

## Build

```bash
source scripts/setup-env.sh   # from repo root; pins the SDK in env/versions.env
cd demos/08-tensor-core-mma
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  -d . TensorCoreMMA.java
```

## Run

```bash
tornado --classpath . TensorCoreMMA
```

Reproducibility form (`java @arg-file`):

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp . TensorCoreMMA
```

To see the generated CUDA source (the actual `mma.sync`/`ldmatrix` asm):

```bash
tornado --printKernel --classpath . TensorCoreMMA
```

## Expected output

```
Single-tile Tensor Core (MMA) demo: C[16x8] = A[16x16] * B[16x8], fp16 -> f32
  [scalar (no MMA)] validation PASSED (max abs err 0.00000, 0/128 cells out of tol)
  [mma.sync (Tensor Core)] validation PASSED (max abs err 0.00000, 0/128 cells out of tol)
Result is correct
```

## What was actually measured (Observed)

Pinned SDK: TornadoVM `6.0.0-jdk22plus-cuda` (SDKMAN), JDK 25.0.2,
RTX 4090, driver `565.57.01`, `nvcc`/`ptxas` 12.6.85.

**Re-verified on 6.0.0:** both kernels still validate exactly, and
`--printKernel` still emits exactly one `mma.sync.aligned` instruction —
the generated-code claim below holds unchanged on the 6.0.0 CUDA backend.
Logs: `results/raw/18-tornadovm-6-migration/08-mma-tornado.log`,
`08-mma-javaargfile.log`, `08-mma-printkernel.log`.

The detailed findings below were captured on the earlier 5.2.1 source-built
pin (logs unmodified in `results/raw/08-tensor-core-mma/`):

- Ran via `tornado --classpath .`, `java @$TORNADOVM_HOME/tornado-argfile`, and
  `tornado --enableProfiler console --classpath .` — all three runs: both
  kernels validate exactly (max abs err `0.00000`, since inputs and the
  16-term dot product are exact in fp16→f32 for these deterministic bounded
  values), and profiler JSON confirms `"BACKEND": "CUDA"`,
  `"DEVICE": "NVIDIA GeForce RTX 4090"` for both `s_scalar.gemm` and
  `s_mma.gemm`. Re-verified from a clean rebuild (`rm *.class`, rebuild,
  rerun) before finalizing. Logs: `results/raw/08-tensor-core-mma/tensorcoremma-run.log`,
  `tensorcoremma-run-javaargfile.log`.
- **Generated-code evidence** (`tornado --printKernel`, log
  `results/raw/08-tensor-core-mma/tensorcoremma-printkernel.log`): the
  compiled `gemmMMASingleTile` kernel contains exactly one
  `asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ...")`
  plus its two `ldmatrix.sync.aligned` operand loads (one `.x4` for the A
  fragment, one `.x2.trans` for the B fragment); the `gemmScalarFp16` kernel
  compiled from the *same run* contains **zero** occurrences of `mma.sync`
  anywhere in its body — a direct source-backed comparison from one
  invocation, not an inference from timing.
- **Nsight Compute hardware-counter evidence** (captured 2026-09-03, log
  `results/raw/23-ncu-tensor-core-counters/`). This was blocked for a long
  time — `ncu` on `PATH` resolves to `2026.2.1.0`, which cannot connect to
  this driver (`565.57.01`) at all, and the matching `2024.3.2` install was
  refused with `ERR_NVGPUCTRPERM`. Both causes are diagnosed and fixed in
  `results/failures/08-nsight-compute-permission.md`. The counters now
  confirm the generated-code evidence above at the hardware level:

  | | `..._pipe_tensor_op_hmma.sum` | tensor-pipe cycles | total inst |
  |---|---|---|---|
  | TornadoVM `gemmScalarFp16` | **0** | **0** | 628 |
  | TornadoVM `gemmMMASingleTile` | **1** | **16** | 183 |
  | CUDA `gemmScalarFp16` | **0** | **0** | 400 |
  | CUDA `gemmMMASingleTile` | **1** | **16** | 40 |

  ```bash
  /opt/nvidia/nsight-compute/2024.3.2/ncu --target-processes all \
    --metrics sm__inst_executed_pipe_tensor_op_hmma.sum,sm__pipe_tensor_op_hmma_cycles_active.sum \
    java @$TORNADOVM_HOME/tornado-argfile -cp . TensorCoreMMA
  ```

  The tensor-core work is **identical on both sides** — one HMMA, 16 active
  tensor-pipe cycles — and the scalar control touches the tensor pipe zero
  times. The honest delta is elsewhere: TornadoVM issues 183 instructions
  around that one HMMA against hand-written CUDA's 40, i.e. the gap is in
  scalar index arithmetic and fragment staging, not in tensor-core selection
  or fragment layout. (Note: the metric names carrying a `_v2` suffix are
  not accepted by the `2024.3.2` install; use the unsuffixed names above.)
- No speedup claim is made for this single-tile demo: 128 output elements
  is far too small a workload to be anything but launch/dispatch-bound, so
  a scalar-vs-MMA wall-clock comparison at this size would not measure the
  Tensor Core's actual throughput advantage. For a size where GEMM
  performance is meaningful, see the realistic-scale reference below.

## Realistic-scale reference (upstream example)

The pinned tree ships its own comprehensive MMA benchmark,
`vendor/tornadovm/tornado-examples/.../compute/MatrixMultiplicationMMA.java`
— six GEMM variants (tiled fp16 baseline with no MMA; multi-warp MMA;
MMA+cp.async; MMA+bf16; MMA with `[N,K]` weight layout; MMA+swizzled shared
memory), each validated against a CPU reference, each timed. Run at
`512 512 512` on this machine (`results/raw/08-tensor-core-mma/upstream-mma-512.log`):
all six variants **PASSED** validation; the same `mma.sync.aligned.m16n8k16`
instruction is confirmed via `--printKernel` for all four MMA variants in
that file (`results/raw/08-tensor-core-mma/mma-printkernel-512.log`, 80
occurrences of `mma.sync`/`ldmatrix` total across the four MMA kernels).
At this problem size the six variants are close in wall-clock (launch- and
memory-bound, not compute-bound — GFLOP/s figures in the log should not be
read as a clean MMA-vs-baseline speedup at 512³; a much larger shape, e.g.
the file's own documented `128 8192 2048` GPULlama3-FFN-shaped example, is
where each variant's GFLOP/s becomes compute-bound and the MMA/cp.async/
bf16 speedups documented in that file's own `Speedups:` block are meant to
be read). Not independently re-run at larger sizes by this task — the
512³ run above already discharges this task's acceptance criterion
(generated-code evidence for real `mma.sync` usage); a larger-scale,
presenter-ready performance narrative is left to a future task if the talk
needs one.

## Fallback if the live demo fails

Both kernels are tiny and deterministic (16×8×16, fixed inputs) — if a live
run doesn't reproduce, the fallback is the captured
`--printKernel` log itself: read the `mma.sync.aligned.m16n8k16` line
directly off `results/raw/08-tensor-core-mma/tensorcoremma-printkernel.log`
without re-running anything live.

## CUDA equivalent

[`TensorCoreMMA.cu`](TensorCoreMMA.cu) is the same demo written directly in CUDA C++, for side-by-side comparison.

```bash
nvcc -arch=sm_89 -o tensor_core_mma TensorCoreMMA.cu && ./tensor_core_mma
```

This is the pair worth studying, because the CUDA version is *shorter* (121
lines vs 129) and much harder to get right.

`ctx.mmaLoadA` / `mmaLoadB` / `mma` / `mmaStore` encapsulate the
register-to-matrix-element mapping fixed by the PTX ISA. Written out, each lane
holds 8 halves of A, 4 of B and 4 floats of C at positions like:

```c
a[0] = A[(groupID) * K + (2 * threadInGroup)];
a[2] = A[(groupID + 8) * K + (2 * threadInGroup)];
b[0] = B[(2 * threadInGroup) * N + groupID];
C[(groupID + 8) * N + (2 * threadInGroup)] = d[2];
```

Get one index wrong and the result is silently incorrect — there is no compiler
error, just a wrong matrix. That is what the Java helpers are for.

Both emit exactly one tensor-core instruction. Verify the CUDA one:

```bash
cuobjdump -sass tensor_core_mma | grep -c HMMA    # -> 1
```

against the Java version's `--printKernel | grep -c mma.sync.aligned` — also 1.
Identical validation output, max abs err `0.00000`.

`bash scripts/run-all-cuda.sh` builds and runs the CUDA equivalent of every demo (needs only the CUDA toolkit, no JDK).

## JBang

Not verified: `jbang` is not installed on this machine (`which jbang` → exit
1, checked 2026-08-20, same finding as demos 00–07). The shape would match
those demos' documented-but-unverified pattern — do not run it live until
tested on the pinned environment:

```bash
jbang TensorCoreMMA.java
```
