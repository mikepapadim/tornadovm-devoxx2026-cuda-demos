# 17 — The matrix-multiply ladder

**Concept (read in ~1 minute):** the same FP32 `C = A * B`, six ways, on the
same buffers, in one JVM, validated against the same CPU reference. The
question is not "does the vendor library win" — it does — but **how much of the
distance a Java programmer can close by hand, and exactly where hand-writing
stops paying.**

| # | Rung | What it is |
|---|---|---|
| 1 | `naive` | `@Parallel`, no GPU knowledge. One thread per output, K global loads each, no reuse |
| 2 | `kcTiled` | `KernelContext` + shared memory. Every staged value reused `TILE` times |
| 3 | `kcRegisterTiled` | + a **register micro-tile**: each thread owns a 4×4 patch of C, so one shared read feeds 4 FMAs. **The rung most people never write** |
| 4 | `cutlassSgemm` | CUTLASS library task |
| 5 | `cublasSgemm` | cuBLAS library task |
| 6 | `cublasSgemmTF32` | Same call, tensor cores. Same FP32 data, ~10 mantissa bits in the multiply |

Rungs 1–3 are Java that TornadoVM JIT-compiles to CUDA. Rungs 4–6 are native
library calls **on the same device buffers, in the same task-graph shape** — no
host round trip anywhere.

Source: [`MatMulLadder.java`](MatMulLadder.java) · hand-written counterpart:
[`MatMulLadder.cu`](MatMulLadder.cu).

## Build and run

```bash
source scripts/setup-env.sh     # from the repo root
cd demos/17-matmul-ladder
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . MatMulLadder.java

tornado --classpath . MatMulLadder                 # defaults: 2048, 20 executions
java @$TORNADOVM_HOME/tornado-argfile -cp . MatMulLadder 2048 10
```

Arguments: `<size> <executions>`. **Size must be a multiple of 64** (the
register-tiled block edge); the demo rejects other sizes with a clear message.

> `source scripts/setup-env.sh` must be run **from the repo root**. Elsewhere it
> silently leaves `JAVA_HOME` on SDKMAN's `current` and every command below
> dies with `UnsupportedClassVersionError`.

## Kernel time — the honest comparison

**Wall clock is the wrong instrument here.** It includes a 16 MB device-to-host
copy per execution, which compresses the whole ladder into a 3.8× spread. Kernel
time is what compares code quality:

```bash
nsys profile --trace=cuda -o ladder \
  $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . MatMulLadder 2048 10
nsys stats --report cuda_gpu_kern_sum --format csv ladder.nsys-rep
```

n = 2048, RTX 4090 (sm_89), CUDA 12.6.85, 10 executions, mean per-kernel:

| Rung | kernel time | GFLOP/s | vs naive |
|---|---|---|---|
| 1. naive | 3434.6 µs | 5,002 | 1.0× |
| 2. KernelContext tiled | 2613.2 µs | 6,574 | 1.3× |
| 3. KernelContext register-tiled | **696.3 µs** | **24,673** | **4.9×** |
| 4. CUTLASS task | 396.3 µs | 43,352 | 8.7× |
| 5. cuBLAS sgemm | 320.1 µs | 53,673 | 10.7× |
| 6. cuBLAS sgemm TF32 | 225.5 µs | 76,173 | 15.2× |

**Tiling alone buys almost nothing (1.3×). The register micro-tile buys 3.8× on
top of it.** That is the single highest-leverage thing a Java programmer can do
here, and it is the step that turns a memory-bound kernel into a compute-bound
one — one shared-memory read feeding four FMAs instead of one.

## Where the generated code actually diverges

The `.cu` runs the same rungs hand-written, so each one can be compared directly
rather than only at the top of the ladder:

| Rung | TornadoVM | hand-written CUDA | ratio |
|---|---|---|---|
| 1. naive | 3434.6 µs | 3375 µs | **1.02×** |
| 2. tiled | 2613.2 µs | 2592 µs | **1.01×** |
| 3. register-tiled | 696.3 µs | 488 µs | **1.43×** |
| 5. cuBLAS sgemm | 320.1 µs | 313 µs | 1.02× |
| 6. cuBLAS TF32 | 225.5 µs | 223 µs | 1.01× |

**On rungs 1 and 2 the JIT-generated code is within 1–2% of hand-written CUDA.**
The library rungs match too, as they must — it is literally the same kernel.

**Rung 3 is where TornadoVM falls behind, at 1.43×.** That is the interesting
number in this demo: not a generic "Java is slower" figure, but a specific
statement that the gap appears *only* once a kernel does register blocking, with
small fully-unrollable inner loops over a per-thread accumulator array. Simple
kernels are already at parity.

## What the kernel names give away

`nsys` names every kernel, and the library rungs are worth reading:

```
 10 x  396287 ns  cutlass::Kernel2<DefaultGemmUniversal<float, RowMajor, ...>>
 10 x  320086 ns  cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>
 10 x  225536 ns  cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>
```

Two things fall out:

**cuBLAS dispatches CUTLASS-derived kernels.** Rungs 5 and 6 are `cutlass_80_*`
— the vendor library and the template library are not separate worlds.

**And it picks a different tile than TornadoVM's CUTLASS task does.** cuBLAS
chose `256x128_8x4` for this shape; TornadoVM instantiates one fixed
128×128×32 tile for every problem. That, not alignment, is why rung 4 trails
rung 5 — cuBLAS dispatches among many tuned kernels and selects per shape.
The `align1` versus `align4` suffixes are the operand alignment showing up in
kernel selection too.

## Hand-written CUDA counterpart

```bash
nvcc -arch=sm_89 -O3 -lcublas -o matmul_ladder MatMulLadder.cu && ./matmul_ladder 2048 10
```

It runs rungs 1, 2, 3, 5 and 6. **CUTLASS is deliberately absent** — it needs a
CUTLASS checkout, and `scripts/run-all-cuda.sh` builds this file with the plain
CUDA toolkit. Demo 12 covers CUTLASS on the CUDA side.

Because it times the kernel with no copy-out, its numbers are directly
comparable to the `nsys` column above and **not** to the Java wall clock.

## Validation

Every rung is checked against the same CPU reference, all `PASSED`:

- rungs 1–5 at `max abs err 0.00000`
- rung 6 at `max abs err 0.00007`, against a deliberately looser 2e-2 tolerance

**That looser tolerance is the demo, not a fudge.** TF32 keeps ~10 mantissa bits
in the multiply and accumulates in FP32, so it is a different numeric contract
reached through an identical API call — one `withTuning` away.

## What this demo does *not* claim

- **No general "TornadoVM is X% of CUDA".** One GPU, one shape, one run. The
  1.43× on rung 3 is a statement about register-blocked kernels on sm_89.
- **The wall-clock table the program prints is not a code-quality measure.** It
  includes dispatch and a 16 MB copy-out per execution. Use the `nsys` numbers.
- **The rungs are not equally tuned.** Rung 3 is a competent register-tiled
  kernel, not a hand-optimised one; a specialist would go further with
  double-buffering, vectorised loads and a swizzled epilogue.

## Related

- **Demo 15** — kernel time vs wall clock, and why the two disagree
- **Demo 12** — CUTLASS fused epilogues, where CUTLASS beats a plain GEMM call
- **Demo 08 / 16** — the tensor-core path from Java, one rung below cuBLAS TF32

Captured evidence: `results/raw/28-matmul-ladder/`.
