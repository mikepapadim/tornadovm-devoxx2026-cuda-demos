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
source ../../scripts/setup-env.sh     # works from here or from the repo root
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . MatMulLadder.java

tornado --classpath . MatMulLadder                 # defaults: 1024, 20 executions
java @$TORNADOVM_HOME/tornado-argfile -cp . MatMulLadder 2048 10
```

Arguments: `<size> <executions>`. **Size must be a multiple of 64** (the
register-tiled block edge); the demo rejects other sizes with a clear message.
Every number below is at **2048, 10 executions** — pass them explicitly.

Rung 4 needs TornadoVM 7.0.0's CUTLASS bridge, which links CUDA 13:
`setup-env.sh` puts a CUDA 13 runtime on `LD_LIBRARY_PATH`. Without one the demo
aborts at rung 4 with `libcudart.so.13: cannot open shared object file`.

## Kernel time — the honest comparison

**Wall clock is the wrong instrument here.** It includes host dispatch and a 16 MB
device-to-host copy per execution, which compresses the whole ladder into a 3.7×
spread against 15.4× in kernel time. Kernel time is what compares code quality:

```bash
nsys profile --trace=cuda --force-overwrite=true -o ladder \
  $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . MatMulLadder 2048 10
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv ladder.nsys-rep
```

TornadoVM 7.0.0, n = 2048, RTX 4090 (sm_89), CUDA 12.6.85, 10 executions, mean
per-kernel. Rungs 1, 2, 5, 6: mean of 3 runs, spread ≤ 0.4%. Rungs 3 and 4:
median of 13 runs (see §Run-to-run variation):

| Rung | kernel time | GFLOP/s | vs naive |
|---|---|---|---|
| 1. naive | 3404.7 µs | 5,046 | 1.0× |
| 2. KernelContext tiled | 2597.5 µs | 6,614 | 1.3× |
| 3. KernelContext register-tiled | **499.8 µs** | **34,373** | **6.8×** |
| 4. CUTLASS task | 426.3 µs | 40,300 | 8.0× |
| 5. cuBLAS sgemm | 314.7 µs | 54,591 | 10.8× |
| 6. cuBLAS sgemm TF32 | 221.1 µs | 77,702 | 15.4× |

**Tiling alone buys almost nothing (1.3×). The register micro-tile buys 5.2× on
top of it.** That is the single highest-leverage thing a Java programmer can do
here, and it is the step that turns a memory-bound kernel into a compute-bound
one — one shared-memory read feeding four FMAs instead of one. Register tiling
written in Java now lands **1.17× from the CUTLASS library task** (499.8 vs
426.3 µs) and 1.59× from tuned cuBLAS (314.7 µs).

## Against hand-written CUDA, rung by rung

The `.cu` runs the same rungs hand-written, so each one can be compared directly
rather than only at the top of the ladder (3 runs, identical to the microsecond
each time):

| Rung | TornadoVM 7.0.0 | hand-written CUDA | ratio | on 6.0.0 |
|---|---|---|---|---|
| 1. naive | 3404.7 µs | 3383 µs | **1.01×** | 1.02× |
| 2. tiled | 2597.5 µs | 2593 µs | **1.00×** | 1.01× |
| 3. register-tiled | 499.8 µs | 489 µs | **1.02×** | **1.43×** |
| 5. cuBLAS sgemm | 314.7 µs | 314 µs | 1.00× | 1.02× |
| 6. cuBLAS TF32 | 221.1 µs | 223 µs | 0.99× | 1.01× |

**On TornadoVM 7.0.0 every JIT-compiled rung is within 2% of hand-written CUDA,**
including the register-tiled one. The library rungs match too, as they must —
they are literally the same kernels.

### Rung 3 used to be 1.43× — and you can still show why, live

On 6.0.0 this demo's headline was that TornadoVM falls behind *only* once a kernel
does register blocking: rung 3 at 1.43× of hand-written CUDA, rungs 1–2 at parity.
That gap was diagnosed with Nsight Compute in this repo — global-memory stalls, not
code volume (`results/raw/29-register-tiled-stalls/`) — and fixed upstream in
[#1079](https://github.com/beehive-lab/TornadoVM/pull/1079), which shipped in 7.0.0.

The cause was the staging loop, not the inner product. TornadoVM emitted one
global load per shared store, so a single load was in flight at a time; ptxas
could not batch them because TornadoVM casts an integer address to a plain
pointer, which lowers to a **generic `LD`** that may target shared memory.
Hand-written CUDA loads through `const float *`, lowers to `LDG`, and ptxas
batches on its own. The fix reorders the loads in the code generator, where the
address space is known.

**The fix is on by default in 7.0.0 and still behind a flag**, so the before and
after run on the same pinned SDK:

```bash
# after (default): rung 3 ~500 µs
nsys profile --trace=cuda --force-overwrite=true -o on \
  $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . MatMulLadder 2048 10

# before: the load-batching pass switched off
nsys profile --trace=cuda --force-overwrite=true -o off \
  $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile \
  -Dtornado.cuda.batchGlobalLoads=false -cp . MatMulLadder 2048 10
```

Measured interleaved with the 6.0.0 release in one session, three rounds, every
run validated:

| | rung 3 kernel time | vs hand-written CUDA |
|---|---|---|
| TornadoVM 6.0.0 | 696.0 / 697.8 / 697.8 µs | 1.43× |
| TornadoVM 7.0.0, `batchGlobalLoads=false` | 659.4 / 678.9 / 688.9 µs | 1.39× |
| TornadoVM 7.0.0, default | 510.6 / 499.8 / 499.8 µs | 1.02× |

Switching the flag off does not land exactly back on 6.0.0 — other changes
between the two releases are in play — but it recovers most of the gap, which
is enough to show the mechanism on stage.

Wall clock moves much less: rung 3 goes 1767 µs → 1637 µs between 6.0.0 and 7.0.0
at n = 2048, because host dispatch and the copy-out dominate this demo. That is the
gap between "the kernel got 1.4× faster" and "the demo got 1.08× faster", and it
is worth saying out loud.

## What the kernel names give away

`nsys` names every kernel, and the library rungs are worth reading (7.0.0):

```
 10 x  426.3 us  cutlass::Kernel2<DefaultGemmUniversal<float, RowMajor, ...>>
 10 x  314.7 us  cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>
 10 x  221.1 us  cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>
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

## Run-to-run variation

Rungs 3 and 4 move more between runs than the others, so their table entries are
medians of 13 runs on 7.0.0:

- **Rung 3:** median 499.8 µs; 11 of 13 runs within 498.6–510.6 µs, two low runs
  at 480.3 and 487.7 µs. A single run can therefore read as low as ~0.98× of
  hand-written CUDA — the settled figure is 1.02×.
- **Rung 4 (CUTLASS):** median 426.3 µs; 10 of 13 within 425.1–427.8 µs, three
  low runs at 397.8–416.2 µs.

**The CUTLASS rung is ~7.6% slower on 7.0.0 than on 6.0.0 in most runs**
(426.3 µs against 396–398 µs, interleaved in the same session), although `nsys`
reports the identical kernel template on both. The naive rung is flat across
every run, which rules out clock drift. **The cause has not been investigated.**

Rungs 1, 2, 5 and 6 vary by ≤ 0.4%.

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

- **No general "TornadoVM is X% of CUDA".** One GPU, one shape. "Within 2% on
  every rung" is a statement about these five kernels on sm_89 at n = 2048.
- **The wall-clock table the program prints is not a code-quality measure.** It
  includes dispatch and a 16 MB copy-out per execution. Use the `nsys` numbers.
- **The rungs are not equally tuned.** Rung 3 is a competent register-tiled
  kernel, not a hand-optimised one; a specialist would go further with
  double-buffering, vectorised loads and a swizzled epilogue.

## Related

- **Demo 15** — kernel time vs wall clock, and the other gap this repo found and
  saw fixed in 7.0.0 (array-header alignment)
- **Demo 18** — the same ladder in FP16, where load batching buys nothing
- **Demo 12** — CUTLASS fused epilogues, where CUTLASS beats a plain GEMM call
- **Demo 08 / 16** — the tensor-core path from Java, one rung below cuBLAS TF32

Captured evidence: `results/raw/36-demo15-demo17-on-7.0.0/` (7.0.0, including the
6.0.0 / 7.0.0 / flag-off interleave); `results/raw/28-matmul-ladder/` and
`results/raw/31-load-batching-reorder/` (the 6.0.0 ladder and the fix's own evidence).
