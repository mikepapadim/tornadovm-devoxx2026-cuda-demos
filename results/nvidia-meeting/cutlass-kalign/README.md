# Does CUTLASS `kAlign = 8` help? No.

TornadoVM instantiates its CUTLASS FP16/BF16 kernels with `kAlign = 4`
(64-bit operand loads) rather than the 8 (128-bit) that FP16 tensor-core GEMM
normally uses. The stated reason is the 24-byte array header, which meant the
payload could never be 16-byte aligned.

[TornadoVM#1066](https://github.com/beehive-lab/TornadoVM/pull/1066) (merged)
now pads the device allocation so the payload lands on a 32-byte boundary, which
is 16-byte aligned by construction. That removes the stated blocker, so
`kAlign = 8` became reachable — and it was flagged as a promising follow-up.

**Measured, it is not worth doing.**

## Method

TornadoVM's exact CUTLASS configuration held constant — `Sm80`, threadblock
128×128×32, warp 64×64×32, instruction 16×8×16, 3 stages, RowMajor × 3, identity
swizzle, FP32 accumulate — with **only `kAlign` varying** between two template
instantiations in one binary. CUDA events, 5 warm-up launches, median of 50.
RTX 4090 (sm_89), CUDA 12.6.85, CUTLASS v3.5.1.

Source: `KAlignBenchmark.cu`. Raw: `kalign-results.txt`.

## Result

| shape (m,n,k) | kAlign=4 | kAlign=8 | speedup |
|---|---|---|---|
| 1024, 1024, 1024 | 0.0307 ms | 0.0307 ms | **1.000×** |
| 2048, 2048, 2048 | 0.1114 ms | 0.1116 ms | **0.998×** |
| 4096, 4096, 4096 | 0.8487 ms | 0.8492 ms | **0.999×** |
| 256, 256, 256 | 0.0109 ms | 0.0111 ms | **0.986×** |
| 4096, 4096, 512 | 0.1137 ms | 0.1106 ms | **1.028×** |
| 1024, **1020**, **1020** | 0.0300 ms | **rejected** | n/a |

**No meaningful gain anywhere.** The best case is a skinny shape at 2.8%, within
run-to-run noise of this harness; two shapes are marginally slower.

## And it would be an API break

The last row is the one that decides it. CUTLASS's alignment constraint applies
to the **leading dimension**, not only the base pointer, so `kAlign = 8` requires
`k` and `n` to be multiples of **8**. `k = n = 1020` is a multiple of 4 but not
of 8: it runs today and `can_implement` **rejects** it at `kAlign = 8`.

So a bump would tighten the documented restriction from "multiples of 4" to
"multiples of 8" and turn working shapes into rejected ones — for no measurable
gain. A non-breaking dual instantiation with runtime dispatch would avoid the
break but still buy nothing.

**Recommendation: leave `kAlign = 4` alone.**

## The finding that matters more

`results/raw/` records CUTLASS FP16 at roughly **0.9× cuBLAS** on the same shape
and GPU, and attributes it to two causes: `kAlign = 4` *and* the single
128×128×32 tile used for every problem.

This measurement isolates them. **Alignment is not a factor.** Whatever the gap
is, it is the single tile configuration — cuBLAS dispatches among many tuned
kernels and the CUTLASS profiler would select per shape, while TornadoVM
instantiates one.

That redirects the lever: per-shape tile selection, not the array header. It is
also a much larger piece of work than an alignment constant, and it is the
honest thing to say when someone asks why the CUTLASS path trails cuBLAS on a
plain GEMM.
