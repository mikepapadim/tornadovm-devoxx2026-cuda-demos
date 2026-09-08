# 18 — the FP16 matrix-multiply ladder

Demo 17 climbs one GEMM in FP32. This is the same climb in FP16, and it has one
rung FP32 cannot have: **tensor cores reached directly from Java**.

| # | Rung | What it is |
|---|---|---|
| 1 | naive `@Parallel` | one thread per output element |
| 2 | `KernelContext` tiled | shared-memory tiles, FP32 accumulate |
| 3 | **`KernelContext` MMA** | `ctx.mma` → a real `mma.sync.aligned.m16n8k16` |
| 4 | CUTLASS `hgemm` | library task, tensor cores |
| 5 | cuBLAS `GemmEx` FP16 | FP16 in, FP16 out, FP32 accumulate |
| 6 | cuBLAS `GemmEx` FP16→FP32 | the standard inference configuration |

Every rung computes the same `C = A * B` and is validated against a CPU
reference **computed over the same FP16-rounded inputs**, so the comparison
measures the kernel and not the quantisation.

Source: [`MatMulLadderFP16.java`](MatMulLadderFP16.java).

## Run

```bash
source scripts/setup-env.sh      # from the repo root
cd demos/18-matmul-ladder-fp16
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . MatMulLadderFP16.java
tornado --classpath . MatMulLadderFP16            # defaults: n=1024, 20 executions
```

`n` must be a multiple of 64.

## Read the kernel numbers, not the wall clock

**This is the demo's main lesson.** At `n = 512` the wall-clock summary reports
the tiled rung as *slower than naive* — 0.5×. That is not what the GPU did; it
is ~100 µs of per-execution host dispatch swamping a kernel that takes less than
that.

```bash
bash scripts/compare-ladder.sh 18 1024 10
```

That runs the ladder under Nsight Systems for per-kernel GPU time and under
Nsight Compute for the counters that explain the ranking, and prints both:

```
   kernel                                 GPU avg     GFLOP/s  vs slowest
   ----------------------------------------------------------------------
   naive                                  496.6us        4324        1.0x
   kcTiled                                337.9us        6355        1.5x
   kcMma                                  192.3us       11167        2.6x
   CUTLASS                                 30.4us       70737       16.4x
   ampere_fp16_s1688gemm_fp16_128x...      21.5us       99788       23.1x
   ampere_s1688gemm_fp16_128x64_sl...      21.4us      100250       23.2x
```

Tiled is **1.5× faster** than naive at kernel level, against 0.5× on the wall
clock. Same run, opposite conclusion.

## What the counters show

```
   kernel                          instrs        regs       spill   bank-conf  gmem-stall      issue%
   naive                          4450304          40           0          61       38.89        9.30
   kcTiled                        2701312          40           0       65161       10.40       19.50
   kcMma                           810752          48           0      299463       17.77        4.57
   void cutlass::Kernel2<c...       42720         230           0       12435        0.48       11.46
```

Three things worth pointing at:

- **`spill` is 0 everywhere.** The private accumulator arrays stay in registers;
  nothing falls back to local memory.
- **`naive` is not compute-bound, it is latency-bound** — a 38.9 global-memory
  stall at a 9.3% issue rate. Tiling drops the stall to 10.4 and doubles the
  issue rate, which is the whole point of the rung.
- **`kcMma` executes 5.5× fewer instructions than `kcTiled`** (810,752 against
  2,701,312) because the tensor core does the arithmetic — but its 299,463 bank
  conflicts and 4.6% issue rate say the staging around it, not the `mma`, is
  what costs.

## The honest gap

`kcMma` reaches 11,167 GFLOP/s against CUTLASS's 70,737 and cuBLAS's ~100,000.
The Java rung emits the same `mma.sync` instruction the library kernels use —
confirm with `tornado --printKernel` — so the instruction is not the difference.
What the libraries add is the surrounding machinery: multi-stage pipelining,
swizzled shared-memory layouts that avoid the bank conflicts above, and a tile
shape chosen per problem.

That is the useful message. Reaching the tensor core from Java is a few lines;
reaching cuBLAS's throughput is a different and much larger problem, and this
ladder shows exactly how much of it the library is doing for you.

### The load-batching fix does not help this ladder — and that is informative

TornadoVM PR [#1079](https://github.com/beehive-lab/TornadoVM/pull/1079) batches
the global loads of a shared-memory staging sequence, and it takes demo 17's
fp32 register-tiled rung from 1.42× to 1.02× of hand-written CUDA. Re-measured
here, n=2048, it does essentially nothing:

| Rung | before | after | hand-written | speedup |
|---|---|---|---|---|
| 1. naive | 5115.4 µs | 5128.0 µs | 3412.4 µs | 1.00× |
| 2. KernelContext tiled | 2609.0 µs | 2596.1 µs | 2586.3 µs | 1.005× |
| 3. KernelContext MMA | 1017.6 µs | 1003.9 µs | 690.0 µs | 1.014× |
| 5. cuBLAS GemmEx FP16 | 113.7 µs | 115.4 µs | 111.7 µs | 0.985× |

**Why:** the pass fires correctly on both kernels here — it is not blocked. The
win is simply bounded by how many outstanding loads the batching can create, and
that equals the staging fan-out per thread:

| kernel | staged per thread | loads in flight, before → after | speedup |
|---|---|---|---|
| `kcRegisterTiled` (demo 17, fp32) | 8 | 1 → 8 | **1.39×** |
| `kcTiled` (here) | 2 | 1 → 2 | ~1.01× |
| `kcMma` (here) | 2, packed into one store | 2 → 4 in one block | 1.014× |

`kcTiled` is genuinely interleaved — load, store, load, store — and the pass sinks
the first store below the second load exactly as intended. Going from one
outstanding load to two is just worth very little in a kernel dominated by a
64-iteration inner product.

`kcMma` is the more interesting case: the fp16 path packs two `__half` values into
a single 32-bit shared store, so the emitted order is *already* load, load, pack,
store. Check it with `tornado --printKernel` — the two `*(( __half *) ...)` loads
sit back to back with only address arithmetic between them. There is no
interleaving left to remove.

So the 1.475× gap on rung 3 above is **not** the staging schedule. It is the
pipelining, swizzling and tile selection described in this section, which is a
much larger piece of work. Evidence:
[`results/raw/31-load-batching-reorder/`](../../results/raw/31-load-batching-reorder/).

Same caveat as demo 17: these numbers come from a source build of `develop` plus
PR #1079, not from the pinned `6.0.0` release every other number here uses.

## Related

- **Demo 17** — the same ladder in FP32, with a register-tiled rung
- **Demo 16** — every tensor-core operand type, validated and counted
- **Demo 08** — the smallest possible `mma.sync`, with a scalar control
