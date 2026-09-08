# 31 — CUDA global-load batching: closing the register-tiled gap

Evidence for the TornadoVM code-generator fix that batches the global loads of a
shared-memory staging sequence (`CUDAGlobalLoadBatching`, upstream PR against
`beehive-lab/TornadoVM` develop).

Captured **2026-09-08**, RTX 4090 (sm_89), CUDA 12.6, TornadoVM built from
`origin/develop` + the fix. Every TornadoVM number comes from **one build** with
`-Dtornado.cuda.batchGlobalLoads` toggled at run time, so the two arms differ only
in the order of the emitted instructions — not the compiler, driver, machine or
data.

## The result

Demo 17 rung 3, `kcRegisterTiled`, n=1024, **100 launches per arm**, `nsys`
per-launch durations:

| arm | median | mean | stddev | speedup |
|---|---|---|---|---|
| TornadoVM, batching off | 112.895 µs | 112.916 | 0.152 | 1.000x |
| TornadoVM, batching on | 68.416 µs | 68.238 | 0.541 | **1.650x** |
| hand-written CUDA | 67.455 µs | 67.483 | 0.388 | 1.674x |

The fix takes TornadoVM from **1.67x slower** than hand-written CUDA to **within
1.4%** of it.

## Why it was slow

`ncu`, same kernel, same launch:

| counter | off | on |
|---|---|---|
| `long_scoreboard` stalls per issue | 12.10 | 1.45 |
| issue rate (% of peak sustained) | 20.82% | 50.90% |
| FFMA instructions | 1,073,741,824 | 1,073,741,824 |
| registers per thread | 72 | 72 |
| LSU bank conflicts | 602,227 | 603,209 |

Same work, same occupancy, same registers, no spills — the kernel was simply
waiting on memory, one load at a time.

## The root cause

`sass-schedules.txt` has the whole story in three lines:

```
tornadovm, batching off    L S L S L S L S L S L S L S L S     (generic LD)
tornadovm, batching on     L L L L L L L L S S S S S S S S     (generic LD)
hand-written CUDA          L L L L L L L L S S S S S S S S     (LDG)
```

Hand-written CUDA loads through `const float *`, which ptxas lowers to **`LDG`** —
a load proven to target global memory, and therefore proven not to alias the
shared store next to it, so ptxas batches the loads on its own. TornadoVM casts an
integer address to a plain pointer, which lowers to a **generic `LD`**. A generic
load may target shared memory, so ptxas has to keep it ordered against every
shared store, and the staging chain serialises.

`ReorderProbe.cu` confirms the other half: in hand-written CUDA the interleaved
schedule **cannot be written at all**. Three source orders (interleaved,
interleaved with an asm memory barrier, batched) compile to byte-identical SASS —
448 instructions, `LDG` x8 then `STS` x8 in every case. ptxas picks the schedule,
not the source.

TornadoVM knows the address space at LIR level (`CUDAMemorySpace`) even though its
emitted C does not encode it, so the batching has to happen in the code generator.

A separate follow-up worth having: emit `__ldg()` or a global-qualified pointer so
ptxas gets the address space directly. That would fix the schedule at the source
and likely help elsewhere too. Not attempted here.

## What else benefits

Every shared-memory kernel in this repo, flag off vs on (`kernel-sweep.csv`). The pass
fires on **8 of 14** generated kernels, and **only one moves outside noise**:

| kernel | stores sunk | speedup |
|---|---|---|
| `kcRegisterTiled` | **7** | **1.518x** |
| `gemmBF16` / `gemmFP8E4M3` / `gemmFP8E5M2` / `gemmInt8` | 2–4 | 1.016–1.032x (single launch, noise) |
| `kcTiled` (x2), `kcMma` | 1 | 1.011–1.018x (noise) |

The number of stores sunk predicts the win, which is the mechanism restated: a kernel
staging **one** element per thread per tile has one load in flight either way and there is
nothing to batch. The six untouched kernels produce bit-identical code and move
0.979x–1.004x, the noise floor here.

So the pattern that benefits is **blocked / register-tiled GEMM** and anything else whose
staging fan-out per thread exceeds one — batched gather-into-shared, multi-row stencil
halo loads, tiled transpose, K-blocked attention staging. Not one-element-per-thread
reductions, scans or histograms. And it grows with tile size: fan-out is
`(BM x BK) / threads-per-block`, so an 8x8 micro-tile would stage 16 elements per thread
against the 4x4's eight.

Not measured: whether workloads outside this repo benefit. GPULlama3.java's matmuls are
the obvious candidate and are unmeasured.

## What it costs

Sinking stores **extends live ranges by construction** — *n* staged values are live where
one was. That is inherent to batching loads, not an artefact. `register-cost.csv`:

| kernel | registers/thread | spill loads | spill stores |
|---|---|---|---|
| `kcRegisterTiled` | 72 → **72** | 0 → 0 | 0 → 0 |
| `kcTiled`, `kcMma`, `kcTiled` (fp16) | unchanged | 0 → 0 | 0 → 0 |
| `gemmBF16` | 33 → **34** | 0 → 0 | 0 → 0 |
| `gemmInt8` | 25 → **28** | 0 → 0 | 0 → 0 |
| `naive` (pass does not fire) | 94 → 94 | 0 → 0 | 0 → 0 |

Zero spill traffic anywhere, both modes. The cost is bounded — a region ends at the first
barrier, so the number of simultaneously live staged values equals the kernel's staging
fan-out, which the kernel author chose.

**But it is not *provably* spill-free.** A kernel already at the register ceiling with a
long staging region could spill and come out slower. None was found here; none was proved
impossible. That asymmetry is the argument for keeping
`-Dtornado.cuda.batchGlobalLoads` rather than hard-wiring the behaviour.

## Files

| file | what |
|---|---|
| `per-launch-durations.csv` | every individual kernel launch, 3 arms — **plot this** |
| `kernel-summary.csv` | median/mean/stddev/percentiles per arm |
| `ncu-counters.csv` | stall and issue counters, off vs on |
| `sass-schedules.txt` | the L/S schedules above |
| `pure-cuda-probe.csv` | `ReorderProbe.cu` output (all three identical, as expected) |
| `kernel-sweep.csv` | every shared-memory kernel in the repo, flag off vs on |
| `register-cost.csv` | register pressure and spill traffic, off vs on |
| `ReorderProbe.cu` | the three-schedule probe |
| `collect.sh` | regenerates every CSV here |

## Reproducing

```bash
source scripts/setup-env.sh
./collect.sh "$TORNADOVM_HOME" 1024 100
```

Needs a TornadoVM build that contains the fix (any build works for the `off` arm;
without the fix both arms give the `off` numbers). `ncu` is optional — the script
skips the counter table if it is missing. Point `NCU=` at a working `ncu` if the
default path is wrong.

## Caveats

- One GPU, one kernel shape. The 1.65x is what this staging pattern recovers; a
  kernel that does not stage through shared memory is unaffected by the fix.
- `nsys` and `ncu` disagree on ratios for this kernel family (see
  `results/nvidia-meeting/measurement-mode/`). Everything above is `nsys`
  per-launch duration except the counter table, which is `ncu` and is used only
  for *why*, never for *how much*.
- The two arms were captured in sequence on an otherwise idle machine, not
  interleaved. Spread across launches is under 0.8% in every arm.
