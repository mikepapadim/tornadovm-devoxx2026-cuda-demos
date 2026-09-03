# TornadoVM on the CUDA backend — a brief for compiler engineers

This repository is a measurement harness disguised as a demo repo. Every Java
demo ships a hand-written CUDA C++ equivalent in the same folder, both run on
the same GPU in the same session, and every performance claim is traced to a
cause rather than left as a ratio.

This page is the start-here for someone who works on compilers. It covers what
the lowering path actually is, what has been measured, and where the gaps are.
Hardware throughout: RTX 4090 (sm_89), driver 565.57.01, CUDA 12.6.85, JDK
25.0.2, TornadoVM 6.0.0 (`6.0.0-jdk22plus-cuda`, SDKMAN release).

## The lowering path

TornadoVM does not interpret or emit PTX directly from bytecode. It reuses
Graal as a front end and emits **CUDA C**, then hands that to NVRTC:

```
Java bytecode
  -> Graal IR                     (via JVMCI)
  -> TornadoVM Graal phases       CUDAHighTier / CUDAMidTier / CUDALowTier
                                  (parallelisation, memory-space assignment,
                                   task specialisation on runtime arg values)
  -> CUDA C source                ...cuda.graal.asm.CUDAAssembler
                                  (inline PTX asm for MMA / cp.async)
  -> cubin                        NVRTC at runtime, via the Java FFM API:
                                  nvrtcGetCUBIN, --gpu-architecture=sm_<cc>
     (fallback) -> PTX            nvrtcGetPTX, --gpu-architecture=compute_<cc>,
                                  then JIT-compiled by the driver at load
  -> cuModuleLoadDataEx -> cuLaunchKernel
```

Worth being precise about the last step, since it is easy to assume otherwise:
the common path is **CUDA C straight to cubin in a single NVRTC call** — there
is no separate `ptxas` invocation. PTX is the fallback for when the installed
toolkit cannot target the GPU directly.

NVRTC is reached through `java.lang.foreign` (Panama FFM), not JNI:
`...cuda.ffm.NVRTCAPI` binds `nvrtcCompileProgram`, `nvrtcGetCUBIN`,
`nvrtcGetPTX` and `nvrtcGetSupportedArchs` with `SymbolLookup` and
`MemorySegment`. `...cuda.ffm.CUDACompiler` drives compile-and-load.

A full class-by-class walk of this pipeline, and an analysis of where a second
emitter would plug in, is in [`compilation-pipeline.md`](compilation-pipeline.md).

Two consequences worth naming, because they shape everything below:

- **What is handed to NVIDIA is C, not an IR.** Anything TornadoVM wants that C
  cannot express — MMA, `cp.async` — is emitted as **inline PTX `asm volatile`**
  in that generated C. That works, and demo 08 and demo 14 both prove it works,
  but it means the interesting instruction selection is happening as string
  emission in a LIR statement rather than in a lowering pass with a type system
  behind it. It is also why the reachable tensor-core instruction set is bounded
  by two enums (below) rather than by the hardware.
- **Compilation happens when the argument values are known.** That is a real
  and measurable advantage over AOT nvcc, quantified below.

`tornado --printKernel` prints the generated CUDA for any demo here, so any
claim on this page can be checked at the source level in one command.

## What has been measured

### 1. The generated arithmetic is equivalent to hand-written CUDA

[Demo 15](../demos/15-kernel-time-comparison/) compares **kernel time only** —
every other timed demo in this repo reports wall clock, which on TornadoVM is
dominated by host-side dispatch and tells you nothing about code generation.
Three kernels, deliberately different bottlenecks, identical block and grid
sizes, identical arithmetic including bounds checks, no `-use_fast_math` on
either side.

| Kernel | TornadoVM | CUDA | |
|---|---|---|---|
| `elementwise` (memory-bound) | 13.94 µs | 10.62 µs | CUDA 1.31x faster |
| `stencil` (memory-bound) | 14.32 µs | 11.55 µs | CUDA 1.24x faster |
| `polynomial` (compute-bound) | 35.24 µs | 39.93 µs | **TornadoVM 1.13x faster** |

Both deltas are attributed, and **neither is arithmetic quality**:

**The memory-bound gap is a data-layout bug.** `FloatArray` places its payload
16 bytes after the allocation base — visible in the generated CUDA as
`l_5 = l_4 + 4L` on every access. A warp-wide 32x4-byte access is 128 bytes,
exactly 4 sectors when aligned; offset by 16 bytes it straddles a fifth.
Nsight Compute counts it directly:

| Kernel | metric | CUDA | TornadoVM | CUDA forced to offset 4 |
|---|---|---|---|---|
| `elementwise` | load sectors | 524,288 | **655,360** | 655,360 |
| `elementwise` | store sectors | 524,288 | **655,360** | 655,360 |
| `polynomial` | load sectors | 524,288 | **655,360** | — |
| `stencil` | load sectors | 1,835,006 | **1,966,080** | 1,966,080 |

Every TornadoVM global access reports **5.00 sectors per request against 4.00**,
a count *bit-identical* to the same CUDA kernel deliberately misaligned. Filed
upstream as [TornadoVM#1065](https://github.com/beehive-lab/TornadoVM/issues/1065).
Padding the payload to a 128-byte boundary recovers it. Data:
`results/raw/22-ncu-alignment-counters/`.

Note `polynomial` pays the identical 1.25x penalty **and it is the kernel
TornadoVM wins** — being compute-bound, it hides the extra transactions. The
defect is in every generated kernel; it only surfaces in time when the kernel
is bandwidth-bound.

**The compute-bound win is JIT specialisation.** `degree` is a task argument, so
Graal compiles the kernel after its value is known and fully unrolls the FMA
chain — no loop counter, no branch. nvcc compiles ahead of time and must emit a
real loop (`cuobjdump -sass` shows 11 branch instructions). Give nvcc the same
value as a template parameter and it lands at 34.7 µs against TornadoVM's
35.24 µs — within 1.6%.

**Controlling for both, the generated arithmetic is equivalent.**

### 2. Tensor cores are real, and identical to hand-written

[Demo 08](../demos/08-tensor-core-mma/) computes one `M16N8K16` fp16 tile via
`ctx.mma(...)` and, from the same run, a scalar control. By hardware counter:

| | HMMA instructions | tensor-pipe cycles | total instructions |
|---|---|---|---|
| TornadoVM `gemmScalarFp16` | **0** | **0** | 628 |
| TornadoVM `gemmMMASingleTile` | **1** | **16** | 183 |
| CUDA `gemmScalarFp16` | **0** | **0** | 400 |
| CUDA `gemmMMASingleTile` | **1** | **16** | 40 |

The tensor-core work is identical on both sides. **The delta is scalar overhead
around the MMA — 183 instructions versus 40** for index arithmetic, bounds
checks and fragment staging. On a one-warp kernel that overhead *is* the kernel.
That is the more interesting number than "does it use tensor cores", and it is
the kind of thing a lowering pass would attack. Data:
`results/raw/23-ncu-tensor-core-counters/`.

### 3. The same defect appears in a third, unrelated kernel class

[Demo 14](../demos/14-warp-async-shared/) is a per-row **int8** reduction using
`cp.async` + shared memory + warp shuffle — a different element type reaching
memory through a different instruction. It shows both halves of the story:

| metric | TornadoVM naive | CUDA naive | TornadoVM opt | CUDA opt |
|---|---|---|---|---|
| sectors per request | 32.00 | 32.00 | **5.00** | **4.00** |
| global load sectors | 4,194,304 | 4,194,304 | **163,840** | **131,072** |
| bank conflicts | 3,944,216 | 3,678,799 | 190 | 270 |
| instructions | 487,040 | 259,456 | **1,134,592** | 1,658,880 |

**The naive kernels are identical to the byte** — same 32.00 sectors per
request, same 4,194,304 sectors, kernel times within 0.05%. **The optimised
kernels differ by exactly 1.250x**, which is #1065 again, in int8, through
`cp.async`. The header offset is a property of the array layout, not of a kernel
shape or an element type.

Two secondary observations for a compiler audience: the optimised TornadoVM
kernel executes **32% fewer instructions than the hand-written CUDA one** and is
still marginally slower, so instruction count is not the lever on this workload;
and within TornadoVM the optimisation is a **2.3x increase in instructions for
an order-of-magnitude speedup**, because it moves 25.6x fewer sectors. Data:
`results/raw/24-ncu-demo14-counters/`.

### 4. The host-side dispatch cost is large and separable

Demo 14's TornadoVM *kernel* is 26.6x faster than its naive variant; its
wall-clock is only 2.17x faster. Differencing two execution counts under `nsys`
(so fixed start-up cancels and the slope is the real per-execution cost) gives
the driver-side breakdown:

| CUDA API | per execution | ns per execution |
|---|---|---|
| `cuStreamSynchronize` | 3 | 54,446 |
| `cuLaunchKernel` | 1 | 2,414 |
| `cuMemcpyDtoHAsync_v2` | 1 | 2,407 |
| `cuMemcpyHtoDAsync_v2` | 1 | 1,654 |
| `cuEventCreate` / `Record` / `SetCurrent` | 3 each | 1,843 combined |

The 54,446 ns of `cuStreamSynchronize` is genuine device wait, not overhead —
the two kernels average 54,792 ns. **Excluding it, per-execution CUDA API
overhead is ~8.3 µs**, and two items in it are already-diagnosed upstream
issues: the single H2D is a 24-byte kernel-argument stack frame re-uploaded on
every launch although it never changes, and three `cuStreamSynchronize` where
one would do.

Call volume against hand-written CUDA is still the striking number — 1,620
memory-transfer calls versus 41 for the same 40 launches, plus 1,656 stream
synchronisations and 6,392 event calls against zero of each. But 94.8% of
TornadoVM's transfers are 256-byte **start-up** traffic, constant regardless of
data size or execution count and complete before the first kernel launches, so
they are not a per-execution cost.

The CUDA driver API therefore explains only ~8–12 µs of demo 14's ~114 µs
per-execution wall-clock gap. **The remainder is host-side runtime work that a
CUDA-only trace cannot see.** Characterising it needs JFR, and that work has
been done upstream — see
[TornadoVM#1028](https://github.com/beehive-lab/TornadoVM/issues/1028), which
measures the same dispatch sequence, attributes the remaining host time, and
carries open PRs against it. Nothing is claimed here beyond the driver-side
numbers above. Data: `results/raw/25-host-dispatch-breakdown/`, including a
minimal reproducer.

**This is a runtime problem, not a code-generation problem**, and it is called
out separately so it is not confused with the code-generation results above.

Worth noting for calibration: in that same steady-state trace TornadoVM's
optimised kernel is **1.08x faster** than the hand-written CUDA one (3,941 vs
4,251 ns), even carrying the #1065 penalty. Under Nsight Compute's cold
single-launch conditions the ordering reverses. Both gaps are small; the two
kernels are equivalent, and the dispatch path is where the real cost is.

## The API surface, and its ceiling

`KernelContext` in 6.0.0 exposes considerably more than the public demos show.
Verified by `javap` against the pinned SDK:

- **MMA:** `mma` (fp16), `mmaBF16`, `mmaInt8`, `mmaFP8E4M3`, `mmaFP8E5M2`
- **Fragment handling:** `mmaLoadA` / `mmaLoadB` / `mmaLoadBSwizzled`,
  `mmaFragment`, `mmaFragmentInt`, `mmaStore`, `mmaStoreInt`, `mmaStoreBSwizzled`
- **Async copy:** `asyncCopyToLocal` / `asyncCopyCommit` / `asyncCopyWaitGroup`
  over `HalfFloatArray`, `FP8Array`, `ByteArray` → `cp.async.ca.shared.global`
- **Warp primitives:** `simdShuffleDown` → `__shfl_down_sync`
- **Shared memory:** `allocate{Int,Byte,Long,Float,Double,HalfFloat,Half2}LocalArray`

The emitter composes MMA instructions from a fixed shape and a fixed operand
suffix — in `CUDALIRStmt$MMAComputeStmt` the literal is
`asm volatile("mma.sync.aligned.`, completed by `MMAShape.getPtxName()` and one
of exactly five operand combinations in `MMAOperand`:

```
.row.col.f32.f16.f16.f32        .row.col.s32.s8.s8.s32
.row.col.f32.bf16.bf16.f32      .row.col.f32.e4m3.e4m3.f32
                                .row.col.f32.e5m2.e5m2.f32
```

with `MMAShape` being exactly `{M16N8K16, M16N8K32}` (`m16n8k16`, `m16n8k32`).
So FP8 lands as `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`.

**Those two enums are the ceiling, and it is sm_89.** Every combination is
`.row.col`; there is no `wgmma` (sm_90) and no `tcgen05` (sm_100) path, and
nothing above `M16N8K32`. Everything measured on this page is therefore an Ada
result, and the architectures NVIDIA most cares about are untested here
**because the hardware is not available, not because the work was scoped out.**

One distinction worth drawing precisely, since it cuts the other way: the
*vendor-library* route already reaches newer hardware. `libtornado-cudnn.so`,
which backs the cuDNN and CUTLASS demos, contains `cp.async.bulk.tensor` (TMA)
and `cta_group` paths. Those are NVIDIA's own compiled kernels, reached through
a library call — nothing TornadoVM's own emitter can generate. The gap is
specifically in the **JIT-emitted** path.

## Where this goes next

Three things would move this forward, roughly in order of how much they unblock:

1. **Hopper/Blackwell access** (DGX Cloud credits or a loaner). Everything above
   stops at sm_89. `wgmma` and `tcgen05` are not reachable from the current
   emitter and cannot even be prototyped without the hardware.

2. **A design review of the lowering path.** The current path emits CUDA C with
   inline PTX for anything C cannot express, then NVRTC. The open question is
   whether to stay there, move to NVVM IR via libNVVM, or target cuTile. That is
   a one-hour conversation with someone who knows the tradeoffs, and it would
   change what gets built next.

3. **A scoped cuTile prototype:** Graal IR → cuTile for one kernel class — GEMM
   with a fused epilogue, which this repo already exercises through CUTLASS in
   [demo 12](../demos/12-cutlass-fused-epilogue/). Narrow enough to evaluate
   honestly, and it tests whether a tile-level IR is a better target for a JIT
   than C-plus-inline-asm.

## Reproducing anything here

```bash
sdk install java 25.0.2-open
sdk install tornadovm 6.0.0-jdk22plus-cuda
git clone https://github.com/mikepapadim/tornadovm-devoxx2026-cuda-demos
cd tornadovm-devoxx2026-cuda-demos
source scripts/setup-env.sh
bash scripts/run-all-demos.sh        # Java side, 36/36 checks
bash scripts/run-all-cuda.sh         # CUDA side, no JDK needed
```

Raw profiler output for every claim is under `results/raw/`, and
`docs/claims.md` maps each claim in the repo to the log that supports it.
Failures are kept too — `results/failures/` records what was attempted and did
not work, including the two separate causes that blocked Nsight Compute on this
machine for weeks.

## Three bugs found upstream while building this

| Issue | Summary |
|---|---|
| [#1063](https://github.com/beehive-lab/TornadoVM/issues/1063) | `CuDnn.sdpaForward` launches no kernel and silently returns an all-zero result |
| [#1064](https://github.com/beehive-lab/TornadoVM/issues/1064) | CUDA lowering crashes with `Node implementing Lowerable not handled: NewInstance` when a ternary precedes an allocation |
| [#1065](https://github.com/beehive-lab/TornadoVM/issues/1065) | `FloatArray`'s 16-byte header misaligns warp-coalesced accesses — the 5.00-vs-4.00 sectors above |
