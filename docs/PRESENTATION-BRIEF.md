# Presentation brief — what to say, and which file proves it

Written to be handed to an assistant with this repo checked out, to build slides
and plots. Every claim below is backed by a committed file, named inline. If a
claim has no file, it is not in this repo and should not go on a slide.

**Audience assumed:** NVIDIA's SW-Compiler team. Framing throughout is
**TornadoVM against hand-written CUDA**, both on the same GPU, in the same
session, on the same workload.

**Hardware for every number:** RTX 4090 (sm_89), driver 565.57.01, CUDA 12.6.85,
JDK 25.0.2, TornadoVM 6.0.0-jdk22plus-cuda. One machine, one GPU. Say so.

---

## Start here: the two data files

| File | Shape | Use |
|---|---|---|
| `results/raw/27-profiler-metrics/comparison.csv` | 52 paired rows: `demo, kernel, metric, unit, tornadovm, cuda, ratio_tornadovm_over_cuda` | **most bar charts read this directly** |
| `results/raw/27-profiler-metrics/metrics.csv` | 172 rows, long: `demo, kernel, implementation, metric, unit, value, launches, source, file` | anything the paired view misses |

Every row carries the `file` it came from, so any number on a slide traces back
to its raw capture. Regenerate with `python3 scripts/build-metrics-csv.py`.

`docs/PROFILER-METRICS.md` is the companion: metric definitions, which direction
is better, and seven worked figures.

---

## The argument, in order

### 1. The generated arithmetic is equivalent to hand-written CUDA

**The headline. Lead with it.**

Demo 15 compares **kernel time only** across three kernels with different
bottlenecks, and root-causes both directions instead of leaving a ratio.

| Kernel | TornadoVM | CUDA | |
|---|---|---|---|
| `elementwise` (memory-bound) | 13.94 µs | 10.62 µs | CUDA 1.31x |
| `stencil` (memory-bound) | 14.32 µs | 11.55 µs | CUDA 1.24x |
| `polynomial` (compute-bound) | 35.24 µs | 39.93 µs | **TornadoVM 1.13x** |

Controlling for both causes below, **the generated arithmetic is equivalent**.

- Data: `results/raw/21-kernel-time-comparison/MANIFEST.md`, and
  `comparison.csv` where `metric == "kernel_time_nsys"`
- Chart: grouped bars, µs, three kernel pairs

### 2. The memory-bound gap is a data-layout bug, not code generation

`FloatArray` puts its payload 16 bytes after the base, so every warp-wide
128-byte access straddles a fifth 32-byte sector. Nsight Compute counts it:

| Demo | Kernel | TornadoVM | CUDA | Ratio |
|---|---|---|---|---|
| 01-vector-add | `vectorAdd` | **5.00** | 4.00 | 1.250 |
| 14 | `rowSumOptimised` | **5.00** | 4.00 | 1.250 |
| 15 | `elementwise` | **5.00** | 4.00 | 1.250 |
| 15 | `polynomial` | **5.00** | 4.00 | 1.250 |
| 15 | `stencil` | **5.00** | 4.67 | 1.071 |
| 14 | `rowSumNaive` | 32.00 | 32.00 | 1.000 |

**Four demos, six kernels, float and int8, ordinary loads and `cp.async`.** Filed
as [TornadoVM#1065](https://github.com/beehive-lab/TornadoVM/issues/1065).

Keep the last two rows — they are the credibility. `rowSumNaive` is identical on
both sides (and 32.00 is the worst possible number), which shows the generated
code is not inherently worse. `stencil`'s CUDA side is already 4.67 because its
neighbour accesses straddle lines anyway.

- Data: `comparison.csv`, `metric == "sectors_per_request_load"`
- Chart: grouped bars with a horizontal line at 4.00 marked "perfectly coalesced"

### 3. The defect is always present; it only sometimes costs time

`polynomial` pays the same 1.25x sector penalty **and is the kernel TornadoVM
wins**. Being compute-bound, it hides the extra transactions.

Demo 01 makes the same point harder: TornadoVM issues **4x the instructions**,
and is **3.4% slower** — because both sit at ~95% of peak DRAM bandwidth. The
3.4% closes exactly as `bytes / bandwidth`:

| | TornadoVM | CUDA | ratio |
|---|---|---|---|
| total DRAM bytes | 174,726,912 | 170,544,512 | 1.0245 |
| achieved bandwidth | 94.63% | 95.46% | 0.9913 |
| **predicted time ratio** | | | **1.0335** |
| **measured time ratio** | 187,840 ns | 181,760 ns | **1.0335** |

- Data: `results/raw/27-profiler-metrics/MANIFEST.md`
- Chart: two-panel — instruction count vs kernel time, same scale, to show they
  do not track

> **Do not put demo 01's occupancy on a slide as a codegen result.** TornadoVM
> ran 1024 threads/block, CUDA 256. On sm_89 a 1024-thread block tiles once
> (32 of 48 warps, 66.7% ceiling); 256 gives six blocks and 100%. It is a launch
> configuration difference. Demo 15 pins both sides to 256 *precisely* so only
> codegen differs; demo 01 does not.

### 4. The compute-bound win is JIT specialisation, not better arithmetic

`degree` is a task argument, so Graal compiles after its value is known and
unrolls the FMA chain. Give nvcc the same value via a template parameter and it
lands at 34.7 µs against TornadoVM's 35.24 µs — equal.

- Data: `results/raw/21-kernel-time-comparison/probe-jit-specialisation.log`
- Chart: three bars — TornadoVM, nvcc runtime arg, nvcc compile-time constant

### 5. Tensor cores: all five operand types, proven twice

Demo 16 plus demo 08 cover every operand combination the backend can emit, each
validated at **max abs err 0.00000 over 256 cells**, with counters matching the
emitted PTX instruction for instruction.

| Kernel | tensor inst | HMMA | IMMA |
|---|---|---|---|
| `gemmBF16` | 4 | **4** | 0 |
| `gemmInt8` | 2 | 0 | **2** |
| `gemmFP8E4M3` | 2 | **2** | 0 |
| `gemmFP8E5M2` | 2 | **2** | 0 |
| demo 08 `gemmMMASingleTile` | 1 | **1** | 0 |
| demo 08 `gemmScalarFp16` | **0** | **0** | 0 |

int8 dispatches to the **IMMA** pipe; BF16 and both FP8 formats to HMMA.

- Data: `results/raw/26-tensor-core-datatypes/`, `results/raw/23-ncu-tensor-core-counters/`
- Chart: stacked bars HMMA/IMMA per kernel, with the scalar control at zero

Demo 08 also shows where the real gap is: identical tensor-core work (1 HMMA,
16 cycles both sides), but **183 scalar instructions around it against CUDA's
40**. That is the interesting number for this audience.

### 6. Host-side dispatch is the largest single cost — and it is a runtime problem

Demo 14's kernel is 26.6x faster than its naive variant; wall-clock only 2.17x.

Per-execution CUDA driver overhead is **~8.3 µs**, measured by differencing two
execution counts so fixed start-up cancels. Two items in it are known upstream
issues still present in 6.0.0: a 24-byte kernel-argument stack frame re-uploaded
every launch, and three `cuStreamSynchronize` where one would do
([#1028](https://github.com/beehive-lab/TornadoVM/issues/1028), PR
[#1022](https://github.com/beehive-lab/TornadoVM/pull/1022)).

- Data: `results/raw/25-host-dispatch-breakdown/MANIFEST.md`
- Chart: stacked bar of per-execution ns by API call

> **Do not divide the 1,620 total transfer calls by the execution count.** 94.8%
> of them are one-time start-up traffic. That manifest documents the mistake
> because the first analysis made it and overstated the cost ~12x.

### 7. The ceiling is sm_89, and it is an emitter limit

`MMAShape` has exactly two entries and `MMAOperand` five suffixes, all
`.row.col`. No `wgmma` (sm_90), no `tcgen05` (sm_100).

- Data: `docs/compilation-pipeline.md`
- Nuance to pre-empt: `libtornado-cudnn.so` *does* contain `cp.async.bulk.tensor`
  (TMA) and `cta_group` — but those are NVIDIA's own compiled kernels reached via
  a library call, not emitter output. Say this before someone greps the .so.

---

## Supporting material

| Topic | File |
|---|---|
| Compilation pipeline, class by class; where a second emitter plugs in | `docs/compilation-pipeline.md` |
| Start-here page for this audience | `docs/NVIDIA-BRIEF.md` |
| Metric definitions and worked figures | `docs/PROFILER-METRICS.md` |
| Every claim in the repo mapped to evidence | `docs/claims.md` |
| What was attempted and failed | `results/failures/` |

## Upstream issues found while building this

| Issue | Summary |
|---|---|
| [#1063](https://github.com/beehive-lab/TornadoVM/issues/1063) | `CuDnn.sdpaForward` launches no kernel, returns all-zero |
| [#1064](https://github.com/beehive-lab/TornadoVM/issues/1064) | CUDA lowering crash on a ternary before an allocation |
| [#1065](https://github.com/beehive-lab/TornadoVM/issues/1065) | `FloatArray` header misaligns coalescing — the 5.00-vs-4.00 above |
| [#1067](https://github.com/beehive-lab/TornadoVM/issues/1067) | A `KernelContext` kernel that fails to compile silently returns **wrong results** |

Four reproducible bugs, all with minimal reproducers. #1067 is a silent
data-corruption path and is arguably the most serious.

---

## Rules for any slide built from this

1. **Never put `kernel_time_ncu` and `kernel_time_nsys` on the same axis.** ncu
   serialises launches, flushes caches and locks clocks; its absolute times run
   several times higher. Use nsys for timing claims.
2. **`ratio_tornadovm_over_cuda` is always `tornadovm / cuda`.** Above 1 is worse
   for time, sectors and instructions — and *better* for throughput and occupancy.
   It is not a slowdown column.
3. **One GPU, one machine, one run.** Label it. Nothing here generalises to
   "TornadoVM is X% of CUDA".
4. **Do not generalise the three demo-15 kernels into a benchmark suite.** They
   were chosen to expose specific effects.
5. **Do not claim a speedup for demos 08 or 16.** 128-256 output elements on one
   warp is far too small for wall-clock to mean anything, and none is reported.
6. **Track B (demos 09, 10) is historical** — captured 2026-08-21 on TornadoVM
   5.2.1 / JDK 21, and does not describe the current runtime.
