# Presentation brief — what to say, and which file proves it

Written to be handed to an assistant with this repo checked out, to build slides
and plots. Every claim below is backed by a committed file, named inline.

**Audience:** NVIDIA's SW-Compiler team. Framing is **TornadoVM against
hand-written CUDA**, same GPU, same session, same workload.

**Every number:** RTX 4090 (sm_89), driver 565.57.01, CUDA/NVRTC 12.6.85,
JDK 25.0.2, TornadoVM 6.0.0-jdk22plus-cuda (SDKMAN release). One machine, one
architecture. Say so on the slide.

> **Authoritative source is [`results/nvidia-meeting/summary.md`](../results/nvidia-meeting/summary.md)**
> and `manifest.json` beside it. If anything here disagrees with those, they win.

---

## Read this before quoting any ratio

Three rules, each learned by getting it wrong in this repo first.

**1. `ncu` and `nsys` disagree on the *ratio*, not just absolute time.** The same
demo 15 kernels at identical verified geometry give memory-bound ratios of
**1.02–1.04 under `ncu`** and **1.24–1.31 under `nsys`**. `ncu` serialises
launches and flushes caches, which makes both implementations DRAM-latency-bound
and hides the sector penalty. **Quote `nsys` for steady-state performance;
quote `ncu` only for counters and for ratios *within* `ncu`.** Never put them on
one axis. → `results/nvidia-meeting/measurement-mode/`

**2. Check launch geometry before attributing anything to code generation.**
Demo 15 pins both sides to block=256/grid=16384 (verified from
`launch__block_size`, not from source comments) — its numbers are sound.
**Demo 12 and demo 01 do not** — TornadoVM's default worker grid picks
1024-thread blocks there. Ratios from those two are launch-config artefacts.

**3. One GPU, one run.** Nothing here is a general "TornadoVM is X% of CUDA".

---

## The argument, in order

### 1. Generated arithmetic is competitive — steady state, matched geometry

**Lead with this.** Demo 15, `nsys`, 20 executions, block=256 both sides.

| Kernel | TornadoVM | CUDA | |
|---|---|---|---|
| `elementwise` (memory-bound) | 13,944 ns | 10,621 ns | CUDA 1.31× |
| `stencil` (memory-bound) | 14,322 ns | 11,546 ns | CUDA 1.24× |
| `polynomial` (compute-bound) | 35,236 ns | 39,926 ns | **TornadoVM 1.13×** |

Both deltas are attributed, and **neither is arithmetic quality** — sections 2
and 3. Controlling for both, the generated arithmetic is equivalent.

→ `results/raw/21-kernel-time-comparison/`, `results/nvidia-meeting/summary.md`
**Chart:** grouped bars, ns, three kernel pairs.

### 2. The memory gap is data layout, proven on one binary

Sweeping **only the base-pointer offset** of a single compiled CUDA binary — if
code generation mattered, this could not reproduce the effect:

| base offset | ld sec/req | st sec/req | median of 30 |
|---|---|---|---|
| 0 B | 4.00 | 4.00 | 11.7 µs |
| **16 B — TornadoVM's `FloatArray` payload** | **5.00** | **5.00** | **15.0 µs** |
| 32 B | 4.00 | 4.00 | 12.0 µs |
| 64 B | 4.00 | 4.00 | 12.0 µs |
| 128 B | 4.00 | 4.00 | 11.7 µs |

A warp request is 128 contiguous bytes; the L1↔L2 path is addressed in **32-byte
sectors**. Only *sub-sector* misalignment adds a fifth sector.

**Slide-worthy consequence: the fix needs 32-byte alignment, not 128.** Open PR
[#1066](https://github.com/beehive-lab/TornadoVM/pull/1066) implements 128 B —
4× the padding for the same sector count. Posted as a measured comment on
[#1065](https://github.com/beehive-lab/TornadoVM/issues/1065).

→ `results/nvidia-meeting/AlignmentSweep.cu`, `alignment-sweep-*.{csv,log}`
**Chart:** bars by offset with a line at 4.00 marked "perfectly coalesced".

### 3. The compute-bound win is JIT specialisation — now visible in SASS

`degree` is a task argument, so Graal compiles after it is bound.

| | SASS instructions | branch-class | FFMA |
|---|---|---|---|
| TornadoVM `polynomial`, `degree=256` | 288 | **1** | **256** |
| nvcc, `degree` as runtime scalar | — | **13** | — |

**256 FFMA and one branch — the loop is gone.** Give nvcc the same information
via a template constant and it lands at ~34,700 ns against TornadoVM's 35,236 —
parity. The advantage is *when* compilation happens, not arithmetic quality.

TornadoVM writes every cubin to `$TORNADOVM_HOME/var/cuda-codecache/device-0-0/`
by default, so `cuobjdump -sass` reads it directly.

→ `results/nvidia-meeting/task-C-jit-specialisation/`
**Chart:** three bars (TornadoVM / nvcc runtime arg / nvcc template constant).

### 4. Tensor cores: all five operand types, counters matching emitted PTX

| Kernel | tensor inst | HMMA | IMMA |
|---|---|---|---|
| `gemmBF16` | 4 | **4** | 0 |
| `gemmInt8` | 2 | 0 | **2** |
| `gemmFP8E4M3` | 2 | **2** | 0 |
| `gemmFP8E5M2` | 2 | **2** | 0 |
| demo 08 `gemmMMASingleTile` | 1 | **1** | 0 |
| demo 08 `gemmScalarFp16` (control) | **0** | **0** | 0 |

All validate at **max abs err 0.00000 over 256 cells**. int8 dispatches to the
**IMMA** pipe; BF16 and both FP8 formats to HMMA.

**Label these "code-generation validation only" — no timing claim.** 128–256
output elements on one warp is far too small.

Demo 08 also shows identical tensor-core work on both sides (1 HMMA, 16 cycles)
with **183 scalar instructions around it against hand-written CUDA's 40** — the
interesting number for this audience.

→ `results/raw/23-ncu-tensor-core-counters/`, `results/raw/26-tensor-core-datatypes/`

### 5. Host dispatch is the largest single cost — and CUDA Graphs remove it

Per-execution cost obtained by **differencing execution counts** (10 vs 100), so
fixed start-up cancels. Dividing a total by iterations overstates it ~12×.

| | nograph | graph |
|---|---|---|
| `cuLaunchKernel` per execution | **6.00** | **0.00** |
| `cuMemcpyHtoDAsync` per execution | **7.00** | **0.00** |
| event create/record/destroy | 42 | 3 |
| host API ns/exec, excl. `cuStreamSynchronize` | **33,774** | **6,283** |

**5.4× less host API time.** Those seven H2D copies are the per-launch 24-byte
kernel-argument stack frames ([#1028](https://github.com/beehive-lab/TornadoVM/issues/1028)
finding 1); replay bakes them into the graph.

`cuStreamSynchronize` is excluded and reported separately — it is device wait and
is *higher* in graph mode because the host blocks once on the batched chain.
Folding it in would read as a regression.

→ `results/nvidia-meeting/task-E-graph-composition/`
**Chart:** stacked bar of per-execution ns by API call, nograph vs graph.

### 6. Library-task integration is free; buffers stay on the device

**Same CUTLASS kernel, same launch config, identical performance:**

| Kernel | 1024³ | 4096³ | ratio |
|---|---|---|---|
| `LinearCombinationRelu` | 30,359.8 / 30,466.9 | 858,509.7 / 853,759.4 | **1.00 / 1.01** |
| `LinearCombination` | 30,055.8 / 29,734.2 | 856,770.7 / 823,562.6 | **1.01 / 1.04** |

Fusion saves **6.2% / 11.8% / 7.9%** of GPU time at 256 / 1024 / 4096.

And in a `JIT → cuDNN → JIT → cuDNN` graph, the input crosses **once**, the
output **once per execution**, and the three intermediates **never** — device
buffers are reused across the boundary with no host round trip.

> Do **not** use demo 12's wall clock to compare fused vs unfused: both plans run
> in one JVM and the fused one runs first, absorbing process start-up.
> → `task-F-fused-gemm/wallclock-inversion-resolved.md`

→ `results/nvidia-meeting/task-F-fused-gemm/`, `task-E-graph-composition/`

### 7. JIT compile cost — the budget any Tile path must fit

m=n=k=1024, `-Dtornado.profiler=True`:

| Phase | Cold (first execution) | Warm |
|---|---|---|
| Graal | **42.2 ms** | 0 |
| NVRTC | **16.3 ms** | 0 |
| total task graph | 87.1 ms | 0.39–0.56 ms |

**58.5 ms of the 87.1 ms cold execution is compilation (67%); Graal is 2.6× the
NVRTC cost; cold ≈ 200× warm.** Paid once per task graph — the price of the
section-3 specialisation advantage.

### 8. Three independent ceilings, and no CUDA Tile on this host

| Layer | Ceiling | Evidence |
|---|---|---|
| GPU (RTX 4090) | **sm_89** | `nvidia-smi` compute cap 8.9 |
| NVRTC 12.6 toolkit | **sm_90** | `nvrtcGetSupportedArchs`, 14 archs |
| TornadoVM MMA emitter | **sm_89-class** | `MMAShape` = {M16N8K16, M16N8K32} |

Note the middle row: **this toolkit could already target Hopper; the emitter
could not.** `wgmma` is not reachable by extending that enum.

**CUDA Tile: zero** toolkit headers, libraries, `nvcc` flags or `libnvrtc`
symbols matching *tile* at CUDA 12.6.85. A Tile prototype cannot begin here.
**Make no Tile performance claim.**

The useful find: NVRTC exports **`nvrtcGetNVVM`** and **`nvrtcGetLTOIR`**, which
reframes the ask from "should we target Tile" to "NVRTC already emits NVVM IR — is
there a supported path to feed IR *in* at runtime?"

→ `results/nvidia-meeting/tile-feasibility/inventory.txt`,
`results/nvidia-meeting/open-questions.md`

---

## Data files

| File | Shape |
|---|---|
| `results/raw/27-profiler-metrics/comparison.csv` | 52 paired rows — most bar charts read this directly |
| `results/raw/27-profiler-metrics/metrics.csv` | 172 rows, long format |
| `results/nvidia-meeting/manifest.json` | 9 entries, full provenance, `corrections` and `resolved` fields |
| `results/nvidia-meeting/task-F-fused-gemm/kernel-times.csv` | per-kernel, **with grid/block** |
| `results/nvidia-meeting/alignment-sweep-ncu.csv` | the offset sweep |

Regenerate the first two with `python3 scripts/build-metrics-csv.py`.

## Supporting documents

| Topic | File |
|---|---|
| Methodology, matrix, acceptance criteria, confounders | `docs/nvidia-meeting-metrics-plan.md` |
| Measured values only, kernel-only vs end-to-end | `results/nvidia-meeting/summary.md` |
| Questions for NVIDIA | `results/nvidia-meeting/open-questions.md` |
| Compilation pipeline class by class | `docs/compilation-pipeline.md` |
| Metric definitions and figure recipes | `docs/PROFILER-METRICS.md` |
| Every repo claim mapped to evidence | `docs/claims.md` |

## Upstream issues found while building this

| Issue | Summary |
|---|---|
| [#1063](https://github.com/beehive-lab/TornadoVM/issues/1063) | `CuDnn.sdpaForward` launches no kernel, returns all-zero |
| [#1064](https://github.com/beehive-lab/TornadoVM/issues/1064) | CUDA lowering crash on a ternary before an allocation |
| [#1065](https://github.com/beehive-lab/TornadoVM/issues/1065) | `FloatArray` header misaligns coalescing (PR #1066 open) |
| [#1067](https://github.com/beehive-lab/TornadoVM/issues/1067) | A `KernelContext` kernel that fails to compile silently returns **wrong results** |

#1067 is a silent data-corruption path and is arguably the most serious.

## Slide rules

1. Never put `ncu` and `nsys` times on one axis, and never compare a ratio from
   one with a ratio from the other.
2. `ratio_tornadovm_over_cuda` is always `tornadovm / cuda`. Above 1 is worse for
   time, sectors and instructions; **better** for throughput and occupancy.
3. Label every figure "one RTX 4090, sm_89, single run".
4. Demos 08 and 16 are **code-generation validation** — no speedup claims.
5. Demo 01's occupancy is a **launch-config artefact** (1024 vs 256 block), not a
   codegen result. Same for demo 12's JIT-kernel ratios.
6. Track B (demos 09, 10) is **historical** — 2026-08-21, TornadoVM 5.2.1, JDK 21.
7. Demo 12's wall clock cannot compare fused vs unfused (ordering artefact).
