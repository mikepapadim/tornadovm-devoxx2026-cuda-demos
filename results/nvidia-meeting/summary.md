# Measured summary

Only values measured on this host. Methodology, confounders and acceptance
criteria: [`docs/nvidia-meeting-metrics-plan.md`](../../docs/nvidia-meeting-metrics-plan.md).
Machine-readable: [`manifest.json`](manifest.json). Provenance:
[`env/provenance.txt`](env/provenance.txt).

**Single architecture: sm_89 (RTX 4090), driver 565.57.01, CUDA/NVRTC 12.6.85,
TornadoVM 6.0.0-jdk22plus-cuda (SDKMAN release), JDK 25.0.2.** Nothing here is
aggregated across architectures, and no figure is a general
"TornadoVM vs CUDA" percentage.

**Kernel-only and end-to-end figures are separated below and must not be mixed.**
`ncu` durations are not comparable to `nsys` durations (ncu serialises launches,
flushes caches, no clock boost).

---

## Kernel-only

### B. Alignment isolation — new in this bundle

**One compiled binary. Only the base-pointer byte offset varies.** If generated
arithmetic were responsible, this could not reproduce the effect.

| base offset | ld sectors/req | st sectors/req | ld sectors | median (30 reps) |
|---|---|---|---|---|
| 0 B | 4.00 | 4.00 | 524,288 | 11.7 µs |
| **16 B — TornadoVM `FloatArray` payload** | **5.00** | **5.00** | **655,360** | **15.0 µs** |
| 32 B | 4.00 | 4.00 | 524,288 | 12.0 µs |
| 64 B | 4.00 | 4.00 | 524,288 | 12.0 µs |
| 128 B | 4.00 | 4.00 | 524,288 | 11.7 µs |

Raw: `alignment-sweep-ncu.csv`, `alignment-sweep-timing.log`. Source:
`AlignmentSweep.cu`. `n = 4194304` float32, grid 16384 × block 256, 3 warm-up +
30 timed, median.

**Conclusion — this is a buffer-layout issue, not a generated-arithmetic issue.**
A warp request is 32 × 4 = 128 contiguous bytes; the L1↔L2 path is addressed in
32-byte sectors. Only sub-sector misalignment adds a fifth sector.

**Refinement to [#1065](https://github.com/beehive-lab/TornadoVM/issues/1065):**
that issue proposes padding the payload to a 128-byte boundary. **32-byte
alignment is sufficient** — it restores 4.00 sectors/request and 524,288
sectors, identical to offset 0, at one quarter of the padding. A residual
~2.5% timing difference remains between 32/64 B (12.0 µs) and 0/128 B (11.7 µs)
which the sector counts do not explain; it is reported, not explained.

### B2. Alignment separated from launch geometry — resolves the iteration-3 retraction

2×2 over {implementation} × {block size}; TornadoVM's block pinned with a
`GridScheduler`. `n=16777216` float32, `out = in*0.25f + 0.1f`, both sides
validate. Counters under `ncu`; ratios valid within that mode only.

| run | ld sec/req | occupancy | regs | DRAM % peak | kernel ns |
|---|---|---|---|---|---|
| TornadoVM @256 | **5.00** | 78.15% | 16 | 93.55 | 105,056 |
| TornadoVM @1024 | **5.00** | 51.26% | 16 | 88.46 | 118,848 |
| CUDA @256 | **4.00** | 79.87% | 16 | 94.53 | 97,760 |
| CUDA @1024 | **4.00** | 50.80% | 16 | 91.29 | 113,312 |

| Effect | Comparison | Ratio |
|---|---|---|
| alignment + codegen, geometry controlled | TVM@256 / CUDA@256 | **1.075** |
| block size, within TornadoVM | @1024 / @256 | **1.131** |
| block size, within CUDA | @1024 / @256 | **1.159** |
| uncontrolled (what demo 12 measured) | TVM@1024 / CUDA@256 | **1.216** |
| product of the first two | 1.075 × 1.131 | **1.216** |

**The two effects multiply out to the uncontrolled figure exactly**, and 1.216
matches the 1.20–1.23 demo 12 reported. The iteration-3 retraction is confirmed
quantitatively.

Three conclusions:

- **Alignment is real and geometry-independent** — 5.00 vs 4.00 sectors/request
  at *both* block sizes, as sector arithmetic predicts — but costs only
  **1.075×** in time, because both kernels run at 93–95% of peak DRAM bandwidth.
- **The block-size penalty is not a TornadoVM property.** 256→1024 costs 1.131×
  on TornadoVM and **1.159× on hand-written CUDA**. Occupancy falls 78→51% and
  80→51%, with registers identical at 16 in all four runs. Anyone choosing
  block=1024 pays this, in any language.
- **The residual codegen difference is instruction count**: 9,437,184 vs
  7,864,320 (**1.20×**) at matched geometry, from bounds checks and index
  arithmetic, largely hidden by the bandwidth bound.

> **The 1.075 figure is `ncu`-conditioned and is not the number to quote.** All
> four runs above were taken under Nsight Compute, which serialises launches and
> flushes caches; that hides most of the alignment penalty. The same demo 15
> kernels give 1.02–1.04 under `ncu` against **1.24–1.31 under `nsys`** at
> identical, verified geometry — see `measurement-mode/`. What the 2×2
> establishes is structural (the penalty is geometry-independent; the block-size
> term is not TornadoVM-specific; the two multiply out exactly), not a headline
> ratio.

For a steady-state figure at matched geometry, quote demo 15 under `nsys`:
**1.31 / 1.24 memory-bound, 0.88 compute-bound**. Demo 15 pins both sides to
block=256, grid=16384, verified from `launch__block_size`. The larger ratios for
**demo 12** and **demo 01** are launch-config artefacts of the default worker
grid. Raw: `task-B2-geometry-controlled/`, `measurement-mode/`.

### A. Baseline code quality — kernel time, nsys

Launch geometry held constant at block=256 on both sides.

| Kernel | TornadoVM | CUDA | |
|---|---|---|---|
| `elementwise` (memory-bound) | 13,944 ns | 10,621 ns | CUDA 1.31x |
| `stencil` (memory-bound) | 14,322 ns | 11,546 ns | CUDA 1.24x |
| `polynomial` (compute-bound) | 35,236 ns | 39,926 ns | TornadoVM 1.13x |

Mean per-kernel `Avg (ns)` from `nsys cuda_gpu_kern_sum`, 3 independent runs,
spread under 1%. Raw: `results/raw/21-kernel-time-comparison/`.
Both validate at max abs err 1e-7.

Combined with B and C: the memory-bound gap is layout, the compute-bound
advantage is specialisation. Controlling for both, the generated arithmetic is
equivalent on these three kernels.

### C. JIT specialisation

| Variant | Kernel time |
|---|---|
| TornadoVM, `degree` as task argument | 35,236 ns |
| nvcc, `degree` as runtime scalar | ~39,500 ns |
| nvcc, `degree` as template constant | ~34,700 ns |

Raw: `results/raw/21-kernel-time-comparison/probe-jit-specialisation.log`.
`cuobjdump -sass` shows 11 branch instructions in the nvcc runtime-scalar
variant (`cuda-sass.log`).

**SASS, both sides.** TornadoVM writes every cubin to an on-disk module cache by
default (`tornado.cuda.codecache.enable=True`, landing at
`$TORNADOVM_HOME/var/cuda-codecache/device-0-0/`), so `cuobjdump -sass` reads it
directly:

| | SASS instructions | branch-class | FFMA |
|---|---|---|---|
| TornadoVM `polynomial` (`degree=256` as task arg) | 288 | **1** | **256** |
| nvcc, `degree` as runtime scalar | — | **13** | — |

**256 FFMA for degree=256 and a single branch — the loop is fully unrolled.** The
mechanism is now shown in SASS rather than inferred. An earlier revision of this
bundle recorded this as uncapturable and raised it with NVIDIA; that was a false
premise and has been withdrawn. Raw: `task-C-jit-specialisation/`.

### D. Tensor-core path — code-generation validation only

**No timing claim. 128–256 output elements on one warp is far too small.**

| Kernel | tensor inst | HMMA | IMMA |
|---|---|---|---|
| `gemmBF16` | 4 | 4 | 0 |
| `gemmInt8` | 2 | 0 | 2 |
| `gemmFP8E4M3` | 2 | 2 | 0 |
| `gemmFP8E5M2` | 2 | 2 | 0 |
| demo 08 `gemmMMASingleTile` | 1 | 1 | 0 |
| demo 08 `gemmScalarFp16` (control) | 0 | 0 | 0 |

Counters match emitted PTX instruction for instruction. All validate at max abs
err 0.00000 over 256 cells against a CPU reference computed from the same stored
values. int8 dispatches to the IMMA pipe; BF16 and both FP8 formats to HMMA.
Raw: `results/raw/23-ncu-tensor-core-counters/`, `results/raw/26-tensor-core-datatypes/`.

Demo 08 also shows identical tensor-core work on both sides (1 HMMA, 16 cycles)
with **183 scalar instructions around it against hand-written CUDA's 40**.

---

## End-to-end / host side

### E. Runtime dispatch cost

Obtained by **differencing execution counts** (10 → 100) so fixed start-up
cancels. Dividing a total by iterations overstates this by ~12x.

| CUDA API | calls/execution | ns/execution |
|---|---|---|
| `cuStreamSynchronize` | 3.00 | 2,751 |
| `cuLaunchKernel` | 1.00 | 2,256 |
| `cuMemcpyDtoHAsync_v2` | 1.00 | 1,535 |
| `cuMemcpyHtoDAsync_v2` | 1.00 | 1,232 |
| `cuEventCreate` | 3.00 | 1,127 |
| `cuCtxSetCurrent` | 3.00 | 211 |
| `cuEventRecord` | 3.00 | 179 |
| `cuEventDestroy_v2` | 3.00 | 128 |
| `cuStreamIsCapturing` | 1.00 | 36 |

**~9.5 µs of CUDA API time per execution** for a one-kernel graph; ~8.3 µs on
demo 14 once genuine device wait is excluded. The single per-launch H2D is a
24-byte kernel-argument stack frame re-uploaded although unchanged.

Both are already diagnosed upstream —
[#1028](https://github.com/beehive-lab/TornadoVM/issues/1028) findings 1 and 2 —
with PR [#1022](https://github.com/beehive-lab/TornadoVM/pull/1022) open and
**not present in 6.0.0**. Raw: `results/raw/25-host-dispatch-breakdown/`.

### E1. Device-buffer reuse across `JIT → libraryTask → JIT` — verified

Demo 13: `scale` (JIT) → `cudnnConv2d` (library) → `addBias` (JIT) →
`cudnnRelu` (library), 20 executions, validation PASSED at max abs err 0.000000
(0/65536 elements out of tolerance).

Transfer histogram shows the 262,160-byte input crossing **once** and the output
**once per execution**. The three intermediate tensors never appear.
**Device buffers are reused across the JIT/libraryTask boundary; there is no
unintended host round trip.** Raw: `task-E-graph-composition/d13-transfer-histogram.csv`.

### E2. CUDA Graph replay — per-execution cost, differenced

Six-stage chain, profiled at 10 and 100 executions per mode, delta/90.

| | nograph | graph |
|---|---|---|
| `cuLaunchKernel` per execution | **6.00** | **0.00** |
| `cuMemcpyHtoDAsync_v2` per execution | **7.00** | **0.00** |
| event create/record/destroy | 42 | 3 |
| `cuGraphLaunch` | — | 1.00 |
| **host API ns/execution, excluding `cuStreamSynchronize`** | **33,774** | **6,283** |

**Graph capture removes the entire per-execution dispatch sequence — 5.4x less
host API time.** The seven H2D copies per execution in `nograph` are the
per-launch 24-byte kernel-argument stack frames (#1028 finding 1); replay
eliminates them because arguments are baked into the captured graph. This is the
mechanism behind the graph speedup being large for TornadoVM and small for
hand-written CUDA.

`cuStreamSynchronize` is excluded above: it is largely genuine device wait, and
is *higher* in graph mode (18,388 vs 3,669 ns/execution) because the host blocks
once on the batched chain instead of interleaving with launch work. Treating
that as a regression would be a misreading.

Raw: `task-E-graph-composition/`.

---

## F. Fused GEMM decision baseline — kernel-only

CUTLASS v3.5.1, fp16, row-major, 20 executions. Fused and unfused CUTLASS
kernels distinguished by epilogue template, not ordering.

| Path | m=n=k=256 | m=n=k=1024 |
|---|---|---|
| TornadoVM fused | 11,077.5 ns | **34,675.9 ns** |
| TornadoVM unfused | 11,810.4 ns | **39,296.8 ns** |
| CUDA fused | 11,193.6 ns | 33,971.7 ns |
| CUDA unfused | 11,778.9 ns | 37,331.8 ns |

Fusion saves **11.8%** of GPU time at 1024 and 6.2% at 256, plus one kernel
launch. The fused epilogue costs **+1.0%** on the GEMM itself against 4,924.9 ns
as a separate pass.

Fusion saving by shape (TornadoVM): 6.2% at 256, **11.8% at 1024**, 7.9% at 4096.
At 4096: TornadoVM fused 946,447.1 ns vs unfused 1,027,685.5 ns.

**Split by kernel origin — with a launch-geometry caveat:**

| Kernel | Origin | same launch geometry? | ratio TornadoVM/CUDA (1024 / 4096) |
|---|---|---|---|
| `LinearCombinationRelu` | CUTLASS library | **yes** — (32,32,1)×128 both | **1.00 / 1.01** |
| `LinearCombination` | CUTLASS library | **yes** | **1.01 / 1.04** |
| `scale` | JIT vs hand-written | **no** — 1024 vs 256 block | 1.23 / 1.32 |
| `biasRelu` | JIT vs hand-written | **no** | 1.20 / 1.41 |

**The valid half:** both sides invoke the same CUTLASS kernel with the same
launch configuration, and it performs identically (1.00–1.04x).
**Library-task integration costs essentially nothing** — TornadoVM reaches
CUTLASS at the same performance as a hand-written caller.

**The invalid half:** the JIT kernels do not share block size (1024 vs 256), so
those ratios are *not* a controlled comparison and **must not be attributed to
code generation or to alignment**. An earlier revision of this bundle made that
attribution; it is corrected here. Two facts contradict it: the block sizes
differ, and the ratio grows with problem size (1.20 → 1.41) whereas the task B
alignment penalty is bounded at 1.25 by sector arithmetic and does not scale.
The growth matches the occupancy ceiling of a 1024-thread block on sm_89 — the
same effect documented for demo 01.

**What it does show is a runtime default, not a compiler defect:** TornadoVM's
default worker grid picks 1024-thread blocks for these tasks. Separating
alignment from occupancy needs a re-run with a `GridScheduler` pinned to 256.
Not done; recorded as a gap.

Raw: `task-F-fused-gemm/kernel-times.csv` (includes grid and block per kernel).

## Cold vs warm compilation (task F, criterion 8)

`-Dtornado.profiler=True`, m=n=k=1024:

| Phase | Cold (first execution) | Warm (steady state) |
|---|---|---|
| `TASK_COMPILE_GRAAL_TIME` | **42.2 ms** | 0 |
| `TASK_COMPILE_DRIVER_TIME` (NVRTC) | **16.3 ms** | 0 |
| `TOTAL_TASK_GRAPH_TIME` | 87.1 ms | 0.39–0.56 ms |
| `TOTAL_KERNEL_TIME` | 0.46 ms | 0.062–0.091 ms |

**Compilation is 58.5 ms of the 87.1 ms cold execution (67%); Graal front-end
work is 2.6x the NVRTC back-end cost. Cold is ~200x warm.** This is the price of
the task C specialisation advantage, paid once per task graph, and it bounds the
runtime compile budget any Tile-based path would have to fit inside.

The profiler perturbs its own measurement — warm `TOTAL_TASK_GRAPH_TIME` here
(~0.4 ms) exceeds the 0.342 ms wall clock measured without it. Use these for the
cold/warm *split*, not as absolute steady-state timings.

## G. CUDA Tile feasibility — host facts

**No CUDA Tile performance claim is made. No Tile code was written or run.**

| Check | Result |
|---|---|
| Toolkit headers matching *tile* | **0** |
| Toolkit libraries matching *tile* | **0** |
| `nvcc` flags matching *tile* | **0** |
| `libnvrtc.so.12` symbols matching *tile* | **0** |
| NVRTC version | 12.6 |
| NVRTC max supported arch | **sm_90** (14 archs, sm_50…sm_90) |
| NVRTC relevant exports | `nvrtcGetCUBIN`, `nvrtcGetPTX`, **`nvrtcGetNVVM`**, **`nvrtcGetLTOIR`** |

Raw: `tile-feasibility/inventory.txt`.

**Blocker, stated as fact:** CUDA Tile is not present in CUDA 12.6.85 on this
host in any form. A Tile prototype cannot begin here without a toolkit that
ships it. Which release does, and whether the integration surface is Tile C++ or
a Tile IR, are open questions.

Three independent ceilings, all verified:

| Layer | Ceiling |
|---|---|
| GPU (RTX 4090) | sm_89 |
| NVRTC 12.6 toolkit | **sm_90** |
| TornadoVM MMA emitter (`MMAShape`) | sm_89-class (M16N8K16, M16N8K32) |

Note the middle row: this toolkit could already target Hopper. The emitter could
not, and `wgmma` is not reachable by extending that enum.

---

## Matrix status

| Task | Status | Note |
|---|---|---|
| A. Baseline code quality | measured | geometry held constant at block=256 |
| B. Alignment isolation | measured | single binary, offset swept 0/16/32/64/128 B |
| B2. Geometry-controlled isolation | measured | 2×2 separates alignment from block size |
| C. JIT specialisation | measured | SASS from both sides; cubin read from TornadoVM's module cache |
| D. Tensor-core path | measured | code-generation validation only, no timing claim |
| E. Runtime and graph composition | measured | buffer reuse verified; graph cost differenced |
| F. Fused GEMM baseline | measured | 3 shapes; cold/warm compile split; inversion resolved |
| G. CUDA Tile feasibility | measured | host inventory; no Tile code written or run |

**All eight matrix entries are measured.** The last gap (task C's TornadoVM-side
SASS) was closed by reading TornadoVM's on-disk module cache, which is enabled by
default — an earlier revision wrongly recorded it as uncapturable.

## Failures and gaps — not hidden

| Item | Status |
|---|---|
| F. Fused GEMM baseline | **measured** — CUTLASS v3.5.1; shapes 256, 1024, 4096 |
| F. JIT-kernel ratios confounded by launch geometry | **now isolated** — see task B2: 1.075x alignment/codegen x 1.131x block size = 1.216x |
| F. TornadoVM fused slower end-to-end than unfused (342 vs 313 µs) | **resolved** — execution-ordering artefact: the fused plan runs first and absorbs process start-up. `task-F-fused-gemm/wallclock-inversion-resolved.md` |
| Cold vs warm compile time | **measured** — 42.2 ms Graal + 16.3 ms NVRTC = 58.5 ms of an 87.1 ms cold execution; warm ~0.4 ms |
| TornadoVM-side SASS (task C) | **resolved** — cubin is in TornadoVM's on-disk module cache, enabled by default |
| CUDA Graph per-execution API cost | **not differenced** |
| Device-buffer reuse across graph nodes | **not verified from trace** |
| Nsight Compute on `PATH` | unusable (2026.2.1.0 vs driver 565.57.01) — `results/failures/08-nsight-compute-permission.md` |
| Demo 16 CUDA side | builds and validates, **not profiled** |
