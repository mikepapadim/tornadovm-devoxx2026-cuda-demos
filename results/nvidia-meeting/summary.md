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

**Gap:** SASS for the TornadoVM side is not captured — its cubin is produced
in-process by NVRTC and not written to disk. Task C is therefore asymmetric.
Raised as an open question.

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

**Gaps:** CUDA Graph replay has wall-clock numbers from earlier batches but no
per-execution API differencing; device-buffer reuse across graph nodes is not
verified from the trace.

---

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

## Failures and gaps — not hidden

| Item | Status |
|---|---|
| F. Fused GEMM baseline | **not measured** — demo 12's CUDA side needs a CUTLASS checkout not vendored here |
| Cold vs warm compile time | **not separated anywhere** in this bundle |
| TornadoVM-side SASS (task C) | **not capturable** — cubin produced in-process by NVRTC |
| CUDA Graph per-execution API cost | **not differenced** |
| Device-buffer reuse across graph nodes | **not verified from trace** |
| Nsight Compute on `PATH` | unusable (2026.2.1.0 vs driver 565.57.01) — `results/failures/08-nsight-compute-permission.md` |
| Demo 16 CUDA side | builds and validates, **not profiled** |
