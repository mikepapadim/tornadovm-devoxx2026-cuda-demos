# NVIDIA compiler meeting — measurement plan and methodology

Evidence pack for a technical discussion with NVIDIA compiler / CUDA-toolchain
engineers. The objective is to establish the current CUDA baseline, **separate
code-generation effects from memory-layout and host-runtime effects**, and
assemble the facts needed to decide whether a narrowly scoped CUDA Tile
prototype is justified.

Not a benchmark. No headline speedup claims. Where a number is not measured on
this host, it is absent rather than estimated.

Bundle: [`results/nvidia-meeting/`](../results/nvidia-meeting/) —
`manifest.json` (machine-readable), `summary.md` (measured values only),
`open-questions.md`, `tile-feasibility/`, `env/provenance.txt`.

## System under test

Full capture: `results/nvidia-meeting/env/provenance.txt`.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, compute capability **8.9**, 24564 MiB, max SM clock 3105 MHz, max mem clock 10501 MHz, power limit 450 W, persistence mode enabled |
| Driver | 565.57.01 |
| CUDA toolkit / NVRTC | 12.6.85 (`cuda_12.6.r12.6/compiler.35059454_0`) |
| OS / kernel | Ubuntu 22.04.5 LTS, Linux 6.8.0-58-generic x86_64 |
| JDK | OpenJDK 25.0.2 |
| TornadoVM (measured) | **`6.0.0-jdk22plus-cuda`, SDKMAN release** |
| Nsight Compute | 2024.3.2.0 |
| Nsight Systems | 2024.5.1.113 |
| Demo repo commit | `f76b01fc67e1fb61c97973388155b3db6428b5ad` |

### Which TornadoVM is measured, and why it is not the local checkout

The local source checkout (`/home/michalis/TornadoVM`) is at
`5bbe99cb0`, on branch `perf/mma-shape-operand-validation`, **one commit behind
`origin/develop` (`e06a2e56f`), with three uncommitted files** under
`tornado-drivers/cuda`. It is therefore not a reproducible measurement baseline
and is used **for source reading only**.

**All measurements use the pristine SDKMAN release `6.0.0-jdk22plus-cuda`**,
which is versioned, redistributable and identical to what a third party would
install. Every existing capture in this repo used the same SDK, so old and new
evidence are directly comparable.

> A local `dist/` build exists in that checkout and contains uncommitted
> changes. It must not be used for measurement and is not referenced anywhere in
> this bundle.

## Profiler configuration

- Nsight Compute **must** be invoked as `/opt/nvidia/nsight-compute/2024.3.2/ncu`.
  The `ncu` on `PATH` resolves to 2026.2.1.0, which cannot connect to driver
  565.57.01 at all. See `results/failures/08-nsight-compute-permission.md`.
- Counter access requires `NVreg_RestrictProfilingToAdminUsers=0`
  (`/proc/driver/nvidia/params` reads `RmProfilingAdminOnly: 0`).
- Under `nsys` and `ncu`, invoke `$JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile`
  directly. The `tornado` launcher resolves a different JDK under those tools and
  fails with `UnsupportedClassVersionError`.

## Known confounders — read before interpreting any number

1. **`ncu` kernel times are not comparable to `nsys` or wall-clock times.** ncu
   serialises launches, flushes caches and does not allow clock boost; absolute
   durations run several times higher. Kernel-time claims use `nsys`. The two are
   kept as separate metrics and must never share an axis.
2. **Launch geometry must be held constant or the comparison is void.** Demo 15
   pins both implementations to 256-thread blocks for this reason. Demo 01 does
   **not** — TornadoVM defaults to 1024 threads/block there against the CUDA
   version's 256, which changes achieved occupancy independently of code
   quality. Any demo-01 occupancy figure is labelled as a launch-configuration
   artefact, not a code-generation result.
3. **Start-up cost cannot be separated from per-execution cost in a single
   trace.** Differencing two execution counts is required; a total divided by
   iterations overstates per-execution cost (it did so by ~12x on first attempt —
   recorded in `results/raw/25-host-dispatch-breakdown/MANIFEST.md`).
4. **One GPU, one machine, one architecture.** Results are never aggregated
   across architectures and no figure is presented as a general
   "TornadoVM vs CUDA" percentage.
5. **Tiny kernels are code-generation validation, not performance.** Demos 08 and
   16 compute 128–256 output elements on one warp. No timing claim is made from
   them, and they are labelled accordingly.
6. **Write-back is not fully captured inside a kernel window.** `dram__bytes_write`
   for short kernels reads below the bytes the kernel logically writes; ratios
   from it are stated as consistent-with, not proven.

## Benchmark matrix

Status: **measured** = captured on this host in this bundle or a cited batch;
**gap** = not measured, with the reason recorded.

### A. Baseline code quality — measured

TornadoVM CUDA JIT vs hand-written CUDA, identical problem size, identical launch
geometry (256-thread blocks), identical arithmetic including bounds checks, no
`-use_fast_math` on either side.

| Kernel | Character | Evidence |
|---|---|---|
| `elementwise` | memory-bound, 1 read + 1 write | `results/raw/21-kernel-time-comparison/`, `results/raw/22-ncu-alignment-counters/` |
| `stencil` | memory-bound, 3 reads + 1 write, neighbour access | same |
| `polynomial` | compute-bound, dependent FMA chain | same |
| `vectorAdd` | bandwidth-saturating | `results/raw/27-profiler-metrics/` |

Collected: correctness and tolerance, kernel duration (nsys), DRAM bytes,
sectors/request, global load/store sectors, instruction count, achieved
occupancy, achieved DRAM throughput, registers/thread, launch geometry.

```bash
# TornadoVM
nsys profile --trace=cuda -o tornado15 \
  $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . KernelTimeComparison 4194304 256 20
# hand-written CUDA
nvcc -arch=sm_89 -O3 -o cuda15 KernelTimeComparison.cu && nsys profile --trace=cuda -o cuda15 ./cuda15 4194304 256 20
nsys stats --report cuda_gpu_kern_sum --format csv <rep>.nsys-rep
```

### B. Alignment isolation — measured, extended for this bundle

One compiled kernel, one problem size, one launch geometry; **the only variable
is the byte offset of the base pointer**. If generated arithmetic were the
cause, changing an offset on the same binary could not reproduce the effect.

Offsets swept: 0 B, **16 B (TornadoVM's `FloatArray` payload offset)**, 32 B,
64 B, 128 B.

Source: [`results/nvidia-meeting/AlignmentSweep.cu`](../results/nvidia-meeting/AlignmentSweep.cu)

```bash
nvcc -arch=sm_89 -O3 -o alignment_sweep AlignmentSweep.cu
/opt/nvidia/nsight-compute/2024.3.2/ncu --csv \
  --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,... \
  ./alignment_sweep --counters
./alignment_sweep     # 3 warm-up + 30 timed reps, median reported
```

Acceptance: the offset sweep must reproduce the TornadoVM-vs-CUDA sector ratio
using a single binary. It does.

### C. JIT specialisation — measured

TornadoVM with a runtime-known task argument vs nvcc with a runtime scalar vs
nvcc with a compile-time template constant.
Evidence: `results/raw/21-kernel-time-comparison/probe-jit-specialisation.log`,
`cuda-sass.log`, `tornado-printkernel.log`.

**Gap:** SASS captured for the CUDA side (`cuobjdump -sass`, branch count) but
not for the TornadoVM side; TornadoVM's cubin is produced in-process by NVRTC and
is not written to disk by default. Recorded in `summary.md`.

### D. Tensor-core / CUDA primitive path — measured, validation only

MMA fp16 / BF16 / INT8 / FP8-e4m3 / FP8-e5m2, `cp.async`, shared memory, warp
shuffle. Numerical validation, emitted PTX confirmation, tensor-pipe counters,
scalar instruction counts.
Evidence: `results/raw/23-ncu-tensor-core-counters/`,
`results/raw/26-tensor-core-datatypes/`, `results/raw/24-ncu-demo14-counters/`.

**Explicitly marked code-generation validation only.** No timing headline.

### E. Runtime and graph composition — partially measured

Per-execution CUDA driver API cost by differencing execution counts; H2D/D2H
counts and sizes; synchronisation and event call counts; nsys timeline.
Evidence: `results/raw/25-host-dispatch-breakdown/`, `results/raw/19-*` (demo 13
`JIT → cuDNN → JIT`), `results/raw/11-integrated-showcase/`.

**Gap:** CUDA Graph replay and stream concurrency have wall-clock numbers from
earlier batches but no per-execution API-cost differencing, and device-buffer
reuse between graph nodes has not been verified from the trace.

### F. Fused GEMM decision baseline — gap

**Not measured.** Demo 12's hand-written CUDA side requires a CUTLASS checkout
(`CUTLASS_DIR`) that is not vendored in this repo. Only nsys kernel times for the
TornadoVM side exist (`results/raw/19-cutlass-cudnn-warp-demos/`). Cold vs warm
compilation time is not separated anywhere yet.

### G. CUDA Tile feasibility inventory — measured

Host inventory only; **no CUDA Tile implementation and no Tile performance
claim**. Evidence: `results/nvidia-meeting/tile-feasibility/inventory.txt`.

## Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| 1 | Every reported number traceable to a committed raw capture | met |
| 2 | Layout effect demonstrated with a single binary, varying only the offset | met (B) |
| 3 | Kernel-only and end-to-end figures never mixed | met |
| 4 | Per-execution cost obtained by differencing, not division | met (E) |
| 5 | Tensor-core claims backed by counters **and** emitted PTX | met (D) |
| 6 | Tile feasibility stated as host facts with command output | met (G) |
| 7 | Failures recorded, not hidden | met (`results/failures/`, F and C gaps) |
| 8 | Cold vs warm compile time separated | **not met** (F) |
| 9 | Device-buffer reuse across graph nodes verified from trace | **not met** (E) |
