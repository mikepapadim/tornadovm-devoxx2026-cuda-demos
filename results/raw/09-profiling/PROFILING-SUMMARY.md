# Task 09 — Profiling summary (Nsight Systems + Nsight Compute)

Date: 2026-08-21. `vendor/tornadovm` HEAD `99549c9862eda8d584e35e99924f9c865501eb3a` (unchanged).
GPU: NVIDIA GeForce RTX 4090, driver `565.57.01`. `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total`
before every run: `0 %, 4 MiB, 24564 MiB` — no competing process on the device.

These are dedicated **profiled** runs, kept separate from the timed runs already
captured for each demo's own task (`results/raw/04-cublas-hybrid/`,
`05-cufft-hybrid/`, `07-cuda-graph-benefit/`, `08-tensor-core-mma/`). Workload
sizes here were chosen for a representative profiler trace, not for the
headline timing numbers reported in those tasks — do not merge the two.

## Tool versions used

- Nsight Systems: `nsys --version` → `NVIDIA Nsight Systems version 2024.5.1.113-245134619542v0` (`/usr/local/cuda-12.6/bin/nsys`).
- Nsight Compute (default on `PATH`): `ncu --version` → `Version 2026.2.1.0 (build 38283040) (public-release)` (`/usr/local/cuda-12.6/bin/ncu`, a version-selector wrapper resolving to `/opt/nvidia/nsight-compute/2026.2.1`).
- Nsight Compute (driver-era match, invoked directly): `/opt/nvidia/nsight-compute/2024.3.2/ncu --version` → `Version 2024.3.2.0 (build 34861637) (public-release)`.

## Demos profiled

| Demo | Class | Profiling workload | Correctness |
|---|---|---|---|
| `demos/04-cublas-hybrid` | `CuBlasSgemvHybrid` | `512 512 10` | 10/10 iterations correct |
| `demos/05-cufft-hybrid` | `CuFftLowPassHybrid` | `4096 16 20` | 20/20 iterations correct |
| `demos/07-cuda-graph-benefit` | `CudaGraphBenefit` | `4096 6 50 both` | all 300 stage-executions (50 nograph + 50 graph, 6 stages/exec) correct |
| `demos/08-tensor-core-mma` | `TensorCoreMMA` | default (single M16N8K16 tile) | both kernels (scalar + MMA) correct |

Exact profiling commands (run after `source vendor/tornadovm/setvars.sh` in the demo's directory,
after a clean `javac --release 21 --enable-preview` rebuild against the jars each demo's own README documents):

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt -o /tmp/nsys-04-cublas \
  tornado --classpath . CuBlasSgemvHybrid 512 512 10

nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt -o /tmp/nsys-05-cufft \
  tornado --classpath . CuFftLowPassHybrid 4096 16 20

nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt -o /tmp/nsys-07-graph \
  tornado --classpath . CudaGraphBenefit 4096 6 50 both

nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt -o /tmp/nsys-08-mma \
  tornado --classpath . TensorCoreMMA
```

Parsed summary reports extracted with:

```bash
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_kern_exec_sum \
  --format csv --output results/raw/09-profiling/nsys-<tag> /tmp/nsys-<tag>.nsys-rep
```

Raw artifacts (`.nsys-rep`, per-demo `*-run.log` stdout, and the four CSV reports per demo)
are committed in this directory. Report definitions: `cuda_api_sum` = CUDA driver/runtime
API call summary (launch overhead lives here, e.g. `cuLaunchKernel`/`cuGraphLaunch`);
`cuda_gpu_kern_sum` = per-kernel GPU execution time; `cuda_gpu_mem_time_sum` = H2D/D2H
copy time; `cuda_kern_exec_sum` = per-kernel launch (API), queue, and execution latency
(`AAvg`=API call time, `QAvg`=queue wait, `KAvg`=kernel exec time).

## Kernel duration (GPU-side, `cuda_gpu_kern_sum`)

| Demo | Kernel | Instances | Total time (ns) | Avg (ns) | Med (ns) |
|---|---|---|---|---|---|
| 04-cublas | `gemv2T_kernel_val<...>` (cuBLAS SGEMV) | 10 | 25,567 | 2,556.7 | 2,528.0 |
| 04-cublas | `scale` (JIT) | 10 | 21,985 | 2,198.5 | 2,144.5 |
| 04-cublas | `bias` (JIT) | 10 | 11,328 | 1,132.8 | 1,120.0 |
| 05-cufft | `vector_fft_c2r<4096,...>` (cuFFT C2R) | 20 | 60,895 | 3,044.8 | 3,040.0 |
| 05-cufft | `vector_fft_r2c<4096,...>` (cuFFT R2C) | 20 | 57,983 | 2,899.2 | 2,880.0 |
| 05-cufft | `scaleBy` (JIT) | 20 | 24,575 | 1,228.8 | 1,216.0 |
| 05-cufft | `lowPass` (JIT) | 20 | 21,727 | 1,086.3 | 1,087.5 |
| 07-graph | `stage` (JIT, both modes combined) | 300 | 360,062 | 1,200.2 | 1,184.0 |
| 08-mma | `gemmMMASingleTile` (Tensor Core `mma.sync`) | 1 | 2,656 | 2,656.0 | 2,656.0 |
| 08-mma | `gemmScalarFp16` (no-MMA reference) | 1 | 1,600 | 1,600.0 | 1,600.0 |

Note on 08-mma: single-tile workload is 1 warp / 1 launch each — a duration
comparison at this size reflects launch/sync overhead more than compute
throughput; it is not a performance claim (task 08 already restricts its
performance claim to generated-code evidence for this reason).

## Launch overhead and CUDA API summary (`cuda_api_sum`, `cuda_kern_exec_sum`)

- **04-cublas**: total trace dominated by one-time `cuCtxCreate_v2` (98.4 ms,
  89.1%) and `cuLibraryLoadData` (8.6 ms, 7.8%, 4 calls — cuBLAS handle/library
  init) — both one-time setup costs excluded from steady-state kernel timing.
  `cuLaunchKernel`: 20 calls, avg 5,130.8 ns. Per-kernel launch/queue/exec
  breakdown (`cuda_kern_exec_sum`, `scale` JIT task): API (`cuLaunchKernel`)
  avg 7,185.1 ns, Queue avg 1,169.9 ns, Kernel-exec avg 2,198.5 ns — i.e. the
  CPU-side API call to issue a JIT kernel costs more than 3x the GPU-side
  execution time for this workload size, consistent with demos 02/04/05/07's
  documented "compile once, reuse" pattern operating at microsecond scale.
- **05-cufft**: `cuCtxCreate_v2` 96.8 ms (96.5%, one-time). `cuLaunchKernel`:
  80 calls (4 kernels × 20 iterations), avg 2,834.1 ns. `cuMemcpyHtoD_v2`
  (blocking) 3 calls avg 226,612 ns dominated by one 633,697 ns outlier
  (first-iteration pinned-buffer setup); steady-state `cuMemcpyHtoDAsync_v2`
  avg 2,776.2 ns (60 calls).
- **07-graph** (`nograph`+`graph` mode, 300 total stage executions):
  `cuCtxCreate_v2` 99.3 ms (93.6%, one-time). `cuLaunchKernel`: 306 calls
  (nograph mode, 6 stages × 50 executions + 6 warm-up/JIT calls, avg
  2,026.1 ns) vs. `cuGraphLaunch`: 50 calls (graph mode, one call replays
  all 6 captured stages, avg 11,070.9 ns) — i.e. one `cuGraphLaunch` call
  replaces 6 `cuLaunchKernel` calls per execution; `cuStreamSynchronize`
  appears 306 times (avg 3,894.9 ns) tied to the per-execution validation
  copy-back. This is the CPU-API-side counterpart to the GPU-side steady-state
  speedup already measured in task 07 (`266.3us → 77.8us`, 3.42x, this run —
  see `nsys-07-graph-run.log`, consistent with task 07's own committed
  evidence of a multi-x speedup, exact multiplier varies by run as documented
  there).
- **08-mma**: `cuCtxCreate_v2` 110.6 ms (99.2%, one-time — dominates because
  the workload itself is tiny, a single warp). `cuLaunchKernel`: 2 calls, avg
  18,626.0 ns (one per kernel, no steady-state loop in this demo).

`cuCtxCreate_v2` (JVM/TornadoVM CUDA context + driver init) is consistently
the largest single entry (89–99% of the trace) in every demo profiled here —
expected, since these traces capture the whole JVM process including JIT
warmup, and is separated out here explicitly rather than left folded into
"total time" the way a naive read of the CSV's `Time (%)` column would
suggest.

## Memory transfer time (`cuda_gpu_mem_time_sum`)

| Demo | Op | Count | Total (ns) | Avg (ns) |
|---|---|---|---|---|
| 04-cublas | H2D | 40 | 149,681 (`cuMemcpyHtoDAsync_v2`, from `cuda_api_sum`) | 3,742.0 |
| 05-cufft | H2D | 63 | 609,143 | 9,668.9 (skewed by one 559,677 ns first-touch outlier; median 384 ns) |
| 05-cufft | D2H | 20 | 27,425 | 1,371.3 |

## Concurrency / synchronization

Cross-stream concurrency evidence for this API surface (`withIntraPlanConcurrency()`)
was already captured with a dedicated sequential-vs-concurrent A/B design and full
timeline evidence in task 06 (`results/raw/06-cuda-streams/nsys-{sequential,concurrent}.nsys-rep`,
`nsys-timeline-evidence.txt`) — not repeated here to avoid duplicate raw evidence;
that finding (1 stream / no overlap sequential vs. 4 streams / genuine overlap
concurrent) stands as this study's concurrency evidence.

## GPU utilization, occupancy, memory throughput (%), instruction mix, tensor-pipe metrics: BLOCKED

These require Nsight Compute (`ncu`) hardware performance counters. Nsight
Systems (`nsys`, used above) does not expose occupancy, SM/tensor-pipe
activity percentages, or instruction mix — only API/kernel/memcpy timing and
concurrency from the CUDA driver's own event stream.

Re-verified on 2026-08-21 (same finding as `results/failures/08-nsight-compute-permission.md`,
originally recorded 2026-08-20 — re-run today, not assumed to still hold):

1. Default `ncu` on `PATH` (`2026.2.1.0`) against `demos/08-tensor-core-mma`:
   ```
   ncu --target-processes all -k "regex:^gemmMMASingleTile$" --launch-count 1 \
     --metrics gpu__time_duration.sum tornado --classpath . TensorCoreMMA
   ==ERROR== Nsight Compute failed to connect to the CUDA driver (stub libcuda.so[.1] on path?).
   ==ERROR== The application returned an error code (1).
   ```
   Log: `ncu-2026.2.1.0-connect-error.log`. Same driver/toolkit-era mismatch
   as batch 08 (`2026.2.1.0` cannot connect to driver `565.57.01`).
2. Driver-era-matching `/opt/nvidia/nsight-compute/2024.3.2/ncu` (`2024.3.2.0`)
   against the same target: connects to the process successfully, then:
   ```
   ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
   NVIDIA GPU Performance Counters on the target device 0.
   ```
   Log: `ncu-2024.3.2.0-nvgpuctrperm-error.log`. Same `NVreg_RestrictProfilingToAdminUsers=1`
   restriction as batch 08; `sudo -n true` still requires a password in this
   unattended run — no privilege escalation attempted, no kernel-module
   parameter changed (system-wide, needs reboot, out of scope per CLAUDE.md's
   reversible-local-action bar).

No GPU utilization/occupancy/memory-throughput-%/instruction-mix/tensor-pipe
number is claimed anywhere in this task's output. If GPU performance-counter
access becomes available in a future environment, the two commands above
(swap `/opt/nvidia/nsight-compute/2024.3.2/ncu` for whichever install matches
the driver at that time) are ready to rerun against any of the four demos
profiled here.

## What this does and does not establish

- **Observed** (this task, `nsys`, all 4 demos): per-kernel GPU execution
  duration, CUDA API call counts/timings (launch overhead), H2D/D2H memcpy
  timing, one-time context/library-init cost isolated from steady-state
  kernel activity.
- **Observed** (task 06, `nsys`, referenced not repeated): multi-stream
  concurrency timeline (1 vs. 4 streams, serial vs. overlapping).
- **Blocked** (this task and task 08, `ncu`, both installed versions):
  GPU utilization %, occupancy, memory throughput %, instruction mix,
  tensor-pipe activity/instruction-count counters — environment-level
  permission/version issue, not a TornadoVM limitation; reproduced on a
  trivial standalone CUDA C program in task 08, confirming it is unrelated
  to TornadoVM's JNI layer.
