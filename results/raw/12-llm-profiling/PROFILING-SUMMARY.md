# Task 12 — Profile GPULlama3.java inference on TornadoVM CUDA

Date: 2026-08-21. Machine/versions per `env/versions.env`
(`TORNADO_SHA=99549c9862eda8d584e35e99924f9c865501eb3a`,
`GPULLAMA3_SHA=bbe42fdc8cd475bb6104cefa42118dd6e068538b`, RTX 4090, driver
`565.57.01`). Nsight Systems `2024.5.1.113-245134619542v0`. Model:
`beehive-llama-3.2-1b-instruct-fp16.gguf` (same file used in task 10/11), FP16
path (only path verified end-to-end working in task 11; Q8_0/Q4_0 not
re-profiled here — out of this task's scope, would be a straightforward
follow-up using the same method).

All numbers below are Observed, from this run, this model, this GPU — not a
general claim. Every number has its exact source file/command next to it.

## 1. Commands

Exact run configuration: `--gpu --cuda --model beehive-llama-3.2-1b-instruct-fp16.gguf --temperature 0.1 --top-p 0.95 --seed 7 --max-tokens 64 --stream true --echo false --prompt "Explain what a GPU kernel is in one paragraph." --instruct`, captured via `llama-tornado --show-command` (`show-command.log`) and executed directly (bypassing the Python wrapper, which has no profiler-prefix option) as the exact `java ...` command it prints (`java-command.txt`).

1. Sanity run, no profiler (`sanity-run.log`): confirms correct/coherent output and baseline `achieved tok/s` before any profiler overhead.
2. Nsight Systems trace (`nsys-12-llm-fp16-run.log`, `nsys-12-llm-fp16.nsys-rep`):
   ```
   nsys profile --trace=cuda,nvtx,osrt \
     --output=results/raw/12-llm-profiling/nsys-12-llm-fp16 --force-overwrite=true \
     bash -c "$(cat java-command.txt)"
   ```
3. Parsed reports (`nsys-12-llm-fp16-stats.log` + 4 CSVs, `nsys-12-llm-fp16-trace-report.log` + 1 CSV):
   ```
   nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_kern_exec_sum --format csv --output nsys-12-llm-fp16 nsys-12-llm-fp16.nsys-rep
   nsys stats --report cuda_gpu_trace --format csv --output nsys-12-llm-fp16 nsys-12-llm-fp16.nsys-rep
   ```
4. Nsight Compute re-check against the actual LLM binary (`ncu-2024.3.2.0-llm-attempt.log`):
   ```
   /opt/nvidia/nsight-compute/2024.3.2/ncu --target-processes all \
     -k "regex:^fusedRmsNormFFNGateUp$" --launch-count 1 \
     --metrics gpu__time_duration.sum bash -c "$(cat java-command.txt)"
   ```

`nvidia-smi` confirmed the GPU idle (`0 %`, `4 MiB`) before every run in this batch.

## 2. Correctness and throughput (no profiler vs. Nsight Systems overhead)

| Run | tok/s (achieved, this run) | Output |
|---|---|---|
| Sanity (no profiler) | 162.62 | coherent, correct completion |
| Under `nsys profile` | 153.54 | coherent, correct completion (same prompt, same seed → near-identical text, same direction as expected for FP16 non-greedy sampling with fixed seed) |
| Under `ncu` (1 kernel, `--launch-count 1`) | 111.73 | coherent, correct completion |

Overhead direction matches every prior task/demo's profiler-overhead note
(profiling always slows the run; `nsys` trace overhead here is modest,
~6%; `ncu`'s per-launch instrumentation is heavier even for a single
launch, ~31% — consistent with `ncu`'s reputation for far higher per-kernel
overhead than `nsys`).

## 3. Dominant GPU kernels (`nsys-12-llm-fp16_cuda_gpu_kern_sum.csv`)

| Kernel | Time % | Total (ns) | Instances | Avg (ns) |
|---|---|---|---|---|
| `fusedRmsNormFFNGateUp` | 39.0 | 87,150,688 | 1040 | 83,798.7 |
| `matrixVectorGenericWithResidual` | 22.9 | 51,155,827 | 2080 | 24,594.1 |
| `matrixVectorGeneric` | 16.1 | 35,986,551 | 65 | 553,639.2 |
| `processHeadsFlashAttention` | 11.5 | 25,711,388 | 1040 | 24,722.5 |
| `fusedQKVMatmulX` | 7.1 | 15,803,215 | 1040 | 15,195.4 |
| `reductionOneBlockWithLayer` | 1.7 | 3,746,484 | 2145 | 1,746.6 |
| `ropeRotationWithCacheCopy` | 1.0 | 2,211,127 | 1040 | 2,126.1 |
| `mapContextWithQuantize` | 0.6 | 1,386,022 | 1040 | 1,332.7 |
| `mapContextWithQuantizeLogits` | 0.0 | 85,953 | 65 | 1,322.4 |
| `convertFP16toFP32` | 0.0 | 85,084 | 65 | 1,309.0 |

**Interpretation (presenter-friendly):** the FFN gate/up projection
(`fusedRmsNormFFNGateUp`) is the single most expensive GPU kernel by total
time, taking almost 2× the next-largest kernel — expected for a decoder
FFN block, which has a larger matmul than attention's Q/K/V or output
projections for this model size. `1040 = 65 × 16`: 65 forward passes (1
prompt + 64 generated tokens, matching `--max-tokens 64`) × 16 per-pass
instances, matching this model's per-layer TornadoVM task-graph
construction (`AbstractTransformerLayerTaskGraphs.java:43-45`, one
task-graph instantiated per `config.numberOfLayers()`) — i.e. 16
transformer layers, derived from the observed instance-count ratio and
confirmed against the source, not asserted from external model-card
knowledge. `matrixVectorGenericWithResidual` at `2080 = 65 × 32` (2 per
layer — attention-output and FFN-down projections). `matrixVectorGeneric`
at exactly 65 instances (once per forward pass) is the final
output/logits projection — much larger per-call time (553.6 µs avg, the
single longest-running kernel type) since it projects to the full
vocabulary, but low total-time share because it only runs once per token
rather than once per layer.

## 4. CUDA API overhead / launch cost (`nsys-12-llm-fp16_cuda_api_sum.csv`)

| API | Time % | Total (ns) | Calls | Avg (ns) |
|---|---|---|---|---|
| `cuMemHostRegister_v2` | 53.6 | 576,663,529 | 162 | 3,559,651.4 |
| `cuStreamSynchronize` | 25.6 | 275,165,253 | 2405 | 114,413.8 |
| `cuCtxCreate_v2` | 9.0 | 96,931,515 | 1 | 96,931,515.0 |
| `cuMemHostUnregister` | 4.5 | 48,063,118 | 162 | 296,685.9 |
| `cuMemAlloc_v2` | 2.6 | 27,498,094 | 198 | 138,879.3 |
| `cuLaunchKernel` | 1.6 | 16,915,419 | 9620 | 1,758.4 |
| `cuMemcpyHtoDAsync_v2` | 1.3 | 14,469,542 | 10129 | 1,428.5 |
| `cuModuleLoadDataEx` | 0.7 | 7,253,130 | 148 | 49,007.6 |

**Interpretation:** the "Time %" column here is dominated by one-time
setup, not steady-state decode cost — `cuCtxCreate_v2` (a single 96.9 ms
call, CUDA context creation) and `cuMemHostRegister_v2`/`cuMemHostUnregister`
(162 calls each, pinning host memory for the model's weight tensors so
they can be DMA'd to the device — a load-time cost, not a per-token cost)
together account for ~67% of all captured CUDA-API time, because the
trace spans the whole JVM run (JIT warm-up + weight loading + inference),
the same "don't mis-read one-time setup as compute cost" caveat task 09
raised for the non-LLM demos. The steady-state, per-token-relevant rows
are `cuLaunchKernel` (9620 calls, 1.76 µs avg dispatch overhead — small
relative to the kernels' own execution time above) and
`cuMemcpyHtoDAsync_v2` (10129 calls, 1.43 µs avg — small per-call
transfers, consistent with per-token KV-cache/activation updates rather
than bulk weight transfer). `cuModuleLoadDataEx` (148 calls, 49.0 µs avg,
7.25 ms total) is TornadoVM JIT-compiling/loading PTX modules for each
distinct task-graph — a compile-time cost, amortized once per task-graph
shape, not repeated per token (148 ≈ the number of distinct compiled
task-graph/kernel variants for this model+backend combination, not the
number of forward passes).

## 5. Memory behavior (`nsys-12-llm-fp16_cuda_gpu_mem_time_sum.csv`)

| Operation | Time % | Total (ns) | Count | Avg (ns) |
|---|---|---|---|---|
| Host-to-Device memcpy | 99.1 | 140,467,127 | 10129 | 13,867.8 |
| Device-to-Host memcpy | 0.9 | 1,339,457 | 65 | 20,607.0 |
| memset | 0.0 | 2,464 | 2 | 1,232.0 |

H2D traffic (10129 copies) vastly outnumbers D2H (65 — exactly one per
forward pass, the sampled/decoded token result copied back to the host).
This asymmetry is expected for autoregressive decode: each step re-sends
small per-layer inputs/activations to the device but only reads back a
single scalar/token id per step.

## 6. Warm-up vs. steady-state decode (from `nsys-12-llm-fp16_cuda_gpu_trace.csv`, per-launch timestamps)

Using `matrixVectorGeneric` (the final-logits kernel, exactly 65
instances = 1 prompt pass + 64 decode steps) as a per-forward-pass marker:

- Trace kernel/memcpy activity span: 1,176,118,212 ns (~1.18 s) from first
  captured GPU op to last.
- First GPU kernel launch (`convertFP16toFP32`) at +42,297,427 ns
  (~42.3 ms) after the first captured op — CUDA context/module/JIT setup
  before any kernel runs.
- First `matrixVectorGeneric` (end of prompt-processing forward pass) at
  +755,564,416 ns (~755.6 ms) — this includes one-time JIT compilation of
  every distinct task-graph shape (the 148 `cuModuleLoadDataEx` calls
  above happen mostly here), not just prompt-token compute.
- Gap from prompt pass → first generated token: 13,130,556 ns (~13.1 ms).
- Gap from token 1 → token 2: 6,770,366 ns (~6.8 ms).
- Steady-state per-token gaps (tokens 3-64, n=62):
  min 5,007,359 ns, max 20,777,818 ns, avg 6,453,092 ns (~6.45 ms/token).
- Cross-check: avg steady-state gap (6.45 ms/token) ≈ 1000/153.54 =
  6.51 ms/token from this same profiled run's reported `achieved tok/s`
  (§2) — the two independent measurements (Nsight Systems kernel
  timestamps vs. the application's own end-to-end timer) agree within
  ~1%.
- The dominant kernel's (`fusedRmsNormFFNGateUp`) own per-launch duration
  does **not** show a JIT warm-up effect: first-pass-of-16 avg 81,030.1 ns
  vs. all-later-launches avg 83,842.0 ns — i.e. once a task-graph is
  compiled, its GPU execution time is stable from the very first launch;
  the ~755 ms "time to first token" cost above is JIT/setup latency
  sitting *before* GPU kernel execution, not slower early kernels.

**Presenter takeaway:** separate "time to first token" (~756 ms here,
dominated by one-time JIT compilation of ~148 task-graph variants plus
CUDA context setup) from steady-state decode (~6.45 ms/token, ~155 tok/s)
— the two have very different causes and a live demo audience should be
told to expect the first-token pause.

## 7. Occupancy / GPU utilization % / tensor-core activity — BLOCKED

Same root cause as `results/failures/08-nsight-compute-permission.md` and
task 09's re-verification, **re-verified a third time here against the
actual LLM binary** rather than assumed: `/opt/nvidia/nsight-compute/2024.3.2/ncu`
connects to the running `java` process (`==PROF== Connected to process
683680`) but immediately fails with `ERR_NVGPUCTRPERM` — the user does not
have permission to access NVIDIA GPU performance counters on this
machine (`NVreg_RestrictProfilingToAdminUsers=1`, unchanged since task 08).
Log: `ncu-2024.3.2.0-llm-attempt.log`. No occupancy, GPU-utilization-%,
memory-throughput-%, instruction-mix, or tensor-pipe-activity number is
claimed anywhere in this file. FP16 GEMV/attention kernels on this model
do not use `mma.sync` Tensor Core instructions (that is demo 08's
dedicated, separately-verified topic via `--printKernel`, not re-checked
here) — no tensor-core-activity claim is made for this LLM path either
way without hardware-counter evidence.

## 8. What would close this gap

If GPU performance-counter access becomes available in a future
environment (root/sudo, or `NVreg_RestrictProfilingToAdminUsers=0`), the
exact `ncu` command in §1.4 can be re-run with the full metric set from
`results/failures/08-nsight-compute-permission.md` against
`fusedRmsNormFFNGateUp`, `matrixVectorGeneric`,
`processHeadsFlashAttention`, and `fusedQKVMatmulX` (the four dominant
kernels from §3) to add occupancy/utilization/throughput numbers without
redoing anything else in this file.
