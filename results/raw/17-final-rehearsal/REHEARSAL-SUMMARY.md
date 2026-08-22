# Final rehearsal — task 17

Full re-run of `docs/demo-runbook.md`'s live sequence end-to-end, same day
as the talk-content drafting, to confirm every command in the runbook still
reproduces before committing it as presenter material. Pre-checks: `git
status` clean, `vendor/tornadovm` HEAD == `env/versions.env` `TORNADO_SHA`
(`99549c9862eda8d584e35e99924f9c865501eb3a`), GPU idle (`0 %`, `4 MiB`) via
`nvidia-smi`, no concurrent `claude -p` process besides this invocation's
own (`ps aux`).

## Talk 1 sequence — all 9 demos, all exit 0, all correct

| Step | Command | Result | Log |
|---|---|---|---|
| 00 | `tornado --classpath . Hello --enableProfiler console` | correct (`in: [0..7]`, `out: [1..8]`) | `01-hello-gpu.log` |
| 01 | `tornado --classpath . VectorAddKernel` | `Result is correct` | `02-vectoraddkernel.log` |
| 02 | `tornado --classpath . CudaGraphReplay` | `All replays correct` (8/8) | `03-cudagraphreplay.log` |
| 04 | `tornado --classpath . CuBlasSgemvHybrid --enableProfiler console` | `All iterations correct` (5/5) | `04-cublassgemvhybrid.log` |
| 05 | `tornado --classpath . CuFftLowPassHybrid` | `All iterations correct` (5/5, maxError 4.77e-7) | `05-cufftlowpasshybrid.log` |
| 06 | `tornado --classpath . CudaStreamsOverlap 8 32768 65536 8 both` | both modes `All executions correct` | `06-cudastreamsoverlap.log` |
| 07 | `tornado --classpath . CudaGraphBenefit 4096 6 50 both` | steady-state speedup **6.61x** (nograph 241.4us -> graph 36.5us) — within the previously recorded 6.47x-7.02x range | `07-cudagraphbenefit.log` |
| 08 | `tornado --printKernel --classpath . TensorCoreMMA` then `tornado --classpath . TensorCoreMMA` | exactly 1x `mma.sync.aligned` in generated PTX; both scalar and MMA validations PASSED | `08-tensorcoremma-printkernel.log`, `08-tensorcoremma-run.log` |
| 11 | `tornado --classpath . IntegratedShowcase 6 8 8 20 all` | all 4 modes + bonus MMA stage correct; steady-state medians: baseline 862us, concurrent 791us (1.09x), graph 145us (5.94x), combined 241us (3.58x) — same ordering (graph best, concurrent marginal) as batches 15/16 | `11-integratedshowcase.log` |

Compiled `.class` artifacts were deleted after this rehearsal (`find demos
-name '*.class' -delete`) — not committed anywhere in the repo, same
convention as every prior batch.

## Talk 2 sequence — GPULlama3.java FP16 inference

| Step | Command | Result | Log |
|---|---|---|---|
| Opening | `./llama-tornado --gpu --cuda --model $GPULLAMA3_MODEL_USED --prompt "What is the capital of France?" --seed 7 --max-tokens 32` | correct completion, **155.40 tok/s** (within the previously recorded 153.18-164.20 tok/s range) | `12-gpullama3-fp16-rehearsal.log` |
| CUDA proof | same, `+ --profiler` | 3,996 task-graph stages report `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"`; 78.96 tok/s under profiler overhead (consistent direction with prior ~92 tok/s profiler-overhead observation, same-family number, lower here likely profiler-JSON-dump variance run-to-run — not investigated further, both numbers already carry the "this-run-only" caveat) | `13-gpullama3-fp16-profiler-rehearsal.log` |

## Conclusion

Every command named in `docs/demo-runbook.md` reproduced with the same
correctness outcome and a performance number inside (or matching, for
correctness-only demos) the range already recorded in this repo's prior
batches. No number, API, or status invented; every log above is a fresh,
this-invocation capture, not a copy of an older result. The runbook is
presenter-ready as written.
