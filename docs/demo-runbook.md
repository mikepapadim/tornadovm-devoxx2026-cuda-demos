# Demo runbook — TornadoVM CUDA demos

Exact live-demo sequence for both talks: what to type, what should appear on
screen, what to say, and what to do if a stage fails. Built from
`docs/profiling-quickstart.md` (verified command reference) and every
demo's own `README.md` "Fallback if the live demo fails" section — this file
sequences them into a talk-length run-of-show, it does not replace them.

Environment for the Talk 1 / Track A commands below: `env/versions.env` —
TornadoVM `6.0.0-jdk22plus-cuda` (SDKMAN release), JDK 25.0.2, RTX 4090,
driver 565.57.01, CUDA 12.6.85.

Talk 2 / Track B (GPULlama3.java) was **not** migrated to 6.0.0 and still runs
against the earlier source-built pin (`vendor/tornadovm` @
`99549c9862eda8d584e35e99924f9c865501eb3a`, JDK 21.0.2). Its sections below
are marked; do not mix the two setups in one shell.

## Pre-talk setup (do this before walking on stage, not live)

```bash
cd /path/to/tornadovm-devoxx2026-cuda-demos
source scripts/setup-env.sh
echo "$TORNADOVM_HOME"                                   # must print a path
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
tornado --devices                                        # must show exactly one CUDA device
bash scripts/run-all-demos.sh                            # all 9 demos, both run paths, must end 27/27
```

Running `scripts/run-all-demos.sh` once before the talk is the cheapest
insurance there is: it compiles every demo and runs it both via `tornado` and
via `java @argfile`, and exits non-zero if anything is broken.

Expect `0 %, 4 MiB, 24564 MiB` from `nvidia-smi`. If the GPU is not idle,
another process is using it — every number in this repo was captured on an
idle GPU; do not trust a busy-GPU number, kill the competing process first.

Have two terminal tabs ready: one in the repo root (for `demos/`), one
`cd`'d into `vendor/GPULlama3.java` with `source ./set_paths` already run
(Talk 2 only).

Have `results/raw/` open in a file browser or a third tab as the "if this
doesn't reproduce live" fallback — every step below names its exact
pre-captured log.

---

## Talk 1 — TornadoVM Hybrid API: Java + NVIDIA CUDA Libraries

### Opening — prove it's really the GPU

```bash
cd demos/00-hello-gpu
tornado --classpath . Hello --enableProfiler console
```

**Say:** "One `@Parallel` Java method, no JNI, no handwritten kernel code."
**Point at:** the profiler JSON's `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"`.
This is the cheapest, fastest way to prove GPU execution — use it before any
other claim in the talk.

**If it fails:** re-run `tornado --devices` first — if it doesn't show
exactly one CUDA device, the environment is broken, not the demo; narrate
from `results/raw/02-hello-kernel/hello-run*.log` instead.

### Show the generated CUDA source

```bash
cd ../01-first-cuda-kernel
tornado --printKernel --classpath . VectorAddKernel
```

**Say:** "This is the literal `extern "C" __global__ void` CUDA source the
Java method compiled to — inspectable, not a black box."
**Fallback:** `results/raw/02-hello-kernel/vectoraddkernel-run.log`.

### CUDA runtime API from Java — graph capture/replay correctness

```bash
cd ../02-cuda-runtime-api
tornado --classpath . CudaGraphReplay
```

**Say:** "`TornadoExecutionPlan.withCUDAGraph()` — real `cuStreamBeginCapture`/
`cuGraphLaunch` under the hood, 8 replays with mutated inputs, each validated
against a CPU reference."
**Fallback:** `results/raw/03-cuda-runtime-api/cudagraphreplay-run.log`.

### cuBLAS — vendor library + JIT kernel, one graph, shared buffers

```bash
cd ../04-cublas-hybrid
tornado --classpath . CuBlasSgemvHybrid --enableProfiler console
```

**Say:** "Three stages, one `TaskGraph`: JIT `scale` → cuBLAS `sgemv` →
JIT `bias`. The point isn't speed, it's that a hand-written kernel and a
vendor-optimized library call share device buffers with zero host
round-trips in between — check every stage reports the same `DEVICE`."
**Fallback:** `results/raw/04-cublas-hybrid/cublassgemvhybrid-run.log`.

### cuFFT — same pattern, a 4-stage pipeline

```bash
cd ../05-cufft-hybrid
tornado --classpath . CuFftLowPassHybrid
```

**Say:** "cuFFT forward (R2C) → JIT low-pass → cuFFT inverse (C2R) → JIT
normalize — a GPU-resident filter, one graph, four stage types interleaved."
**Do not pass `n=65536`** — known CPU-fallback bug, use the default/small `n`
from the README. **Fallback:** `results/raw/05-cufft-hybrid/cufftlowpasshybrid-run.log`.

### CUDA streams — real concurrent execution, not just correctness

```bash
cd ../06-cuda-streams
tornado --classpath . CudaStreamsOverlap 8 32768 65536 8 both
```

**Say:** "8 independent pipelines. Sequential mode: 1 CUDA stream,
back-to-back. Concurrent mode: 4-stream pool, genuinely overlapping kernel
windows — this is a Nsight Systems timeline claim, not a wall-clock number
I'm promising you." Show the pre-captured timeline if a projector/terminal
context switch is too slow live: `results/raw/06-cuda-streams/nsys-timeline-evidence.txt`.
**Caveat to state out loud:** concurrency is not a universal speedup —
demo 11 later in this run shows the same mechanism giving 0.78x–1.21x on a
launch-overhead-bound workload. Say that now so it isn't a surprise later.
**Fallback:** `results/raw/06-cuda-streams/cudastreamsoverlap-run.log`.

### CUDA graph replay — the actual speedup number

```bash
cd ../07-cuda-graph-benefit
tornado --classpath . CudaGraphBenefit 4096 6 50 both
```

**Say:** "Same 6-stage graph, 50 executions, `nograph` vs. `graph`. On this
machine, this run: 6.5–7x steady-state speedup — one `cuGraphLaunch` replaces
6 `cuLaunchKernel` calls per execution." **State explicitly:** "this number is
this-machine/this-workload, not a general TornadoVM claim" — three separate
runs landed at 6.47x/6.58x/7.02x, quote the range, not one cherry-picked
number. **Fallback:** `results/raw/07-cuda-graph-benefit/cudagraphbenefit-run.log`.

### Tensor Core — generated-code proof, not a benchmark

```bash
cd ../08-tensor-core-mma
tornado --printKernel --classpath . TensorCoreMMA
```

**Say:** "One warp, one `M16N8K16` fp16 tile, exactly one `mma.sync.aligned`
PTX instruction — next to a scalar reference kernel from the same compile
with zero `mma.sync` occurrences." **If asked about occupancy/utilization %:**
say up front that Nsight Compute hardware counters are blocked on this
machine (`ERR_NVGPUCTRPERM`, driver permission default, not a TornadoVM
limitation) — do not claim a number, point to
`results/failures/08-nsight-compute-permission.md` if pressed.
**Fallback:** `results/raw/08-tensor-core-mma/tensorcoremma-printkernel.log`.

### Closer — everything at once

```bash
cd ../11-integrated-showcase
tornado --classpath . IntegratedShowcase 6 8 8 20 all
```

**Say:** "6 chains, each JIT kernel + cuBLAS, one Tensor Core bonus stage,
run four ways: baseline, concurrent streams, CUDA graph replay, and both
stacked together — every mode validated against the same closed-form CPU
reference." **Recovery order if the live run has any trouble** (cheapest to
riskiest, per the demo's own README): `graph` mode first (most reliable
speedup) → `baseline` (always works, just slower) → `mma` bonus stage alone
→ `concurrent`/`combined` (most workload-sensitive, fine to skip under time
pressure). **Fallback:** `results/raw/11-integrated-showcase/showcase-run-fullverbose.log`.

**Talk 1 total live-demo time budget:** ~8 commands, each under a few
seconds of wall-clock plus narration. If running short on time, cut
02-cuda-runtime-api and 05-cufft-hybrid first (04-cublas-hybrid already
carries the "shared buffers, one graph" point); never cut the opening
(00) or closer (11).

---

## Talk 2 — Java LLM Inference with TornadoVM

### Setup (do before the audience is watching, ~756 ms first-token pause is expected)

```bash
# Track B only — the 5.2.1 source-built pin, NOT the SDKMAN 6.0.0 SDK.
# Use a separate terminal tab from the Talk 1 demos.
cd vendor/GPULlama3.java
source ../tornadovm/setvars.sh
source ./set_paths
```

### Opening — a real end-to-end inference run

```bash
./llama-tornado --gpu --cuda \
  --model "$GPULLAMA3_MODEL_USED" \
  --prompt "What is the capital of France?" --seed 7 --max-tokens 32
```

**Say up front, before pressing enter:** "There will be a visible pause
before the first token — that's one-time JIT compilation of ~148 task-graph
variants, about 756 ms on this machine, not the model being slow. Steady-
state after that is roughly 150–165 tokens/second on this GPU."
**Expect:** `The capital of France is Paris.` and an `achieved tok/s` line.
**Fallback:** `results/raw/10-gpullama3/inference-run-reverify.log`.

### Prove it's really CUDA, not a silent CPU fallback

```bash
./llama-tornado --gpu --cuda --profiler \
  --model "$GPULLAMA3_MODEL_USED" \
  --prompt "What is the capital of France?" --seed 7 --max-tokens 32
```

**Say:** "Every TornadoVM task-graph stage in the profiler JSON reports
`"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"`." **Note:**
throughput drops under `--profiler` (extra per-kernel JSON dump overhead,
~90 tok/s observed here) — say that's profiler overhead, not real
degradation, so the audience doesn't misread it.
**Fallback:** `results/raw/10-gpullama3/inference-run-profiler.log`.

### Quantization story — what actually works today

**Say, do not run all three live unless time allows — narrate from the
table if short on time:**

| Path | Status | Command swap |
|---|---|---|
| FP16 | works, ~150–165 tok/s this GPU | `--model "$GPULLAMA3_MODEL_USED"` |
| Q8_0 | works, ~186 tok/s this GPU | `--model "$GPULLAMA3_MODEL_Q8_0"` |
| Q4_0 (legacy) | **blocked**, deterministic `UnsupportedOperationException` at model-load, before any GPU work | `--model "$GPULLAMA3_MODEL_Q4_0"` |

If running the Q4_0 failure live: **frame it before running it** — "this is
going to fail, on purpose, to show you what 'not yet implemented' looks like
in this codebase, not a bug I hit by accident." It fails in under a second
(GGUF metadata parse, before GPU init), safe to run live.
**Fallback logs:** `results/raw/11-quantization/{fp16,q8_0,q4_0}-inference-run*.log`.

### Optimization story (if time / audience wants "why is it fast")

**Say, with `results/raw/12-llm-profiling/PROFILING-SUMMARY.md` open, not a
live profiler run (Nsight Systems trace capture takes longer than a talk
slot affords):** "The dominant GPU kernel during decode is the FFN gate/up
projection, ~39% of GPU time. Memory traffic is 99% host-to-device — small
per-token transfers, not bulk weight movement, because weights stay
resident on the GPU across tokens." **If asked about occupancy/tensor-core
utilization %:** blocked on this machine, same `ERR_NVGPUCTRPERM` as Talk 1 —
do not claim a number.

### Integration story (Quarkus / LangChain4j) — narrate, do not attempt live

**Do not run `demos/09-quarkus-langchain4j-gpullama3` or
`demos/10-langchain4j-gpullama3` live** — both are blocked by a real,
interesting root cause, and running them live only produces a stack trace.

**Say:** "Both the Quarkus extension and the LangChain4j provider build and
link against our pinned CUDA jar. Both hit the same wall at runtime: our
GPULlama3.java build uses Java preview features under JDK 21, and preview
bytecode can only be loaded by the exact JDK release that compiled it — never
older, never newer. The integration modules themselves need JDK 23+. There
was no single JVM that satisfied both." If asked whether TornadoVM 6.0.0
changes this: its `jdk22plus` SDK is a non-preview build, so the
preview-bytecode half of the blocker is gone in principle — but Track B has
not been rebuilt or re-tested on 6.0.0, so say that it is now unblocked *in
principle* and untested, not that it works. Show the exact error transcript
from `docs/quarkus-langchain4j-integration.md` §3 if asked for proof — it's
a real, reproduced failure, not a guess.

**Talk 2 total live-demo time budget:** 2 required commands (opening +
profiler proof) plus the quantization table (run Q8_0 live if time allows,
Q4_0 failure only if the audience seems engaged). Integration story is
narration-only, zero live commands.

---

## Global recovery rules (both talks)

1. `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader`
   before anything — a busy GPU invalidates every number below it.
2. `tornado --devices` must show exactly one CUDA device. If not, stop
   running live commands and narrate from `results/raw/` for the rest of the
   section.
3. Watch stdout for a red `[Bailout] Running the sequential implementation`
   line — silent CPU fallback. The demo still prints a correct result but is
   not on the GPU; never claim GPU execution from output correctness alone.
4. Every number quoted in either talk is scoped "this run, this machine" —
   never restate a demo's speedup as a general TornadoVM claim (see
   `docs/claims.md`, "No-claim list").
5. If a whole section is unrecoverable mid-talk, skip to the next demo and
   narrate the skipped one from its `results/raw/` log — every demo above has
   one named explicitly.
