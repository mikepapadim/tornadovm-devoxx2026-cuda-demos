# Talk 2 — Java LLM Inference with TornadoVM

Draft content. Every technical claim below carries an evidence tag
(**Observed** / **Source-backed** / **Documented** / **Hypothesis** /
**Blocked**) per `PLAN.md` §6 — cross-reference `docs/claims.md` for the full
evidence map. Live sequence and exact commands: `docs/demo-runbook.md`.
System under test: this repo's pinned TornadoVM CUDA build
(`TORNADO_SHA=99549c9862eda8d584e35e99924f9c865501eb3a`), GPULlama3.java
pinned at `GPULLAMA3_SHA=bbe42fdc8cd475bb6104cefa42118dd6e068538b`
(`main`, 2026-08-15), RTX 4090.

## Opening (2–3 min)

Local LLM inference in Java, on the GPU, without leaving the JVM — no
subprocess to a Python runtime, no ONNX export step. GPULlama3.java is the
practical case study: a real Llama-3 inference engine that runs its
forward pass as TornadoVM task-graphs on the CUDA backend.

**Live, first thing:** a real prompt, real completion, visible pause before
the first token. Tell the audience about the pause *before* running it —
"that's one-time JIT compilation of the task-graph variants, not the model
being slow" — so it reads as expected behavior, not a glitch.

## Narrative arc

1. **It builds and runs as documented — with one real integration wrinkle.**
   GPULlama3.java's own `make`/`mvnw install` succeeds out of the box
   (**Observed**, `results/raw/10-gpullama3/build.log`). But its `pom.xml`
   pins TornadoVM 5.0.0 from Maven Central, while this repo's pinned CUDA
   build is 5.2.1-jdk21-dev — running the documented-build jar against the
   pinned SDK fails with a `TornadoInternalError` about a lambda missing
   `writeReplace()` (**Observed failure**, root cause: whether a `TaskN`
   lambda gets a synthetic `writeReplace()` is decided by `javac` at compile
   time, based on the `tornado-api` release used for that compile — a real
   cross-version incompatibility, not a local misconfiguration). Fix: a
   build-property override (`-Dtornadovm.base.version=5.2.1
   -Djdk.version.suffix=-jdk21-dev`), source unmodified, per this study's own
   "never patch upstream in place" rule. Worth telling this story — it's the
   kind of integration friction every audience member doing this for real
   will hit.

2. **Minimal inference works, and the CUDA execution is provable, not just
   claimed.** `--gpu --cuda` produces correct, coherent completions
   (**Observed**, 4 reproductions). The `--profiler` flag's JSON confirms
   `"BACKEND": "CUDA"` per task-graph stage — because `llama-tornado`'s own
   "Detected TornadoVM backend: cuda" message is a static config-file read,
   not runtime proof, and this talk doesn't rest a GPU-execution claim on a
   log line that could be printed even if execution silently fell back to
   CPU.

3. **Quantization: two paths work today, one is explicitly unimplemented.**
   Source-level dispatch gates on `GGMLType` at two independent
   checkpoints — model load and forward-plan construction
   (**Source-backed**, `docs/quantization-paths.md` §1). FP16 and Q8_0 both
   work (**Observed**, profiler-confirmed CUDA execution, ~150–186 tok/s this
   GPU/this model). Legacy Q4_0 fails deterministically with
   `UnsupportedOperationException: Unsupported quantization format: 2`
   before any GPU work starts (**Observed**, matches the source-level
   prediction exactly — two independent guards agree). Frame this as "the
   codebase tells you what it doesn't support, loudly and early" rather than
   as a shortcoming — a clean `UnsupportedOperationException` at
   metadata-parse time beats a silent wrong-answer or a segfault deep in a
   kernel. K-quants (Q4_K/Q5_K/Q6_K) are **documented, not reproduced** — the
   source shows a dequant-to-Q8_0 path, but no K-quant GGUF file was
   available on this machine to actually run it. Say "documented, not yet
   verified here" if asked, not "works."

4. **Where the time actually goes.** Nsight Systems profiling
   (**Observed**, `results/raw/12-llm-profiling/PROFILING-SUMMARY.md`):
   dominant GPU kernel is the FFN gate/up projection
   (`fusedRmsNormFFNGateUp`, 39.0% of GPU kernel time) — matches the
   textbook expectation that FFN layers dominate transformer decode cost.
   Memory traffic is 99.1% host-to-device, consistent with small per-token
   transfers, not bulk weight reload — weights stay GPU-resident across the
   generation loop. Time-to-first-token (~756 ms) is separated explicitly
   from steady-state decode (~6.45 ms/token) because they have different
   causes (one-time JIT warm-up of ~148 task-graph variants vs. actual
   per-token compute) — cross-checked against the run's own reported tok/s
   within ~1%, not an isolated number pulled from one column of a CSV.
   Hardware-counter metrics (occupancy, tensor-pipe activity) remain
   **Blocked** on this machine, same `ERR_NVGPUCTRPERM` root cause as Talk 1.

5. **Framework integration: real, and honestly blocked today.** Both the
   Quarkus LangChain4j extension (`quarkus-langchain4j-gpu-llama3:1.13.0`,
   a real non-beta Quarkiverse release) and the upstream LangChain4j
   provider (`langchain4j-gpu-llama3:1.19.0-beta29` — **every** published
   version to date is a beta, despite "officially supported" framing in
   GPULlama3.java's own README) build and link against this repo's pinned
   CUDA jar via a direct dependency override (**Observed**). Both then hit
   the identical wall at runtime: this repo's CUDA build compiles
   GPULlama3.java with `--enable-preview` under JDK 21, and preview-flagged
   bytecode is loadable *only* by the exact JDK feature release that
   produced it — never older, and, by JVM spec, never newer either. Both
   integration modules need JDK 23+ just to load their own deployment
   classes. No single JVM process satisfies both requirements today
   (**Observed, root cause identified from actual class-file bytes, not
   guessed** — `unzip -p ... | xxd` on the real jars).
   A JDK-25 build of everything might resolve this (**Hypothesis, not
   attempted** — a non-trivial second TornadoVM CUDA rebuild, flagged as
   follow-up). Present this section as "here's exactly where the wall is and
   why," which is more useful to a Java audience evaluating this stack than
   a demo that silently avoids the integration story.

## Live-coding sequence

See `docs/demo-runbook.md` "Talk 2" for exact commands and fallbacks.
Order: opening inference run → `--profiler` CUDA proof → quantization table
(run Q8_0 live if time allows; Q4_0 failure live only if time/audience
allow, framed in advance as an intentional "watch it fail correctly" demo) →
optimization story (narrated from the pre-captured Nsight Systems summary,
not a live profiler capture) → integration story (narration only, zero live
commands — running either blocked demo live only produces a stack trace).

## Technical explanation (for the "how does this actually work" segment)

- GPULlama3.java's forward pass is expressed as TornadoVM task-graphs, one
  per transformer layer/stage shape — dispatch is gated per-request on the
  loaded model's `GGMLType`, resolved once at model-load time
  (**Source-backed**, `AbstractModelLoader.getModelQuantization`,
  `docs/quantization-paths.md` §1). This is why an unsupported quantization
  fails at load, before any kernel runs, rather than mid-generation.
- Weight tensors for FP16/Q8_0 models get dedicated TornadoVM execution-plan
  component classes (`tornadovm/plan/components/{fp16,q8_0}/`) — there is no
  generic "any quantization" kernel path; each supported format is its own
  concrete implementation (**Source-backed**, `docs/quantization-paths.md` §1).
- The ~756 ms time-to-first-token is JIT compilation of ~148 distinct
  task-graph variants (one per layer/stage combination actually exercised by
  the loaded model), not GPU compute — this is a one-time cost per process,
  not per token, which is why steady-state decode is two orders of
  magnitude faster per step (**Observed**, cross-checked against reported
  tok/s within ~1%, `results/raw/12-llm-profiling/PROFILING-SUMMARY.md` §6).
- The Quarkus/LangChain4j blocker is a JVM classfile-versioning constraint,
  not an application bug: preview-flagged class files (JVMS `minor_version`
  preview marker) are bound to the exact JDK feature release that produced
  them by specification, in both directions — this is why "just use a newer
  JDK" does not work here (**Source-backed**, verified against the actual
  compiled class file bytes, `docs/quarkus-langchain4j-integration.md` §3).

## Conclusion

A real Java LLM inference engine, genuinely running its forward pass on the
GPU via TornadoVM, with two working quantization paths and a clearly
understood boundary for a third. The interesting parts of this story for a
Java audience aren't the tokens/second number — they're the two version-pin
frictions (TornadoVM release mismatch, JDK preview-feature boundary) that
show up the moment you try to compose this with the rest of a real Java
stack, and how each was root-caused rather than worked around blindly.

## What to do if a demo fails live

Every command in this talk has a named pre-captured log in
`docs/demo-runbook.md` "Talk 2." Never run the Quarkus/LangChain4j demos
live expecting success — they are blocked by design of this evidence base,
not by presenter error; narrate from `docs/quarkus-langchain4j-integration.md`
§3's exact transcript instead.
