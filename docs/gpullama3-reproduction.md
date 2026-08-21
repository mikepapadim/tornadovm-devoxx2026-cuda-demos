# GPULlama3.java — current-state reproduction (Track B0/B1/B2)

Status: **Observed** unless marked otherwise. Environment: this repo's pinned TornadoVM CUDA build (`env/versions.env`, `TORNADO_SHA=99549c9862eda8d584e35e99924f9c865501eb3a`), RTX 4090, driver 565.57.01, CUDA 12.6.85, JDK 21.0.2. GPULlama3.java pinned at `GPULLAMA3_SHA=bbe42fdc8cd475bb6104cefa42118dd6e068538b` (`main`, 2026-08-15) — see `env/versions.env`.

## 1. Clone

```bash
git clone https://github.com/beehive-lab/GPULlama3.java.git vendor/GPULlama3.java
```

Cloned into `vendor/GPULlama3.java` (gitignored, upstream checkout — not committed in place, same convention as `vendor/tornadovm`).

## 2. Build

Documented instructions (`README.md` §Install & build, `Makefile`):

```bash
source vendor/tornadovm/setvars.sh   # this repo's pinned TornadoVM env, NOT sdkman
source vendor/GPULlama3.java/set_paths
make    # == ./mvnw install -DskipTests
```

This **builds successfully as documented** (`results/raw/10-gpullama3/build.log`, `BUILD SUCCESS`) and produces `target/gpu-llama3-1.0.0-jdk21.jar` (a shaded/uber jar).

### Blocker found and worked around: compile-time TornadoVM version pin does not match this repo's pinned build

`pom.xml` hard-codes `tornadovm.base.version=5.0.0` (comment: "CUDA backend is only available after 5.0.0 TornadoVM version"), so the documented build pulls `tornado-api`/`tornado-runtime` **5.0.0-jdk21 from Maven Central**, not this repo's pinned `5.2.1-jdk21-dev` build. `llama-tornado` runs the resulting jar's classes with `--module-path` pointed at `$TORNADOVM_HOME/share/java/tornado` (this repo's pinned 5.2.1-jdk21-dev SDK), so the **compiled classes** and the **runtime TornadoVM modules** are two different TornadoVM releases.

Running the documented build against this repo's pinned SDK failed **every time** with:

```
Exception in thread "main" uk.ac.manchester.tornado.api.exceptions.TornadoInternalError:
Kernel entry org.beehive.gpullama3.tornadovm.layers.Activation$$Lambda/... has no writeReplace():
this task lambda was compiled against a tornado-api release whose TornadoFunctions.TaskN
interfaces were not Serializable. Recompile the application against this TornadoVM's tornado-api.
```

Full trace: `results/raw/10-gpullama3/inference-run.log`.

**Root cause** (source-backed): whether a `TaskN` lambda gets a synthetic `writeReplace()` (needed by `TaskUtils.resolveViaSerializedLambda`, `tornado-runtime`) is decided by javac at the point the lambda expression is *compiled*, based on whether the target functional interface (`TornadoFunctions.TaskN`) extended `Serializable` in the `tornado-api` jar used for that compile. Maven Central's `5.0.0-jdk21` `tornado-api` predates the change that makes it Serializable-required by this repo's `5.2.1-jdk21-dev` `tornado-runtime` — a real cross-version incompatibility, not a local misconfiguration.

**Workaround used** (source is unmodified — build-property override only, per `CLAUDE.md`'s "never modify the upstream checkout in place; use scripts/patches"):

```bash
source vendor/tornadovm/setvars.sh
cd vendor/GPULlama3.java && source ./set_paths
./mvnw clean install -DskipTests \
  -Dtornadovm.base.version=5.2.1 -Djdk.version.suffix=-jdk21-dev
```

This resolves `tornado-api`/`tornado-runtime` `5.2.1-jdk21-dev` from the local `~/.m2` (installed there by this repo's task 00 `make BACKEND=cuda`, i.e. the exact pinned-SHA build, not just a same-numbered release from elsewhere) and produces `target/gpu-llama3-1.0.0-jdk21-dev.jar`. `mvn clean` is required — a non-clean `install` after only changing `-D` properties leaves stale `.class` files (Maven's compiler plugin only checks source-file mtimes, not dependency changes) and silently reproduces the same failure (`results/raw/10-gpullama3/inference-run-pinned-tornado-api.log`, same stack trace, confirmed as a real footgun during this task).

Logs: `results/raw/10-gpullama3/build-pinned-tornado-api-clean.log` (`BUILD SUCCESS`, `Including io.github.beehive-lab:tornado-api:jar:5.2.1-jdk21-dev`).

**Note for a future run.py glob fix**: `llama-tornado`'s `_find_llama_jar()` globs `target/gpu-llama3-*.jar` and picks the lexicographically-last match. `gpu-llama3-1.0.0-jdk21.jar` sorts *after* `gpu-llama3-1.0.0-jdk21-dev.jar` (`.` > `-` in ASCII), so if both jars exist in `target/` the script silently picks the wrong (5.0.0-linked) one. This run deleted the stray `-jdk21.jar`/`original-*.jar` from a prior default build before testing — not a code change, just build-artifact hygiene, documented here so a future invocation doesn't get bitten blind.

## 3. Minimal inference — works

Model: `beehive-llama-3.2-1b-instruct-fp16.gguf` (2,479,591,200 bytes), the exact model named in the upstream README quickstart. Pre-existing on this machine at `/home/michalis/test_install/GPULlama3.java/beehive-llama-3.2-1b-instruct-fp16.gguf` (outside this repo — not downloaded by this task, not committed; path recorded in `env/versions.env`).

```bash
source vendor/tornadovm/setvars.sh
cd vendor/GPULlama3.java && source ./set_paths
./llama-tornado --gpu --cuda \
  --model /home/michalis/test_install/GPULlama3.java/beehive-llama-3.2-1b-instruct-fp16.gguf \
  --prompt "What is the capital of France?" --seed 7 --max-tokens 32
```

Result (`results/raw/10-gpullama3/inference-run-reverify.log`):

```
The capital of France is Paris.

==== Performance Metrics ====
achieved tok/s: 153.18. Tokens: 25, seconds: 0.16
```

Ran three times across this task (`inference-run-pinned-clean.log`, `inference-run-profiler.log`, `inference-run-reverify.log`, different prompts/seeds/token counts) — correct, coherent output every time, all on the CUDA backend.

**CUDA-execution evidence, not just a claimed flag**: `--profiler` run (`results/raw/10-gpullama3/inference-run-profiler.log`) shows every TornadoVM task-graph stage reporting `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"` in its profiler JSON — e.g. `logits.rms_apply_fp16` stage, `TASK_KERNEL_TIME` in microseconds. `llama-tornado`'s own `Detected TornadoVM backend: cuda (from .../etc/tornado.backend)` message is a static file read, not execution evidence on its own; the profiler JSON is the actual runtime confirmation.

Throughput observed this run: ~150-155 tok/s (`--gpu --cuda`, no `--profiler`) and ~92 tok/s with `--profiler` enabled (extra per-kernel JSON dump overhead, same direction as demos 04/05/07/08's profiler-overhead notes) for the 1B FP16 model, single RTX 4090, 32-64 generated tokens. **This-run/this-model/this-GPU numbers only** — not a general benchmark claim (same convention as `demos/06-cuda-streams`, `demos/07-cuda-graph-benefit`).

## 4. Exact reproducible command (`--show-command`)

`results/raw/10-gpullama3/show-command.log` captures the full `java` invocation `llama-tornado` builds under the hood (module-path, `--add-modules`, `-Dtornado.*` properties, `-cp target/gpu-llama3-1.0.0-jdk21-dev.jar org.beehive.gpullama3.LlamaApp ...`) — usable as a `java @...`-equivalent reproducibility record without needing the Python launcher, per this repo's `docs/run-conventions.md` convention (no `@arg-file` is generated by GPULlama3.java itself, so the full command is captured verbatim instead).

## 5. JBang path — not verified

`which jbang` → exit 1 on this machine (same finding as every prior demo task, 00-09). README's `jbang gpullama3@beehive-lab -m model.gguf -p "..."` and `jbang LlamaTornadoCli.java -m model.gguf --interactive` shapes are documented but untested; do not claim they work here.

## 6. Prerequisite discrepancy (Documented, not Observed)

README states "GCC/G++ 13+ — to build TornadoVM's native components" as a GPULlama3.java prerequisite. This repo's pinned environment has GCC 11.4.0 (`env/versions.env`), and TornadoVM's native components were already built successfully with it in task 00. GPULlama3.java's own Maven build does not itself invoke a C/C++ compiler (pure Java + shading) — the GCC prerequisite applies to building the TornadoVM SDK it links against, not to GPULlama3.java's own build step. Not independently tested against GCC 13, since TornadoVM was already built and pinned before this task started; flagged here as a documentation nuance, not a reproduced blocker.

## Acceptance

Current build/run instructions are captured (§2-3, with the real cross-repo version-pin blocker and its exact workaround documented rather than silently sidestepped), and minimal inference works: `./llama-tornado --gpu --cuda` produces correct, coherent, CUDA-profiler-confirmed output against this repo's pinned TornadoVM build, reproduced across three separate runs.

## 7. Quantized paths (FP16 / Q8_0 / Q4_0) — see `docs/quantization-paths.md`

Task 11 probes which weight-quantization formats TornadoVM execution actually supports on this
pinned SHA: FP16 and Q8_0 are both working (source-backed dispatch in `ForwardPlanFactory.java` plus
independently reproduced inference runs); legacy Q4_0 is blocked with a deterministic
`UnsupportedOperationException` at model-load time, matching the source-level prediction exactly;
K-quants (Q4_K/Q5_K/Q6_K, dequantized to Q8_0 at load) are documented from source but not
independently reproduced (no test file available on this machine). Full detail, exact commands, and
evidence log paths in `docs/quantization-paths.md`.
