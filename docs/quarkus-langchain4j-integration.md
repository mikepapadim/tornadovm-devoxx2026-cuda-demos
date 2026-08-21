# Quarkus & LangChain4j integration — probe (Track B5/B6)

Status: as marked per claim. Environment: this repo's pinned TornadoVM CUDA build (`env/versions.env`,
`TORNADO_SHA=99549c9862eda8d584e35e99924f9c865501eb3a`), RTX 4090, driver 565.57.01, CUDA 12.6.85.
GPULlama3.java pinned at `GPULLAMA3_SHA=bbe42fdc8cd475bb6104cefa42118dd6e068538b`, rebuilt locally as
`gpu-llama3-1.0.0-jdk21-dev` (see `docs/gpullama3-reproduction.md`). Probe date: 2026-08-21.

## Summary

Both the Quarkus LangChain4j extension and the upstream LangChain4j provider for GPULlama3.java are
real, actively maintained, Maven-Central-published integrations — not vaporware. Both **build and
link successfully** against this repo's pinned CUDA GPULlama3.java artifact once the transitive
`gpu-llama3` dependency is overridden to our local build. Both are **blocked at model-load time** by
the same root cause: a JDK-version split that cannot be satisfied within a single JVM process.

| Integration | Artifact (current, 2026-08-21) | Build vs pinned CUDA build | Run |
|---|---|---|---|
| Quarkus | `io.quarkiverse.langchain4j:quarkus-langchain4j-gpu-llama3:1.13.0` | Observed: builds (JDK 25 required) | Blocked (same root cause) |
| LangChain4j | `dev.langchain4j:langchain4j-gpu-llama3:1.19.0-beta29` | Observed: builds (JDK 25 required) | Blocked (same root cause) |

## 1. Current published versions (Source-backed — Maven Central, checked live)

```bash
curl -s https://repo1.maven.org/maven2/dev/langchain4j/langchain4j-gpu-llama3/maven-metadata.xml
curl -s https://repo1.maven.org/maven2/io/quarkiverse/langchain4j/quarkus-langchain4j-gpu-llama3/maven-metadata.xml
```

- `dev.langchain4j:langchain4j-gpu-llama3` latest is `1.19.0-beta29` (2026-08-14). **Every version ever
  published to this artifact is a `-betaN` version** — despite the GPULlama3.java README's badge
  ("LangChain4j 1.7.1+") and prose ("Since LangChain4j v1.7.1 ... officially supported model
  provider"), there has never been a non-beta release on Maven Central. The "official" framing refers
  to the module living in the `langchain4j/langchain4j` reactor, not to release stability.
- `io.quarkiverse.langchain4j:quarkus-langchain4j-gpu-llama3` latest is `1.13.0` (2026-08-18) — a
  proper, non-beta Quarkiverse extension release.
- `quarkus-langchain4j-gpu-llama3:1.13.0`'s parent POM pins
  `<gpu-llama3.version>1.0.0-jdk21</gpu-llama3.version>` (`1.0.0-jdk25` under its `jdk25` profile) —
  the **exact same GPULlama3.java release line** (`1.0.0-jdk21`/`-jdk25`) that this repo's pinned SHA
  builds as `gpu-llama3-1.0.0-jdk21-dev`. `langchain4j-gpu-llama3:1.19.0-beta29` instead pins the much
  older `gpu-llama3:0.4.0-jdk25`.

## 2. Demos built

- `demos/09-quarkus-langchain4j-gpullama3/` — minimal Quarkus app: one `@RegisterAiService` interface
  (`GpuExplainerService`, fixed deterministic prompt "In exactly one sentence, explain what a GPU
  kernel is.") + a `@QuarkusMain` CLI runner. `pom.xml` imports `quarkus-bom:3.33.2` and
  `quarkus-langchain4j-bom:1.13.0` (the versions the real `quarkus-langchain4j-parent:1.13.0` POM
  itself declares), depends on `quarkus-langchain4j-gpu-llama3:1.13.0`, and **directly declares**
  `io.github.beehive-lab:gpu-llama3:1.0.0-jdk21-dev` — Maven dependency mediation ("nearest wins": a
  directly declared dependency always beats a transitively declared one) makes this override the
  extension's own transitive `1.0.0-jdk21` pin. `application.properties` points
  `quarkus.langchain4j.gpu-llama3.chat-model.model-name` at `unsloth/Llama-3.2-1B-Instruct-GGUF`
  (F16), which the extension's `GPULlama3ModelRegistry` resolved against an **already-cached** local
  copy at `/home/michalis/.langchain4j/models/unsloth_Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-F16.gguf`
  (pre-existing on this machine from earlier manual work, not downloaded by this task; same model
  family as `GPULLAMA3_MODEL_USED` in `env/versions.env`).
- `demos/10-langchain4j-gpullama3/` — minimal standalone (no Quarkus) Java program calling the
  README's own documented pattern: `GPULlama3ChatModel.builder().modelPath(Path).temperature(0.0)
  .topP(0.9).maxTokens(64).onGPU(Boolean.TRUE).build()` then `.chat(String)`, against
  `GPULLAMA3_MODEL_USED` (the FP16 model from `env/versions.env`). Same `gpu-llama3` override
  strategy as above (direct dependency beats the transitive `0.4.0-jdk25`).

## 3. Blocked — Observed root cause (both integrations, same failure class)

**Reproduce (Quarkus):**
```bash
cd demos/09-quarkus-langchain4j-gpullama3
source ../../vendor/tornadovm/setvars.sh   # JDK 21.0.2, this repo's pinned env
mvn -q clean package -DskipTests
```
Fails with `UnsupportedClassVersionError: ... LangChain4jGPULlama3FixedRuntimeConfig has been
compiled by a more recent version of the Java Runtime (class file version 67.0), this version of the
Java Runtime only recognizes class file versions up to 65.0` — full log:
`results/raw/13-quarkus-langchain4j/quarkus-build-jdk21-fail.log`.

**Root cause, part 1 (Source-backed, verified by inspecting the actual class files):** the
`quarkus-langchain4j-gpu-llama3-deployment:1.13.0` module (and `langchain4j-gpu-llama3:1.19.0-beta29`,
class file version `69.0` = Java 25) require **JDK 23+** to even load their build/deployment classes.
This repo's pinned environment is JDK 21.0.2 (`env/versions.env`), so `mvn package` cannot run at all
under the pinned JDK.

**Switching the build/run JVM to JDK 25** (`~/.sdkman/candidates/java/25.0.2-open`, available on this
machine but *not* this repo's pinned JDK) lets both demos build and, for Quarkus, boot successfully —
log: `results/raw/13-quarkus-langchain4j/quarkus-build-jdk25-ok.log`. The Quarkus app starts, CDI/ArC
wiring resolves, and the boot log confirms
`Installed features: [cdi, langchain4j, langchain4j-gpu-llama3, qute, smallrye-context-propagation, vertx]`
and correctly resolves the cached local model path — this part of the integration (config, DI,
AiService proxy generation, model registry/cache lookup) is genuinely **Observed working** against
the current code.

Both then fail identically at the moment GPULlama3.java's `ModelLoader` class is actually loaded:

```
java.lang.UnsupportedClassVersionError: org/beehive/gpullama3/model/loader/ModelLoader
(class file version 65.65535) was compiled with preview features that are unsupported.
This version of the Java Runtime only recognizes preview features for class file version 69.65535
```

Full logs: `results/raw/13-quarkus-langchain4j/quarkus-run-jdk25-blocked.log`,
`results/raw/13-quarkus-langchain4j/langchain4j-standalone-jdk25-blocked.log`.

**Root cause, part 2 (Source-backed, verified with `unzip -p ... | xxd` on the actual jars):** this
repo's pinned CUDA build of GPULlama3.java (`gpu-llama3-1.0.0-jdk21-dev`, from task 10's `mvn ...
-Dtornadovm.base.version=5.2.1 -Djdk.version.suffix=-jdk21-dev` workaround) was compiled with
`--enable-preview` under JDK 21 — its class files carry `major=65 (Java 21), minor=0xFFFF` (the JVMS
preview-feature marker). By JVM specification, a preview-flagged class file can **only** be loaded by
the exact JDK feature release that produced it, run with `--enable-preview` — never by an older JDK,
and, critically, **never by a newer one either** (this is intentional: preview features carry no
compatibility guarantee across releases). Both current integration modules need JDK 23+/25 just to
load their own classes. **These two requirements cannot both be satisfied by one JVM process** — there
is no single `java` invocation that can run this repo's pinned-CUDA GPULlama3.java build together with
the current Quarkus or LangChain4j GPULlama3 integration.

Plain `java -jar`/`java -cp` runs of both demos under the pinned JDK 21.0.2 (no JDK switch) also failed,
but with a different, unresolved symptom: the process exited immediately with status 1 and **zero**
stdout/stderr output (confirmed via `strace -f`; a self-test `SIGSEGV`/`rt_sigreturn` pair appears
early and is JVM-internal CPU-feature-probing noise, not a crash — the JVM's own "VM Thread" then
performs normal shutdown and calls `exit_group(1)`). This anomaly is recorded but not explained
further; it does not change the conclusion above, since the JDK-25 run already gives a complete,
attributable failure and the JDK-21 path is blocked earlier anyway (Quarkus's own deployment classes
require JDK 23+ to load, so JDK 21 cannot get past `mvn package`).

## 4. Hypothesis — not verified this batch

`quarkus-langchain4j-gpu-llama3:1.13.0`'s own `jdk25` Maven profile pins `gpu-llama3:1.0.0-jdk25`,
and GPULlama3.java's `pom.xml` supports building that exact artifact. The plausible *supported*
combination is GPULlama3.java `-jdk25` + a TornadoVM CUDA backend also built for JDK 25 + a JDK 25
runtime throughout. This repo's pinned TornadoVM CUDA build (task 00) targets JDK 21 only; building a
second, JDK-25 TornadoVM CUDA backend from the same pinned SHA to test this hypothesis is a
non-trivial rebuild, out of scope for this probe. Flagged as a candidate follow-up task, not attempted.

## 5. Reference material (not the basis of the results above)

Pre-existing local checkouts on this machine — `/home/michalis/jcon/quarkus-langchain4j` (a
`quarkiverse/quarkus-langchain4j` clone) and `/home/michalis/jcon/gpullama3-quarkus-langchain4j-demo`
— were inspected read-only to learn the current config-property surface
(`quarkus.langchain4j.gpu-llama3.*`) and API shape before writing the demos above. They were not
built or run as part of this probe; they predate the versions probed here (`999-SNAPSHOT` /
`gpu-llama3:0.2.2`, built Nov 2025) and are unrelated to this repo's pinned SHAs.
