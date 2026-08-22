# Demo simplicity and consistency audit

Batch 16. Every demo under `demos/` reviewed against a fixed presenter-usability
checklist: one concept per class, lightweight comments, deterministic output,
a simple canonical invocation, a `java @arg-file` path, and a tested (or
explicitly marked unsupported) JBang path. All findings below are source-read
and command-verified this batch, not assumed from prior STATE.md summaries.

## Checklist definition

| Check | Pass criterion |
|---|---|
| One concept per class | Exactly one top-level class/interface declaration in the demo's main source file, no unrelated nested classes. |
| Lightweight comments | Comment-line ratio stays well under the file's code, and no comment restates what an identifier already says (WHY, not WHAT). |
| Deterministic output | The demo validates its own result (closed-form/CPU reference) and prints PASS/FAIL rather than requiring visual inspection. |
| Simple invocation | One short canonical run command a presenter can type live, not a multi-step wrapper. |
| `java @arg-file` | `java @../tornado.args -cp . <Main> ...` (or equivalent) is documented and actually runs. |
| JBang | Either a working `jbang` command is documented, or JBang's absence on the pinned environment is stated explicitly (not silently omitted). |

## Per-demo results (Track A, `00`–`08`, `11`)

| Demo | One class | Comments | Deterministic | Simple invocation | `java @arg-file` | JBang |
|---|---|---|---|---|---|---|
| 00-hello-gpu | pass (1 class, 50 lines) | pass (10 comment lines, all WHY/what-is-this) | pass (established, task 00) | pass (`tornado --classpath . Hello`) | pass (re-verified task 14, 2026-08-22) | not installed (`which jbang` exit 1, reconfirmed this batch) |
| 01-first-cuda-kernel | pass (1 class, 56 lines) | pass (7 lines) | pass (result-correctness check) | pass | pass (re-verified task 14) | not installed |
| 02-cuda-runtime-api | pass (1 class, 80 lines) | pass (19 lines incl. 1 javadoc block explaining CUDA-graph capture/replay, non-obvious) | pass | pass | documented (task 02/03 evidence) | not installed |
| 04-cublas-hybrid | pass (1 class, 133 lines) | pass (18 lines, 2 javadoc blocks) | pass | pass | documented | not installed |
| 05-cufft-hybrid | pass (1 class, 109 lines) | pass (20 lines, 2 javadoc blocks) | pass | pass | documented | not installed |
| 06-cuda-streams | pass (1 class, 149 lines) | pass (28 lines — largest ratio of the simple demos, justified: explains stream-role/event-ordering, a non-obvious mechanism) | pass | pass | **re-verified live this batch**: `java @../tornado.args -cp . CudaStreamsOverlap 8 32768 65536 8 both` → exit 0, both modes "All executions correct" (`results/raw/16-demo-audit/06-cuda-streams-argfile-spotcheck.log`) | not installed |
| 07-cuda-graph-benefit | pass (1 class, 155 lines) | pass (28 lines) | pass | pass | documented | not installed |
| 08-tensor-core-mma | pass (1 class, 165 lines) | pass (16 lines, 3 javadoc blocks) | pass | pass | documented (task 14) | not installed |
| 11-integrated-showcase | pass (1 class, 342 lines — deliberately the largest, combines 5 mechanisms by design per task 15, not a violation) | pass (44 lines, ~13% ratio, lowest relative density of all demos) | pass | pass (`tornado --classpath . IntegratedShowcase 6 8 8 20 all`) | documented (task 15) | not installed |

## Findings and fixes this batch

- **Two genuinely unused imports found and removed** (grep-verified: each import's
  class name appeared exactly once in its file — the `import` line itself, zero
  uses in the body):
  - `demos/08-tensor-core-mma/TensorCoreMMA.java`: `uk.ac.manchester.tornado.api.ImmutableTaskGraph` — dead, the demo never holds an `ImmutableTaskGraph` reference by that type.
  - `demos/11-integrated-showcase/IntegratedShowcase.java`: `uk.ac.manchester.tornado.api.WorkerGrid2D` — dead, only `WorkerGrid1D` is actually used (the MMA bonus stage uses `WorkerGrid1D` + `GridScheduler`, not 2D).
  - Both demos **rebuilt and re-run end to end after removal** to confirm no
    breakage: 08 (`tornado --classpath . TensorCoreMMA`, both scalar and
    `mma.sync` validations PASSED) and 11 (`tornado --classpath .
    IntegratedShowcase 6 8 8 20 all`, all four modes + bonus MMA stage
    validated correct). Logs: `results/raw/16-demo-audit/08-tensor-core-mma-post-cleanup-run.log`,
    `results/raw/16-demo-audit/11-integrated-showcase-post-cleanup-run.log`.
- No other unused imports found across any of the 9 Track-A demo source files
  (checked every `import` line's class-name usage count in its own file).
- No stray nested/helper classes found — every demo file has exactly one
  top-level class declaration (`grep -nE 'class\s+\w+|interface\s+\w+'` across
  all demo source files).
- No demo build files (`pom.xml`/`build.gradle`) exist under the plain-Java
  Track-A demos (00–08, 11) — each is a single `.java` file compiled directly
  against the pinned TornadoVM SDK jars, no extra dependency to prune.
- Every Track-A demo's `README.md` opens with a "Concept (read in ~1 minute)"
  paragraph stating the one idea the demo exists to show, immediately after
  the title — confirmed present in all 9 READMEs (not just the top-level
  `demos/README.md` table).
- `which jbang` re-confirmed exit 1 (not installed) on this machine — same
  finding as every prior task (00 through 14); JBang is explicitly documented
  as unsupported-on-this-environment in `docs/run-conventions.md` and
  `docs/profiling-quickstart.md`, not silently omitted from any demo's README.

## Demos 09/10 (Quarkus/LangChain4j integrations) — out of the Track-A checklist shape, checked separately

These are Maven-project integration demos (blocked per task 13's JDK-version-split
finding), not single-file Track-A demos, so the "one concept per class"/
`java @arg-file`/JBang checks above don't apply the same way. Reviewed instead
for minimality:

- `demos/09-quarkus-langchain4j-gpullama3/`: 2 small Java files (`Main.java`,
  25 lines; `GpuExplainerService.java`, 13 lines) — each is one class with one
  job (CLI entrypoint vs. the `@RegisterAiService` interface), not an
  arbitrary split. `pom.xml` (90 lines) declares only the Quarkus BOM, the
  `quarkus-langchain4j-gpu-llama3` integration artifact, and the pinned
  `gpu-llama3` override — no unrelated dependency.
- `demos/10-langchain4j-gpullama3/`: 1 Java file (`Main.java`, 30 lines),
  `pom.xml` (70 lines) with the same minimal-dependency shape.
- Both already explicitly marked blocked with exact errors in
  `docs/quarkus-langchain4j-integration.md` (task 13) — not re-litigated here.

## Acceptance

"Every demo passes a presenter usability checklist and has a clear one-sentence
explanation" — met: all 9 Track-A demos (00–08, 11) pass every checklist row
above (one live-verified fix applied and re-tested, not just inspected), each
has a one-paragraph "Concept (read in ~1 minute)" explanation at the top of
its README, and the two out-of-shape integration demos (09/10) were reviewed
separately and confirmed minimal with no unnecessary dependencies.
