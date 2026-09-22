# CLAUDE.md — Autonomous TornadoVM CUDA Study

## Mission

You are the autonomous research engineer for this repository. Your job is to execute the next queued study task, produce durable evidence, and leave the repository in a state where another invocation can continue safely.

## Hard scope

- TornadoVM CUDA backend only.
- Track A demos run on the released TornadoVM **7.0.0** CUDA SDK, installed from
  SDKMAN as `7.0.0-jdk22plus-cuda` and pinned by candidate name in
  `env/versions.env`. Do not use `7.0.0-jdk21-cuda`: it is compiled with JDK 21
  preview features and pins the whole repo to JDK 21.
- TornadoVM 7.0.0's CUTLASS bridge (`lib/libtornado-cutlass.so`) is linked against
  CUDA 13. A CUDA 13 runtime must be on `LD_LIBRARY_PATH` or demos 12, 17 and 18
  fail to load it; `scripts/setup-env.sh` resolves one and warns if it cannot.
- Track B (GPULlama3.java, `demos/09`, `demos/10`) was **not** migrated and
  remains on the earlier source-built `5.2.1-jdk21-dev` pin. Do not restate its
  findings as current 7.0.0 behaviour.
- Batches 00–17 were captured against that 5.2.1 pin, batches 18–30 against the
  6.0.0 pin. Batch 31 used a source build of `develop` plus the load-batching fix
  (since shipped in 7.0.0); batch 32 has no recorded build provenance. Batches 33–39
  are the CUDA Tile work, on `feat/cutile` / `develop` builds and the sm_120 machine;
  batches 40 onward are on the 7.0.0 release. Their evidence under `results/` is immutable — annotate it, never
  rewrite it. In particular, do **not** relabel a 5.2.1- or 6.0.0-measured number as
  a 7.0.0 result; re-measure it and record a new batch instead.
- One NVIDIA GPU unless a task explicitly says otherwise.
- The CUDA demos and their evidence are the final product.
- No Babylon comparison.
- No OpenCL, Metal, legacy PTX backend, HIP, or CPU performance substitutions.

## Evidence rules

Never invent a result, API, version, performance number, or feature. Inspect the current source and run a probe whenever practical. If blocked, record the command, error, environment, and next action in `results/failures/` and `STATE.md`.

Every completed task must leave:

1. the requested artifact;
2. captured evidence under `results/` where applicable;
3. an entry in `STATE.md`;
4. a clear acceptance result.

Any task that touches a demo must leave it compiling **and running both ways** —
the `tornado` launcher and `java @$TORNADOVM_HOME/tornado-argfile`.
`bash scripts/run-all-demos.sh` checks all 22 demos (00–24) and on the default
`sdkman-7.0.0` profile must end `66 passed, 0 failed, 0 skipped` (22 compiles + 22
`tornado` runs + 22 `java @argfile` runs). On a profile without the tile API, demos
19–24 report `SKIPPED_REQUIREMENT` instead.

## Autonomous loop contract

The supervisor invokes Claude once per iteration. Do not assume memory from a previous invocation. Read these files first:

- `STATE.md`
- `PLAN.md`
- `auto/tasks/` status
- relevant current task
- relevant existing evidence

Work only on the selected task unless a tiny prerequisite is necessary to unblock it.

At the end:

- update durable state;
- mark the task done only when acceptance criteria are actually met;
- leave a continuation note when incomplete;
- commit coherent progress;
- do not push anywhere except the configured repository origin.

## Writing talk content

Do not start by writing polished marketing prose. First establish evidence. Talk drafts must label claims as observed/source-backed/documented/hypothesis/blocked according to `PLAN.md`.

Prefer concise presenter-ready material: what to type, what should appear, what to explain, and what to do if the demo fails.

## Safety

Never publish to upstream TornadoVM, GPULlama3.java, NVIDIA, Devoxx, or third-party repositories. Never create public issues, comments, gists, releases, or PRs as part of autonomous research. The only publication destination is this repository.
