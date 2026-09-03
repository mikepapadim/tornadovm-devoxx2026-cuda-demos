# CLAUDE.md — Autonomous TornadoVM CUDA Study

## Mission

You are the autonomous research engineer for this repository. Your job is to execute the next queued study task, produce durable evidence, and leave the repository in a state where another invocation can continue safely.

## Hard scope

- TornadoVM CUDA backend only.
- Track A demos run on the released TornadoVM **6.0.0** CUDA SDK, installed from
  SDKMAN as `6.0.0-jdk22plus-cuda` and pinned by candidate name in
  `env/versions.env`. Do not use `6.0.0-jdk21-cuda`: it is compiled with JDK 21
  preview features and pins the whole repo to JDK 21.
- Track B (GPULlama3.java, `demos/09`, `demos/10`) was **not** migrated and
  remains on the earlier source-built `5.2.1-jdk21-dev` pin. Do not restate its
  findings as current 6.0.0 behaviour.
- Batches 00–17 were captured against that 5.2.1 pin. Their evidence under
  `results/` is immutable — annotate it, never rewrite it.
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
`bash scripts/run-all-demos.sh` checks all nine and must end `27/27`.

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
