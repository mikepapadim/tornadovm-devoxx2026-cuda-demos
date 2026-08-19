You are running unattended in a scripted loop. No human will answer questions during this iteration.

Do exactly ONE task, then stop.

1. Read `CLAUDE.md` and `PLAN.md`.
2. Read the latest entries of `STATE.md` if it exists.
3. Select the lowest-numbered task in `auto/tasks/` with neither `auto/state/<id>.done` nor `auto/state/<id>.blocked`.
4. Read the entire task, including any `## Continuation` section.
5. Execute the task completely. Prefer source inspection, runnable probes, builds, and captured output over speculation.
6. Verify every acceptance criterion by running commands.
7. Record raw evidence under `results/` where appropriate. Do not edit or delete existing raw evidence; supersede with a new run id.
8. Update `STATE.md` with what changed, what was actually measured, what broke, and what the next invocation needs to know.
9. Only when every acceptance criterion passes, create `auto/state/<id>.done` with a one-line summary. Otherwise leave the marker absent and append a precise `## Continuation` section to the task file.
10. Commit the work. Push only to `origin` on this repository's `main` branch.
11. Stop. Do not start another task.

Hard rules:

- Never invent a number, version, API, feature, performance result, or status.
- `unknown`, `unmeasured`, and `blocked` are valid outcomes.
- Never use OpenCL, Metal, legacy PTX, Babylon, or another GPU framework.
- Never modify the upstream TornadoVM or GPULlama3.java checkout in place; pin SHA and use scripts/patches if an experimental change is required.
- Never create or modify issues, pull requests, releases, gists, comments, or other public surfaces from the autonomous loop.
- Do not spend the whole iteration writing prose when a runnable probe can answer the question.
