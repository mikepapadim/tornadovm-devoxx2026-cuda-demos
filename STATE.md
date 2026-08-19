# Autonomous Study State

## Batch 00 — TornadoVM CUDA baseline pinned (2026-08-19)

Task `auto/tasks/00.md` is done. What was measured, all Observed:

- Cloned `beehive-lab/TornadoVM` `develop` at SHA `99549c9862eda8d584e35e99924f9c865501eb3a` into `vendor/tornadovm` (gitignored — upstream checkout, not committed in place; pinned via `env/versions.env`). `develop` and `master` pointed at the same commit at clone time.
- Machine: RTX 4090 (24564 MiB, compute cap 8.9), driver 565.57.01, CUDA toolkit (nvcc/ptxas) 12.6.85, JDK 21.0.2 (sdkman), Maven 3.6.3, gcc 11.4.0, Ubuntu 22.04.5, kernel 6.8.0-58-generic. Full detail in `env/versions.env` and `results/raw/00-baseline/MANIFEST.md`.
- `make BACKEND=cuda` in `vendor/tornadovm` builds successfully (all Maven modules `SUCCESS`, installer reports `Backend : CUDA`, `Commit : 99549c9`). Log: `results/raw/00-baseline/tornadovm-build-cuda.log`.
- `tornado --devices` (after `source vendor/tornadovm/setvars.sh`) shows exactly one CUDA driver, one device (`0:0`), `NVIDIA GeForce RTX 4090`. Log: `results/raw/00-baseline/tornado-devices.log`.
- Smoke example `uk.ac.manchester.tornado.examples.VectorAddInt` (size 256) run via `tornado --enableProfiler console`: 10/10 iterations report `Result is correct`; profiler JSON confirms `BACKEND=CUDA`, `DEVICE=NVIDIA GeForce RTX 4090`, `DEVICE_ID=0:0` for every run. Log: `results/raw/00-baseline/vectoradd-int-cuda-run.log`.

Acceptance criteria for task 00 (env/versions.env + manifest exist; CUDA backend builds; smoke example runs on CUDA; exact SHA recorded) are all met and verified by the commands above.

### Next invocation

Start with `auto/tasks/01.md` (verify current Hybrid API against the pinned `vendor/tornadovm` source at SHA `99549c9862eda8d584e35e99924f9c865501eb3a`). The checkout at `vendor/tornadovm` is left in place (built, CUDA backend ready, `setvars.sh` available) so later tasks do not need to rebuild from scratch — but it is gitignored, so a fresh clone/rebuild is required if the working directory is reset. Re-verify the SHA still matches `env/versions.env` before reusing it; if TornadoVM `develop` has moved and a task needs the newer tree, re-pin deliberately and record the new SHA rather than silently drifting.
