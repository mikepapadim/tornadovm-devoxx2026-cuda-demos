# Batch 18 — TornadoVM 6.0.0 CUDA migration

Captured 2026-09-02. Every Track A demo re-built and re-run on the TornadoVM
6.0.0 CUDA release installed from SDKMAN, replacing the source-built
5.2.1-jdk21-dev pin used by batches 00–17.

## Environment

| | |
|---|---|
| TornadoVM | `6.0.0-jdk22plus-cuda` (SDKMAN), release commit `fadb20b` |
| JDK | OpenJDK 25.0.2 (`25.0.2-open`) |
| GPU | NVIDIA GeForce RTX 4090, driver 565.57.01 |
| CUDA toolkit | 12.6.85 (`nvcc`/`ptxas`) |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-58-generic |

`tornado.release`, `tornado.jdk`, `tornado-devices.log`, `java-version.log`,
`nvidia-smi.log` and `tornado-argfile.generated` are verbatim copies taken at
capture time.

## Result

All nine Track A demos compile and run correctly under **both** supported run
paths — the `tornado` launcher and `java @$TORNADOVM_HOME/tornado-argfile`.
27/27 checks pass (`run-all-demos.log`, produced by `scripts/run-all-demos.sh`).

**No demo source needed an API change.** All nine compile unmodified against
`tornado-api-6.0.0`; the migration is entirely in packaging, JDK level, and
launch flags.

## Files

| File | What it is |
|---|---|
| `run-all-demos.log` | `scripts/run-all-demos.sh` — 9 compiles + 9 `tornado` runs + 9 `java @argfile` runs |
| `00-hello-*.log` … `11-showcase-*.log` | per-demo runs, `-tornado` and `-javaargfile` variants |
| `07-graphbenefit-profiler.log` | demo 07 under `--enableProfiler console` (5 executions) |
| `08-mma-printkernel.log` | demo 08 under `--printKernel`; generated-code evidence |
| `11-showcase-repeat3.log` … `repeat5.log` | three extra demo 11 runs (5 total) for the combined-mode finding |

## Measured (Observed — this machine, this run)

| Demo | Metric | Result |
|---|---|---|
| 04 | correctness | 5/5 iterations match the sequential Java reference |
| 05 | correctness | 5/5 iterations, max abs error `4.7683716E-7` |
| 06 | sequential vs. concurrent steady-state median | 2174 µs vs. 960 µs (`tornado`); 2176 µs vs. 936 µs (`java @argfile`) — ~2.3× |
| 07 | nograph vs. graph steady-state median | 364.2/36.4 µs = 10.00× (`tornado`); 292.4/36.2 µs = 8.08× (`java @argfile`); 1070.9/121.4 µs = 8.82× (profiler) |
| 08 | `mma.sync.aligned` instruction count | exactly 1 in the MMA kernel, 0 in the scalar reference |
| 11 | mode comparison vs. baseline, 5 runs | baseline 811–851 µs; concurrent 1.08–1.13×; graph 5.37–5.87×; combined 5.55–5.71× |

## Changes vs. the 5.2.1 pin

- **Demo 11's `combined` mode caught up with `graph`.** On 5.2.1, stacking
  `withIntraPlanConcurrency()` on top of `withCUDAGraph()` cost more than
  `graph` alone in every run (3.91× vs 6.56×; 2.95× vs 4.90×). On 6.0.0 the two
  are indistinguishable within run-to-run noise (5.55–5.71× vs 5.37–5.87×,
  n=5). Observed; mechanism not investigated.
- **Demo 11's `concurrent` mode is no longer sometimes-slower.** 5.2.1 produced
  both a 1.21× speedup and a 0.81× slowdown across runs; all five 6.0.0 runs
  landed in 1.08–1.13×. Still marginal next to `graph` mode.
- **Demo 07's speedup ratio rose** (6.47–7.02× → 8.08–10.00×), but the `graph`
  steady-state median is ~36 µs on both pins. What moved is the `nograph`
  baseline, so read this as dispatch-overhead variance, not a 6.0.0
  optimisation.
- Demo 06's ~2.3× concurrency benefit and demo 08's single-`mma.sync` codegen
  are unchanged.

## Not re-captured on 6.0.0

- **Nsight Systems traces.** The stream-overlap timelines and profiling
  summaries in `results/raw/06-cuda-streams/`, `results/raw/09-profiling/` and
  `results/raw/11-integrated-showcase/` are from the 5.2.1 pass. They remain
  valid as *mechanism* evidence (1 stream vs. a 4-stream pool, where launch
  cost goes) and are still cited as such.
- **Nsight Compute hardware counters** remain blocked on this machine —
  `ERR_NVGPUCTRPERM`, `NVreg_RestrictProfilingToAdminUsers=1`, no passwordless
  sudo. Re-confirmed unchanged. Not a TornadoVM limitation.
  `results/failures/08-nsight-compute-permission.md`.
- **Track B (GPULlama3.java, demos 09/10)** was not migrated. Its pins and
  evidence stay on 5.2.1 — see `env/versions.env` and `README.md`.
- **JBang** is still not installed here (`which jbang` → exit 1).
