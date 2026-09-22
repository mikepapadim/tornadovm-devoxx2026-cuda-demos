# Batch 35 — Accuracy audit of the repo against TornadoVM 7.0.0

> **Correction (batch 36):** the demo 17 rung 3 figure below (480.3 µs, "~0.98x,
> parity") came from a single low run. Over 13 runs on 7.0.0 the median is
> **499.8 µs, 1.02x** of hand-written CUDA — see
> `results/raw/36-demo15-demo17-on-7.0.0/MANIFEST.md`. The rest of this audit stands.

Captured 2026-09-22. Checks the repo's claims against its own evidence, against
upstream state, and — where a claim depends on the TornadoVM version — against
fresh measurements on the pinned `7.0.0-jdk22plus-cuda` SDK. Same machine and
environment as batches 33–34 (RTX 4090, driver 610.57.04, CUDA 12.6.85, JDK 25.0.2).

## Root cause of most findings

Three upstream fixes that this repo's headline gaps were built around **shipped in
7.0.0**. Each merge commit is an ancestor of tag `v7.0.0` (`upstream-refs.txt`):

| PR | What | Merged | Effect on the repo's claims |
|---|---|---|---|
| #1066 | align native array payload to 32 bytes (fixes #1065) | 2026-09-07 | demo 15 memory-bound gap, #1065 counter rows |
| #1079 | batch global loads of a shared-memory staging sequence | 2026-09-08 | demo 17 register-tiled rung |
| #1022 | skip per-launch kernel stack-frame upload when unchanged | 2026-09-07 | plausible cause of demo 06/11 host-side speedups (not isolated) |

#1079's pass is default-on: `TornadoOptions` reads `tornado.cuda.batchGlobalLoads`
with default `"True"` (read from the 7.0.0 bytecode).

## Stale on 7.0.0 — measured

| Claim (where) | Repo says (6.0.0) | 7.0.0, measured here | Evidence |
|---|---|---|---|
| Demo 17 rung 3 vs hand-written CUDA (`demos/17-matmul-ladder/README.md`, `demos/README.md` table) | TornadoVM **1.43x** slower, "where TornadoVM falls behind" | **480.3 µs vs 489–490 µs → ~0.98x, parity** | `compare-ladder-17.log`, `17-cuda-handwritten-3runs.log` |
| Demo 15 elementwise, kernel time (`demos/15-kernel-time-comparison/README.md`) | CUDA **1.31x** faster | **1.02x** (10.95 vs 10.70 µs) | `15-nsys-kernsum-3runs.csv` |
| Demo 15 stencil, kernel time | CUDA **1.24x** faster | **1.03x** (11.92 vs 11.63 µs) | same |
| #1065 sectors/request (demo 15 README §Why, README table) | TornadoVM **5.00** vs CUDA 4.00 | **4.00** on elementwise and polynomial; stencil load sectors equal hand-written CUDA's | `15-ncu-sectors.csv` |

`15-ncu-sectors.csv` was captured at n = 1,048,576 (the harness size), not the
README's n; compare the per-request ratio and the scaled sector counts, not the
absolute totals. The demo 15 nsys run used the demo's defaults, as the README does.

Consequently the demo 15 narrative ("Memory-bound gap is the `FloatArray` header
offset") and the demo 17 narrative ("Rung 3 is where TornadoVM falls behind") describe
6.0.0. Neither README is rewritten here: both are talk narratives, and the new story
("the gap the talk diagnosed is closed in the next release") is an editorial call.

## Not re-measured, but mechanism now gone

These README rows rest on the #1065 misalignment and were not re-captured: demo 14's
"TornadoVM 163,840 load sectors vs CUDA's 131,072 — 1.250x", and demo 01's
"4x the instructions, only 3.4% slower". Expect both to have changed.

## Still accurate on 7.0.0 — measured

- Demo 15 `polynomial` (compute-bound): TornadoVM **1.15x** faster (35.01 vs 40.26 µs);
  repo says 1.13x. Holds.
- Demo 18 FP16 kernel-time ladder: every rung within ±8% of batch 30
  (`compare-ladder-18.log`) — consistent with batch 31's finding that the FP16 ladder
  gains nothing from load batching.
- Demo 17's other rungs: naive, tiled, CUTLASS and cuBLAS within ~2% of batch 28.
- Every upstream issue/PR number cited in README, docs and demos resolves, with a
  title matching how the repo describes it.
- Every `results/` path cited anywhere outside `results/` resolves on disk.

## `CuDnn.sdpaForward` on 7.0.0

Fails loudly before running (`sdpa-benchmark.log`): SDPA now goes through
`lib/libtornado-cudnn.so`, which needs `libnvrtc.so.13` **and** `GLIBC_2.38`; Ubuntu
22.04 has 2.35 (`libtornado-cudnn-glibc-requirements.txt`). So whether 7.0.0 fixed
#1063's silent all-zero result is untestable on this machine. This also corrects
batch 33's statement that 7.0.0 "never loads" `libtornado-cudnn.so` — true only for
the conv2d/relu path demo 13 uses.

## Stale counts and instructions (text only, no measurement needed)

| Where | Says | Should say |
|---|---|---|
| `docs/demo-runbook.md:26` | `all 9 demos ... must end 27/27` | 16 demos, 48/48 — a presenter following this would think a pass is a failure |
| `docs/NVIDIA-BRIEF.md:291` | `36/36 checks` | 48/48 |
| `README.md` repo-layout list | `run-all-demos.sh — ... all 12 demos` | 16 |
| `docs/hybrid-api-inventory.md:5` | Track A "now run on the TornadoVM 6.0.0 CUDA release" | 7.0.0 |
| `demos/README.md` CUDA-equivalents | "all thirteen compile", "13 compiles + 13 runs + 2 probes" | `run-all-cuda.sh` lists 16 demos |
| `demos/README.md` demo table | rows 00–17 | no rows for 16 or 18 (both exist, both have READMEs) |

## Demo 18's CUDA equivalent never validates

`bash scripts/run-all-cuda.sh` (CUTLASS 3.5.1) ends **33 passed, 1 failed**: every
compile and run passes except `18-matmul-ladder-fp16 -- run`, flagged `no verdict in
output`. That is correct behaviour by the script — `MatMulLadderFP16.cu` prints a timing
table and contains no correctness check at all (`18-cuda-run-no-verdict.log`). Every
other `.cu` validates against a reference. Present since demo 18 was added (2e8fa2f).

## Evidence-trail gaps

- `STATE.md` has **no entries for batches 24–32**, though all nine directories exist
  under `results/raw/`. The ledger jumps from batch 23 to batch 33.
- `results/raw/32-host-overhead/` has three CSVs and **no MANIFEST** — no recorded
  environment, build, or method.

## Errors in batches 33–34 found and fixed by this audit

- Batch numbers **31 and 32 collided** with existing `31-load-batching-reorder` and
  `32-host-overhead`. Renamed to 33 and 34; every reference updated.
- The cuDNN statement above ("never loads") and an unsupported contrast in the README
  migration table (that 6.0.0 loaded `libtornado-cudnn.so` via JNI — never observed).
- Demo 13's SDPA paragraph, edited in batch 33 to assert a 7.0.0 all-zero result that
  was never tested.

## Files

| File | What it is |
|---|---|
| `compare-ladder-17.log` | `scripts/compare-ladder.sh 17 2048 10` on 7.0.0 — nsys kernel time + ncu counters |
| `compare-ladder-18.log` | `scripts/compare-ladder.sh 18 1024 10` on 7.0.0 |
| `17-cuda-handwritten-3runs.log` | `MatMulLadder.cu` at 2048/10, three runs, same session |
| `15-nsys-kernsum-3runs.csv` | demo 15, TornadoVM and CUDA, three nsys runs each |
| `15-ncu-sectors.csv` | demo 15 TornadoVM kernels, sectors/request and sector totals |
| `sdpa-benchmark.log` | the SDK's own `BenchmarkSdpa` on 7.0.0 |
| `libtornado-cudnn-glibc-requirements.txt` | glibc symbol versions the shim needs, and this box's glibc |
| `18-cuda-run-no-verdict.log` | `MatMulLadderFP16.cu` output from `run-all-cuda.sh` — timings, no verdict |
| `upstream-refs.txt` | every cited upstream issue/PR, state and title; fix-commit ancestry vs `v7.0.0` |
