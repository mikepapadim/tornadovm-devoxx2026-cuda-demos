# Regression report — `feat/cutile` branch → upstream `develop`, sm_89 → sm_120

## What changed

| | Before (batches 33, 34) | After |
| --- | --- | --- |
| TornadoVM (tile demos) | `feat/cutile` branch, `6.1.1-jdk21-dev`, PR #1083 unmerged | upstream **`develop` `8d592d6fbaafe42d66e1e22c27057cb1cdcf9097`**, `6.1.1-jdk22plus-dev` |
| TornadoVM (other demos) | 6.0.0 SDKMAN release | same build as the tile demos — one SDK for everything |
| JDK for tile demos | 21, `--release 21 --enable-preview` | **25, no preview flags** |
| Runner | `run-all-demos.sh` + a separate `run-cutile-demos.sh` | one runner, profile-selected |
| GPU | RTX 4090, sm_89, driver 610.57.04 | RTX 5070 Ti, **sm_120**, driver 580.142 |
| CUDA (tile) | nvcc 13.3.73 | nvcc 13.4.92 |
| OS | Ubuntu 22.04.5 | Ubuntu 23.10 |

This is a **different machine** as well as a different TornadoVM. No timing comparison is
made against the batch 33/34 numbers, and none should be: demo 22's recorded kernel times
(naive 537.2 µs … cuBLAS 20.2 µs) were measured on the 4090 and are untouched by this batch.
This report covers **correctness only**.

## Result: no regressions

| Suite | Before | After |
| --- | --- | --- |
| `run-all-demos.sh` | 48/48 (16 demos, 6.0.0) | **66/66** (22 demos, develop) |
| `run-cutile-demos.sh` | 12/12 (4 demos, feat/cutile + JDK 21) | folded into the above, no preview flags |

Every demo that passed before passes now. No existing demo's `.java` was modified in this
batch; the only edits to demos 19–22 are to their READMEs, removing build flags that are no
longer required.

Evidence: `results/raw/37-demo-matrix-develop/` (develop),
`results/raw/38-demo-matrix-6.0.0/` (the same demos on the released SDK — 48 passed, 0
failed, 6 skipped).

`38-demo-matrix-6.0.0` is the honest control: it isolates the **TornadoVM** variable by
running both SDKs on the same GPU, driver and OS.

## The material improvement

The tile demos no longer need a personal feature branch. PR #1083's merge commit
`ec970e26d` is an ancestor of upstream `develop`, so three constraints disappear at once:

1. **No feature branch** — `git clone --branch develop https://github.com/beehive-lab/TornadoVM.git`.
2. **No JDK 21, no `--enable-preview`** — `develop` is a jdk22plus build. Verified: all four
   existing tile demos compile with plain `javac` and run on JDK 25.
3. **No second runner** — one `run-all-demos.sh` covers all 22 demos, and a demo the active
   SDK cannot run is skipped rather than failed.

## New: demos 23 and 24

| Demo | Result |
| --- | --- |
| 23-cutile-row-scan | PASS — bit-exact prefix sums and row totals, both run modes |
| 24-cutile-histogram | PASS — every bin exact, all values counted, both run modes |

Neither existed before, so neither can regress. Both are `SKIPPED_REQUIREMENT` on the 6.0.0
profile. They were chosen to cover what demos 19–22 do not: before them nothing in the repo
used a scan, a masked load or store, a loop-carried reduction, any predicate, or any of the
tile API's ten atomic operations.

## Upstream tile test suite on sm_120

First record of this suite on Blackwell. **212 of 219 pass.** All 7 failures are FP8
arithmetic in `TestTileOpLevel` (`testFp8Add/Sub/Mul/Div/Maximum/Exp/Sum`).

Root-caused: **CUDA Tile C++ 13.4 defines no arithmetic operators for FP8 tile element
types**, and the backend emits the operator form directly. Reduced to pure C++ in
`results/raw/35-develop-cutile-baseline/fp8-probe/`: `fp8.cu` fails with
`no operator "+" matches these operands`; the same kernel with an `element_cast` round-trip
through f32 compiles clean.

This is **not** the documented "fp8 below compute capability 9.0" limitation — this GPU is
CC 12.0. Written up in `docs/cutile-api.md`. Not filed upstream pending a decision.

## Harness changes that affect how results are read

1. **`-Dtornado.recover.bailout=False` is now passed to every demo run.** Previously a
   kernel that failed to compile could fall back to the host, compute the right answer,
   print `correct`, and be scored a pass. This matters most for the tile demos, where every
   `TileContext` method has a JVM fallback — a wholly blocked GPU path would have scored
   green. The old `run-cutile-demos.sh` did **not** pass it.
2. **A third verdict, `SKIP`.** The runner had only pass/fail, so a demo whose requirement
   the active SDK cannot meet was scored a *failure*.

## Known open defect (pre-existing, not fixed here)

`scripts/verify.sh` fails with `1 demo(s) have no .cu equivalent`:
`demos/22-matmul-ladder-fp16-tile/` has no hand-written CUDA twin, while every other demo
does. Confirmed pre-existing on pristine `origin/main`.
