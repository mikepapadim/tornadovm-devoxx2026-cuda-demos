# Demo status matrix

Every demo, built and run on the TornadoVM `develop` baseline. No status here is inferred:
each `PASS` means the demo was compiled, launched on the CUDA backend **both ways** (the
`tornado` launcher and `java @argfile`), and printed its own correctness verdict.

```text
Date          : 2026-09-18
TORNADO_SHA   : 8d592d6fbaafe42d66e1e22c27057cb1cdcf9097  (develop, built from source)
SDK profile   : develop  (env/sdk/develop.env)
GPU           : NVIDIA GeForce RTX 5070 Ti, sm_120, 16303 MiB
Driver        : 580.142
CUDA          : 13.0.88 system / 13.4.92 userspace (tile path)
JDK           : 25.0.2-open
OS            : Ubuntu 23.10, kernel 6.5.0-44-generic
Command       : source scripts/setup-env.sh && bash scripts/run-all-demos.sh
Result        : 66 passed, 0 failed, 0 skipped
Evidence      : results/raw/37-demo-matrix-develop/
```

Every run passes `-Dtornado.recover.bailout=False`, so a kernel that fails to compile can no
longer fall back to the host and still print `correct`.

| Demo | Status | Correctness | Evidence |
| --- | --- | --- | --- |
| 00-hello-gpu | PASS | own verdict | `raw/37-demo-matrix-develop/00-hello-gpu/` |
| 01-first-cuda-kernel | PASS | own verdict | `raw/37-.../01-first-cuda-kernel/` |
| 02-cuda-runtime-api | PASS | own verdict | `raw/37-.../02-cuda-runtime-api/` |
| 04-cublas-hybrid | PASS | own verdict | `raw/37-.../04-cublas-hybrid/` |
| 05-cufft-hybrid | PASS | own verdict | `raw/37-.../05-cufft-hybrid/` |
| 06-cuda-streams | PASS | own verdict | `raw/37-.../06-cuda-streams/` |
| 07-cuda-graph-benefit | PASS | own verdict | `raw/37-.../07-cuda-graph-benefit/` |
| 08-tensor-core-mma | PASS | own verdict | `raw/37-.../08-tensor-core-mma/` |
| 09-quarkus-langchain4j-gpullama3 | BLOCKED_ENVIRONMENT | not attempted | see note |
| 10-langchain4j-gpullama3 | BLOCKED_ENVIRONMENT | not attempted | see note |
| 11-integrated-showcase | PASS | own verdict | `raw/37-.../11-integrated-showcase/` |
| 12-cutlass-fused-epilogue | PASS | own verdict | `raw/37-.../12-cutlass-fused-epilogue/` |
| 13-cudnn-jit-convblock | PASS | own verdict | `raw/37-.../13-cudnn-jit-convblock/` |
| 14-warp-async-shared | PASS | own verdict | `raw/37-.../14-warp-async-shared/` |
| 15-kernel-time-comparison | PASS | own verdict | `raw/37-.../15-kernel-time-comparison/` |
| 16-tensor-core-datatypes | PASS | own verdict | `raw/37-.../16-tensor-core-datatypes/` |
| 17-matmul-ladder | PASS | own verdict | `raw/37-.../17-matmul-ladder/` |
| 18-matmul-ladder-fp16 | PASS | own verdict | `raw/37-.../18-matmul-ladder-fp16/` |
| 19-cutile-matmul | PASS | own verdict | `raw/37-.../19-cutile-matmul/` |
| 20-cutile-hybrid | PASS | own verdict | `raw/37-.../20-cutile-hybrid/` |
| 21-cutile-flash-attention | PASS | own verdict | `raw/37-.../21-cutile-flash-attention/` |
| 22-matmul-ladder-fp16-tile | PASS | own verdict | `raw/37-.../22-matmul-ladder-fp16-tile/` |
| **23-cutile-row-scan** | **PASS** | bit-exact: 0/4096000 scan, 0/4096 totals | `raw/36-cutile-demos-23-24/`, `raw/37-.../23-cutile-row-scan/` |
| **24-cutile-histogram** | **PASS** | every bin exact, 1048576/1048576 counted | `raw/36-cutile-demos-23-24/`, `raw/37-.../24-cutile-histogram/` |

## Demos 09 and 10 — `BLOCKED_ENVIRONMENT`

Track B. Maven projects that pull TornadoVM transitively through GPULlama3.java and are
pinned to the earlier source-built `5.2.1-jdk21-dev` baseline, not to `TORNADOVM_HOME`.
Running them needs multi-GB GGUF model files and would drag the old pin in alongside the
develop build. Deliberately not attempted; their pins stay recorded in `env/versions.env`.

## Demos 19–22 no longer need JDK 21 or preview flags

They were pinned to the `feat/cutile` branch, a jdk21-dev build, and compiled with
`--release 21 --enable-preview`. PR #1083 is merged into upstream `develop`, so on the
`develop` profile all four compile with no preview flags and run on the same JDK 25 as every
other demo. Verified in this run; their READMEs and the runner were updated accordingly.

## The tile demos did more than print a verdict

A passing verdict is not sufficient evidence for a tile demo, so the GPU path was confirmed
directly for the two new ones:

| Check | 23-cutile-row-scan | 24-cutile-histogram |
| --- | --- | --- |
| Generated kernel | `__tile_global__`, `ct::partial_sum` ×8, `ct::sum` ×8, `ct::partition_view` | `__tile_global__`, `ct::atomic_add` ×8, `ct::select` ×8, `ct::iota`, `ct::broadcast` |
| Pure cuTile C++ twin | compiles to a cubin | compiles; `cuobjdump -sass` shows atomic instructions |
| Bailout | `-Dtornado.recover.bailout=False` | `-Dtornado.recover.bailout=False` |

## Profile switching

The same matrix on the released 6.0.0 SDK, which has no tile API:

```text
export TORNADO_SDK_PROFILE=sdkman-6.0.0 && source scripts/setup-env.sh
bash scripts/run-all-demos.sh
→ 48 passed, 0 failed, 6 skipped
  SKIP 19-cutile-matmul … 24-cutile-histogram  -- profile sdkman-6.0.0 lacks: tile
```

Evidence: `results/raw/38-demo-matrix-6.0.0/`. The tile demos **skip**, they do not fail.

## Known open defect (pre-existing)

`scripts/verify.sh` reports `1 demo(s) have no .cu equivalent`:
**`demos/22-matmul-ladder-fp16-tile/` has no hand-written CUDA twin.** Every other demo has
one. This predates this batch — `git ls-tree origin/main demos/22-matmul-ladder-fp16-tile/`
on the pristine branch shows only the `.java` and the `README.md` — so `verify.sh` has been
failing on `main` since demo 22 landed. Not fixed here: writing a faithful CUDA twin for a
six-rung ladder is the demo author's call, not a drive-by.
