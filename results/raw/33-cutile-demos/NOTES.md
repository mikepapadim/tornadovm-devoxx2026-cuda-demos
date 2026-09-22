# CUDA Tile demos — captured evidence (2026-09-11)

Environment: RTX 4090 (sm_89), driver **610.57.04**, nvcc **13.3.73** (userspace pip wheel),
tileiras 13.3.36, JDK 21.0.2, TornadoVM **6.1.1-jdk21-dev** built from branch `feat/cutile`
([PR #1083](https://github.com/beehive-lab/TornadoVM/pull/1083)). Pins in
`env/versions.env`, section *CUDA Tile*.

## What was run

`bash scripts/run-cutile-demos.sh results/raw/33-cutile-demos` — three demos, each compiled
and then run **both ways** (the `tornado` launcher and `java @$TORNADOVM_HOME/tornado-argfile`):

```
== Summary: 9 passed, 0 failed ==
```

Per-demo logs: `19-cutile-matmul/`, `20-cutile-hybrid/`, `21-cutile-flash-attention/`
(each with `javac.log`, `tornado.log`, `javaargfile.log`).

## Observed numbers (all Observed, all validated against a CPU reference)

| Demo | Measurement | Result |
|---|---|---|
| 19 | wall clock per execution, n=256 | naive 0.446 ms, hand-tiled 0.321 ms (1.39x), `TileContext` 0.221 ms (2.02x) |
| 19 (CUDA) | kernel time, CUDA events, n=256 | naive 0.031 ms, hand-tiled 0.048 ms (0.65x), CUDA Tile 0.014 ms (2.24x) |
| 20 | wall clock per execution, n=256 | no graph 0.396 ms, `withCUDAGraph()` 0.073 ms → **5.44x** |
| 20 (CUDA) | kernel time, CUDA events | stream 0.021 ms, CUDA graph 0.018 ms → **1.14x** |
| 21 | wall clock per execution, Q=[128,64] KV=[256,64] | materialised 0.378 ms, flash 0.253 ms → **1.49x** |
| 21 (CUDA) | kernel time, CUDA events | materialised 0.025 ms, flash 0.022 ms → **1.18x** |

The Java and CUDA rows of each pair measure different things: the Java ones include JVM-side
dispatch, the CUDA ones are kernel time only. Quote them together. `cuda-equivalents.log`
holds the raw output of the three hand-written `.cu` programs.

## Generated code and hardware evidence

`printkernel-19.log`: one `extern "C" __tile_global__` entry, **zero** `asm volatile`
occurrences — the tile rung reaches tensor cores without a line of hand-written PTX.

`hmma-counts.log`, from `cuobjdump -sass` on the cached cubins:

```
tiles-*.cubin   HMMA=8 / 16 / 32   (demo 19, at n=64 / 128 / 256)
gemm-*.cubin    HMMA=8 / 16 / 32   (demo 20)
flash-*.cubin   HMMA=128           (demo 21, all HMMA.16816.F32)
```

## Side finding, not tile-related

`HalfWrite.java` here is a 25-line reproducer for a silent wrong answer in the CUDA backend:
an in-place `new HalfFloat(...)` write from a `KernelContext` kernel leaves the buffer zeroed.
Written up in `results/failures/09-kernelcontext-halffloat-write.md`. It cost demo 20 its
first correct run and contains no tile code.
