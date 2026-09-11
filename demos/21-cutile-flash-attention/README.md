# 21 — flash attention in Java, compiled through CUDA Tile

Attention two ways, both as CUDA Tile kernels:

| # | Path | Kernels | Score matrix |
|---|---|---|---|
| 1 | materialised | 3 (`Q Kt`, row softmax, `P V`) | `[S_Q, S_KV]` in device memory |
| 2 | **flash** | 1 | never exists |

The fused kernel is the reason a tile-level API is worth having. In one loop body it needs a
matmul, a row reduction, a row broadcast against a full tile, an exponential, a narrowing
cast back to FP16 and a second accumulating matmul — with three tiles carried across a loop
whose trip count is a runtime value. Written with threads that is a few hundred lines of
staging and barriers; written with tiles it is the fifteen lines of `flash(...)`.

Ported from NVIDIA's TileGym `ops/tilecpp/attention.cuh`. Both paths are validated against a
sequential two-pass CPU reference over the same FP16-rounded inputs.

Source: [`TileFlashAttention.java`](TileFlashAttention.java), hand-written CUDA in
[`TileFlashAttention.cu`](TileFlashAttention.cu).

## Requirements

Needs a TornadoVM built from the cuTile branch (not the pinned 6.0.0 SDK), CUDA Toolkit
**13.3+** and driver **R580+**. See `env/versions.env`, section *CUDA Tile*.

## Run

```bash
export TORNADOVM_HOME=<cutile SDK>
export JAVA_HOME=$HOME/.sdkman/candidates/java/21.0.2-open
cd demos/21-cutile-flash-attention
javac --release 21 --enable-preview -proc:none -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . TileFlashAttention.java
tornado --classpath . TileFlashAttention 128 256 20      # queryRows, kvRows, executions
```

Observed on an RTX 4090 (sm_89), driver 610.57.04, nvcc 13.3.73:

```
Q = [128, 64], K = V = [256, 64], 20 executions

path                                             ms/execution      max error   result
1. materialised (3 kernels + score matrix)              0.378         0.0000   correct
2. flash (1 kernel, online softmax)                     0.253         0.0000   correct

fusion: 0.378 ms -> 0.253 ms per execution, 1.49x faster
and 196608 fewer bytes of device traffic
```

Both paths run entirely on the GPU — the materialised path is three tile tasks in one
`TaskGraph` on shared buffers, so the comparison is fusion against fusion, not GPU against
host. The hand-written CUDA version measures the same two paths with CUDA events and gets
1.18x, which is the fusion alone with host dispatch removed:

```bash
nvcc --enable-tile -std=c++20 -arch=sm_89 -o tileflash TileFlashAttention.cu
./tileflash 128 256 20
```

`queryRows` must be a multiple of 32; the materialised path is compiled for `kvRows = 256`
(a tile shape must be a compile-time constant, so the full-row softmax kernel is specialised
for one KV length — the fused kernel is not, which is a real advantage of it).

## The generated kernel

```bash
tornado --printKernel --classpath . TileFlashAttention 32 64 1
```

The body is `ct::mma`, `ct::reduce_max`, `ct::max`, `ct::exp`, `ct::sum`,
`ct::element_cast<__half>`, `ct::transpose` — and no inline PTX at all. The cubin:

```bash
cuobjdump -sass $TORNADOVM_HOME/var/cuda-codecache/device-0-0/flash-*.cubin | grep -c HMMA
# 128   (all HMMA.16816.F32, at Q = [128, 64] / KV = [256, 64])
```

## What is missing, and why

**Causal masking is not implemented.** It needs a boolean tile from a comparison plus
`ct::select`, which the Java API declares but does not lower yet. A zero-padded masked load
is not a substitute: a padded key contributes `exp(0 - m)` to the softmax denominator rather
than nothing. The same gap rules out a ragged KV tail, which is why `kvRows` must be a
multiple of the KV block here.

The demo also uses a natural `exp` where TileGym folds `1/ln 2` into the QK scale and uses
`ct::exp2` — the cheaper instruction. The softmax is identical either way.

## If it fails

* `TornadoDeviceTileNotSupported` — nvcc older than 13.3 on the path.
* A NaN output — check `NEGATIVE_LIMIT` is still a large negative constant and not `-inf`;
  with a true `-inf` the first rescale is `exp(-inf - m)` multiplying a zero accumulator.
* `The tile rows must be a compile-time constant` — a shape reached the kernel as a
  parameter. Tile shapes are part of the kernel's type; use a literal or `static final int`.
