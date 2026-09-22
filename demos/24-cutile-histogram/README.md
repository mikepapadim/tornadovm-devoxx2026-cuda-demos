# 24 — CUDA Tile histogram

A value histogram built with the tile API's **atomics**: every tile block folds a per-bin
count into every bin, and the adds collide by construction.

## What does this demonstrate?

**The one part of the tile API nothing else here touches.** `PartitionView` exposes ten
atomic operations — `atomicAdd/Sub/Min/Max/And/Or/Xor/Exchange/Load/Store`. Demos
[19](../19-cutile-matmul/), [20](../20-cutile-hybrid/), [21](../21-cutile-flash-attention/),
[22](../22-matmul-ladder-fp16-tile/) and [23](../23-cutile-row-scan/) use **none** of them:
every one of those kernels owns its output tile outright. This one does not.

```java
Tile count = tc.sum(tc.select(hit, ones, zeros), 1);
bv.atomicAdd(count, 0, bin);
```

With 4096 tile blocks and 256 bins that is **1,048,576 contended adds** per execution.
Replace `atomicAdd` with `store` and the answer collapses to whatever the last block wrote.

**Data-dependent selection without a gather.** A histogram picks its bin from the *value*,
which normally means a scatter — and CUDA Tile has no gather/scatter at all. The tile
formulation replaces the per-element index with a predicate over the whole tile:

```java
Tile atOrAbove = tc.greaterOrEqual(chunk, bin);   // PRED tile
Tile below     = tc.lessThan(chunk, bin + 1);     // PRED tile
Tile hit       = tc.logicalAnd(atOrAbove, below);
Tile count     = tc.sum(tc.select(hit, ones, zeros), 1);
```

This is the demo that shows what you do *instead of* a scatter, which is the question
anyone porting a real kernel hits first.

**The bin index is a block index.** `counts` is viewed as `[1, bins]` and partitioned `1×1`,
so bin *b* is simply block *b* of that view — which is how a runtime-chosen destination is
expressed when tile shapes must be compile-time constants.

## What TornadoVM feature/API does it use?

`uk.ac.manchester.tornado.api.tile` — `TileContext.partition/view/full/zeros/greaterOrEqual/
lessThan/logicalAnd/select/sum/bidX` and **`PartitionView.atomicAdd`**.

Confirmed in the generated kernel: `ct::atomic_add`, `ct::select`, `ct::sum`, `ct::iota`,
`ct::broadcast`, `ct::memory_order_relaxed_t`, `ct::thread_scope_device_t`. The scalar
comparisons lower to an `ct::iota`/`ct::broadcast` pair — the backend materialises the
scalar as a tile rather than exposing a tile-scalar compare in C++.

See [`docs/cutile-api.md`](../../docs/cutile-api.md).

## Prerequisites

| | |
| --- | --- |
| GPU | compute capability **8.0+** (validated on sm_120, RTX 5070 Ti) |
| Driver | **R580+** to load a CUDA 13 cubin |
| CUDA Toolkit | **13.3+** — `CUDATileCompiler.MINIMUM_TOOLKIT = 13003` |
| TornadoVM | a build with the tile API — the `develop` profile |

Atomic element types are **F16, F32, F64 or S32**; this demo uses F32.

```bash
pip install --user nvidia-cuda-nvcc 'cuda-tile[tileiras]' nvidia-cuda-cccl
```

`scripts/setup-env.sh` puts that toolchain on `PATH` — `nvcc` spawns `tileiras` by bare
name, so without it a tile compile fails late with `sh: 1: tileiras: not found`.

## How do I run it?

```bash
source scripts/setup-env.sh
cd demos/24-cutile-histogram
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . CuTileHistogram.java

tornado --jvm="-Dtornado.recover.bailout=False" --classpath . CuTileHistogram
java @$TORNADOVM_HOME/tornado-argfile -Dtornado.recover.bailout=False -cp . CuTileHistogram
```

Arguments: `[n] [bins] [iterations]`, default `1048576 256 20`. `n` must be a multiple of
the 256-wide tile.

> **Always pass `-Dtornado.recover.bailout=False`.** Every `TileContext` method also has a
> plain-Java implementation, and `TornadoOptions.RECOVER_BAILOUT` defaults to **true**. With
> the default, a tile kernel that fails to compile silently falls back to the host, computes
> the right answer, and prints `correct` — while the GPU did nothing.

## How do I validate the result?

Every input is a whole number in `[0, bins)`, so each lands in exactly one bin and every
count is an exact integer in fp32 — the check is `==`, not a tolerance, no matter how the
atomic adds interleave. Two conditions must both hold: every bin matches a sequential
reference, **and** the counts sum to `n` (which is what catches a lost update).

```
CuTileHistogram: 1048576 values into 256 bins, tile width 256, 4096 tile blocks
CuTileHistogram: every one of the 4096 blocks folds into every one of the 256 bins -- 1048576 contended adds
CuTileHistogram: bins wrong 0/256, counted 1048576 of 1048576 values
CuTileHistogram: result is correct (every bin exact, all values counted)
```

That verdict alone is not proof the GPU ran it. Confirm the codegen:

```bash
tornado --printKernel --jvm="-Dtornado.recover.bailout=False" --classpath . CuTileHistogram 1024 8 1 \
  | grep -E '__tile_global__|ct::atomic_add|ct::select'
```

Evidence: [`results/raw/36-cutile-demos-23-24/`](../../results/raw/36-cutile-demos-23-24/).

## What GPU/CUDA requirements exist?

See Prerequisites. Below any of them the task throws `TornadoDeviceTileNotSupported` naming
the missing requirement; the demo prints `[UNSUPPORTED]` and exits, probing **once** and
never retrying — repeating a failed tile launch has been observed to take the JVM down with
a SIGSEGV.

## What should the presenter point out?

1. **`bv.atomicAdd(count, 0, bin)`** — and the printed contention count. Say what `store`
   would do instead.
2. **There is no scatter.** Show the predicate trio. This is the single most useful thing
   to know when porting an existing CUDA kernel to tiles.
3. **A bin index is a block index** — the trick that lets a runtime-chosen destination
   coexist with compile-time tile shapes.
4. **The accumulator is cleared on the host each execution.** It has to be: the host array
   is what gets uploaded, so without the reset the bins keep summing across iterations. This
   bit the demo during development and the code says so.
5. Scans and reductions are **rank-2 only**, which is why the tile stays `[1,TILE]` and the
   reduction axis is `1`.

## Provenance

```text
Pure cuTile C reference : NVIDIA/TileGym, src/tilegym/ops/tilecpp/moe_align_block.cuh
                          (stage 1: per-block counts folded into a shared array)
Reference revision      : ec339c0dbac3efe61e73ac2b782f818d253eaff2 (MIT)
Reference source        : https://github.com/NVIDIA/TileGym
TornadoVM develop SHA   : 8d592d6fbaafe42d66e1e22c27057cb1cdcf9097
TornadoVM cuTile API    : TileContext.partition/view/full/zeros/greaterOrEqual/lessThan/
                          logicalAnd/select/sum/bidX, PartitionView.load/atomicAdd
Runtime result          : PASS — every bin exact, both run modes, sm_120
```

[`CuTileHistogram.cu`](CuTileHistogram.cu) is the pure cuTile C++ twin, adapted (not copied)
from the reference. Build with
`nvcc -tilecubin --tile-only -std=c++20 -arch=sm_120 -o CuTileHistogram.cubin CuTileHistogram.cu`;
`cuobjdump -sass` on the result shows the atomic instructions.

### Key translation differences

| TileGym `moe_align_block.cuh` | TornadoVM | Why |
| --- | --- | --- |
| `ct::atomic_add(tile, ct::memory_order_relaxed_t{}, ct::thread_scope_device_t{}, idx...)` | `PartitionView.atomicAdd(tile, 0, bin)` | The Java API fixes relaxed ordering at device scope — the javadoc argues correctness comes from per-element atomicity, not from ordering between elements, so a stricter order would cost more and buy nothing |
| `chunk >= ct::broadcast<shape>(float(bin))` | `tc.greaterOrEqual(chunk, bin)` | The Java API takes a `double` scalar directly; the backend materialises the `ct::broadcast`/`ct::iota` pair itself |
| `ct::sum(t, ct::integral_constant<1>{})` | `tc.sum(t, 1)` | Axis is a plain `int`, constant-folded by the plugin |
| counts sized to `PADDED_PROGRAMS` (next power of two) in Python before launch | `bins` is a runtime extent; only the `1×1` partition shape is constant | Extents are runtime in this API |
