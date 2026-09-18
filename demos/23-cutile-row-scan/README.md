# 23 — CUDA Tile row scan

A per-row prefix sum over a **deliberately ragged** row: 1000 columns against a 128-wide
tile. Written against TornadoVM's **CUDA Tile API**.

## What does this demonstrate?

Two things the existing tile demos do not show. Demos [19](../19-cutile-matmul/),
[20](../20-cutile-hybrid/), [21](../21-cutile-flash-attention/) and
[22](../22-matmul-ladder-fp16-tile/) are all matmul-shaped and all divisible-by-the-tile;
between them they use no scan, no masked load or store, and no loop-carried reduction.

**A scan is one call.** The prefix sum is the GPU primitive that normally costs a whole
lecture — Blelloch, up-sweep, down-sweep, bank conflicts. Here:

```java
tc.prefixSum(values, 1)
```

**Ragged extents are the API's two rules colliding.** Tile *shapes* are compile-time
constants; tensor *extents* are runtime. 1000 is not a multiple of 128, so the last tile of
every row hangs 24 lanes off the end of the data. `loadMasked` zero-pads it, `storeMasked`
writes back only the lanes that exist, and a `[1,1]` carry tile rides the loop to join the
tiles into one running total:

```java
Tile carry = tc.zeros(DType.F32, 1, 1);
for (int block = 0; block < colBlocks; block++) {
    Tile values = iv.loadMasked(row, block);
    ov.storeMasked(tc.add(tc.prefixSum(values, 1), carry), row, block);
    carry = tc.add(carry, tc.sum(values, 1));
}
tv.store(carry, row, 0);
```

No tensor cores here at all.

## What TornadoVM feature/API does it use?

`uk.ac.manchester.tornado.api.tile` — `TileContext.prefixSum/sum/add/zeros/partition/view`
and `PartitionView.loadMasked/storeMasked/store`. See [`docs/cutile-api.md`](../../docs/cutile-api.md).

Three behaviours it leans on, each already exercised by upstream tests:

- implicit **broadcast** of a `[1,1]` operand against a `[1,TILE]` tile;
- a partition tile **wider than the extent**, walked with masked load/store;
- a **loop-carried tile**, which the backend turns into a named `ct::tile` rather than an `auto`.

## Prerequisites

CC 8.0+, driver R580+, CUDA Toolkit 13.3+, and a TornadoVM build with the tile API
(the `develop` profile — see [SDK profiles](../../README.md#sdk-profiles)).

```bash
pip install --user nvidia-cuda-nvcc 'cuda-tile[tileiras]'
```

## How do I run it?

```bash
source scripts/setup-env.sh
cd demos/23-cutile-row-scan
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . CuTileRowScan.java

tornado --jvm="-Dtornado.recover.bailout=False" --classpath . CuTileRowScan
java @$TORNADOVM_HOME/tornado-argfile -Dtornado.recover.bailout=False -cp . CuTileRowScan
```

Arguments: `[rows] [cols] [iterations]`, default `4096 1000 20`. `cols` is deliberately
**not** a multiple of the tile width — that is the point of the demo.

> **Always pass `-Dtornado.recover.bailout=False`.** Every `TileContext` method also has a
> plain-Java implementation, and `TornadoOptions.RECOVER_BAILOUT` defaults to **true**. With
> the default, a tile kernel that fails to compile silently falls back to the host, computes
> the right answer, and prints `correct` — while the GPU did nothing.

## How do I validate the result?

Two independent exact checks. Inputs are integers in [-3,3], so every prefix is an integer
bounded by 3000 — exact in fp32 on both paths, so the comparison uses `==`, not a tolerance.
The scan is checked element by element against a sequential running sum, and the row totals
are checked separately.

```
CuTileRowScan: 4096 rows x 1000 cols, tile width 128, 8 tiles per row
CuTileRowScan: last tile of each row has 24 padded lane(s) handled by loadMasked/storeMasked
CuTileRowScan: scan mismatches 0/4096000, row-total mismatches 0/4096
CuTileRowScan: result is correct (bit-exact prefix sums and row totals)
```

That verdict alone is not proof the GPU ran it. Confirm the codegen:

```bash
tornado --printKernel --jvm="-Dtornado.recover.bailout=False" --classpath . CuTileRowScan 8 1000 1 \
  | grep -E '__tile_global__|ct::partial_sum|ct::sum'
```

Observed on sm_120: `ct::partial_sum`, `ct::sum`, `ct::partition_view`, `ct::tile`, `ct::bid`.
Evidence: [`results/raw/36-cutile-demos-23-24/`](../../results/raw/36-cutile-demos-23-24/).

## What GPU/CUDA requirements exist?

See Prerequisites. Below any of them the task throws `TornadoDeviceTileNotSupported` naming
the missing requirement; the demo prints `[UNSUPPORTED]` and exits, probing **once** and
never retrying — repeating a failed tile launch has been observed to take the JVM down with
a SIGSEGV.

## What should the presenter point out?

1. **`tc.prefixSum(values, 1)`** — say out loud what that replaces.
2. **The 24 padded lanes.** The demo prints the count. This is the concrete moment where
   "shapes are compile-time, extents are runtime" stops being a slogan.
3. **The carry is a tile, not a scalar** — `[1,1]`, and `tc.add` broadcasts it across
   `[1,TILE]` with no explicit broadcast call.
4. **Why `totals` needs its own view**: a reduced `[1,1]` tile cannot be stored back into the
   `[1,TILE]` view it came from, so it gets a `[rows,1]` view partitioned `1×1`.
5. **Scans and reductions are rank-2 only**, which is why the tile stays `[1,TILE]` and the
   axis argument is `1`.

## Provenance

```text
Pure cuTile C reference : NVIDIA/TileGym, src/tilegym/ops/tilecpp/moe_align_block.cuh
                          (stage 2 ct::partial_sum scan; stage 3 carried running total)
Reference revision      : ec339c0dbac3efe61e73ac2b782f818d253eaff2 (MIT)
Reference source        : https://github.com/NVIDIA/TileGym
TornadoVM develop SHA   : 8d592d6fbaafe42d66e1e22c27057cb1cdcf9097
TornadoVM cuTile API    : TileContext.partition/view/zeros/add/prefixSum/sum/bidX,
                          PartitionView.loadMasked/storeMasked/store
Runtime result          : PASS — bit-exact, both run modes, sm_120
```

[`CuTileRowScan.cu`](CuTileRowScan.cu) is the pure cuTile C++ twin, adapted (not copied)
from the reference. Build with
`nvcc -tilecubin --tile-only -std=c++20 -arch=sm_120 -o CuTileRowScan.cubin CuTileRowScan.cu`.

### Key translation differences

| TileGym `moe_align_block.cuh` | TornadoVM | Why |
| --- | --- | --- |
| `ct::partial_sum(v, ct::integral_constant<0>{})` | `tc.prefixSum(v, 1)` | Axis is a plain `int` argument, constant-folded by the plugin |
| Masking via a **pointer tile + predicate** (`ptr + ct::iota<...>()`, `mask = iota < N`) | `PartitionView.loadMasked` / `storeMasked` | **Not the same call.** This API exposes no gather/scatter, so the masked partition-view load is how the same intent is expressed — the reference's own `matmul.cuh` uses that form too |
| Tile padded to `PADDED_PROGRAMS` (next power of two) in Python before launch | the power-of-two constraint is on the `partition` shape; the padding is the mask | |
| `last_cumsum = last_cumsum + padded_cnt` inside `ct::irange` | `carry = tc.add(carry, tc.sum(values, 1))` | Same loop-carried `[1,1]` tile |
