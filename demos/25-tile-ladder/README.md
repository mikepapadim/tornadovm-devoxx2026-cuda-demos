# 25 — The TileContext ladder vs. an optimised KernelContext and native CUDA Tile

**Concept (read in ~1 minute):** the same FP16 `C = A * B` (FP32 accumulate, FP32 out)
written three ways and measured in kernel time on one GPU:

- **KernelContext**, simple and then **fully optimised** — the most a Java programmer can do
  by hand: warps, fragments, shared-memory layout, a `cp.async` pipeline.
- **TileContext** — a ten-line kernel that never names a thread, climbed as a ladder over the
  only two knobs a tile kernel has: its **tile shape** and a **launch hint**.
- **Native CUDA Tile** — the same tile kernels hand-written in CUDA Tile C++
  ([`TileLadder.cu`](TileLadder.cu)), to see what TornadoVM adds or costs.

| # | Rung | What it is |
|---|---|---|
| 1 | `kcSimple` | KernelContext, one warp per 16×16 tile, operands packed from global memory each k-step (demo 22's rung 3) |
| 2 | `kcOptimised` | KernelContext, 128×128 block tile, 8 warps × 64×32 warp tiles, 16 register fragments per warp, `cp.async` double buffering |
| 3–6 | `tile…` | TileContext, the same kernel at tile shapes 32×32×32, 64×64×64, 128×128×32, 128×128×64 |
| 7 | `tile128x128x64Hinted` | rung 6 compiled with the CUDA Tile launch hint `occupancy=2` |
| 8 | cuBLAS | `GemmEx` FP16→FP32, the vendor ceiling |

Source: [`TileLadder.java`](TileLadder.java) · hand-written counterpart:
[`TileLadder.cu`](TileLadder.cu).

## Build and run

```bash
source ../../scripts/setup-env.sh          # default profile sdkman-7.0.0: has the tile API
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . TileLadder.java

tornado --jvm="-Dtornado.recover.bailout=False" --classpath . TileLadder 2048 10
java @$TORNADOVM_HOME/tornado-argfile -Dtornado.recover.bailout=False -cp . TileLadder 2048 10
```

Arguments: `<n> <executions>`; `n` must be a multiple of 128 (default 2048). Every rung is
validated against a full CPU reference over the same FP16 inputs; the program ends with
`All rungs produced the same, correct result`.

**Always pass `-Dtornado.recover.bailout=False`.** Every `TileContext` method has a plain-Java
fallback, so without it a tile kernel that fails to compile runs on the host, prints
`PASSED`, and looks like a very slow pass.

Needs: an SDK with the tile API (TornadoVM 7.0.0+), CUDA Toolkit 13.3+ with `tileiras` for
the tile path (`pip install --user nvidia-cuda-nvcc 'cuda-tile[tileiras]' nvidia-cuda-cccl`),
tensor cores (sm_80+). Rung 8 uses TornadoVM's cuBLAS bridge.

## Kernel time — the honest comparison

The program prints wall clock. **Do not compare rungs on it**: at n=2048 every execution
also copies the 16 MB result back, which flattens the ladder (rung 2 and rung 6 both read
~1.2 ms). Kernel time is what compares code:

```bash
nsys profile --trace=cuda --force-overwrite=true -o ladder \
  $JAVA_HOME/bin/java @$TORNADO_ARGFILE -Dtornado.recover.bailout=False -cp . TileLadder 2048 10
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv ladder.nsys-rep
```

RTX 4090 (sm_89), TornadoVM 7.0.0, n = 2048, 10 executions; mean of 3 runs, spread ≤ 0.9%
(`results/raw/45-tile-ladder/`):

| Rung | kernel time | TFLOP/s | vs cuBLAS |
|---|---|---|---|
| 1. KernelContext, simple | 1006.0 µs | 17.1 | 8.58× |
| 2. **KernelContext, optimised** | **143.8 µs** | **119.5** | **1.23×** |
| 3. TileContext 32×32×32 | 1123.7 µs | 15.3 | 9.58× |
| 4. TileContext 64×64×64 | 502.5 µs | 34.2 | 4.28× |
| 5. TileContext 128×128×32 | 529.1 µs | 32.5 | 4.51× |
| 6. TileContext 128×128×64 | 201.7 µs | 85.2 | 1.72× |
| 7. **TileContext 128×128×64 + `occupancy=2`** | **137.4 µs** | **125.0** | **1.17×** |
| 8. cuBLAS `GemmEx` FP16→FP32 | 117.3 µs | 146.5 | 1.00× |

Three things to take from it:

- **Hand-tuning KernelContext is worth 7.0×** (rung 1 → rung 2), and lands at 1.23× of cuBLAS.
- **The tile shape alone is worth 5.6×** (rung 3 → rung 6) without changing a line of the
  kernel body. The k-depth matters as much as the block: 128×128 at k=32 is no faster than
  64×64×64; at k=64 it is 2.6× faster.
- **With one launch hint, the ten-line tile kernel edges past the optimised KernelContext
  kernel** (137.4 vs 143.8 µs, 1.05×) and gets within 1.17× of cuBLAS.

### How rung 2 was tuned

Rung 2 is not a first draft. It is the best of 11 tilings of the same pipelined kernel,
each validated and timed under `nsys` (`results/raw/45-tile-ladder/tuning/kernelcontext-tilings.csv`):

| block tile | warps | warp tile | k per stage | kernel time |
|---|---|---|---|---|
| **128×128** | **8 (2×4)** | **64×32** | **16** | **143.4 µs** |
| 128×128 | 4 (2×2) | 64×64 | 16 | 145.6 µs |
| 128×128 | 4 (2×2) | 64×64 | 32 | 146.8 µs |
| 64×128 | 4 (2×2) | 32×64 | 16 | 153.4 µs |
| 128×128 | 8 (4×2) | 32×64 | 16 | 153.1 µs |
| 128×128 | 8 (2×4) | 64×32 | 32 | 159.5 µs |
| 128×64 | 4 (2×2) | 64×32 | 16 | 167.6 µs |
| 128×256 / 256×128 | 8 | 64×64 | 16 | 193.9 / 194.5 µs |
| 128×128 | 8 (2×4) | 64×32 | 64 | 197.4 µs |
| 64×64 | 4 (2×2) | 32×32 | 16 | 200.1 µs |

Deeper k-stages and bigger blocks lose. The likely reason is that they raise shared memory
and registers per block and cut how many blocks an SM can hold — not profiled.

**What the KernelContext API cannot express, as of 7.0.0**, that tuned CUDA GEMMs commonly
use: `cp.async` copies wider than 4 bytes (`asyncCopyToLocal` moves one packed pair), a
swizzled A layout (`mmaLoadBSwizzled` exists for B only), and deeper pipelines without
hand-writing each buffer. They are the likely candidates for the remaining 1.23× to cuBLAS,
along with its per-shape kernel selection — **not measured**.

The fragment loads (`mmaLoadA/B` with a byte offset) lower to `ldmatrix.x4` / `ldmatrix.x2.trans`
and use `threadIdx.x & 31` as the lane, so multi-warp blocks are safe — verified in
`CUDALIRStmt.java` at the `v7.0.0` tag before the kernel was written.

### How the tile shapes were chosen

14 shapes were probed, each in its own JVM (a failed tile launch has been known to take the
JVM down), each validated (`tuning/tilecontext-shapes.csv`). Beyond the four in the demo:
64×128×64 is 204.9 µs, 128×64×64 217.0, 128×128×128 241.8, 64×64×128 255.9; larger blocks
fall off (256×128×64 595.6, 256×256×64 897.5, 256×128×32 1174.8). 128×128×64 is the best.

The launch hint is set with `-Dtornado.cuda.tile.hints`. TornadoVM reads it each time it
compiles a tile kernel, so the demo sets it for rung 7 only; `--printKernel` shows it landing
on that kernel alone as `[[ using cutile : hint(0, occupancy=2) ]]`. The keys are not
validated — a misspelt key compiles and does nothing — so check the attribute. Measured on
rung 6's kernel (`tuning/tilecontext-launch-hints.csv`): none 201.1 µs, `occupancy=1` 201.2,
**`occupancy=2` 137.8**, `occupancy=4` 902.6.

## Native CUDA Tile: why the hand-written kernels are slower — and how to fix them

`TileLadder.cu` writes the same tile kernels in CUDA Tile C++ the way this repo's other tile
twins (demos 19 and 22) write them: `n` is a kernel argument.

```bash
nvcc --enable-tile -std=c++20 -arch=sm_89 -O3 -o tile_ladder TileLadder.cu -lcublas
nsys profile --trace=cuda --force-overwrite=true -o native ./tile_ladder 2048 10
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv native.nsys-rep
```

Same GPU, n = 2048, mean of 3 runs, spread ≤ 0.9%:

| Rung | native CUDA Tile | TileContext (Java) | native / Java |
|---|---|---|---|
| 32×32×32, runtime `n` | 1976.6 µs | 1123.7 µs | 1.76× |
| 64×64×64, runtime `n` | 1068.8 µs | 502.5 µs | 2.13× |
| 128×128×32, runtime `n` | 931.9 µs | 529.1 µs | 1.76× |
| 128×128×64, runtime `n` | 916.7 µs | 201.7 µs | **4.54×** |

**Written this way, native CUDA Tile is 1.8–4.5× slower than TileContext.** The reason is in
TornadoVM's generated source (`tornado --printKernel`,
`results/raw/45-tile-ladder/tuning/tornadovm-generated-128x128x64.cu.txt`). TornadoVM compiles a
tile kernel after its arguments are known, so the kernel it hands to `nvcc` carries three
facts the hand-written one does not:

```cpp
auto tview_3 = ct::partition_view{ct::tensor_span{
    ct::assume_aligned(reinterpret_cast<__half *>(ul_0 + 16), 16_ic),   // (b) alignment promise
    ct::extents{2048, 2048}},                                           // (a) n as a constant
    ct::shape{128_ic, 64_ic}};
...
auto tile_11 = ct::mma(tile_9, tile_10, tile_8);                        // (c) k-loop written out:
auto tile_12 = tview_3.load(i_6, 1);                                    //     32 load/load/mma
...                                                                     //     steps, no loop
```

The kernel's `int n` argument is still there, and unused. Part 2 of `TileLadder.cu` gives the
native 128×128×64 kernel those facts one at a time:

| Native 128×128×64 | kernel time | matches |
|---|---|---|
| N4. runtime `n` | 916.7 µs | |
| N5. + constant `n` + `assume_aligned` | 327.0 µs | |
| N6. + k-loop as straight-line code | **200.8 µs** | TileContext rung 6 (201.7) |
| N7. + `occupancy=2` hint | **136.2 µs** | TileContext rung 7 (137.4) |
| cuBLAS, same process | 115.4 µs | |

**With the same information, native CUDA Tile and TileContext are the same speed.** TornadoVM
adds no overhead in the kernel and compiles it no differently: its exact generated kernel,
compiled natively, runs at 200.2 µs (`tuning/native-attribution-128x128x64.csv`, "replica").
The whole difference is what the JIT knows.

Each fact alone is worth little, which is why this is easy to miss:

- the alignment promise alone makes it *slower* (1024.9 µs); constant `n` alone gives 799.4 µs;
  both together 325–327 µs;
- `#pragma unroll` on the loop does **not** give (c): 325.8 µs, the same as no pragma. Only
  writing the steps out does.

An AOT program pays for (a) and (c) with a kernel per problem size — `TileLadder.cu`
compiles its specialised kernels for n = 256 and n = 2048 only and skips them otherwise. A JIT
gets them for every `n` it is called with. It is the same structural advantage demo 15 shows
on a SIMT kernel.

## Validation

Every rung, both files, is checked against a CPU reference over the same FP16-rounded inputs;
all report `max abs err 0.0000` at n = 256 and n = 2048. Tolerance scales with the k-length:
`0.05 · n / 256`, relative to `max(1, |expected|)`.

## What this demo does *not* claim

- **One GPU, one problem shape.** sm_89, square n = 2048, FP16 in / FP32 out. The best tile
  shape, the best KernelContext tiling and the value of `occupancy=2` are all tuned for this;
  `occupancy=4` was 4.5× *slower* here. Re-tune on other hardware.
- **Rung 2 is the best of the tilings tried, not a proof of optimality.** The API limits
  above are the ones that were hit, not an exhaustive list.
- **The `occupancy` hint's mechanism was not profiled.** It is measured, not explained.
- **cuBLAS picks different kernels in the two processes** (`ampere_s1688gemm…` from
  TornadoVM's bridge, `cutlass_80_tensorop_s16816gemm…` from the `.cu`), at 117.3 vs 115.4 µs.

## If the demo fails on stage

- `[UNSUPPORTED] TileLadder` — the SDK profile has no tile API; `source scripts/setup-env.sh`
  with the default `sdkman-7.0.0` profile.
- `sh: 1: tileiras: not found` — the CUDA 13.3 wheel's `bin` is not on `PATH`;
  `setup-env.sh` adds it from `TORNADO_NVCC`.
- A tile rung prints `PASSED` but takes seconds — you forgot
  `-Dtornado.recover.bailout=False` and it ran on the host.
- Fall back to the captured logs in `results/raw/45-tile-ladder/`.

## Related

- **Demo 22** — the FP16 ladder with one tile rung; this demo's rung 1 is its rung 3
- **Demo 19** — the smallest TileContext GEMM
- **Demo 15** — JIT specialisation beating AOT on a SIMT kernel, the same effect as part 2 above
- **Demo 17 / 18** — the FP32 and FP16 SIMT ladders

Captured evidence: `results/raw/45-tile-ladder/`.
