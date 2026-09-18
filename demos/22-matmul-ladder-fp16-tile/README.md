# 22 — the FP16 ladder, with a CUDA Tile rung

Demo [18](../18-matmul-ladder-fp16/) is this ladder without rung 4. Rung 4 is why this demo
exists: the same FP16 GEMM written against `TileContext` and compiled through **NVIDIA CUDA
Tile**, measured the same way as the hand-written tensor-core rung above it and the vendor
libraries below it.

| # | Rung | What the author writes |
|---|---|---|
| 1 | naive `@Parallel` | one thread per output element |
| 2 | `KernelContext` tiled | shared-memory tiles, staged and barriered by hand |
| 3 | `KernelContext` MMA | fragment packing + `ctx.mma` → `mma.sync.aligned.m16n8k16` |
| 4 | **`TileContext`** | `tc.mma(a, b, acc)` → `ct::mma`; the tile compiler picks the HMMA |
| 5 | CUTLASS `hgemm` | library task |
| 6 | cuBLAS `GemmEx` FP16 | library task |
| 7 | cuBLAS `GemmEx` FP16→FP32 | library task, the inference configuration |

Rungs 3 and 4 are the comparison worth reading: same language, same hardware, same output.
Rung 3 packs two fp16 values per `int`, stages shared memory, names a fixed `m16n8k16` shape
and issues two `mma` calls per 16×16 tile. Rung 4 is nine lines and names no shape at all.

Every rung is validated against a CPU reference computed over the same FP16-rounded inputs.

Source: [`MatMulLadderFP16Tile.java`](MatMulLadderFP16Tile.java).

## Requirements

Needs a TornadoVM with the tile API (the `develop` SDK profile, not the pinned 6.0.0 SDK), CUDA Toolkit
**13.3+** and driver **R580+**. See [SDK profiles](../../README.md#sdk-profiles) and [`docs/cutile-api.md`](../../docs/cutile-api.md).

## Run

> The `--release 21 --enable-preview` flags this demo used to need are **gone**. They existed
> because the tile demos were pinned to the `feat/cutile` branch, a jdk21-dev build. PR #1083
> is merged into upstream `develop`, so the `develop` SDK profile is a jdk22plus build and
> this demo compiles and runs on the same JDK 25 as every other demo.
> Verified 2026-09-18: `results/raw/37-demo-matrix-develop/`.


```bash
source ../../scripts/setup-env.sh           # profile `develop` -- has the tile API
cd demos/22-matmul-ladder-fp16-tile
javac -proc:none -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . MatMulLadderFP16Tile.java
tornado --jvm="-Dtornado.recover.bailout=False" --classpath . MatMulLadderFP16Tile 1024 20
```

`n` must be a multiple of 64.

## Kernel time is the number that means something

Observed on an RTX 4090 (sm_89), driver 610.57.04, nvcc 13.3.73, `n = 1024`, via
`bash scripts/compare-ladder.sh 22 1024 20` (Nsight Systems, steady state):

| Kernel | GPU avg | GFLOP/s | vs naive |
|---|---|---|---|
| naive | 537.2 µs | 3 997 | 1.0× |
| `kcTiled` | 331.3 µs | 6 482 | 1.6× |
| `kcMma` (hand-written `mma.sync`) | 181.7 µs | 11 821 | 3.0× |
| **`tiles` (CUDA Tile)** | **65.7 µs** | **32 666** | **8.2×** |
| CUTLASS | 29.9 µs | 71 916 | 18.0× |
| cuBLAS `ampere_*_s1688gemm_fp16` | 20.2 µs | 106 488 | 26.6× |

So the tile rung is **2.8× faster than the hand-written MMA rung** it sits next to, and still
**3.3× slower than cuBLAS**. Both halves of that matter: the tile compiler beats what a
careful author writes by hand against the same tensor cores, and a vendor library tuned per
architecture is still ahead of a single 32×32 tile shape with no pipelining hints.

The wall-clock summary the demo prints is much flatter — 2.0× for the tile rung instead of
8.2× — because at `n = 1024` every rung pays the same JVM-side dispatch. Quote the nsys table.

Nsight Compute explains the ranking (same command, counters section):

| Kernel | instructions | bank conflicts | gmem stall | registers |
|---|---|---|---|---|
| `kcMma` | 810 752 | 292 558 | 15.90 | 48 |
| `tiles` | 214 784 | 7 717 | 0.50 | 110 |

The hand-written rung spends its time on the staging the tile compiler does not need: a
quarter of the instructions, a fortieth of the bank conflicts, and essentially no unhidden
global-memory latency — paid for with more registers.

## Verify it is really CUDA Tile

```bash
tornado --printKernel --classpath . MatMulLadderFP16Tile 64 1 | grep -c "asm volatile"   # 0 for the tile rung
cuobjdump -sass $TORNADOVM_HOME/var/cuda-codecache/device-0-0/tiles-*.cubin | grep -c HMMA
```

Rung 3's kernel contains `asm volatile("mma.sync.aligned.m16n8k16...")`; rung 4's contains
`ct::mma` and no PTX, and both cubins hold `HMMA` instructions.

## The hand-written CUDA equivalent

`MatMulLadderFP16Tile.cu` is the same ladder in CUDA C++, so each rung can be read against
the Java one. It is demo 18's equivalent plus rung 4:

```bash
pip install --user nvidia-cuda-nvcc 'cuda-tile[tileiras]' nvidia-cuda-cccl
nvcc --enable-tile -std=c++20 -arch=sm_120 -O3 -o matmul_ladder_fp16_tile \
     MatMulLadderFP16Tile.cu -lcublas
./matmul_ladder_fp16_tile 256 20
```

`--enable-tile` mixes tile kernels and host code in one translation unit, which is what lets
this be an executable at all. TornadoVM instead drives `nvcc -tilecubin --tile-only` to a
bare cubin and loads it itself, because it has no host translation unit to put the launch in.

Read rung 3 against rung 4 in that file: rung 3 packs fragments by lane, indexes a 32-thread
warp and names `m16n8k16` in inline PTX; rung 4 is `ct::mma` and nothing else. Both reach the
tensor cores.

Unlike demo 18's equivalent, every rung here is validated against a CPU reference, so a fast
wrong rung cannot pass as a win. CUTLASS is omitted so the file builds with the plain toolkit.

> The timings this binary prints are **not** comparable to the table above. That table is
> `nsys` kernel time at n=1024 on an RTX 4090; this is CUDA-event time at whatever size you
> pass, on whatever GPU you run. Do not quote one as if it confirmed the other.


## If it fails

* `TornadoDeviceTileNotSupported` — nvcc older than 13.3 on the path; point
  `-Dtornado.cuda.nvcc` at a 13.3+ one.
* Rung 3 fails while rung 4 passes — that is the MMA intrinsic path, not the tile path; check
  `n` is a multiple of 64.
* `uses preview features of Java SE 21` — you are on the old `feat/cutile` SDK. Use the `develop` profile (`source scripts/setup-env.sh`); no preview flags are needed there.
