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

Needs a TornadoVM built from the cuTile branch (not the pinned 6.0.0 SDK), CUDA Toolkit
**13.3+** and driver **R580+**. See `env/versions.env`, section *CUDA Tile*.

## Run

```bash
export TORNADOVM_HOME=<cutile SDK>
export JAVA_HOME=$HOME/.sdkman/candidates/java/21.0.2-open
cd demos/22-matmul-ladder-fp16-tile
javac --release 21 --enable-preview -proc:none -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . MatMulLadderFP16Tile.java
tornado --classpath . MatMulLadderFP16Tile 1024 20
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

## If it fails

* `TornadoDeviceTileNotSupported` — nvcc older than 13.3 on the path; point
  `-Dtornado.cuda.nvcc` at a 13.3+ one.
* Rung 3 fails while rung 4 passes — that is the MMA intrinsic path, not the tile path; check
  `n` is a multiple of 64.
* `uses preview features of Java SE 21` — add `--release 21 --enable-preview`.
