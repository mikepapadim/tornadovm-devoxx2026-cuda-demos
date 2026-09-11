# 19 — the same GEMM written with threads, then with tiles

Three rungs, one FP16 `C = A * B`, increasing in how much of the GPU stays hidden:

| # | Rung | What the author writes |
|---|---|---|
| 1 | `@Parallel`-style naive | one thread per output element |
| 2 | `KernelContext` tiled | shared-memory tiles, indexed and barriered by hand |
| 3 | **`TileContext`** | `acc = tc.mma(a, b, acc)` — no thread, warp, fragment or barrier |

Rung 3 compiles through **NVIDIA CUDA Tile**, not the SIMT path: TornadoVM emits CUDA Tile
C++, `nvcc -tilecubin` compiles it, and NVIDIA's `tileiras` backend chooses the tensor-core
instruction, the layouts and the thread count. All three rungs are validated against a CPU
reference computed over the same FP16-rounded inputs.

Source: [`TileMatMul.java`](TileMatMul.java), hand-written CUDA in
[`TileMatMul.cu`](TileMatMul.cu).

## Requirements

This demo does **not** run on the pinned TornadoVM 6.0.0 SDK — `TileContext` does not exist
there. It needs a TornadoVM built from the cuTile branch, CUDA Toolkit **13.3+** and driver
**R580+** (R610+ for the in-process NVRTC route). See `env/versions.env`, section
*CUDA Tile*.

## Run

```bash
export TORNADOVM_HOME=<cutile SDK>          # not the 6.0.0 SDK
export JAVA_HOME=$HOME/.sdkman/candidates/java/21.0.2-open
cd demos/19-cutile-matmul

# --enable-preview: the cuTile branch is a jdk21-dev build (the released
# 6.0.0-jdk22plus-cuda SDK does not need it)
javac --release 21 --enable-preview -proc:none -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . TileMatMul.java
tornado --classpath . TileMatMul 256 10
```

Observed on an RTX 4090 (sm_89), driver 610.57.04, nvcc 13.3.73:

```
rung                                         ms/execution     vs naive   max error   result
1. @Parallel-style naive (KernelContext)            0.446        1.00x      0.0000   correct
2. KernelContext tiled (shared memory)              0.321        1.39x      0.0000   correct
3. TileContext (CUDA Tile, ct::mma)                 0.221        2.02x      0.0000   correct
```

`n` must be a multiple of 32.

## Read the generated code, not only the clock

The wall-clock column includes JVM-side dispatch, which at these sizes is most of it. The
hand-written CUDA version measures the same three kernels with CUDA events, and there the
picture is different — the hand-tiled rung is *slower* than naive at `n = 256`:

```
$ nvcc --enable-tile -std=c++20 -arch=sm_89 -o tilematmul TileMatMul.cu && ./tilematmul 256 10
1. naive (one thread per element)                   0.031         1.00x      0.0000   correct
2. tiled (shared memory, by hand)                   0.048         0.65x      0.0000   correct
3. CUDA Tile (ct::mma)                              0.014         2.24x      0.0000   correct
```

What the demo is really for is the generated source:

```bash
tornado --printKernel --classpath . TileMatMul 64 1
```

Rung 3 prints `extern "C" __tile_global__ void tiles(...)` whose body is `ct::partition_view`,
`ct::zeros` and `ct::mma` — and **no** `asm volatile("mma.sync...")`. That absence is the
point: the tensor cores are still there, chosen by the tile compiler, not spelled out by the
author. In the cached cubin:

```bash
cuobjdump -sass $TORNADOVM_HOME/var/cuda-codecache/device-0-0/tiles-*.cubin | grep -c HMMA
# 32 at n = 256 (HMMA.16816.F32), 8 at n = 64
```

Compare with demo [08](../08-tensor-core-mma/) and demo [18](../18-matmul-ladder-fp16/),
where reaching the same tensor cores means writing `ctx.mma` against a fixed
`M16N8K16` shape and packing fragments by hand.

## If it fails

* `TornadoDeviceTileNotSupported` naming the toolkit — the nvcc on the path is older than
  13.3. Point `-Dtornado.cuda.nvcc` at a newer one; a userspace
  `pip install --user 'cuda-tile[tileiras]' nvidia-cuda-cccl` is enough and needs no root.
* `CUDA_ERROR_INVALID_IMAGE` or a load failure — the driver is older than R580.
* `uses preview features of Java SE 21` from `javac` — add `--release 21 --enable-preview`.
* Falls back to sequential Java with a bailout warning — run with `--debug` to see why;
  rung 3 bails out rather than miscompiling if a tile shape is not a compile-time constant.
