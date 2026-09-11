# 20 — a JIT kernel, a CUDA Tile kernel and cuBLAS in one graph

One `TaskGraph`, four stages, one stream, shared device buffers, no host round-trip:

| Stage | Task | Path |
|---|---|---|
| 1 | `scale` | `KernelContext` — thread-level Java kernel, SIMT |
| 2 | `gemm` | **`TileContext`** — CUDA Tile kernel, tensor cores via `ct::mma` |
| 3 | `project` | `libraryTask` — cuBLAS `sgemv` |
| 4 | `biasRelu` | `@Parallel` — thread-level Java kernel again |

This is the claim the whole cuTile integration rests on: a tile kernel is an ordinary module
launch on the plan's stream, so it chains with SIMT tasks and with vendor libraries under the
existing rules — including `withCUDAGraph()`, which captures all four stages, tile task
included, and replays them as one submission.

Source: [`TileHybridPipeline.java`](TileHybridPipeline.java), hand-written CUDA in
[`TileHybridPipeline.cu`](TileHybridPipeline.cu).

## Requirements

Needs a TornadoVM built from the cuTile branch (not the pinned 6.0.0 SDK), CUDA Toolkit
**13.3+** and driver **R580+**. See `env/versions.env`, section *CUDA Tile*.

## Run

```bash
export TORNADOVM_HOME=<cutile SDK>
export JAVA_HOME=$HOME/.sdkman/candidates/java/21.0.2-open
cd demos/20-cutile-hybrid
javac --release 21 --enable-preview -proc:none -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . TileHybridPipeline.java
tornado --classpath . TileHybridPipeline 256 20 both      # n, executions, nograph|graph|both
```

Observed on an RTX 4090 (sm_89), driver 610.57.04, nvcc 13.3.73:

```
pipeline: KernelContext scale -> CUDA Tile gemm -> cuBLAS sgemv -> @Parallel biasRelu

   max error 0.0000 of max |value| 21.1709 (tolerance 0.2217)
no CUDA graph       0.396 ms/execution   correct
   max error 0.0000 of max |value| 21.1709 (tolerance 0.2217)
withCUDAGraph()     0.073 ms/execution   correct

CUDA graph replay: 0.396 ms -> 0.073 ms per execution, 5.44x faster
0.323 ms of host dispatch saved per execution of this 4-stage graph
```

## What that 5.44x is and is not

It is **host dispatch**, not faster kernels. The hand-written CUDA version runs the identical
four stages and measures them with CUDA events:

```bash
nvcc --enable-tile -std=c++20 -arch=sm_89 -lcublas -o tilehybrid TileHybridPipeline.cu
./tilehybrid 256 20
# stream launches      0.021 ms/execution   correct
# CUDA graph           0.018 ms/execution   correct
# CUDA graph replay: 0.021 ms -> 0.018 ms per execution, 1.14x faster
```

At the CUDA level graph capture buys 1.14x; the Java-level 5.44x is that plus the JVM-side
submission of four tasks that capture removes. Both numbers are real and they measure
different things — quote them together, as demo [07](../07-cuda-graph-benefit/) does.

## Verify the composition

```bash
tornado --printBytecodes --classpath . TileHybridPipeline 256 2 graph
```

Expect one `EXECUTION_GRAPH_BEGIN_CAPTURE` / `END_CAPTURE` pair around all four launches and
`EXECUTION_GRAPH_LAUNCH` on later executions, with **no** `TRANSFER_DEVICE_TO_HOST` between
stages — the intermediates never leave the GPU. Under `nsys`, the four stages appear as four
kernels on one stream, the cuBLAS one inside an `nvidia/cublas/...` NVTX range.

## One trap, found here and since fixed

An earlier version of stage 1 scaled the FP16 matrix in place with
`inOut.set(i, new HalfFloat(inOut.get(i).getFloat32() * factor))` inside the `KernelContext`
kernel. The pipeline then produced a zero result with no error reported.

The cause was in the backend, not the demo: a half-float read lowered to a node the
sketch-tier dataflow analysis did not recognise, so the array was classified `WRITE_ONLY`
and the runtime skipped its host-to-device copy — the kernel read an uninitialised buffer.
Fixed in TornadoVM `75ae022` on the cuTile branch (a `MarkReadNode` marker, plus the
`TestHalfFloatInPlaceUpdate` regression suite); write-up in
[`results/failures/09-kernelcontext-halffloat-write.md`](../../results/failures/09-kernelcontext-halffloat-write.md).

The demo still scales the FP32 vector, which is where a pre-pass belongs anyway, and it keeps
its two thread-level kernels either way.

## If it fails

* `TornadoDeviceTileNotSupported` — nvcc older than 13.3 on the path; point
  `-Dtornado.cuda.nvcc` at a 13.3+ one.
* The cuBLAS stage throws `UnsatisfiedLinkError` or "provider not found" — the launcher adds
  the provider modules automatically; the `java @argfile` path needs them on
  `--add-modules`, which `tornado --generate-argfile` already includes.
* `WRONG` with a large error — check `n` is a multiple of 32; a ragged `n` needs the masked
  load variants, which this demo does not use.
