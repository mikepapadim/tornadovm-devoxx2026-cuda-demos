# 30 — FP32 GEMM: three KernelContext kernels and four tile shapes

**Concept (read in ~1 minute):** the same FP32 `C = A * B` written seven ways, validated
against the full CPU reference on random inputs.

| # | Rung | What it is |
|---|---|---|
| 1 | `oneThreadPerOutput` | KernelContext, one thread per output element, 16×16 work-groups |
| 2 | `sharedTiled` | KernelContext, 16×16 shared-memory tiles, two barriers per k-step |
| 3 | `registerTiled` | KernelContext, 64×64 block tile with k-steps of 16; 16×16 threads, each owning a 4×4 register patch of C |
| 4–7 | `tile…` | TileContext with **FP32 operands** into `tc.mma`, at 32×32×32, 64×64×64, 128×128×32, 128×128×64 |

Rungs 1–3 are the classic SIMT progression, the same steps as demo 17's rungs 1–3 written
with KernelContext throughout: rung 1 is the KernelContext form of demo 17's `@Parallel`
rung, and rung 3 stages its tiles with strided cooperative loads (each thread loads one
element per `stride` rows) instead of demo 17's flat index split. Rungs 4–7 are demo 25's
tile ladder with `FloatArray` instead of `HalfFloatArray`: same body, FP32 tensors.

Source: [`MatMulFP32Variants.java`](MatMulFP32Variants.java).

## Build and run

```bash
source ../../scripts/setup-env.sh
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . MatMulFP32Variants.java

tornado --jvm="-Dtornado.recover.bailout=False" --classpath . MatMulFP32Variants 1024 10
java @$TORNADOVM_HOME/tornado-argfile -Dtornado.recover.bailout=False -cp . MatMulFP32Variants 1024 10
```

Arguments: `<n> <executions>`; `n` must be a multiple of 128 (default 1024). The program
ends with `All seven rungs produced the same, correct result`. As in every tile demo, pass
`-Dtornado.recover.bailout=False` so a tile kernel that fails to compile errors out instead
of silently running on the host.

## Kernel time

The printed numbers are wall clock (dispatch + kernel + copy-out). Compare kernels with
`nsys`:

```bash
nsys profile --trace=cuda --force-overwrite=true -o fp32 \
  java @$TORNADOVM_HOME/tornado-argfile -Dtornado.recover.bailout=False -cp . MatMulFP32Variants 2048 10
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv fp32.nsys-rep
```

Kernel time of the same kernels on an RTX 5070 Ti (sm_120), TornadoVM `develop` with
PR [#1194](https://github.com/beehive-lab/TornadoVM/pull/1194), nsys median, measured with
the harness in
[beehive-lab/tornado-babylon-comparison](https://github.com/beehive-lab/tornado-babylon-comparison)
(variants `kcNaive`, `kcTiled`, `kcRegTiled`, `tileF_*`):

| Rung | n = 2048 | TFLOP/s | n = 8192 | TFLOP/s |
|---|---:|---:|---:|---:|
| 1. one thread per output | 6,089 µs | 2.8 | 436.7 ms | 2.5 |
| 2. shared-memory tiled | 4,149 µs | 4.1 | 299.2 ms | 3.7 |
| 3. **register-tiled** | **834 µs** | **20.6** | **61.8 ms** | **17.8** |
| 4. TileContext 32×32×32 | 6,790 µs | 2.5 | 459.8 ms | 2.4 |
| 5. TileContext 64×64×64 | 6,804 µs | 2.5 | 429.9 ms | 2.6 |
| 6. TileContext 128×128×32 | 9,698 µs | 1.8 | 735.5 ms | 1.5 |
| 7. TileContext 128×128×64 | 18,390 µs | 0.9 | 1,237 ms | 0.9 |

Two things to take from it:

- **The register micro-tile is the step that matters**: 5× over shared-memory tiling, 7×
  over one thread per output.
- **FP32 tiles are not the tensor-core path.** With `float` operands `tc.mma` gets no
  tensor-core speed-up here, and the larger tile shapes are slower, not faster. For tensor
  cores use FP16/BF16 operands (demos 19, 22, 25).

## Requirements

TornadoVM 7.0.0+ with the tile API (rungs 4–7), CUDA Toolkit 13.3+ with `tileiras`, driver
R580+. On a TornadoVM without PR #1194, tile k-loops are fully unrolled: at n = 2048 the
128×128 shapes take 45–105 s to compile (or hit the 120 s tile-compiler timeout). Use
n ≤ 1024 there, or a TornadoVM with #1194.

## Validation

Every rung is checked against a CPU reference over the **whole** output matrix, on random
inputs in [−0.5, 0.5), with a relative tolerance of `1e-4 · √n` (FP32 accumulation in a
different order).
