# 13 — cuDNN + JIT conv block

**Concept (read in ~1 minute):** the classic CNN block is
`conv → bias → activation`. cuDNN owns the hard parts — the convolution and the
activation — but the bias add is trivial and application-specific, exactly the
kind of step you would rather write in Java than look up in a vendor API.

TornadoVM's hybrid API lets both live in **one** `TaskGraph`, on the same device
buffers, with no round trip to the host between stages:

```
1. JIT   "scale"    Java @Parallel kernel — normalises the input
2. cuDNN "conv2d"   NCHW convolution, 3x3, pad 1, stride 1
3. JIT   "addBias"  Java @Parallel kernel — per-output-channel bias
4. cuDNN "relu"     activation
```

Every execution is validated against a sequential Java reference that runs the
identical four steps on the CPU.

Source: [`CuDnnConvBlockHybrid.java`](CuDnnConvBlockHybrid.java).

## Build

```bash
source scripts/setup-env.sh   # from repo root; pins the SDK in env/versions.env
cd demos/13-cudnn-jit-convblock
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . CuDnnConvBlockHybrid.java
```

## Run

```bash
tornado --classpath . CuDnnConvBlockHybrid                       # defaults: 4 16 32 32 16 10
java @$TORNADOVM_HOME/tornado-argfile -cp . CuDnnConvBlockHybrid
```

Arguments: `<N> <C> <H> <W> <K> <executions>` — batch, input channels, height,
width, output channels, executions. Filters are fixed at 3x3, pad 1, stride 1,
so the output keeps the input's spatial dimensions.

## Expected output

```
cuDNN + JIT conv block: NCHW 4x16x32x32, 16 filters 3x3 (pad 1, stride 1)
  graph: JIT scale -> cuDNN conv2d -> JIT addBias -> cuDNN relu
  10 executions, steady-state median reported (first execution excluded)

  first execution (JIT compile + cuDNN plan setup): <large> us
  steady-state median wall-clock (n=9): 367 us
  validation PASSED (max abs err 0.000000, 0/65536 elements out of tol)
Result is correct
```

`max abs err 0.000000` is the headline: a four-stage pipeline alternating
between a vendor library and JIT-compiled Java reproduces the CPU reference
exactly, in fp32.

## Profiling with Nsight Systems

```bash
nsys profile --trace=cuda --force-overwrite=true -o cudnn \
  tornado --classpath . CuDnnConvBlockHybrid 4 16 32 32 16 10

nsys stats --report cuda_gpu_kern_sum --format csv cudnn.nsys-rep
```

Example output (RTX 4090, TornadoVM 6.0.0, 10 executions):

```
Time (%)  Total Time (ns)  Instances  Avg (ns)   Name
    61.4            63841         10    6384.1   implicit_convolve_sgemm<float, ...>
    14.6            15136         10    1513.6   op_generic_tensor_kernel<(int)1, ...>
    12.1            12575         10    1257.5   scale
    11.9            12352         10    1235.2   addBias
```

This one table *is* the demo. Four kernels, ten instances each — one per
execution — and they are exactly the four graph stages:

| Kernel in the trace | Where it came from |
|---|---|
| `implicit_convolve_sgemm` | cuDNN's convolution — a vendor kernel you did not write |
| `op_generic_tensor_kernel` | cuDNN's ReLU |
| `scale` | **your Java method name**, JIT-compiled to CUDA |
| `addBias` | **your Java method name**, JIT-compiled to CUDA |

Your Java methods appear in an NVIDIA profiler's timeline under their own
names, interleaved with cuDNN's kernels, sharing device buffers. Nothing copies
back to the host between stages — confirm that with:

```bash
nsys stats --report cuda_gpu_mem_time_sum --format csv cudnn.nsys-rep
```

The convolution dominates at 61% of GPU kernel time, and the two JIT tasks
together cost about as much as cuDNN's ReLU — a useful, concrete answer to
"is writing that bit in Java going to cost me?"

Open it visually with `nsys-ui cudnn.nsys-rep`.

Captured evidence: `results/raw/19-cutlass-cudnn-warp-demos/13-cudnn-nsys-kernsum.csv`,
`13-cudnn.nsys-rep`, run logs `13-cudnn-tornado.log` / `13-cudnn-javaargfile.log`.

## If the demo fails on stage

- `validation FAILED` with a large error usually means N/C/H/W/K were changed
  in a way the reference and the cuDNN call disagree on — re-run with the
  defaults first.
- If cuDNN cannot be loaded, check the SDK is the CUDA build
  (`cat $TORNADOVM_HOME/etc/tornado.backend` → `cuda-backend`) and that
  `libcudnn` is on the system library path.
- Fall back to the captured logs and CSV in
  `results/raw/19-cutlass-cudnn-warp-demos/`.

## Not used here: `CuDnn.sdpaForward`

The 6.0.0 SDK also exposes a fused scaled-dot-product-attention task. It is
**not** used in this demo because the SDK's own shipped benchmark
(`tornado -m tornado.cudnn/uk.ac.manchester.tornado.cudnn.tests.BenchmarkSdpa`)
returns an all-zero result on this machine and prints
`Results DO NOT match`. Reported upstream — see the repo README's
"Upstream issues filed" section. Do not demo it live.

## CUDA equivalent

[`CuDnnConvBlockHybrid.cu`](CuDnnConvBlockHybrid.cu) is the same demo written directly in CUDA C++, for side-by-side comparison.

```bash
nvcc -arch=sm_89 -I/usr/include/x86_64-linux-gnu -lcudnn -o cudnn_conv_block CuDnnConvBlockHybrid.cu && ./cudnn_conv_block
```

This is the pair where the hybrid API earns the most. Two `libraryTask` calls in
Java become, in CUDA: a handle, two tensor descriptors, a filter descriptor, a
convolution descriptor, an activation descriptor, an algorithm choice, a
workspace-size query, a workspace allocation, and six matching `destroy` calls.

```c
cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, R, S);
cudnnSetConvolution2dDescriptor(convDesc, PAD, PAD, STRIDE, STRIDE, 1, 1,
                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
cudnnGetConvolutionForwardWorkspaceSize(cudnn, inDesc, filterDesc, convDesc,
                                        outDesc, algo, &workspaceBytes);
```

`CUDNN_CROSS_CORRELATION` is the trap: pick `CUDNN_CONVOLUTION` instead and
cuDNN flips the filter, so the result stops matching any reference written the
obvious way. The Java `cudnnConv2d` picks cross-correlation for you — which is
why the Java demo's reference matches with no filter flip.

Both versions validate at **max abs err `0.000000`**. Steady-state median
wall-clock: TornadoVM 367 µs, CUDA 69 µs.

138 lines of Java against 186 of CUDA, and nearly all of the difference is
descriptor plumbing.

`bash scripts/run-all-cuda.sh` builds and runs the CUDA equivalent of every demo (needs only the CUDA toolkit, no JDK).

## JBang

Not verified: `jbang` is not installed on this machine (`which jbang` → exit 1,
re-checked 2026-09-02). Documented-but-untested shape — do not run live:

```bash
jbang -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  --java-opts="@$TORNADOVM_HOME/tornado-argfile" \
  CuDnnConvBlockHybrid.java
```
