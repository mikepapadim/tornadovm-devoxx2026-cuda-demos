# 01 — First CUDA Kernel

**Concept (read in ~1 minute):** a slightly more realistic kernel —
element-wise vector addition over 1024 floats — and proof that TornadoVM
really compiles your Java loop down to CUDA. `--printKernel` prints the
generated `extern "C" __global__ void vectorAdd(...)` source TornadoVM's
JIT produced from the `@Parallel` loop in [`VectorAddKernel.java`](VectorAddKernel.java),
before it is compiled with `nvcc`/`ptxas` and launched on the device.

## Build

```bash
source scripts/setup-env.sh   # from repo root; pins the SDK in env/versions.env
cd demos/01-first-cuda-kernel
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  -d . VectorAddKernel.java
```

## Run

Canonical, with the generated CUDA source and profiler output:

```bash
tornado --enableProfiler console --printKernel --classpath . VectorAddKernel 1024
```

Reproducibility form (`java @arg-file`):

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp . VectorAddKernel 1024
```

## Expected output

Generated CUDA kernel (captured verbatim in
`results/raw/02-hello-kernel/vectoraddkernel-run.log`; re-verified on
TornadoVM 6.0.0 in `results/raw/18-tornadovm-6-migration/01-vectoradd-tornado.log`
and `01-vectoradd-javaargfile.log`):

```c
extern "C" __global__ void vectorAdd(long long *_kernel_context, unsigned char *_constant_region, unsigned char *_local_region, int *_atomics, unsigned char *arg0, unsigned char *arg1, unsigned char *arg2)
{
  ...
  f_14  =  f_10 + f_12;
  *(( float *) ul_13)  =  f_14;
  ...
}
```

Followed by:

```
Result is correct
Total time: <ns>
```

and a profiler JSON block confirming `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"`.

## JBang

Not verified: `jbang` is not installed on this machine (checked 2026-08-19,
same as [`../00-hello-gpu/README.md`](../00-hello-gpu/README.md)). The
shape would be the same `jbang -cp ... --java-opts="@$TORNADOVM_HOME/tornado-argfile"
VectorAddKernel.java` pattern — do not run it live until it has been tested
on the pinned environment.

## If the demo fails on stage

- If `--printKernel` output looks wrong or missing, fall back to the
  captured log at `results/raw/02-hello-kernel/vectoraddkernel-run.log`.
- If the run reports `Result is WRONG`, stop and do not claim GPU
  correctness live; re-run `tornado --devices` to check for a device/driver
  regression since `env/versions.env` was recorded.
