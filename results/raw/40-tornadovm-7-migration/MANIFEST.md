# Batch 33 — TornadoVM 7.0.0 CUDA migration

Captured 2026-09-22. Every Track A demo re-built and re-run on the TornadoVM
7.0.0 CUDA release installed from SDKMAN, replacing the `6.0.0-jdk22plus-cuda`
pin used by batches 18–30.

## Environment

| | |
|---|---|
| TornadoVM | `7.0.0-jdk22plus-cuda` (SDKMAN), release commit `65eb834` (tag `v7.0.0`) |
| JDK | OpenJDK 25.0.2 (`25.0.2-open`) |
| GPU | NVIDIA GeForce RTX 4090, driver 610.57.04 |
| CUDA toolkit | 12.6.85 (`nvcc`/`ptxas`) |
| CUDA 13 runtime | `libcudart.so.13` from the pip `nvidia-cu13` wheel, on `LD_LIBRARY_PATH` |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-58-generic |

`tornado.release`, `tornado.jdk`, `tornado-devices.log`, `java-version.log`,
`nvidia-smi.log`, `tornado-argfile.generated` and `native-lib-ldd.log` are
verbatim copies taken at capture time.

## Result

All sixteen Track A demos compile and run correctly under **both** supported run
paths — the `tornado` launcher and `java @$TORNADOVM_HOME/tornado-argfile`.
48/48 checks pass (`run-all-demos.log`, produced by `scripts/run-all-demos.sh`).

**No demo source needed an API change.** All sixteen compile unmodified against
`tornado-api-7.0.0`. Every API the demos use is still present: `MMAShape` is
byte-for-byte the same enum (`M16N8K16`, `M16N8K32`), and `CuDnn` still exposes
`cudnnConv2d`, `cudnnRelu`, `cudnnMaxPool2d`, `cudnnSigmoid`, `cudnnSoftmax`,
`cudnnTanh` and `sdpaForward`.

## The one migration blocker: CUTLASS is linked against CUDA 13

7.0.0 ships `lib/libtornado-cutlass.so` linked against `libcudart.so.13`. This
machine's CUDA toolkit is 12.6, so on a bare `LD_LIBRARY_PATH` the three demos
that use a CUTLASS library task — **12, 17, 18** — abort at that task with:

```
TornadoBailoutRuntimeException: [TornadoVM] Error - Recover option disabled
Caused by: TornadoRuntimeException: [ERROR] Unable to load libtornado-cutlass.
  ... /lib/libtornado-cutlass.so: libcudart.so.13: cannot open shared object file
```

That is 42/48 — `run-all-demos-without-cuda13.log`. Putting any CUDA 13 runtime
on `LD_LIBRARY_PATH` fixes all three with no source change: 48/48,
`run-all-demos.log`. `scripts/setup-env.sh` now resolves one and warns if it
cannot (`CUDA13_RUNTIME_LIB` in `env/versions.env`).

`lib/libtornado-cudnn.so` is linked against CUDA 13 **and** `GLIBC_2.38` (this
box has 2.35), so it is unloadable here too — but demo 13 passes anyway, because
demo 13's conv2d/relu path never loads it (`CuDnn.sdpaForward` does — see the
batch 35 audit): `strace` shows the process dlopening the system
`/lib/x86_64-linux-gnu/libcudnn.so.9` and its `libcudnn_{graph,ops,cnn,...}.so.9`
engines directly. Only the CUTLASS path still goes through a TornadoVM-shipped
JNI `.so`. See `native-lib-ldd.log`.

## Scope of this batch

Correctness only. **No timing was re-measured on 7.0.0.** Every performance
number elsewhere in this repository retains its 5.2.1 or 6.0.0 provenance and is
labelled as such; none of it has been relabelled as a 7.0.0 result.

## Files

| File | What it is |
|---|---|
| `run-all-demos.log` | `scripts/run-all-demos.sh` with a CUDA 13 runtime present — 48/48 |
| `run-all-demos-without-cuda13.log` | the same harness without one — 42/48, demos 12/17/18 failing |
| `native-lib-ldd.log` | `ldd` of `libtornado-cudnn.so` and `libtornado-cutlass.so`, showing the CUDA 13 and GLIBC_2.38 links |
| `tornado-devices.log` | `tornado --devices` on 7.0.0 |
| `tornado.release`, `tornado.jdk` | SDK release/JDK-contract files, verbatim |
| `tornado-argfile.generated` | the argfile `tornado --generate-argfile` produced for JDK 25 |
| `java-version.log`, `nvidia-smi.log` | toolchain and GPU state at capture time |
