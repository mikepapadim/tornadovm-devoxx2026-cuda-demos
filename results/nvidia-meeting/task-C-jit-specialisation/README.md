# Task C — JIT specialisation, with SASS from both sides

> **Correction.** Earlier revisions of this bundle recorded task C as *partial*,
> stating that TornadoVM-side SASS was "not capturable because the cubin is
> produced in-process by NVRTC and never written to disk", and raised that with
> NVIDIA as an open question. **That was wrong.** TornadoVM writes every compiled
> cubin to an on-disk module cache **by default**, and `cuobjdump` reads them
> directly. The open question has been withdrawn.

sm_89 / RTX 4090, driver 565.57.01, CUDA 12.6.85, JDK 25.0.2,
TornadoVM 6.0.0-jdk22plus-cuda (SDKMAN release).

## Where TornadoVM puts the cubin

`CUDACodeCache` maintains an on-disk module cache so a warm start can skip NVRTC
entirely. It is controlled by:

| Property | Default |
|---|---|
| `tornado.cuda.codecache.enable` | **`True`** |
| `tornado.cuda.codecache.dir` | `/var/cuda-codecache` |

The cache key is a digest over the kernel source, device, driver and compiler
flags, so a hit on a different GPU or after a toolkit upgrade is impossible.
Files are named `<entryPoint>-<key>.cubin`.

**One trap worth recording.** `resolveDirectory` builds the path as

```java
Paths.get(System.getenv("TORNADOVM_HOME") + "/" + dir + "/" + deviceDir)
```

so the configured directory is **concatenated onto `$TORNADOVM_HOME`** rather
than used as given. Passing an absolute path such as
`-Dtornado.cuda.codecache.dir=/tmp/cc` does not write to `/tmp/cc`; it writes to
`$TORNADOVM_HOME/tmp/cc/device-0-0`. On this host the default lands at:

```
$TORNADOVM_HOME/var/cuda-codecache/device-0-0/
```

which held 78 cubins from prior runs with no flags set at all.

```bash
cuobjdump -sass "$TORNADOVM_HOME/var/cuda-codecache/device-0-0/polynomial-<key>.cubin"
```

## Result — the specialisation is visible in SASS

`polynomial` computes a dependent FMA chain of length `degree`, passed as a task
argument (`degree = 256`).

| Kernel (TornadoVM) | SASS instructions | branch-class instrs | FFMA |
|---|---|---|---|
| `polynomial` | 288 | **1** | **256** |
| `elementwise` | 32 | 1 | 1 |
| `stencil` | 56 | 1 | 0 |

**256 FFMA for `degree = 256`, and one branch.** The loop is gone: Graal compiled
the kernel after the argument was bound, so the chain is fully unrolled and the
only remaining branch is the bounds-check exit.

For comparison, the hand-written CUDA variant taking `degree` as a **runtime
scalar** contains **13** branch-class instructions
(`results/raw/21-kernel-time-comparison/cuda-sass.log`) — a real loop, as an AOT
compiler must emit without knowing the value.

Timing already recorded in `summary.md`:

| Variant | Kernel time |
|---|---|
| TornadoVM, `degree` as task argument | 35,236 ns |
| nvcc, `degree` as runtime scalar | ~39,500 ns |
| nvcc, `degree` as template constant | ~34,700 ns |

**The mechanism is now shown, not inferred**: unrolled straight-line FMA chain
against a loop, in SASS, on both sides. Give nvcc the same compile-time
information and it reaches parity (34.7 µs vs 35.24 µs), so the advantage is the
*timing* of compilation, not the quality of the arithmetic.

## Why this matters for the CUDA Tile discussion

The specialisation advantage exists because compilation happens **after argument
values are known**. Any alternative lowering path — Tile C++, Tile IR, NVVM —
would have to preserve that property to keep it, and would have to fit inside
the compile budget measured in task F (42.2 ms Graal + 16.3 ms NVRTC per task
graph, cold).

## Files

| File | Contents |
|---|---|
| `tornadovm-polynomial-sass.txt` | full `cuobjdump -sass` output |
| `tornadovm-polynomial.cubin` | the cubin it was taken from |
