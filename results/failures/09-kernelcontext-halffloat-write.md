# OPEN (2026-09-11) — in-place `HalfFloat` write from a `KernelContext` kernel yields zeros

Date: 2026-09-11. Found while building `demos/20-cutile-hybrid`.
**Not a CUDA Tile issue** — the reproducer contains no tile code.

## What happens

A `KernelContext` kernel that writes a `HalfFloatArray` element it has just read,
constructing the value with `new HalfFloat(...)`, leaves the buffer **zeroed**. No exception,
no bailout warning, no wrong-size complaint: the kernel runs and the result is zeros.

```java
public static void scale(KernelContext ctx, HalfFloatArray a, float f) {
    int i = ctx.globalIdx;
    a.set(i, new HalfFloat(a.get(i).getFloat32() * f));
}
```

Input `1..8`, factor `2.0`:

```
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0   (expected 2 4 6 8 10 12 14 16)
```

## Reproducer

`results/raw/33-cutile-demos/HalfWrite.java` — 25 lines, one task, one `WorkerGrid1D`.

```bash
export TORNADOVM_HOME=<cutile SDK>      # any CUDA SDK; nothing here is tile-specific
export JAVA_HOME=$HOME/.sdkman/candidates/java/21.0.2-open
javac --release 21 --enable-preview -proc:none -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . HalfWrite.java
tornado --classpath . HalfWrite
```

## Environment

TornadoVM built from the cuTile branch (6.1.1-jdk21-dev), CUDA backend, RTX 4090 (sm_89),
driver 610.57.04, JDK 21.0.2. Observed with the `tornado` launcher.

## How it surfaced

`demos/20-cutile-hybrid` originally scaled its FP16 matrix in place in stage 1. Every
execution reported `WRONG` with the output equal to the bias constant alone, i.e. the GEMM
consumed a zero matrix. Diagnosis went: check whether the scale ran at all (the error matched
neither "scaled once" nor "never scaled"), then print per-row ratios (`actual - BIAS` over
expected was exactly 0), then reduce to the 25-line probe above, which reproduces with no
tile task, no library task and no graph.

## Not diagnosed further

The generated SIMT kernel has not been inspected (`--printKernel`) and no narrowing of
whether the fault is in the read, the `HalfFloat` construction, the write, or the dataflow
analysis that marks the buffer for transfer back. Recorded here so the demo's workaround has
a reason on file; worth a focused look before it is reported upstream, since it is a silent
wrong answer rather than a failure.

## Workaround in the demo

Stage 1 scales the FP32 vector instead of the FP16 matrix — the pre-pass belongs on the
vector anyway, and the pipeline keeps its four stages and its two thread-level kernels.
