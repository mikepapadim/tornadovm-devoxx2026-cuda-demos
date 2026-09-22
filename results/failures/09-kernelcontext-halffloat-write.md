# RESOLVED (2026-09-11) — in-place `HalfFloat` write from a `KernelContext` kernel yields zeros

Date: 2026-09-11. Found while building `demos/20-cutile-hybrid`.
**Not a CUDA Tile issue** — the reproducer contains no tile code.

**Resolved the same day**, in TornadoVM commit `75ae022` on branch `feat/cutile`. The cause
was not the write: a half-float **read** lowers to a backend `ReadHalfFloatNode` rather than
to a `ReadNode`/`JavaReadNode`, and `TornadoDataflowAnalysis` did not recognise it, while it
did recognise the matching write through the `MarkWriteNode` marker. A kernel reading and
writing the same `HalfFloatArray` was therefore classified `WRITE_ONLY`, the runtime skipped
the host-to-device copy for that parameter, and the kernel read an uninitialised device
buffer — which is why the answer was zeros rather than an error. Evidence: `--debug` printed
`access: parameter 1 -> WRITE_ONLY`, and `--printBytecodes` showed **no**
`TRANSFER_HOST_TO_DEVICE` despite `transferToDevice(EVERY_EXECUTION, a)`.

The fix adds the missing counterpart marker, `MarkReadNode`, implemented by the CUDA, OpenCL
and Metal `ReadHalfFloatNode` and handled in `TornadoDataflowAnalysis` (and in
`TornadoFeatureExtraction`, whose read counter under-counted half-float reads for the same
reason). Regression suite `TestHalfFloatInPlaceUpdate`: four tests that all fail on the
unfixed analysis and pass with it. The analysis below is kept because the diagnosis path —
that the output equalled the bias constant alone, i.e. the GEMM consumed a zero matrix — is
what made the reproducer findable.

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

## Where it was, in the end

The generated CUDA was correct all along — `--printKernel` shows the read, the
`__half2float`, the multiply and the store back to the same address. Nothing was wrong inside
the kernel; the device buffer it read had never been filled. The fault was one missing
`instanceof` in the sketch-tier access classification, and a marker interface that existed for
writes but not for reads.

One detail worth keeping: a test written the obvious way does **not** catch this. Calling
`inOut.getSize()` inside the kernel reads a field of the array, which the analysis counts as a
read of the parameter, so the classification comes out `READ_WRITE` anyway. The regression
tests pass the size in as a parameter for that reason.

## Workaround in the demo

Stage 1 scales the FP32 vector instead of the FP16 matrix — the pre-pass belongs on the
vector anyway, and the pipeline keeps its four stages and its two thread-level kernels.
