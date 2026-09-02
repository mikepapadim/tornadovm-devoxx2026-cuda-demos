# 00 — Hello GPU

**Concept (read in ~1 minute):** the smallest possible TornadoVM program. One
Java method (`addOne`), annotated with `@Parallel`, runs on the GPU instead
of the CPU. A `TaskGraph` describes the three steps TornadoVM needs: copy
data to the device, run the task, copy the result back. No CUDA C, no JNI,
no manual memory management — this is plain Java compiled and dispatched to
the GPU by TornadoVM's CUDA backend.

Source: [`Hello.java`](Hello.java).

## Build

```bash
source scripts/setup-env.sh   # from repo root; pins the SDK in env/versions.env
cd demos/00-hello-gpu
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  -d . Hello.java
```

No `--enable-preview` anywhere: the pinned `6.0.0-jdk22plus-cuda` SDK is a
non-preview build (`etc/tornado.jdk`: floor 22, preview false), so any JDK 22+
compiles and runs it. The JDK-21-only `6.0.0-jdk21-cuda` SDK is the one that
still needs preview flags — this repo does not use it.

## Run

Canonical (`tornado` launcher, resolves the CUDA backend automatically):

```bash
tornado --classpath . Hello
```

Reproducibility form (`java @arg-file`, per `docs/run-conventions.md`):

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp . Hello
```

`tornado --generate-argfile` writes `$TORNADOVM_HOME/tornado-argfile`.
It is generated into the installed SDK rather than committed here because its
flags are absolute-path and JDK-specific; `scripts/setup-env.sh` regenerates it.

## Expected output

```
in:  [0, 1, 2, 3, 4, 5, 6, 7]
out: [1, 2, 3, 4, 5, 6, 7, 8]
```

Add `--enableProfiler console` to `tornado` to confirm the execution device,
e.g. `"BACKEND": "CUDA", "DEVICE": "NVIDIA GeForce RTX 4090"`. Captured logs:
`results/raw/02-hello-kernel/hello-run.log`,
`results/raw/02-hello-kernel/hello-run-profiler.log`,
`results/raw/02-hello-kernel/hello-run-javaargfile.log`.

Re-verified on TornadoVM 6.0.0 / JDK 25, both run paths:
`results/raw/18-tornadovm-6-migration/00-hello-tornado.log`,
`00-hello-javaargfile.log`.

## CUDA equivalent

[`Hello.cu`](Hello.cu) is the same demo written directly in CUDA C++, for side-by-side comparison.

```bash
nvcc -arch=sm_89 -o hello Hello.cu && ./hello
```

Same program without TornadoVM. The `@Parallel` loop becomes a `__global__`
kernel with an explicit bounds check, and everything the `TaskGraph` implied
becomes a line of host code: `cudaMalloc` per buffer, `cudaMemcpy` in both
directions, a launch configuration you compute yourself, and `cudaFree`.

Output is identical:

```
in:  [0, 1, 2, 3, 4, 5, 6, 7]
out: [1, 2, 3, 4, 5, 6, 7, 8]
```

33 lines of Java against 48 of CUDA — the gap here is small, and most of it is
error handling and memory management rather than the computation itself.

`bash scripts/run-all-cuda.sh` builds and runs the CUDA equivalent of every demo (needs only the CUDA toolkit, no JDK).

## JBang

Not verified: `jbang` is not installed on this machine (`which jbang` →
command not found, checked 2026-08-19). Documented-but-unverified shape,
matching TornadoVM's own JBang guidance — do not claim this works until
tested on the pinned environment:

```bash
jbang --version
jbang -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  --java-opts="@$TORNADOVM_HOME/tornado-argfile" \
  Hello.java
```

## If the demo fails on stage

- Re-run `tornado --devices` — if it does not show exactly one CUDA device,
  the environment, not the demo, is broken; fall back to showing the
  captured logs in `results/raw/02-hello-kernel/`.
- If the JVM refuses to start with "built for JDK 21 with preview features
  enabled", `TORNADOVM_HOME` points at `6.0.0-jdk21-cuda` instead of
  `6.0.0-jdk22plus-cuda`; re-run `source scripts/setup-env.sh`.
