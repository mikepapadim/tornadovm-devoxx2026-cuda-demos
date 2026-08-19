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
source vendor/tornadovm/setvars.sh   # from repo root; pins the CUDA-backend build in env/versions.env
cd demos/00-hello-gpu
javac --release 21 --enable-preview \
  -cp "$TORNADOVM_HOME/share/java/tornado/tornado-api-5.2.1-jdk21-dev.jar" \
  -d . Hello.java
```

`--enable-preview` is required because `tornado-api` ships compiled with
Java preview features on this pinned build (JDK 21).

## Run

Canonical (`tornado` launcher, resolves the CUDA backend automatically):

```bash
tornado --classpath . Hello
```

Reproducibility form (`java @arg-file`, per `docs/run-conventions.md`):

```bash
java @../tornado.args -cp . Hello
```

`demos/tornado.args` was generated on this machine with
`tornado --generate-argfile` against the pinned build and committed for
reference; regenerate it with that command if the JDK or TornadoVM build
changes.

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

## JBang

Not verified: `jbang` is not installed on this machine (`which jbang` →
command not found, checked 2026-08-19). Documented-but-unverified shape,
matching TornadoVM's own JBang guidance — do not claim this works until
tested on the pinned environment:

```bash
jbang --version
jbang -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  --javac-opts="--release 21 --enable-preview" \
  --java-opts="@../tornado.args" \
  Hello.java
```

## If the demo fails on stage

- Re-run `tornado --devices` — if it does not show exactly one CUDA device,
  the environment, not the demo, is broken; fall back to showing the
  captured logs in `results/raw/02-hello-kernel/`.
- If compilation fails with a preview-feature error, confirm `--release 21
  --enable-preview` is present on the `javac` command.
