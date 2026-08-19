# Run Conventions

## Java arg-files

When the current TornadoVM checkout or demo provides a Java argument file, prefer the explicit `java @arg-file <MainClass> ...` form in reproducibility notes and presenter runbooks.

Example:

```bash
java @.ffi-cuda-example <example>.Main
```

Do not assume an arg-file name exists on every TornadoVM revision. First inspect the checked-out tree and record the exact file used.

## JBang

For demos that can be expressed as a standalone Java source or a small set of sources, include a JBang invocation in the run instructions when it genuinely works with the pinned TornadoVM CUDA environment.

Example shape:

```bash
jbang --version
jbang Demo.java
```

For TornadoVM-specific demos, the run instructions must show how the TornadoVM runtime/classpath is made available. Do not claim that plain `jbang Demo.java` is sufficient unless it has been tested on the pinned environment.

Every demo task should record both:

1. the canonical Java/arg-file command used for reproducibility; and
2. a JBang command or explicitly documented reason that JBang is unsuitable for that demo.
