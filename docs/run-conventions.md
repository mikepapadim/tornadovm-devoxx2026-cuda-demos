# Run Conventions

## Java arg-files

Every demo must document the explicit `java @arg-file <MainClass> ...` form
alongside the `tornado` launcher, and that form must be *run*, not just
written down — `scripts/run-all-demos.sh` executes both paths for all nine
demos and is the gate.

On TornadoVM 6.0.0 the arg-file is produced by the SDK itself:

```bash
tornado --generate-argfile                              # writes $TORNADOVM_HOME/tornado-argfile
java @$TORNADOVM_HOME/tornado-argfile -cp . <MainClass> [args]
```

**Do not commit a generated arg-file to this repo.** Its contents are absolute
paths into the installed SDK and JDK-specific flags (`-XX:+EnableJVMCI` is
required on JDK ≤ 26 and fatal on JDK 27+), so a committed copy is wrong for
every machine but the one that generated it. `scripts/setup-env.sh`
regenerates it on each setup, which is also what makes switching JDK a no-op.

Do not assume an arg-file name or location is stable across TornadoVM
releases — it changed at 6.0.0, from a repo-local `demos/tornado.args` to
`$TORNADOVM_HOME/tornado-argfile`. Inspect the installed SDK and record the
exact file used.

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
