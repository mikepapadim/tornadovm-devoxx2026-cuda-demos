# 05 — Java kernel + cuFFT in one execution graph

**Concept (read in ~1 minute):** the same Hybrid API pattern as demo `04`
(cuBLAS), but with cuFFT and two JIT-compiled Java kernels sandwiched
between two library calls, all sharing device buffers in one `TaskGraph`,
no host round trip:

1. `forward` — `CuFft.cufftForwardR2C` (`nvidia/cufft`), a real-to-complex
   forward FFT of a real signal into the non-redundant half of its Hermitian
   spectrum (interleaved complex, `2 * (n/2 + 1)` floats).
2. `lowPass` — a plain `@Parallel` Java loop, JIT-compiled to CUDA, zeroes
   every frequency bin at or above `cutoff` directly on the cuFFT output
   buffer.
3. `inverse` — `CuFft.cufftInverseC2R`, the inverse FFT back to a real
   signal (unnormalized, per cuFFT semantics: `c2r(r2c(x)) = n * x`).
4. `normalize` (method name `scaleBy`) — another `@Parallel` Java loop
   divides by `n` to undo cuFFT's unnormalized scaling.

The input is a deterministic sum of two low-frequency tones (kept) and one
high-frequency tone (removed by the filter), so the filtered output has an
exact closed-form expected value — no numeric DFT reference needed.

Source-backed pattern this demo is modeled on:
`vendor/tornadovm/tornado-cufft/.../tests/FrequencyFilterExample.java`
(same R2C → JIT-filter → C2R → JIT-normalize shape, read in full this
batch) and `vendor/tornadovm/tornado-unittests/.../cufft/TestCuFft.java`
(confirms the R2C/C2R pairing and unnormalized-FFT semantics against a
Java DFT reference). Written fresh for this repo with CLI-configurable
`<n> <cutoff> <iterations>` and per-iteration profiler timing, matching the
shape of demo `04`.

Source: [`CuFftLowPassHybrid.java`](CuFftLowPassHybrid.java).

**Gotcha found while building this demo:** naming the fourth task's method
`normalize` fails at sketch time with `[ERROR] Java method name corresponds
to an OpenCL Token. Change the Java method's name: normalize` — TornadoVM's
`TornadoSketcher` rejects that identifier as a reserved OpenCL token even
though this demo only ever targets the CUDA backend (the sketcher check is
shared across backends). Upstream's `FrequencyFilterExample` and `TestCuFft`
both independently use `scaleBy` for the same reason — not a coincidence,
confirmed by hitting the same compiler error here.

## Build

```bash
source scripts/setup-env.sh   # from repo root; pins the SDK in env/versions.env
cd demos/05-cufft-hybrid
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  -d . CuFftLowPassHybrid.java
```

## Run

Canonical, with profiler output showing all four stages (`forward`,
`lowPass`, `inverse`, `normalize`) executing on the CUDA backend:

```bash
tornado --enableProfiler console --classpath . CuFftLowPassHybrid 4096 16 5
```

Reproducibility form (`java @arg-file`):

```bash
java @$TORNADOVM_HOME/tornado-argfile -cp . CuFftLowPassHybrid 4096 16 5
```

Arguments: `<n> <cutoff> <iterations>` (defaults `4096 16 5` if omitted).
`n` is the signal length, `cutoff` the first frequency bin removed by the
filter (must be greater than both tone frequencies, 3 and 7, and less than
or equal to `n/2 + 1`).

## Expected output

```
Low-pass filter via hybrid cuFFT pipeline: n=4096, cutoff bin=16, iterations=5
iteration 0: correct maxError=4.7683716E-7 filtered[0]=0.49999997
  total task-graph time: 86065183 ns
iteration 1: correct maxError=4.7683716E-7 filtered[0]=0.49999997
  total task-graph time: 462627 ns
...
All iterations correct
```

With `--enableProfiler console`, each iteration's JSON block shows
`cufftHybrid.forward`, `cufftHybrid.lowPass`, `cufftHybrid.inverse`, and
`cufftHybrid.normalize` — each with `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA
GeForce RTX 4090"` — confirming both cuFFT library tasks and both JIT
kernels execute on the CUDA device inside a single task graph. The first
iteration's `TOTAL_TASK_GRAPH_TIME` (~86 ms) is dominated by Graal + driver
JIT compilation of the two Java tasks (`TOTAL_GRAAL_COMPILE_TIME` ~48 ms,
`TOTAL_DRIVER_COMPILE_TIME` ~11 ms); iterations 1–4 drop to ~380–460
microseconds once compiled code is reused — the same presenter-visible
compile-then-reuse effect as demo `04`.

Re-verified on TornadoVM 6.0.0 / JDK 25 — 5/5 iterations correct, max abs
error `4.7683716E-7` under both run paths:
`results/raw/18-tornadovm-6-migration/05-cufft-tornado.log`,
`05-cufft-javaargfile.log`.

Earlier 5.2.1 logs:
`results/raw/05-cufft-hybrid/cufftlowpasshybrid-run.log`,
`cufftlowpasshybrid-run-javaargfile.log`. Upstream sanity check (same
pipeline shape, run before writing this demo):
`results/raw/05-cufft-hybrid/upstream-frequencyfilterexample.log`.

## CUDA equivalent

[`CuFftLowPassHybrid.cu`](CuFftLowPassHybrid.cu) is the same demo written directly in CUDA C++, for side-by-side comparison.

```bash
nvcc -arch=sm_89 -lcufft -o cufft_lowpass CuFftLowPassHybrid.cu && ./cufft_lowpass
```

Four stages alternating library and hand-written kernels, by hand. Two cuFFT
plans (one per direction), the shared device buffers, and the same
`1.0f / n` normalisation the Java version needs — cuFFT's inverse is
unnormalised in both.

Output is bit-identical: `maxError=4.76837e-07`, `filtered[0]=0.49999997`,
matching the Java run exactly.

`bash scripts/run-all-cuda.sh` builds and runs the CUDA equivalent of every demo (needs only the CUDA toolkit, no JDK).

## JBang

Not verified: `jbang` is not installed on this machine (`which jbang` →
exit 1, checked 2026-08-20, same finding as demos `00`–`04`). The shape
would match those demos' documented-but-unverified pattern — do not run it
live until tested on the pinned environment:

```bash
jbang --version
jbang -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  --java-opts="@$TORNADOVM_HOME/tornado-argfile" \
  CuFftLowPassHybrid.java
```

## If the demo fails on stage

- If any iteration prints `WRONG`, fall back to the captured log at
  `results/raw/05-cufft-hybrid/cufftlowpasshybrid-run.log` and walk through
  the JSON profiler block explaining the four-stage graph instead.
- Re-run `tornado --devices` first — if it does not show exactly one CUDA
  device, the environment, not the demo, is broken.
- `NoClassDefFoundError` for `uk.ac.manchester.tornado.cufft.*` usually
  means the classpath is missing `tornado-cufft-6.0.0.jar` — add
  it alongside `tornado-api-6.0.0.jar` (see Build above).
- If a rebuild ever reintroduces a `[ERROR] Java method name corresponds to
  an OpenCL Token` sketch failure, rename the offending task method (see
  the gotcha above) — this is a TornadoSketcher restriction, not a runtime
  bug.
