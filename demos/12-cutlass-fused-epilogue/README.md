# 12 — CUTLASS fused epilogue

**Concept (read in ~1 minute):** a GEMM library computes `C = A * B`. CUTLASS
also lets you fold what comes *after* the multiply — the bias add and the
activation — into the same kernel, so the MxN result never leaves registers.
The alternative is a second pass that writes C to global memory and reads it
straight back.

This demo runs both shapes on the same inputs, in one JVM, each as a single
`TaskGraph` that mixes Java JIT tasks with a CUTLASS library task:

```
fused    : JIT "scale" -> CUTLASS gemmBiasRelu                 (1 GPU kernel)
unfused  : JIT "scale" -> CUTLASS hgemm -> JIT "biasRelu"      (2 GPU kernels)
```

Both are validated against the same sequential Java reference.

Source: [`CutlassFusedEpilogue.java`](CutlassFusedEpilogue.java).

## Build

```bash
source scripts/setup-env.sh   # from repo root; pins the SDK in env/versions.env
cd demos/12-cutlass-fused-epilogue
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . CutlassFusedEpilogue.java
```

## Run

```bash
tornado --classpath . CutlassFusedEpilogue                       # defaults: 1024 1024 1024 20
java @$TORNADOVM_HOME/tornado-argfile -cp . CutlassFusedEpilogue
```

Arguments: `<M> <N> <K> <executions>`. **M, N and K must all be multiples of
32** — CUTLASS tiles the problem and the library task rejects anything else.

## Expected output

```
CUTLASS fused epilogue demo: C[1024x1024] = relu(scale(A[1024x1024]) * B[1024x1024] + bias), fp16
  20 executions per mode, steady-state median reported (first execution excluded)

=== FUSED (CUTLASS gemmBiasRelu — one kernel) ===
  first execution (JIT compile + CUTLASS plan setup): <large> us
  steady-state median wall-clock (n=19): 317 us
  [fused  ] validation PASSED (max abs err 0.00781, 0/1048576 cells out of tol)

=== UNFUSED (CUTLASS hgemm + JIT bias/ReLU — two kernels) ===
  first execution (JIT compile + CUTLASS plan setup): <large> us
  steady-state median wall-clock (n=19): 304 us
  [unfused] validation PASSED (max abs err 0.00781, 0/1048576 cells out of tol)

=== Summary (steady-state median us, this run/this GPU) ===
fused   : 317
unfused : 304 (0.96x the fused time)
Both modes produce the same, correct result
```

**Read the wall-clock line honestly: it does not show a fused win.** At this
size a single execution is dominated by the 2 MB device-to-host copy of C and
by host-side dispatch, and the epilogue difference (~1.7 µs of GPU time) is far
below that noise floor. The wall-clock numbers are there to show the two modes
cost about the same end-to-end — the actual evidence for fusion is in the
kernel timeline below. This is exactly the case Nsight Systems exists for.

## Profiling with Nsight Systems

```bash
nsys profile --trace=cuda --force-overwrite=true -o cutlass \
  tornado --classpath . CutlassFusedEpilogue 512 512 512 10

nsys stats --report cuda_gpu_kern_sum --format csv cutlass.nsys-rep
```

Example output (RTX 4090, TornadoVM 6.0.0, 512³, 10 executions per mode —
kernel names abbreviated, full text in the captured CSV):

```
Time (%)  Total Time (ns)  Instances  Avg (ns)   Name
    43.3           165474         10   16547.4   cutlass::Kernel2<... LinearCombinationRelu ...>
    42.1           161058         10   16105.8   cutlass::Kernel2<... LinearCombination ...>
     9.0            34528         20    1726.4   scale
     5.6            21248         10    2124.8   biasRelu
```

Three things to point at on that table:

1. **The fusion is visible in the kernel's own name.** The fused mode's CUTLASS
   kernel is templated on `LinearCombinationRelu`; the unfused mode's on plain
   `LinearCombination`. That is CUTLASS's epilogue, compiled into the GEMM.
2. **The unfused mode really is two kernels.** `biasRelu` — the Java method
   name, JIT-compiled to CUDA — appears as its own 2124 ns kernel. The fused
   mode has no such row.
3. **Per execution: fused 16547 ns vs unfused 16106 + 2125 = 18231 ns.** The
   fused epilogue is ~1.7 µs (~9%) cheaper in GPU time, and it costs one fewer
   kernel launch and one fewer full read-modify-write pass over C.

`scale` shows 20 instances because both graphs run it — it is the JIT task
shared by the two modes, and a useful sanity check that you are reading a
trace containing both.

Open the timeline visually with `nsys-ui cutlass.nsys-rep` to see the fused
mode's single kernel next to the unfused mode's back-to-back pair.

Captured evidence: `results/raw/19-cutlass-cudnn-warp-demos/12-cutlass-nsys-kernsum.csv`,
`12-cutlass.nsys-rep`, run logs `12-cutlass-tornado.log` / `12-cutlass-javaargfile.log`.

## Gotcha worth knowing: `Math.max`, not a ternary

The JIT `biasRelu` task writes its result as `new HalfFloat(Math.max(v, 0f))`.
Writing the same thing as a ternary — `new HalfFloat(v > 0 ? v : 0)` — fails to
compile on the CUDA backend:

```
tornado.graal.compiler.debug.GraalError: should not reach here:
  Node implementing Lowerable not handled: NewInstance
```

The ternary leaves a merge point that defeats escape analysis on the following
allocation, and the backend has no lowering for a surviving `NewInstance`.
`Math.max` keeps the allocation on a straight-line path where it is virtualised
away. Reported upstream — see the repo README's "Upstream issues filed" section.

## If the demo fails on stage

- `Size must be a multiple of the tile size (32)` — M, N or K is not a multiple
  of 32. Use the defaults.
- If the CUTLASS library task is missing, `tornado --devices` will still work
  but the graph will fail to resolve `tornado.cutlass`. Confirm the SDK is the
  CUDA build: `cat $TORNADOVM_HOME/etc/tornado.backend` must say `cuda-backend`.
- Fall back to the captured logs and CSV in
  `results/raw/19-cutlass-cudnn-warp-demos/`.

## CUDA equivalent

[`CutlassFusedEpilogue.cu`](CutlassFusedEpilogue.cu) is the same demo written directly in CUDA C++, for side-by-side comparison.

```bash
nvcc -arch=sm_89 -std=c++17 -I$CUTLASS_DIR/include -I$CUTLASS_DIR/tools/util/include -o cutlass_fused_epilogue CutlassFusedEpilogue.cu && ./cutlass_fused_epilogue
```

CUTLASS is header-only and **not vendored in this repo**. Fetch it once:

```bash
git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git
export CUTLASS_DIR=$PWD/cutlass
```

In Java, choosing the epilogue is choosing a method: `cutlassGemmBiasRelu`
versus `cutlassHgemm`. In CUDA it is a template parameter, and you supply
everything around it yourself:

```c
using GemmRelu = cutlass::gemm::device::Gemm<
    Element, Layout, Element, Layout, Element, Layout,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape,
    cutlass::epilogue::thread::LinearCombinationRelu<Element, 4, float, float>,
    ...>;
```

Those tile shapes (`128x128x32`, `64x64x32`, `16x8x16`, 3 stages) are the ones
TornadoVM's provider picks — you can read them straight out of the kernel name
in the nsys trace above. Choosing them is a decision the Java API makes for you.

One structural difference worth knowing: CUTLASS's fused epilogue applies bias
through its `C` operand with `beta = 1`, so the length-N bias vector has to be
broadcast to a full MxN matrix first. The Java `cutlassGemmBiasRelu` takes the
vector directly and handles that internally.

**Measured, 1024³, 20 executions:** CUDA fused 233 µs vs unfused 241 µs
(fused 1.04x faster); TornadoVM fused 317 µs vs unfused 304 µs. Both are within
the noise floor of wall-clock at this size — see the Nsight Systems section
above for the measurement that actually resolves the epilogue difference.
Validation is identical in both: max abs err `0.00781`.

`bash scripts/run-all-cuda.sh` builds and runs the CUDA equivalent of every demo (needs only the CUDA toolkit, no JDK).

## JBang

Not verified: `jbang` is not installed on this machine (`which jbang` → exit 1,
re-checked 2026-09-02). Documented-but-untested shape — do not run live:

```bash
jbang -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  --java-opts="@$TORNADOVM_HOME/tornado-argfile" \
  CutlassFusedEpilogue.java
```
