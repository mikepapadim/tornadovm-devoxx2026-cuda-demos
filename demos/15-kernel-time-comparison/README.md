# 15 — Kernel time: TornadoVM vs hand-written CUDA

**Concept (read in ~1 minute):** every other timed demo here reports wall-clock,
which on TornadoVM is dominated by host-side dispatch. This one asks the
narrower question: **once a kernel is actually running on the GPU, how does
TornadoVM's generated code compare with CUDA written by hand?**

Three kernels with deliberately different bottlenecks, chained:

| Kernel | Bottleneck | Per element |
|---|---|---|
| `elementwise` | memory | 1 read, 1 write |
| `polynomial` | compute | dependent chain of `degree` FMAs, 1 read, 1 write |
| `stencil` | memory | 3 reads (neighbours), 1 write |

Sources: [`KernelTimeComparison.java`](KernelTimeComparison.java) and
[`KernelTimeComparison.cu`](KernelTimeComparison.cu).

**The comparison is set up so nothing but code generation differs:** identical
kernel names (so `nsys` rows line up), identical 256-thread blocks and grid
sizes, identical arithmetic including bounds checks, and no `-use_fast_math`
(TornadoVM does not enable it either, so turning it on would measure a compiler
flag rather than codegen).

## Result

Mean of the per-kernel `Avg (ns)` reported by `nsys`, over 3 independent runs of
20 executions each, RTX 4090. Spread across runs is under 1%, so these are
stable numbers rather than one-shot readings
(`results/raw/21-kernel-time-comparison/repeat-runs.csv` has every value).

| Kernel | TornadoVM | CUDA | Ratio |
|---|---|---|---|
| `elementwise` (memory-bound) | 13.94 µs | 10.62 µs | CUDA **1.31x** faster |
| `stencil` (memory-bound) | 14.32 µs | 11.55 µs | CUDA **1.24x** faster |
| `polynomial` (compute-bound) | 35.24 µs | 39.93 µs | TornadoVM **1.13x** faster |

TornadoVM loses on both memory-bound kernels and wins on the compute-bound one.
Both results have a single identifiable cause, and neither is "TornadoVM emits
worse arithmetic" — §Why below establishes that with two probes.

## Reproducing it

### 1. Build both

```bash
source scripts/setup-env.sh                     # from repo root
cd demos/15-kernel-time-comparison

javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . KernelTimeComparison.java
nvcc -arch=sm_89 -o kernel_time_comparison KernelTimeComparison.cu
```

### 2. Check both produce the same answer

```bash
tornado --classpath . KernelTimeComparison
./kernel_time_comparison
```

Both print `validation PASSED (max abs err 0.0000001, 0/4194304 elements out of tol)`.
Do **not** compare the wall-clock lines they print — that is the number this
demo exists to look past.

### 3. Measure kernel time with Nsight Systems

```bash
nsys profile --trace=cuda --force-overwrite=true -o tornado \
  tornado --classpath . KernelTimeComparison
nsys stats --report cuda_gpu_kern_sum --format csv tornado.nsys-rep

nsys profile --trace=cuda --force-overwrite=true -o cuda ./kernel_time_comparison
nsys stats --report cuda_gpu_kern_sum --format csv cuda.nsys-rep
```

`Avg (ns)` is the column to compare. Because both sides name their kernels
`elementwise`, `polynomial` and `stencil`, the two tables line up directly:

```
# TornadoVM
Time (%),Total Time (ns),Instances,Avg (ns),Name
55.5,706021,20,35301.1,polynomial
22.5,286531,20,14326.5,stencil
22.0,279360,20,13968.0,elementwise

# CUDA
Time (%),Total Time (ns),Instances,Avg (ns),Name
64.4,804708,20,40235.4,"polynomial(const float *, float *, int, int)"
18.5,231808,20,11590.4,"stencil(const float *, float *, int)"
17.1,213953,20,10697.6,"elementwise(const float *, float *, int)"
```

`Instances` must read 20 on both sides — that confirms you are comparing the
same amount of work.

### 4. Optional — per-kernel hardware counters

`ncu` gives memory throughput and occupancy per kernel:

```bash
ncu --set full --kernel-name elementwise --launch-count 1 ./kernel_time_comparison
```

**This is blocked on this machine** (`ERR_NVGPUCTRPERM`,
`NVreg_RestrictProfilingToAdminUsers=1`, no passwordless sudo — see
`results/failures/08-nsight-compute-permission.md`). Everything reported here
comes from `nsys`, which needs no special permissions. On a machine where `ncu`
works, it would show the transaction-count effect of §Why directly.

## Why: two probes, two causes

Neither difference is arithmetic quality. Both are structural, and both are
reproducible in isolation.

### Memory-bound: TornadoVM's array header misaligns coalescing

`tornado --printKernel` shows every `FloatArray` access offset by four floats:

```c
l_4  =  (long long) i_2;
l_5  =  l_4 + 4L;        // <-- 16-byte FloatArray header
l_6  =  l_5 << 2;
ul_7 =  ul_0 + l_6;
f_8  =  *(( float *) ul_7);
```

A warp reads 32 x 4 = 128 bytes, which is exactly **4 sectors** of 32 bytes when
aligned. Offsetting by 16 bytes makes every warp-wide access straddle a fifth
sector: **5 transactions for the same data, 1.25x.**

That is measurable directly, and Nsight Compute measures it. Every global access
in all three TornadoVM kernels reports 5.00 sectors per request, against 4.00
for the hand-written CUDA:

```bash
/opt/nvidia/nsight-compute/2024.3.2/ncu --csv \
  --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
  <binary>
```

| Kernel | metric | CUDA | TornadoVM | CUDA forced to offset 4 |
|---|---|---|---|---|
| `elementwise` | load sectors | 524,288 | **655,360** | 655,360 |
| `elementwise` | store sectors | 524,288 | **655,360** | 655,360 |
| `polynomial` | load sectors | 524,288 | **655,360** | — |
| `stencil` | load sectors | 1,835,006 | **1,966,080** | 1,966,080 |
| `stencil` | store sectors | 524,288 | **655,360** | 655,360 |

The TornadoVM counts are **identical to the deliberately misaligned CUDA
kernel** — exactly, not approximately. The mechanism is measured, not inferred.

Two things fall out that the timing alone could not show:

- **`polynomial` pays the same 1.25x penalty and it costs nothing.** It is the
  kernel where TornadoVM *wins*. Being compute-bound, it hides the extra
  transactions behind the FMA chain. So the header offset is in every kernel
  TornadoVM generates; it only becomes visible in time when the kernel is
  bandwidth-bound.
- **DRAM traffic is unchanged** (~16.78 MB read on both sides, within 0.03%).
  The cost is extra 32-byte sector transactions inside the cache hierarchy, not
  extra memory traffic.

For the wall-clock consequence, [`ProbeHeaderAlignment.cu`](ProbeHeaderAlignment.cu)
runs the *identical* CUDA kernel at offset 0 and offset 4 floats:

```bash
nvcc -arch=sm_89 -o probe_alignment ProbeHeaderAlignment.cu && ./probe_alignment
```

```
elementwise  offset=0 floats : median 12.5 us
stencil      offset=0 floats : median 13.2 us
elementwise  offset=4 floats : median 16.0 us     <- 1.28x slower
stencil      offset=4 floats : median 16.7 us     <- 1.27x slower
```

1.28x and 1.27x, against the 1.31x and 1.24x measured between the two
implementations. **The header offset accounts for essentially the whole
memory-bound gap.** It is a data-layout property, not a code-generation one, and
padding the payload to a 128-byte boundary would recover it. Reported upstream —
see the repo README's "Upstream issues filed" section.

Sector counts are the right evidence for the mechanism, but they are not a
linear predictor of time: `stencil`'s total sectors rise 1.111x against a
measured 1.24x. Use the counters for *why*, the timing loop for *how much*.
Kernel durations reported under `ncu` itself are not comparable to either —
single cold launches put all four probe configurations at ~19.5 µs.

Full data: `results/raw/22-ncu-alignment-counters/`.

### Compute-bound: TornadoVM JIT-specialises on the runtime value

`degree` is a task argument, so TornadoVM compiles the kernel *after* its value
is known and Graal fully unrolls the loop. `--printKernel` (with `degree=8`)
shows a straight-line FMA chain with no loop counter and no branch:

```c
f_10  =  f_8 + 0.5F;
f_11  =  fma(f_10, f_8, 0.5F);
f_12  =  fma(f_11, f_8, 0.5F);
...                                  // no loop, no branch
```

nvcc compiles ahead of time, cannot know `degree`, and must emit a real loop —
`cuobjdump -sass` shows 11 branch instructions in that kernel.
[`ProbeJitSpecialisation.cu`](ProbeJitSpecialisation.cu) gives nvcc the same
information via a template parameter:

```bash
nvcc -arch=sm_89 -o probe_specialisation ProbeJitSpecialisation.cu && ./probe_specialisation
```

```
degree as runtime argument         median 39.5 us
degree as compile-time constant    median 34.7 us
```

34.7 µs against TornadoVM's 35.24 µs — within 1.6%. **Once nvcc knows what the
JIT knows, the two are equal.** TornadoVM's win here is not better arithmetic;
it is having the value available at compile time, which is a structural
advantage of JIT compilation over AOT, not a TornadoVM trick. It is also a real
advantage: a CUDA programmer only gets it by templating and instantiating every
value they might need.

## What to take from this

Controlling for both effects, **the generated arithmetic is equivalent.** The
honest summary for a talk:

- TornadoVM's kernels are competitive with hand-written CUDA on the same GPU.
- It pays ~25–30% on bandwidth-bound kernels for its array header layout — a
  fixable layout issue, not a compiler limitation.
- It gains on kernels whose shape depends on a runtime value, because it
  compiles when that value is known.
- Wall-clock differences elsewhere in this repo (demos 06, 07, 11, 13) are
  host-side dispatch, a separate cost from anything measured here.

Do not generalise these three kernels to "TornadoVM is X% of CUDA". They were
chosen to expose specific effects, not to be a benchmark suite.

## Expected output

```
Kernel-time comparison: n=4194304, polynomial degree=256, 20 executions
  block size 256 (identical to the CUDA version)
  kernels: elementwise (memory-bound) -> polynomial (compute-bound) -> stencil (memory-bound)
  NOTE: the wall-clock below includes host dispatch and transfers.
        Compare kernel time with nsys -- see this demo's README.

first execution (JIT compile): <large> us
steady-state median wall-clock (n=19): 1170 us
validation PASSED (max abs err 0.0000001, 0/4194304 elements out of tol)
Result is correct
```

Captured evidence: `results/raw/21-kernel-time-comparison/`.

## If the demo fails on stage

- If the two `nsys` tables show different `Instances` counts, one side ran a
  different number of executions — pass the same third argument to both.
- If the CUDA kernel names in `nsys` appear mangled beyond recognition, add
  `--demangle=true` to `nsys stats`.
- `ncu` will fail with `ERR_NVGPUCTRPERM` on this machine; that is expected and
  documented above. Use `nsys`.
- Fall back to the captured CSVs in `results/raw/21-kernel-time-comparison/`.

## JBang

Not verified: `jbang` is not installed on this machine (`which jbang` → exit 1,
re-checked 2026-09-02). Documented-but-untested shape — do not run live:

```bash
jbang -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  --java-opts="@$TORNADOVM_HOME/tornado-argfile" \
  KernelTimeComparison.java
```
