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

## Result (TornadoVM 7.0.0)

Mean of the per-kernel `Avg (ns)` reported by `nsys`, over 3 independent runs of
20 executions each, RTX 4090. Spread across runs is under 0.7%
(`results/raw/36-demo15-demo17-on-7.0.0/15-nsys-kernsum.csv` has every value).

| Kernel | TornadoVM | CUDA | Ratio | on 6.0.0 |
|---|---|---|---|---|
| `elementwise` (memory-bound) | 10.98 µs | 10.69 µs | CUDA **1.03x** faster | CUDA 1.31x |
| `stencil` (memory-bound) | 11.95 µs | 11.63 µs | CUDA **1.03x** faster | CUDA 1.24x |
| `polynomial` (compute-bound) | 35.09 µs | 40.27 µs | TornadoVM **1.15x** faster | TornadoVM 1.13x |

**The story is the last column.** On TornadoVM 6.0.0 this demo found a 24–31% gap
on memory-bound kernels, traced it to one cause with Nsight Compute and a probe,
and reported it upstream. The fix shipped in 7.0.0 and the gap is now ~3%. The
compute-bound win is unchanged, and it has a different cause: JIT specialisation.
§Why below walks through both.

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
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv tornado.nsys-rep

nsys profile --trace=cuda --force-overwrite=true -o cuda ./kernel_time_comparison
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv cuda.nsys-rep
```

`--force-export=true` matters: without it `nsys stats` can silently reuse a stale
`.sqlite` from an earlier run with the same name.

`Avg (ns)` is the column to compare. Because both sides name their kernels
`elementwise`, `polynomial` and `stencil`, the two tables line up directly
(7.0.0, run 1 of 3; middle columns omitted):

```
# TornadoVM
Time (%),Total Time (ns),Instances,Avg (ns),...,Name
60.5,700022,20,35001.1,...,polynomial
20.6,238215,20,11910.8,...,stencil
18.9,219335,20,10966.8,...,elementwise

# CUDA
64.3,805274,20,40263.7,...,"polynomial(const float *, float *, int, int)"
18.6,232328,20,11616.4,...,"stencil(const float *, float *, int)"
17.1,213928,20,10696.4,...,"elementwise(const float *, float *, int)"
```

`Instances` must read 20 on both sides — that confirms you are comparing the
same amount of work.

### 4. Per-kernel hardware counters

Nsight Compute shows the memory side directly. Use the 2024.3.2 build: when last
checked, the newer `ncu` on `PATH` on this machine could not connect to the driver
(`results/failures/08-nsight-compute-permission.md`). The commands below are the ones
that produced this README's counters:

```bash
NCU=/opt/nvidia/nsight-compute/2024.3.2/ncu
M=l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
dram__bytes_read.sum,smsp__inst_executed.sum

$NCU --csv --target-processes all --metrics $M \
  java @$TORNADOVM_HOME/tornado-argfile -cp . KernelTimeComparison 4194304 256 2
$NCU --csv --metrics $M ./kernel_time_comparison 4194304 256 2
```

## Why: two effects, one of them now fixed

Neither effect is arithmetic quality. Both are structural, and both are
reproducible in isolation with a hand-written CUDA probe.

### Memory-bound: the array header used to misalign coalescing — fixed in 7.0.0

`tornado --printKernel` shows every `FloatArray` access offset by four floats,
past the array's 16-byte header:

```c
l_4  =  (long long) i_2;
l_5  =  l_4 + 4L;        // <-- 16-byte FloatArray header
l_6  =  l_5 << 2;
ul_7 =  ul_0 + l_6;
f_8  =  *(( float *) ul_7);
```

A warp reads 32 x 4 = 128 bytes, which is exactly **4 sectors** of 32 bytes when
aligned. On **6.0.0** the buffer itself started on an aligned boundary, so the
16-byte header pushed every warp-wide access across a fifth sector: **5 transactions for the
same data**. Nsight Compute measured exactly that — 5.00 sectors per request in
every TornadoVM kernel against 4.00 for CUDA, and sector totals identical to the
same CUDA kernel run deliberately at a 4-float offset
(`results/raw/22-ncu-alignment-counters/`). Reported upstream as
[#1065](https://github.com/beehive-lab/TornadoVM/issues/1065).

[`ProbeHeaderAlignment.cu`](ProbeHeaderAlignment.cu) shows what that offset
costs, using the *identical* CUDA kernel at offset 0 and offset 4 floats:

```bash
nvcc -arch=sm_89 -o probe_alignment ProbeHeaderAlignment.cu && ./probe_alignment
```

```
elementwise  offset=0 floats : median 11.7 us
stencil      offset=0 floats : median 12.5 us
elementwise  offset=4 floats : median 15.1 us     <- 1.29x slower
stencil      offset=4 floats : median 15.7 us     <- 1.26x slower
```

That matches the 1.31x / 1.24x this demo measured on 6.0.0: **the offset was
essentially the whole memory-bound gap.**

**What changed in 7.0.0.** PR
[#1066](https://github.com/beehive-lab/TornadoVM/pull/1066) pads each allocation
so that `base + 16` — where element 0 lives — is 32-byte aligned. The generated
code is **byte-identical**: the `+ 4L` above is still there on 7.0.0. What moved is
where the buffer starts. Nsight Compute on 7.0.0, both sides at the same n:

| Kernel | load sectors | load sectors/request | DRAM read |
|---|---|---|---|
| `elementwise` | 524,288 **both** | 4 **both** | 16,780,160 vs 16,779,904 B |
| `polynomial` | 524,288 **both** | 4 **both** | 16,784,256 vs 16,780,928 B |
| `stencil` | 1,835,006 **both** | 4.67 **both** | 16,781,056 vs 16,780,288 B |

Every sector count now equals hand-written CUDA's exactly (store sectors too).
On 6.0.0 the TornadoVM load counts were 655,360 and 1,966,080.

One caveat from the PR itself: buffers in **`withBatch` slots are deliberately
not padded**, so batched execution still carries the offset. Not measured here.

### The ~3% that is left

With memory traffic identical, the visible remaining difference is instruction
count. TornadoVM executes more instructions on both memory-bound kernels, with
the same number of global load/store instructions and the same 16 registers:

| Kernel | instructions, TornadoVM / CUDA |
|---|---|
| `elementwise` | 2,359,296 / 1,966,080 = **1.20x** |
| `stencil` | 5,242,880 / 3,670,016 = **1.43x** |
| `polynomial` | 35,782,656 / 44,826,624 = 0.80x |

That is **consistent with** a ~3% gap on bandwidth-bound kernels, but it is
**not shown to cause it** — and the repo has already retracted one attribution of
the 1.20x (to bounds checks; see `docs/ANALYSIS-GUIDE.md`). Treat it as the next
thing to look at, not an explanation.

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
`cuobjdump -sass` shows 11 branch instructions in that kernel. It also executes
1.25x TornadoVM's instruction count (table above).
[`ProbeJitSpecialisation.cu`](ProbeJitSpecialisation.cu) gives nvcc the same
information via a template parameter:

```bash
nvcc -arch=sm_89 -o probe_specialisation ProbeJitSpecialisation.cu
nsys profile --trace=cuda --force-overwrite=true -o spec ./probe_specialisation
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv spec.nsys-rep
```

```
polyRuntime  (degree as runtime argument)      36.77 us
polyConst<>  (degree as compile-time constant) 32.23 us    <- 1.14x
```

Specialisation alone is worth **1.14x** — the same in each of three runs — and
TornadoVM's measured win over hand-written CUDA is **1.15x**. Compare the ratios,
not the absolute times: the probe's kernel is not the demo's kernel.
**Once nvcc knows what the JIT knows, the two are equal.** TornadoVM's win here is
not better arithmetic; it is having the value available at compile time — a
structural advantage of JIT over AOT compilation. It is also a real one: a CUDA
programmer only gets it by templating and instantiating every value they might
need.

## What to take from this

- **On TornadoVM 7.0.0 the generated kernels are within ~3% of hand-written CUDA
  on bandwidth-bound work, and faster on kernels shaped by a runtime value.**
- The 24–31% memory-bound gap on 6.0.0 was a data-layout property, not a
  code-generation one. It was diagnosed here with a counter and a probe, reported
  as #1065, and fixed by #1066 without touching the generated code.
- The compute-bound win is JIT specialisation and survives the upgrade unchanged.
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
steady-state median wall-clock (n=19): 1181 us
validation PASSED (max abs err 0.0000001, 0/4194304 elements out of tol)
Result is correct
```

Captured evidence: `results/raw/36-demo15-demo17-on-7.0.0/` (7.0.0);
`results/raw/21-kernel-time-comparison/` and `results/raw/22-ncu-alignment-counters/`
(the 6.0.0 diagnosis).

## If the demo fails on stage

- If the two `nsys` tables show different `Instances` counts, one side ran a
  different number of executions — pass the same third argument to both.
- If the CUDA kernel names in `nsys` appear mangled beyond recognition, add
  `--demangle=true` to `nsys stats`.
- If `ncu` cannot connect to the driver, use
  `/opt/nvidia/nsight-compute/2024.3.2/ncu` rather than the one on `PATH`.
- If the memory-bound kernels show the old ~1.3x gap, check `tornado --version`:
  you are on a release older than 7.0.0.
- Fall back to the captured CSVs in `results/raw/36-demo15-demo17-on-7.0.0/`.

## JBang

Not verified: `jbang` is not installed on this machine (`which jbang` → exit 1,
re-checked 2026-09-02). Documented-but-untested shape — do not run live:

```bash
jbang -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  --java-opts="@$TORNADOVM_HOME/tornado-argfile" \
  KernelTimeComparison.java
```
