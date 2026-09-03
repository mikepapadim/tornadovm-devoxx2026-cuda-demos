# 14 — Warp shuffle + async copy + shared memory

**Concept (read in ~1 minute):** three optimisations that CUDA programmers reach
for by hand, written in Java, in one kernel:

| Technique | TornadoVM API | PTX/CUDA it generates |
|---|---|---|
| **Asynchronous copy** | `ctx.asyncCopyToLocal` / `asyncCopyCommit` / `asyncCopyWaitGroup` | `cp.async.ca.shared.global` |
| **Shared memory** | `ctx.allocateIntLocalArray` / `allocateFloatLocalArray` / `localBarrier` | `__shared__`, `__syncthreads()` |
| **Warp shuffle** | `ctx.simdShuffleDown` | `__shfl_down_sync` |

The workload is a per-row sum over **int8** data — the shape a quantized
inference kernel actually has. Each thread block owns one row: it `cp.async`s
the row into shared memory as packed int32 words, unpacks four int8 lanes per
word, reduces each warp in registers by shuffling, then combines the four
per-warp partials through a small shared-memory array.

A one-thread-per-row kernel reading straight from global memory is the
baseline. Both are validated against a sequential Java reference.

Source: [`WarpAsyncSharedReduce.java`](WarpAsyncSharedReduce.java).

## Build

```bash
source scripts/setup-env.sh   # from repo root; pins the SDK in env/versions.env
cd demos/14-warp-async-shared
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . WarpAsyncSharedReduce.java
```

## Run

```bash
tornado --classpath . WarpAsyncSharedReduce                       # defaults: 4096 1024 20
java @$TORNADOVM_HOME/tornado-argfile -cp . WarpAsyncSharedReduce

tornado --printKernel --classpath . WarpAsyncSharedReduce 64 256 2   # see the generated CUDA
```

Arguments: `<rows> <rowLen> <executions>`. **`rowLen` must be a multiple of 4** —
int8 values are staged four to an int32 word.

## Expected output

```
Warp-shuffle + cp.async + shared-memory row reduction: 4096 rows x 1024 int8 values
  block = 128 threads (4 warps), one block per row
  20 executions per kernel, steady-state median reported (first execution excluded)

=== NAIVE (one thread per row, global memory only) ===
  first execution (JIT compile): <large> us
  steady-state median wall-clock (n=19): 228 us

=== OPTIMISED (cp.async + shared + shuffle) ===
  first execution (JIT compile): <large> us
  steady-state median wall-clock (n=19): 105 us

  [naive    ] validation PASSED (max abs err 0.00000, 0/4096 rows out of tol)
  [optimised] validation PASSED (max abs err 0.00000, 0/4096 rows out of tol)

=== Summary (steady-state median us, this run/this GPU) ===
naive     : 228
optimised : 105 (2.17x vs naive)
Both kernels produce the same, correct result
```

Observed 2.06x–2.25x across runs. **Observed, this machine, this run** — not a
general claim.

## Generated-code evidence (`--printKernel`)

The point of the demo is that these are real CUDA optimisations, not Java
abstractions that "sort of" map to them. `tornado --printKernel` prints the
CUDA that TornadoVM compiled, and every technique is visible verbatim:

```cuda
  __shared__ int adi_2[128];                      // ctx.allocateIntLocalArray(BLOCK)
  __shared__ float adf_3[4];                      // ctx.allocateFloatLocalArray(WARPS_PER_BLOCK)
  ...
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
               :: "r"(__smem), "l"((const char *) ul_0 + 16u + ((long long) i_9) * 1));
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  ...
  f_26 = __shfl_down_sync(0xffffffff, f_24, i_25);
```

Instruction counts from that dump on this machine:

| Pattern | Count |
|---|---|
| `cp.async` | 5 |
| `cp.async.commit_group` | 1 |
| `cp.async.wait_group` | 1 |
| `__shfl_down_sync` | 1 |
| `__shared__` | 2 |
| `__syncthreads` | 3 |

Captured: `results/raw/19-cutlass-cudnn-warp-demos/14-warp-printkernel.log`.

## Profiling with Nsight Systems

```bash
nsys profile --trace=cuda --force-overwrite=true -o warp \
  tornado --classpath . WarpAsyncSharedReduce 4096 1024 10

nsys stats --report cuda_gpu_kern_sum --format csv warp.nsys-rep
```

Example output (RTX 4090, TornadoVM 6.0.0, 10 executions per kernel):

```
Time (%)  Total Time (ns)  Instances  Avg (ns)    Name
    96.4          1056681         10   105668.1   rowSumNaive
     3.6            39712         10     3971.2   rowSumOptimised
```

**This is the most important number in the demo, and the wall-clock hides it.**
On the GPU the optimised kernel is `105668 / 3971 = ` **26.6x** faster than the
naive one. End-to-end wall-clock only showed 2.17x, because each execution also
pays a fixed ~100 µs of host-side dispatch and device-to-host copy that neither
kernel can avoid.

Both readings are true and they answer different questions:

- **2.17x** is what a caller of this task graph experiences today.
- **26.6x** is what the three optimisations actually did to the kernel, and it
  is the number that tells you the remaining time is somewhere else.

Quote the wall-clock when you talk about the application; quote the kernel time
when you talk about the optimisation. Reading only the first would badly
undersell the techniques; reading only the second would oversell the demo.

Open it visually with `nsys-ui warp.nsys-rep`.

Captured evidence: `results/raw/19-cutlass-cudnn-warp-demos/14-warp-nsys-kernsum.csv`,
`14-warp.nsys-rep`, run logs `14-warp-tornado.log` / `14-warp-javaargfile.log`.

## Why it is faster, in counters (Nsight Compute)

`nsys` says the optimised kernel is faster. `ncu` says *why*, and the answer is
not the one you would guess from the instruction count:

```bash
/opt/nvidia/nsight-compute/2024.3.2/ncu --csv --target-processes all \
  --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu.sum,\
smsp__inst_executed.sum \
  java @$TORNADOVM_HOME/tornado-argfile -cp . WarpAsyncSharedReduce 4096 1024 3
```

| metric | naive | optimised | change |
|---|---|---|---|
| sectors per request | 32.00 | 5.00 | **6.4x fewer** |
| global load sectors | 4,194,304 | 163,840 | **25.6x fewer** |
| bank conflicts | 3,944,216 | 190 | **20,760x fewer** |
| instructions executed | 487,040 | 1,134,592 | 2.3x **more** |

**The optimised kernel executes 2.3x more instructions and runs an order of
magnitude faster.** The naive kernel's 32.00 sectors per request is the worst
possible number — one thread per row, each striding across a whole row, so a
warp's 32 lanes land in 32 different sectors. Nsight Compute puts its useful
bytes per fetched sector at **3.12%**, which is exactly 1/32: one useful byte
per 32-byte sector.

`cp.async` + shared memory converts a bandwidth-bound kernel into a
compute-bound one. DRAM reads are ~4.2 MB either way — the data is read once
regardless. The entire win is in the L1/sector path.

### Two things the comparison with hand-written CUDA adds

Profiling the CUDA equivalent the same way:

- **The naive kernels are identical.** Both report 32.00 sectors per request and
  exactly **4,194,304** load sectors, with kernel times matching to 0.05%. On
  the memory path the generated code and the hand-written code are the same
  kernel.
- **The optimised kernels differ by exactly the header-offset ratio.**
  TornadoVM 163,840 sectors against CUDA's 131,072 — **1.250x**, 5.00 vs 4.00
  sectors per request. That is
  [#1065](https://github.com/beehive-lab/TornadoVM/issues/1065) again, the same
  defect demo 15 measures on float kernels, here in an int8 kernel reaching
  memory through `cp.async`. It is a property of the array layout, not of a
  kernel shape or an element type. It costs only 4.5% in time here, because
  this kernel is no longer bandwidth-bound once optimised.

Full data: `results/raw/24-ncu-demo14-counters/`. Kernel times measured *under*
`ncu` are cold single launches and are not comparable to the `nsys` numbers
above — quote the `nsys` 26.6x for timing.

## How the kernel works

```
row  = ctx.groupIdx            one block per row
tid  = ctx.localIdx            128 threads = 4 warps

for each BLOCK-sized chunk of the row:
    asyncCopyToLocal(tile, tid, data, row*rowLen + word*4)   4 bytes per thread
    asyncCopyCommit(); asyncCopyWaitGroup(0); localBarrier()
    unpack 4 sign-extended int8 lanes from tile[tid], accumulate
    localBarrier()                                            tile is reused

warp reduction:  for delta in 16,8,4,2,1: value += simdShuffleDown(value, delta)
cross-warp:      lane 0 of each warp -> warpPartials[warp]; barrier; tid 0 sums 4 values
```

`asyncCopyToLocal` copies **exactly 4 bytes** per call, and its source offset is
in **source-array elements** (bytes for a `ByteArray`, half-floats for a
`HalfFloatArray`). That is not currently spelled out in the API javadoc; it was
established here by probing and is what the index arithmetic above relies on.

## If the demo fails on stage

- `rowLen must be a multiple of 4` — int8 values are staged four per int32 word.
- If `--printKernel` shows no `cp.async`, the device is pre-Ampere:
  `cp.async` needs compute capability 8.0+. This machine is 8.9 (RTX 4090).
- A `[Bailout] Running the sequential implementation` line means a kernel failed
  to compile and silently ran on the CPU — the timings are then meaningless.
  Re-run; if it persists, fall back to the captured logs in
  `results/raw/19-cutlass-cudnn-warp-demos/`.

## CUDA equivalent

[`WarpAsyncSharedReduce.cu`](WarpAsyncSharedReduce.cu) is the same demo written directly in CUDA C++, for side-by-side comparison.

```bash
nvcc -arch=sm_89 -o warp_async_shared WarpAsyncSharedReduce.cu && ./warp_async_shared
```

Put this side by side with the `--printKernel` dump above — they are nearly
line-for-line the same, which is the claim this demo exists to support:
`KernelContext.asyncCopyToLocal` and `simdShuffleDown` are not an abstraction
*over* `cp.async` and `shfl.sync`, they compile to exactly those instructions.

```c
uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(&tile[tid]));
asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem), "l"(src));
asm volatile("cp.async.commit_group;\n");
asm volatile("cp.async.wait_group 0;\n");
...
value += __shfl_down_sync(0xffffffff, value, delta);
```

The one thing you must do by hand and Java does for you: `cp.async`'s
destination is a **shared-window** address, not a generic pointer, hence the
`__cvta_generic_to_shared` cast. Pass a generic pointer and it fails at runtime.

At 164 lines the CUDA version is *shorter* than the 175-line Java one. The
argument for the Java version here is not brevity — it is that the same source
compiles and runs under an ordinary JVM debugger, and that `int8` unpacking and
validation are plain Java.

**Measured, 4096x1024:** TornadoVM naive 228 µs → optimised 105 µs (2.17x);
CUDA naive 64 µs → optimised 14 µs (4.47x). TornadoVM's *kernel* speedup was
26.6x (see the Nsight section above); CUDA's wall-clock ratio is closer to its
kernel ratio simply because it has far less fixed host overhead to dilute it.

`bash scripts/run-all-cuda.sh` builds and runs the CUDA equivalent of every demo (needs only the CUDA toolkit, no JDK).

## JBang

Not verified: `jbang` is not installed on this machine (`which jbang` → exit 1,
re-checked 2026-09-02). Documented-but-untested shape — do not run live:

```bash
jbang -cp "$TORNADOVM_HOME/share/java/tornado/*" \
  --java-opts="@$TORNADOVM_HOME/tornado-argfile" \
  WarpAsyncSharedReduce.java
```
