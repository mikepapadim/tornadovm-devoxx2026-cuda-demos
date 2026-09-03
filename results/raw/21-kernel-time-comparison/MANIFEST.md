# Batch 21 — kernel-time-only comparison, TornadoVM vs hand-written CUDA

Captured 2026-09-02. `demos/15-kernel-time-comparison`. RTX 4090 (sm_89),
driver 565.57.01, CUDA 12.6.85, JDK 25.0.2, TornadoVM 6.0.0-jdk22plus-cuda,
Nsight Systems 2024.5.1.113.

Every other timed demo in this repo reports wall-clock, which on TornadoVM is
dominated by host-side dispatch. This batch isolates **kernel time only**, to
compare generated code rather than runtime overhead.

## Method

Three kernels chained, with deliberately different bottlenecks: `elementwise`
(memory-bound, 1 read + 1 write), `polynomial` (compute-bound, dependent chain
of `degree` FMAs), `stencil` (memory-bound, 3 reads + 1 write).

Controlled so that only code generation differs: identical kernel names (so the
two `nsys` tables line up row by row), identical 256-thread blocks and grid
sizes, identical arithmetic including bounds checks, and no `-use_fast_math`
on either side.

`n = 4194304`, `degree = 256`, 20 executions per run, 3 independent runs.
Both implementations validate at `max abs err 0.0000001`.

## Result

Mean per-kernel `Avg (ns)` from `nsys stats --report cuda_gpu_kern_sum`.
Spread across the 3 runs is under 1% (`repeat-runs.csv`).

| Kernel | TornadoVM | CUDA | Ratio |
|---|---|---|---|
| `elementwise` (memory-bound) | 13944 ns | 10621 ns | CUDA 1.31x faster |
| `stencil` (memory-bound) | 14322 ns | 11546 ns | CUDA 1.24x faster |
| `polynomial` (compute-bound) | 35236 ns | 39926 ns | **TornadoVM 1.13x faster** |

## Cause 1 — array header offset (memory-bound kernels)

`FloatArray` places its payload 16 bytes after the allocation base; the
generated CUDA shows `l_5 = l_4 + 4L` on every access
(`tornado-printkernel.log`). A warp reads 32 x 4 = 128 bytes, so that offset
makes every warp-wide access straddle a 128-byte boundary — two memory
transactions instead of one.

`ProbeHeaderAlignment.cu` runs the **identical** CUDA kernel at offset 0 and
offset 4 floats (`probe-header-alignment.log`):

```
elementwise  offset=0 floats : median 12.5 us
stencil      offset=0 floats : median 13.2 us
elementwise  offset=4 floats : median 16.0 us     (1.28x)
stencil      offset=4 floats : median 16.7 us     (1.27x)
```

1.28x / 1.27x from the offset alone, against the 1.31x / 1.24x measured between
implementations. **The header offset accounts for essentially the whole
memory-bound gap** — a data-layout property, not code-generation quality.

Filed upstream as
[beehive-lab/TornadoVM#1065](https://github.com/beehive-lab/TornadoVM/issues/1065).

**Superseded by direct measurement (2026-09-03).** The attribution above is an
inference from wall-clock. `results/raw/22-ncu-alignment-counters/` measures the
mechanism on the generated kernels with Nsight Compute: all three TornadoVM
kernels report 5.00 sectors per request on every global access against
hand-written CUDA's 4.00, matching the offset-4 kernel's counts exactly. It also
shows `polynomial` carries the same penalty without paying for it in time.

## Cause 2 — JIT specialisation (compute-bound kernel)

`degree` is a task argument, so TornadoVM compiles the kernel after its value is
known and Graal fully unrolls the FMA chain — `tornado-printkernel.log` shows
straight-line `fma()` calls with no loop counter and no branch. nvcc compiles
ahead of time and must emit a real loop; `cuda-sass.log` shows 11 branch
instructions in that kernel.

`ProbeJitSpecialisation.cu` gives nvcc the same information via a template
parameter (`probe-jit-specialisation.log`):

```
degree as runtime argument         median 39.5 us
degree as compile-time constant    median 34.7 us
```

34.7 µs against TornadoVM's 35.24 µs — within 1.6%. **Once nvcc knows what the
JIT knows, the two are equal.** TornadoVM's win is having the value at compile
time, a structural advantage of JIT over AOT rather than better arithmetic.

## Conclusion

Controlling for both effects, the generated arithmetic is **equivalent**.
TornadoVM pays ~25–30% on bandwidth-bound kernels for a fixable array-layout
issue, and gains on kernels whose shape depends on a runtime value.

These three kernels were chosen to expose specific effects and are **not** a
benchmark suite; do not generalise them to "TornadoVM is X% of CUDA".

Nsight Compute hardware counters (which would show the transaction-count effect
directly rather than by inference from timing) remain blocked on this machine —
`ERR_NVGPUCTRPERM`, see `results/failures/08-nsight-compute-permission.md`.

## Files

| File | What it is |
|---|---|
| `tornado-run.log`, `cuda-run.log` | program output from both implementations |
| `tornado-nsys-kernsum.csv`, `cuda-nsys-kernsum.csv` | the two kernel summaries compared above |
| `tornado.nsys-rep`, `cuda.nsys-rep` | raw traces (open with `nsys-ui`) |
| `repeat-runs.csv` | per-kernel Avg (ns) for all 3 runs of each implementation |
| `probe-header-alignment.log` | cause 1 attribution |
| `probe-jit-specialisation.log` | cause 2 attribution |
| `tornado-printkernel.log` | generated CUDA: the `+ 4L` offset and the unrolled FMA chain |
| `cuda-sass.log` | `cuobjdump -sass` of the CUDA build, showing the polynomial loop's branches |
