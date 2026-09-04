# Step 6 — host dispatch and CUDA Graph (task E), sm_120

`CudaGraphBenefit 4096 6 <execs> <mode>`, nsys `--trace=cuda --sample=none --cpuctxsw=none`.
Per-execution cost obtained by differencing the 100- and 10-execution runs, (100-10)/90,
as the doc requires. Dividing a total by iterations is the mistake recorded in
`results/raw/25-host-dispatch-breakdown/MANIFEST.md` and is not used here.

## Per-execution API counts — differenced

| | sm_120 measured | sm_89 baseline | match |
|---|---|---|---|
| nograph `cuLaunchKernel`   | 6.00 | 6  | yes |
| nograph `cuMemcpyHtoDAsync`| 7.00 | 7  | yes |
| nograph event calls (create+record+destroy) | 14+14+14 = **42** | 42 | yes |
| graph `cuGraphLaunch`      | 1.00 | 1  | yes |
| graph event calls          | 1+1+1 = **3** | 3 | yes |

## Per-execution host API time, excluding sync

| mode | sm_120 | sm_89 baseline | delta |
|---|---|---|---|
| nograph | **32,862 ns** | 33,774 ns | -2.7% |
| graph   | **6,212 ns**  | 6,283 ns  | -1.1% |
| saving  | **5.29x**     | 5.4x      | — |

The CUDA Graph result reproduces on Blackwell. Counts are identical; times are
within 3%.

## Methodology correction — differencing cancels counts, not one-time *time*

Differencing two runs correctly cancels the *call counts* of one-time setup. It
does **not** cancel their *duration*, because the two runs are separate processes
and the fixed cost has run-to-run variance far larger than 90 executions of real
work.

`cuCtxCreate_v2` is called once in both runs, so its differenced count is 0, but
it took 92.2 ms in one run and ~132 ms in the other. Its differenced time is
therefore -91,088 ns/exec for nograph and +40,022 ns/exec for graph — noise that
swamps the real signal. Summing every API's differenced time yields
**-58,177 ns/exec** for nograph and a nonsensical **-1.27x** "speedup".

FIX APPLIED: sum only APIs whose differenced call count is > 0 — i.e. those that
genuinely scale with execution count. If the count did not change, the time is not
per-execution cost by definition. With that restriction the numbers land within 3%
of the sm_89 baseline, which is the evidence the restriction is the right one.

RECOMMENDATION FOR THE DOC: state this explicitly in Step 6. "Difference two
execution counts" is necessary but not sufficient — the filter on count-scaling
APIs is what makes the time figure meaningful.

## Note on cuStreamSynchronize

Excluded from the totals above, per the baseline's "excl. sync" convention. Worth
recording that it is *not* small in graph mode: 3 calls/exec at 16,388 ns/exec,
vs 3 calls at 3,047 ns/exec for nograph. Graph mode moves the wait into sync
rather than removing it — the win is host-side dispatch overhead, not wall clock.
Do not present the 5.29x as an end-to-end speedup.

---

# Step 6b — demo 13 buffer reuse across JIT -> libraryTask -> JIT

`CuDnnConvBlockHybrid 4 16 32 32 16 10` under nsys, trace exported to SQLite and
`CUPTI_ACTIVITY_KIND_MEMCPY` histogrammed by `bytes`, as the doc specifies.
Validation PASSED (max abs err 0.000000, 0/65536 out of tol).

## Memcpy histogram

| kind | bytes | count | what it is |
|---|---|---|---|
| H2D | 262160 | 1   | input NCHW 4x16x32x32 fp32 (+16 B header) — uploaded **once** for 10 executions |
| D2H | 262160 | 10  | final result — one per execution |
| H2D | 9232   | 1   | 16x16x3x3 filters (+header) — once |
| H2D | 80     | 1   | 16-element bias (+header) — once |
| H2D | 24     | 20  | small metadata, 2 per execution |
| H2D | 256    | 512 | cuDNN setup (see below) |
| D2H | 256    | 512 | cuDNN setup (see below) |

## VERDICT: intermediates do not appear. Buffer reuse holds.

The chain is `JIT scale -> cuDNN conv2d -> JIT addBias -> cuDNN relu`. Three
intermediate tensors of 262144 B each pass between those stages. Exactly **one**
262160 B H2D and **one** 262160 B D2H per execution appear in the trace — the
original input and the final output. No intermediate is staged through the host.

## The 512+512 copies of 256 B are not a leak

They looked like the largest count in the histogram, so they were checked rather
than assumed. All 1024 occur between t=0.0 ms and t=256.5 ms; the first
execution's result D2H is at t=896.1 ms. **1024 of 1024 complete before the first
execution finishes, 0 after.** They are cuDNN's plan-selection / autotuning
phase, which is why the demo's own first execution reads 655 ms against a
528 us steady-state median.

Steady-state per-execution memcpy traffic is therefore: 1 x 262160 B D2H and
2 x 24 B H2D. Nothing else.

## Kernels observed (10 executions)

| count | kernel | side |
|---|---|---|
| 10 | `scale` | TornadoVM JIT |
| 10 | `addBias` | TornadoVM JIT |
| 10 | `cutlass__5x_cudnn::...s1688fprop_optimized_tf32_64x64...` | cuDNN conv |
| 10 | `op_generic_tensor_kernel<...>` | cuDNN relu |
| 20 | `nchwToNhwcKernel` | cuDNN layout transform (2/exec) |
| 10 | `nhwcToNchwKernel` | cuDNN layout transform |

Worth flagging for the NVIDIA conversation, though it is **not** a TornadoVM
issue: cuDNN inserts 3 layout transforms per execution because it prefers NHWC
while the demo's tensors are NCHW. That is pure overhead in the hybrid path and
is a fair question to raise — the JIT kernels either side are already NCHW.
