# Profiler metrics: TornadoVM vs hand-written CUDA

Every NVIDIA-profiler measurement in this repo, consolidated into two CSVs you
can plot directly, plus what each number means and which figure it supports.

**Framing throughout is TornadoVM against hand-written CUDA**, on the same GPU,
in the same session, on the same workload.

## The files

| File | Shape | Use it for |
|---|---|---|
| `results/raw/27-profiler-metrics/metrics.csv` | long: `demo, kernel, implementation, metric, unit, value, launches, source, file` | the full record; pivot on any axis |
| `results/raw/27-profiler-metrics/comparison.csv` | paired: `demo, kernel, metric, unit, tornadovm, cuda, ratio_tornadovm_over_cuda` | bar charts, straight in |
| `scripts/build-metrics-csv.py` | — | regenerates both from the raw captures |

```bash
python3 scripts/build-metrics-csv.py     # 172 rows, 52 paired
```

`metrics.csv` is long format on purpose: one row per measurement, so a plotting
tool can pivot however a given figure needs without reshaping the file.

> **Read `ratio_tornadovm_over_cuda` carefully.** It is always
> `tornadovm / cuda`. Above 1 means TornadoVM's number is *larger*, which is
> worse for time, sectors, instructions and bank conflicts — and *better* for
> throughput and occupancy. It is not a "slowdown" column.

## Hardware and toolchain

RTX 4090 (sm_89), driver 565.57.01, CUDA 12.6.85, JDK 25.0.2,
TornadoVM 6.0.0-jdk22plus-cuda. Nsight Compute **2024.3.2**
(`/opt/nvidia/nsight-compute/2024.3.2/ncu` — *not* the `ncu` on `PATH`, which
is 2026.2.1.0 and cannot connect to this driver), Nsight Systems 2024.5.1.

Counter access needs `NVreg_RestrictProfilingToAdminUsers=0`; see
`results/failures/08-nsight-compute-permission.md`.

## The metrics

| Metric | Unit | What it is | Lower is better? |
|---|---|---|---|
| `sectors_per_request_load` / `_store` | sectors | 32-byte sectors moved per warp memory request. **4.00 is perfectly coalesced** (128 B = 4 sectors) | yes |
| `global_load_sectors` / `_store` | sectors | total sector traffic on the L1↔L2 path | yes |
| `dram_bytes_read` / `_write` | bytes | actual DRAM traffic | yes |
| `dram_throughput_pct_peak` | % | achieved DRAM bandwidth as a fraction of peak. **>90% = bandwidth-bound** | no, higher |
| `achieved_occupancy` | % | active warps vs peak sustained | no, higher |
| `instructions_executed` | count | total instructions issued | usually |
| `bank_conflicts` | count | shared-memory bank conflicts | yes |
| `hmma_` / `imma_` / `tensor_instructions` | count | tensor-core instructions issued, by pipe | — |
| `tensor_pipe_cycles` | cycles | cycles the tensor pipe was active | — |
| `kernel_time_ncu` | ns | kernel duration **under ncu** | yes |
| `kernel_time_nsys` | ns | kernel duration under nsys, steady state | yes |
| `api_calls` / `api_total_time` | count / ns | host-side CUDA driver API cost | yes |

### Two caveats that will otherwise produce a wrong slide

1. **`kernel_time_ncu` and `kernel_time_nsys` are not comparable.** ncu
   serialises launches, flushes caches and locks clocks, so its absolute times
   run several times higher. Use `kernel_time_nsys` for any timing claim and
   `kernel_time_ncu` only to compare two kernels measured the same way. Never
   put them on one axis.
2. **One GPU, one machine, one run.** Nothing here is a general claim about
   TornadoVM or about CUDA. Label figures accordingly.

## Figures worth building

### 1. Coalescing: sectors per request — *the headline chart*

`comparison.csv`, `metric == "sectors_per_request_load"`. Grouped bars, one
pair per kernel.

| Demo | Kernel | TornadoVM | CUDA | Ratio |
|---|---|---|---|---|
| 01-vector-add | `vectorAdd` | **5.00** | 4.00 | 1.250 |
| 14-warp-async-shared | `rowSumNaive` | 32.00 | 32.00 | 1.000 |
| 14-warp-async-shared | `rowSumOptimised` | **5.00** | 4.00 | 1.250 |
| 15-kernel-time | `elementwise` | **5.00** | 4.00 | 1.250 |
| 15-kernel-time | `polynomial` | **5.00** | 4.00 | 1.250 |
| 15-kernel-time | `stencil` | **5.00** | 4.67 | 1.071 |

Every TornadoVM kernel issues 5.00 sectors per request against CUDA's 4.00 —
`FloatArray`'s 16-byte header pushing every warp-wide 128-byte access across a
fifth sector. Four demos, six kernels, float and int8, ordinary loads and
`cp.async`. Filed as
[TornadoVM#1065](https://github.com/beehive-lab/TornadoVM/issues/1065).

Two bars carry the nuance and are worth keeping rather than trimming:
`rowSumNaive` is 32.00 on **both** sides — identical, and the worst possible
figure — which proves the generated code is not inherently worse. And
`stencil` is 4.67 on the CUDA side because its neighbour accesses already
straddle lines, so misalignment costs it less.

### 2. The penalty is not always a cost

`comparison.csv`, `kernel_time_nsys` next to `sectors_per_request_load`.

`polynomial` pays the identical 1.25× sector penalty and is still the kernel
TornadoVM **wins** (35.24 µs vs 39.93 µs). Being compute-bound, it hides the
extra transactions. The honest message: the defect is in every generated
kernel; it only shows up in time when the kernel is bandwidth-bound.

### 3. Bandwidth-bound means the gap collapses — demo 01

| Metric | TornadoVM | CUDA | Ratio |
|---|---|---|---|
| `instructions_executed` | 33,554,432 | 8,388,608 | **4.00** |
| `global_load_sectors` | 5,242,880 | 4,194,304 | 1.25 |
| `achieved_occupancy` | 55.66% | 88.72% | 0.63 |
| `dram_throughput_pct_peak` | **94.63%** | **95.46%** | 0.99 |
| `kernel_time_ncu` | 187,840 ns | 181,760 ns | **1.034** |

TornadoVM issues **4× the instructions** at 63% of CUDA's occupancy and is only
**3.4% slower**, because both kernels sit at ~95% of peak DRAM bandwidth. Once
the bus is saturated, neither instruction count nor spare occupancy can buy
time back.

**The 3.4% is fully accounted for, and none of it is the instructions.** Time on
a bandwidth-bound kernel is `bytes / bandwidth`:

| | TornadoVM | CUDA | ratio |
|---|---|---|---|
| total DRAM bytes | 174,726,912 | 170,544,512 | 1.0245 |
| achieved bandwidth | 94.63% | 95.46% | 0.9913 |
| **predicted time ratio** | | | **1.0335** |
| **measured time ratio** | 187,840 ns | 181,760 ns | **1.0335** |

Exact to four decimals. So the gap is ~2.4% from moving more DRAM bytes and
~0.9% from marginally lower achieved bandwidth. The 4× instruction count
contributes nothing measurable — it hides under memory latency.

The extra DRAM bytes are on the **write** side (40.5 MB vs 36.3 MB). That is
consistent with the misaligned stores, since a partially covered 32-byte sector
has to be read before it can be written back; it is not proven here, and both
figures sit below the 67 MB the kernel actually writes, because write-back is
not fully captured inside the kernel window.

> **Do not read the occupancy row as a code-generation result.** It is a launch
> configuration difference. TornadoVM ran **1024 threads per block**, the CUDA
> version **256**. On sm_89 an SM holds 1536 threads, so a 1024-thread block
> tiles only once — 32 of 48 warps, a 66.7% ceiling before register pressure is
> even considered (`launch__occupancy_limit_warps` = 1). At 256 threads, six
> blocks fit: 1536 threads, 48 warps, a 100% ceiling
> (`launch__occupancy_limit_warps` = 6). Register use is close (24 vs 16 per
> thread) and is not the limiter.
>
> Demo 15 pins both sides to 256 deliberately so that only code generation
> differs. **Demo 01 does not**, so use it for the bandwidth-saturation point,
> not for an occupancy comparison. The useful observation is the other one:
> 55.84% achieved occupancy is already enough memory-level parallelism to reach
> 94.6% of peak bandwidth.

### 4. Tensor cores: every operand type — demo 16

`metrics.csv`, `metric in {hmma_instructions, imma_instructions, tensor_pipe_cycles}`.

| Kernel | tensor inst | HMMA | IMMA |
|---|---|---|---|
| `gemmBF16` | 4 | **4** | 0 |
| `gemmInt8` | 2 | 0 | **2** |
| `gemmFP8E4M3` | 2 | **2** | 0 |
| `gemmFP8E5M2` | 2 | **2** | 0 |

Counters match the emitted PTX instruction for instruction. int8 dispatches to
the **IMMA** pipe; BF16 and both FP8 formats to HMMA. With demo 08's fp16 that
is all five operand combinations the backend can emit.

### 5. Tensor-core parity, and where the real gap is — demo 08

| | HMMA | tensor cycles | instructions |
|---|---|---|---|
| TornadoVM `gemmMMASingleTile` | 1 | 16 | **183** |
| CUDA `gemmMMASingleTile` | 1 | 16 | **40** |
| both `gemmScalarFp16` | **0** | **0** | — |

Identical tensor-core work. The delta is 183 scalar instructions around the one
HMMA against CUDA's 40 — index arithmetic and fragment staging, not
instruction selection.

### 6. Host-side dispatch — demo 14

`metrics.csv`, `source == "nsys-api"`, stacked bar of `api_calls` by API name.

For the same 40 launches: TornadoVM **1,620** memory-transfer calls against
CUDA's **41**, plus 1,656 `cuStreamSynchronize` and 6,392 event calls against
zero of each.

**Do not divide those totals by the execution count** — 94.8% of TornadoVM's
transfers are one-time start-up traffic that completes before the first kernel
launches. Real per-execution driver overhead is ~8.3 µs, measured by
differencing two execution counts. See
`results/raw/25-host-dispatch-breakdown/MANIFEST.md`; that manifest documents
the mistake, because dividing the total by executions overstates it ~12×.

### 7. What the optimisation buys — demo 14, TornadoVM only

| Metric | naive | optimised | change |
|---|---|---|---|
| `sectors_per_request_load` | 32.00 | 5.00 | 6.4× fewer |
| `global_load_sectors` | 4,194,304 | 163,840 | 25.6× fewer |
| `bank_conflicts` | 3,944,216 | 190 | 20,760× fewer |
| `instructions_executed` | 487,040 | 1,134,592 | 2.3× **more** |

2.3× more instructions, an order of magnitude faster. `cp.async` plus shared
memory turns a bandwidth-bound kernel into a compute-bound one.

## Coverage, and what is missing

| Demo | ncu | nsys | Comparable to CUDA? |
|---|---|---|---|
| 01-vector-add | ✅ | — | ✅ |
| 08-tensor-core | ✅ | — | ✅ |
| 14-warp-async-shared | ✅ | ✅ | ✅ |
| 15-kernel-time | ✅ | ✅ | ✅ |
| 16-tensor-core-datatypes | ✅ | — | Java only so far |
| 06, 07, 11 | — | ✅ (batches 06/11) | host-side only |
| 12-cutlass | — | ✅ (batch 19) | **gap** |
| 13-cudnn | — | ✅ (batch 19) | vendor kernels both sides |

Two honest gaps:

- **Demo 12 (CUTLASS)** has no counter data. Its CUDA side needs the
  header-only CUTLASS checkout (`CUTLASS_DIR`), which is not vendored here.
  Fused-vs-unfused epilogue would make a strong bar chart; it needs
  `git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git`
  first.
- **Demo 16** has counters for the Java side only. The CUDA equivalent builds
  and validates, but was not profiled — worth adding for symmetry.

Demos 04, 05 and 13 call the same vendor libraries (cuBLAS, cuFFT, cuDNN) from
both sides, so a counter comparison measures NVIDIA's kernels rather than
TornadoVM's code generation. Their value is the *integration* story, not a
codegen chart.

## Regenerating

```bash
source scripts/setup-env.sh          # from the repo root, or JAVA_HOME silently wrong
python3 scripts/build-metrics-csv.py
```

The script reads only committed raw captures, so it is deterministic. To
recapture on other hardware, the per-demo `ncu` and `nsys` invocations are in
each demo's README and in the batch manifests under `results/raw/`.
