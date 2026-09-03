# Batch 22 — #1065 measured with Nsight Compute counters

Captured 2026-09-03. RTX 4090 (sm_89), driver 565.57.01, CUDA 12.6.85,
JDK 25.0.2, TornadoVM 6.0.0-jdk22plus-cuda,
**Nsight Compute 2024.3.2** (`/opt/nvidia/nsight-compute/2024.3.2/ncu`).

Batch 21 attributed demo 15's memory-bound gap to `FloatArray`'s 16-byte header
by *inference*: an identical CUDA kernel run at a 4-float offset reproduced the
wall-clock ratio. This batch measures the mechanism directly, on the
TornadoVM-generated kernels themselves.

## Profiler availability

`ncu` was previously blocked (`results/failures/08-nsight-compute-permission.md`).
Two separate problems were conflated there:

1. `/usr/local/cuda/bin/ncu` is 2026.2.1.0 — newer than driver 565.57.01, and
   fails with "failed to connect to the CUDA driver (stub libcuda.so on path?)".
   That message is a red herring; `LD_LIBRARY_PATH` is empty and `libcuda.so.1`
   resolves correctly. **Use the 2024.3.2 install instead**, which connects fine.
2. `RmProfilingAdminOnly: 1` → `ERR_NVGPUCTRPERM`. Fixed with
   `options nvidia NVreg_RestrictProfilingToAdminUsers=0` in
   `/etc/modprobe.d/nvidia-profiling.conf`, `update-initramfs -u`, reboot.

Now reads `RmProfilingAdminOnly: 0` (`profiling-permission.log`).

## Method

Same binaries and workload as batch 21 (`n = 4194304`, `degree = 256`), block
size 256 on both sides. `ProbeHeaderAlignment.cu` gained a `--counters` mode
that launches each kernel exactly once per offset and skips the timing loop, so
the profiler sees one clean launch per configuration — the kernels themselves
are unchanged, so there is still one source of truth for them.

Metrics: `l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,st}.sum`,
`l1tex__t_requests_pipe_lsu_mem_global_op_{ld,st}.sum`, their
`l1tex__average_t_sectors_per_request_...ratio`, and `dram__bytes_{read,write}.sum`.

A warp-wide 32x4-byte access is 128 bytes. Aligned, that is **4 sectors** of 32
bytes. Offset by 16 bytes it straddles a fifth sector: **5 sectors**, 1.25x the
transactions for the same data.

## Result 1 — the predicted 4 → 5 sectors, exactly

`ncu-probe-offsets.csv`, identical CUDA kernel at both offsets:

| Kernel | offset | ld sectors | ld sec/req | st sectors | st sec/req |
|---|---|---|---|---|---|
| `elementwise` | 0 floats | 524,288 | **4.00** | 524,288 | **4.00** |
| `elementwise` | 4 floats | 655,360 | **5.00** | 655,360 | **5.00** |
| `stencil` | 0 floats | 1,835,006 | 4.67 | 524,288 | **4.00** |
| `stencil` | 4 floats | 1,966,080 | **5.00** | 655,360 | **5.00** |

`elementwise` moves 4.00 → 5.00 sectors per request on both loads and stores:
655,360 / 524,288 = **1.250 exactly**. The `stencil`'s loads already average
4.67 sectors at offset 0 — its `i-1` / `i+1` neighbours straddle lines on their
own — so misalignment costs its loads only 1.071x, while its stores pay the
full 1.25x.

## Result 2 — the TornadoVM kernels match the misaligned CUDA, sector for sector

`ncu-cuda15.csv` vs `ncu-tornado15.csv`, the actual demo 15 kernels:

| Kernel | metric | CUDA | TornadoVM | CUDA @ off=4 |
|---|---|---|---|---|
| `elementwise` | ld sectors | 524,288 | **655,360** | 655,360 |
| `elementwise` | st sectors | 524,288 | **655,360** | 655,360 |
| `polynomial` | ld sectors | 524,288 | **655,360** | — |
| `polynomial` | st sectors | 524,288 | **655,360** | — |
| `stencil` | ld sectors | 1,835,006 | **1,966,080** | 1,966,080 |
| `stencil` | st sectors | 524,288 | **655,360** | 655,360 |

Every TornadoVM kernel reports **5.00 sectors per request** on every global
load and store. The counts are **bit-identical to the deliberately misaligned
CUDA kernel** — not approximately, exactly. This is no longer an inference from
wall-clock: it is the same defect, measured on the generated code.

Both `stencil` launches and all three kernels reproduce across the two
executions profiled (IDs 0–2 and 3–5 in `ncu-tornado15.csv`).

## Result 3 — `polynomial` carries the same defect and it costs nothing

`polynomial` pays the identical 1.25x sector penalty, yet it is the kernel where
TornadoVM *wins* (35.2 µs vs 39.9 µs, batch 21). Being compute-bound, it hides
the extra transactions behind the FMA chain.

That separates two claims batch 21 could not: the header offset is present in
**every** kernel TornadoVM generates, and it becomes visible in time only when
the kernel is bandwidth-bound. Fixing #1065 should be expected to help
memory-bound kernels and do nothing measurable for compute-bound ones.

## Result 4 — the cost is in the L1↔L2 path, not DRAM

`dram__bytes_read.sum` is ~16.78 MB (= `n` x 4 bytes) on both sides for all
three kernels, differing by under 0.03%. The array is streamed once either way;
the misalignment costs extra 32-byte sector transactions inside the cache
hierarchy, not extra DRAM traffic.

## What the counters do and do not settle

They settle the **mechanism** exactly. They are not a linear predictor of the
**time** delta: `stencil`'s total sectors rise 1.111x (2,621,440 vs 2,359,294)
against a measured 1.24x time ratio, while `elementwise`'s 1.250x sits near its
1.31x. Wall-clock attribution still rests on the offset probe, which reproduces
unchanged in this session (`probe-header-alignment-timing.log`):

```
elementwise  offset=0 floats : median 12.6 us
stencil      offset=0 floats : median 13.4 us
elementwise  offset=4 floats : median 16.2 us     (1.29x)
stencil      offset=4 floats : median 16.9 us     (1.26x)
```

against batch 21's 12.5 / 13.2 / 16.0 / 16.7 µs — within run-to-run spread.

Note that kernel durations reported *under* `ncu` are not comparable to the
timed runs: single cold launches with flushed caches land all four probe
configurations at ~19.5 µs. Use the counters for mechanism, the timing loop for
time.

## Files

| File | Contents |
|---|---|
| `ncu-probe-offsets.csv` | `ProbeHeaderAlignment --counters`, one launch per (kernel, offset) |
| `ncu-cuda15.csv` | demo 15 hand-written CUDA, 3 kernels |
| `ncu-tornado15.csv` | demo 15 TornadoVM, 6 launches (2 executions x 3 kernels) |
| `probe-header-alignment-timing.log` | timing probe re-run, this session |
| `profiling-permission.log` | `RmProfilingAdminOnly` + `ncu --version` |
| `*.stderr.log` | profiler stderr for each run |

Upstream: [beehive-lab/TornadoVM#1065](https://github.com/beehive-lab/TornadoVM/issues/1065).
