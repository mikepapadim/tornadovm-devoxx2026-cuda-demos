# Step 2 — alignment sweep, sm_120

`nvcc -arch=sm_120 -O3` on `results/nvidia-meeting/AlignmentSweep.cu` (the same
source as the sm_89 baseline), Nsight Compute 2025.3.1.0, CUDA 13.0.88.
Counters became available only after `RmProfilingAdminOnly` reached 0.

## Sector arithmetic — reproduces the baseline exactly

| offset | ld sec/req | st sec/req | ld sectors | DRAM bytes read |
|---|---|---|---|---|
| 0 B   | 4 | 4 | 524,288 | 16,780,800 |
| 16 B  | **5** | **5** | **655,360** | 16,785,408 |
| 32 B  | 4 | 4 | 524,288 | 16,783,104 |
| 64 B  | 4 | 4 | 524,288 | 16,781,056 |
| 128 B | 4 | 4 | 524,288 | 16,781,312 |

Identical to sm_89, value for value: 4 sectors per request when the base address
is 32 B-aligned, 5 when it is not, and 655,360 / 524,288 = **1.25** exactly. Only
the 16 B offset misaligns; 32 B and above are sector-aligned again, which is what
a 32 B sector predicts.

**DRAM traffic is flat across every offset** — ~16.78 MB read in all five cases,
matching the sm_89 baseline's 16.78 MB. The extra sectors are L1↔L2 traffic and
are absorbed before DRAM. This was true on Ada and is still true on Blackwell.

## Wall-clock cost of the misalignment has shrunk

Program's own median, not under the profiler:

| offset | sm_120 | vs aligned | sm_89 | vs aligned |
|---|---|---|---|---|
| 0 B   | 17.7 us | — | 11.7 us | — |
| 16 B  | 18.9 us | **+6.8%** | 15.0 us | **+28.2%** |
| 32 B  | 18.2 us | +2.8% | 12.0 us | +2.6% |
| 64 B  | 18.2 us | +2.8% | 12.0 us | +2.6% |
| 128 B | 17.9 us | +1.1% | 11.7 us | 0% |

The same 1.25x sector penalty costs 6.8% of wall time here against 28.2% on the
sm_89 host. **Attribution is not established** — GPU, driver, toolkit, gcc and OS
all differ (see `summary.md`), and this host is not bandwidth-saturated on this
kernel to the degree the 4090 was.

## Under `ncu` the effect disappears, on both architectures

Kernel durations from the counter capture: 26.0 / 26.1 / 26.2 / 26.0 / 26.0 us on
sm_120 and 19.52 / 19.52 / 19.49 / 19.52 / 19.49 us on sm_89 — flat, with the
16 B offset indistinguishable from the aligned cases in both. This is the
measurement-mode effect documented in
[`../nvidia-meeting/measurement-mode/README.md`](../nvidia-meeting/measurement-mode/README.md):
Nsight Compute serialises launches, flushes caches and disallows clock boost, and
the alignment penalty is hidden under those conditions. **Sector counts must come
from `ncu`; the time cost must not.**

## Files

| File | Contents |
|---|---|
| `alignment-sweep-ncu.csv` | sector and duration capture |
| `alignment-sweep-dram-ncu.csv` | DRAM bytes, re-captured under the CUDA 13 metric names |
| `alignment-sweep-timing.log` | the program's own medians, no profiler attached |
