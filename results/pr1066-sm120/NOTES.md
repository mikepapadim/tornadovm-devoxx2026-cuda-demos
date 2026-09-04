# PR #1066 re-measured on sm_120 / RTX 5070 Ti

[beehive-lab/TornadoVM#1066](https://github.com/beehive-lab/TornadoVM/pull/1066)
pads every non-batched CUDA device allocation so a `TornadoNativeArray`'s payload
starts on a 32-byte boundary instead of 16 bytes into the buffer. Its measurements
were taken on an RTX 4090 (sm_89). This is the same analysis on Blackwell.

**Headline: the patch does exactly what it claims to the sector counts, and on this
GPU that buys essentially no time.** The 1.29x the PR reports at 16 MB on sm_89
does not reproduce here; what reproduces is 5.00 -> 4.00 sectors per request.

## Method — deliberately the same as the PR's

One build supplies both arms. `-Dtornado.cuda.payloadAlignment=1` restores the
pre-patch layout exactly, so *before* and *after* differ in one property and
nothing else — not two builds, not two checkouts. Hand-written CUDA compiled at
`-arch=sm_120 -O3` from the demo suite's own `.cu` sources is the third arm.

Kernel time only, from `nsys stats --report cuda_gpu_kern_sum`; sector counts from
`ncu`. Wall clock is never used. Reproduce with `run-pr1066-analysis.sh` and
`run-sweep-repeated.sh`; provenance in `provenance.txt`.

| | |
|---|---|
| GPU | RTX 5070 Ti, cc 12.0, driver 580.142 |
| TornadoVM | `b0cc7f231` — PR #1066 on `develop` (`e06a2e56f`) |
| CUDA / JDK | 13.0.88 / 25.0.2 |
| nsys / ncu | 2025.3.2.474 / 2025.3.1.0 |

`--printKernel` emits identical index offsets in both arms, confirming the patch
moves the buffer and not the generated code.

## 1. Sectors per request — reproduces perfectly

`GeometryControlled`, n = 16777216 (64 MB), block 256, 2 executions under `ncu`:

| arm | ld sec/req | st sec/req | ld sectors | st sectors |
|---|---|---|---|---|
| before | 5.00 | 5.00 | 2,621,440 | 2,621,440 |
| **after** | **4.00** | **4.00** | **2,097,152** | **2,097,152** |
| hand-written CUDA | 4.00 | 4.00 | 2,097,152 | 2,097,152 |

2,621,440 / 2,097,152 = 1.25 exactly. **After the patch TornadoVM's sector counts
are identical to hand-written CUDA's**, on both loads and stores. This is the
robust half of the measurement and it is unambiguous.

## 2. Kernel time — the benefit does not follow

Buffer-size sweep, `elementwise` at block 256, 20 executions per run, with the two
arms **interleaved** so drift hits both equally. Median of per-run medians. Run
twice: batch 1 while an unrelated Jenkins CI job was intermittently on the GPU,
and a verification batch on a GPU confirmed to have no other compute process.

| buffer | before | after | batch 1 | verification |
|---|---|---|---|---|
| 1 MB | 1600 ns | 1600 ns | 0.980 | **1.000** |
| 4 MB | 4512 ns | 4384 ns | 0.972 | **0.972** |
| 16 MB | 16088 ns | 15672 ns | 0.974 | **0.974** |
| 64 MB | 162517 ns | 165957 ns | 1.020 | **1.021** |
| 256 MB | 671140 ns | 673700 ns | 1.004 | **1.004** |

Times are the verification batch. The two batches agree to within 0.001 at every
size but 1 MB, which sits on the timer's 32 ns granularity (1600 vs 1568 ns) and
should be read as *no measurable difference* rather than as a 2% gain. That the
contended and idle batches agree is also the evidence that the Jenkins job did not
perturb these numbers.

> Provenance caveat: the verification run reused the same output names, so batch 1's
> repetitions 1-4 were overwritten in place; only its repetition 5 survives, in
> `raw-repeat-batch1/`. The batch-1 column above is the figure computed from all five
> of its repetitions before that happened, and is reported here for comparison only —
> everything quoted elsewhere in this file comes from the verification batch, which is
> committed in full.

Per-repetition values are tight and the arms do not overlap at 4, 16 or 64 MB
(64 MB before = 162309..162724, after = 165733..166229), so the signs are real.
The gain peaks at **~2.7%**, against **29%** at 16 MB on sm_89, and inverts into a
**2.1% regression at 64 MB**.

Two method notes, both of which changed a number:

- The first, *non-interleaved* pass of this sweep ran all of "before" then all of
  "after", once per point, and reported 16 MB as 0.989 — the opposite sign to the
  0.974 that eleven interleaved repetitions give. Only the interleaved numbers are
  quoted; `raw/` keeps the first pass.
- `nsys stats` **silently reuses an existing `.sqlite` export**. Re-running the
  sweep regenerated every `.nsys-rep` but left the CSVs untouched, so the first
  attempt at this verification re-read hour-old numbers and "reproduced" the
  earlier batch to the digit. `--force-export=true`, or deleting the `.sqlite`
  first, is required. This is the same failure shape as `ncu` returning `n/a` for
  a renamed metric: the tool succeeds, exits 0, and hands back something stale.

## 3. Three kernels against hand-written CUDA

Demo 15, n = 4194304 (16 MB), 3 runs x 20 executions, median kernel time. Spread
across runs is under 0.4% except where noted:

| kernel | before | after | hand-written CUDA | after/before | after/CUDA |
|---|---|---|---|---|---|
| `elementwise` (mem-bound) | 38789 ns | 40074 ns | 38149 ns | **1.033** | 1.050 |
| `stencil` (mem-bound) | 17861 ns | 17195 ns | 17738 ns | **0.963** | 0.969 |
| `polynomial` (compute-bound) | 55290 ns | 55146 ns | 60399 ns | 0.997 | 0.913 |

**The two memory-bound kernels move in opposite directions**: `stencil` (3 reads,
1 write) gains 3.7%, `elementwise` (1 read, 1 write) loses 3.3% — and loses it
against hand-written CUDA too, going from 1.017x to 1.050x. `after`'s
`elementwise` is also the only measurement here with visible variance (2.4%
spread against 0.2% for every other row).

Note that `elementwise` at 16 MB is *faster* after the patch when measured alone
(§2, 0.974) and *slower* inside demo 15's three-kernel chain (1.033). Same kernel,
same size, opposite sign. The chain leaves different data in L2 and allocates more
buffers, each shifted 16 bytes by the padding. The effect is real in both cases
and depends on surrounding context, which is itself the finding: on this GPU the
padding is not a reliable win.

## 4. Why — L2 traffic falls, DRAM write traffic rises

Same 64 MB configuration under `ncu`, before vs after:

| metric | before | after | delta |
|---|---|---|---|
| `lts__t_sectors_op_read.sum` | 2,164,510 | 2,098,821 | **-3.03%** |
| `lts__t_sectors_op_write.sum` | 2,466,891 | 2,098,083 | **-14.95%** |
| `dram__bytes_op_read.sum` | 67,128,320 | 67,132,160 | +0.01% |
| `dram__bytes_op_write.sum` | 45,930,752 | 47,039,488 | **+2.41%** |
| `gpu__time_duration.sum` | 150,080 ns | 151,424 ns | +0.90% |

The patch delivers what it targets — L2 write sectors fall 15% — but this kernel
runs at ~86% of peak DRAM bandwidth, so L1<->L2 sector count is not what bounds
it. DRAM *writes* rise 2.4%, and time follows the DRAM traffic rather than the
sector count.

That the regression appears in `ncu` (+0.90%) and in `nsys` (+2.0%) independently,
in the same direction, is the reason it is reported as an effect rather than noise.

## 5. Consistency with the sm_120 evidence pack

This is the same conclusion Step 2 of [`../nvidia-meeting-sm120/`](../nvidia-meeting-sm120/NOTES-step2-alignment.md)
reached from the opposite direction. There, sweeping the offset of a *hand-written*
CUDA kernel, the identical 1.25x sector penalty cost **+6.8%** of wall time on
sm_120 against **+28.2%** on sm_89, and DRAM traffic was flat across every offset.
Blackwell already absorbs sub-sector misalignment. A patch whose entire mechanism
is removing that misalignment therefore has little left to recover here.

## What this does and does not say about the PR

- It does **not** contradict the sm_89 numbers. Different GPU, driver, toolkit,
  gcc and OS; the sm_89 measurements were not reproduced or re-run.
- The mechanism claim is **confirmed**: 5.00 -> 4.00 sectors per request, matching
  hand-written CUDA exactly, generated code unchanged.
- The benefit claim is **architecture-dependent** in a way the PR body does not
  currently note. On this Blackwell host the patch is roughly time-neutral, with a
  reproducible ~2% regression at 64 MB and inside demo 15's `elementwise`.
- **No correctness regression**: `run-all-demos.sh` gives 39/39 on the patched
  build, identical to the unpatched 6.0.0 SDK.
- The cost side of the PR's own argument gets stronger, not weaker: 16 bytes per
  allocation is cheap, and on this architecture it is close to all one is buying.

## Files

| File | Contents |
|---|---|
| `run-pr1066-analysis.sh` | arms 1-4: codegen identity, sectors, demo 15, first sweep |
| `run-sweep-repeated.sh` | the interleaved, repeated sweep — the quotable one |
| `raw-repeat/` | verification batch, idle GPU, forced re-export |
| `raw-repeat-batch1/` | the surviving repetition of batch 1 |
| `provenance.txt` | host, toolchain and commit under test |
| `raw/` | first-pass captures, `ncu` CSVs, demo-15 nsys reports |
| — | sweep `.nsys-rep`/`.sqlite` are not committed: the CSVs carry every quoted number and the scripts regenerate the rest |
| `raw-repeat/` | the 5 interleaved repetitions |
| `run-all-demos-patched.log` | 39/39 on the patched build |
