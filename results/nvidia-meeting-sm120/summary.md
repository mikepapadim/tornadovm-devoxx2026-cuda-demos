# sm_120 / RTX 5070 Ti — evidence pack

Parallel bundle to `results/nvidia-meeting/` (sm_89 / RTX 4090).
**Not merged with it, and no figure averages across the two architectures.**

## Host

| | sm_120 (this run) | sm_89 (baseline) |
|---|---|---|
| GPU | RTX 5070 Ti, cc 12.0, 16303 MiB | RTX 4090, cc 8.9 |
| Driver | 580.142 | 565.57.01 |
| CUDA | 13.0.88 | 12.6.85 |
| JDK / TornadoVM | 25.0.2 / 6.0.0-jdk22plus-cuda | same |
| gcc | 12.3.0 | 11.4 |
| OS | Ubuntu 23.10 (EOL) | Ubuntu 22.04 LTS |
| repo | 82cbc927 | 82cbc927 |

Five variables move at once (GPU, driver, toolkit, gcc, OS). Differences below are
measured on this host; attribution to any single variable is not established.

## Status

| Step | Task | State |
|---|---|---|
| 0b | provenance | DONE — `env/provenance.txt` |
| 1  | correctness, both suites | DONE — 39/39 Java, 28/28 CUDA, 0 failed, 0 skipped |
| 2  | alignment sweep | DONE — `NOTES-step2-alignment.md` |
| 3  | geometry-controlled 2x2 | DONE — `task-B2/` |
| 4  | kernel time under nsys | DONE — `task-A/` |
| 5  | tensor cores, 5 operand types | DONE — `task-D/` |
| 6  | host dispatch + CUDA Graph | DONE — `task-E/` |
| 7  | CUDA Tile inventory | DONE — `tile-feasibility/` |
| 8  | SASS | DONE — `task-C/` |

Steps 2, 3 and 5 needed `ncu` performance counters, which needed
`NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/` and a reboot.
`RmProfilingAdminOnly` now reads **0**; `run-blocked-steps.sh` re-runs them.

## Headline results

**Step 1 — everything passes.** 39/39 TornadoVM, 28/28 pure CUDA. No architecture
failures. CUTLASS 3.5.1 compiles and runs for sm_120 under CUDA 13.0.88 despite
predating consumer Blackwell.

**Step 2 — the alignment penalty reproduces exactly in sectors, and is far
cheaper in time.** 4 sectors per request when 32 B-aligned, 5 when offset by 16 B,
655,360 / 524,288 = 1.25x — value for value identical to sm_89. DRAM traffic is
flat across every offset on both hosts (~16.78 MB), so the extra sectors are
absorbed before DRAM. What changed is the wall-clock cost of that same 1.25x:
**+6.8% here against +28.2% on sm_89**. Five variables move, so this is not
attributed. Under `ncu` the effect vanishes on both architectures — sector counts
come from `ncu`, the time cost must not. See `NOTES-step2-alignment.md`.

**Step 3 — the 2x2 decomposition still multiplies out, and the instruction gap
has closed.**

| Effect | sm_120 | sm_89 |
|---|---|---|
| alignment + codegen, geometry controlled | **1.047** | 1.075 |
| block 256 -> 1024 within TornadoVM | 1.295 | 1.131 |
| block 256 -> 1024 within CUDA | 1.364 | 1.159 |
| uncontrolled (TVM@1024 / CUDA@256) | 1.356 | 1.216 |

1.047 x 1.295 = 1.356 exactly, as on sm_89. Sectors per request stay 5.00 vs 4.00
at *both* block sizes, so the alignment penalty remains geometry-independent, and
the 256 -> 1024 penalty is again worse on hand-written CUDA than on TornadoVM —
it is not a TornadoVM property. Two things differ from the baseline: that
block-size penalty roughly doubled, and TornadoVM's 1.20x instruction-count
disadvantage (9,437,184 vs 7,864,320) is **gone** — both sides now execute
9,437,184, because nvcc 13.0's count rose to meet TornadoVM's, which did not move.
Toolkit and architecture change together, so neither is credited. See `task-B2/`.

**Step 4 — TornadoVM's memory-bound gap has closed on Blackwell.**

| kernel | TVM/CUDA sm_120 | sm_89 |
|---|---|---|
| elementwise | 1.019 | 1.31 |
| polynomial  | 0.914 | 0.88 |
| stencil     | 1.005 | 1.24 |

Geometry verified from the trace: both sides grid 16384x1x1, block 256x1x1, 16 regs.

**Step 5 — tensor-core codegen is unchanged, but FP8 is a distinct op on
Blackwell.** Emitted PTX is identical to sm_89 (4x `m16n8k16` BF16, 2x `m16n8k32`
for each of e4m3, e5m2, s8), and the aggregate counters reproduce it instruction
for instruction (4/2/2/2 tensor instructions, 64/32/32/32 cycles). The dispatch
changed: on sm_89 both FP8 formats counted as **HMMA**, while on sm_120 they count
in the HMMA/QMMA/OMMA umbrella but **not** as HMMA — FP8 issues as **QMMA**. So
`op_hmma + op_imma` no longer sums to the tensor-pipe total on Blackwell; the
umbrella `subpipe_hmma` / `subpipe_imma` pair does. BF16 -> HMMA and int8 -> IMMA
are unchanged. No timing claim: one warp, 128-256 output elements. See `task-D/`.

**Step 6 — the CUDA Graph result reproduces.** Counts identical to baseline
(nograph 6 launches + 7 H2D + 42 event calls; graph 1 `cuGraphLaunch` + 3 event
calls). Host API excl. sync 32,862 -> 6,212 ns/exec = **5.29x** (baseline 5.4x).
Demo 13 buffer reuse holds: no intermediate is staged through the host.

**Step 7 — CUDA 13.0 carries Tile-language plumbing that 12.x does not**, but
exposes no usable API. See `tile-feasibility/inventory.txt`.

**Step 8 — native sm_120 cubin, no PTX fallback.** `polynomial` degree 256 is
fully unrolled at 288 instructions / 1 branch, reproducing the sm_89 baseline
exactly; nvcc keeps the loop rolled at 11 branches because degree is a runtime
scalar. That is the mechanism behind the 0.914 polynomial ratio in Step 4.

## Four corrections to the reproduction doc

1. **Step 7's `find` command yields a false zero.** `find` defaults to `-P` and
   will not follow the `include -> targets/x86_64-linux/include` symlink in
   NVIDIA's standard layout. Needs `find -L`. The sm_89 baseline's "all four
   counts were 0" was produced by the same command and should be re-run.
2. **Step 6's differencing cancels call counts but not one-time durations.**
   `cuCtxCreate_v2` varies ~40 ms run to run, swamping the signal and producing a
   negative "speedup". The time sum must be restricted to APIs whose differenced
   call count is > 0.
3. **Step 3 overwrites the sm_89 baseline.** It says `cd
   results/nvidia-meeting/task-B2-geometry-controlled` then writes `tvm_$b.csv`
   and `cuda_$b.csv` into the working directory — but four files of exactly those
   names are the committed sm_89 baseline. Running Step 3 verbatim destroys them,
   which breaks the doc's own headline rule ("never merge the new results into
   `results/nvidia-meeting/`"). Output must be redirected to
   `results/nvidia-meeting-<arch>/task-B2/`.

All three are written up in full in the respective NOTES.md.

4. **Nsight Compute returns `n/a` for renamed metrics instead of failing.** Four
   metric names the doc specifies were renamed in this profiler generation
   (2025.3.1.0 / CUDA 13.0): `sm__inst_executed_pipe_tensor_op_{hmma,imma}.sum` ->
   `..._subpipe_{hmma,imma}_op_{hmma,imma}.sum`, and
   `dram__bytes_{read,write}.sum` -> `dram__bytes_op_{read,write}.sum`. `ncu`
   emits the row with a value of `n/a`, writes nothing to stderr, and **exits 0**.
   A capture script that checks only the exit status records a column of `n/a` and
   reports success — which is what the first pass here did. Names must be checked
   against `ncu --query-metrics`. Both steps were re-captured under the current
   names; full detail in `task-D/NOTES.md`.
