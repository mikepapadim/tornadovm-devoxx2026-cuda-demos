# How to analyse the findings in this repo

Start here if you want to *reason about* the evidence — check a claim, answer a
new question, or decide what is safe to say. For building slides, use
[`PRESENTATION-BRIEF.md`](PRESENTATION-BRIEF.md) instead; it is a subset shaped
for that.

The repo holds **168 CSVs, 34 Nsight Systems traces and two full evidence
bundles on two GPU architectures**. This tells you what is where, how to read
it, and — most importantly — the six ways it is easy to draw a wrong conclusion
from it.

---

## 1. What exists

### Evidence bundles — structured, provenance-complete

| Bundle | Architecture | Entry point |
|---|---|---|
| `results/nvidia-meeting/` | **sm_89** — RTX 4090, CUDA 12.6.85, driver 565.57.01 | `summary.md`, `manifest.json` |
| `results/nvidia-meeting-sm120/` | **sm_120** — RTX 5070 Ti, CUDA 13.0.88 | `summary.md` |

Both follow the same task matrix **A–G** (see
[`nvidia-meeting-metrics-plan.md`](nvidia-meeting-metrics-plan.md)):

| Task | Question it answers |
|---|---|
| A | Baseline code quality, kernel time, matched geometry |
| B | Alignment isolation — is the memory gap layout or code generation? |
| B2 | Alignment separated from launch geometry |
| C | JIT specialisation, with SASS |
| D | Tensor-core path, all operand types |
| E | Runtime dispatch, CUDA Graph, buffer reuse |
| F | Fused GEMM baseline, cold/warm compile split |
| G | CUDA Tile feasibility inventory |

`results/nvidia-meeting/manifest.json` is the machine-readable index: **9
entries**, each with the claim, method, problem size, statistic, tolerance, raw
file paths, and `corrections` / `resolved` fields where a finding changed.

### Raw batches — chronological, 27 of them

`results/raw/00-…` through `27-profiler-metrics`. Each has a `MANIFEST.md`
stating what was captured, how, and what it means. The ones that carry current
findings:

| Batch | Contents |
|---|---|
| `21-kernel-time-comparison` | demo 15 kernel times under `nsys`, JIT-specialisation probe, CUDA SASS |
| `22-ncu-alignment-counters` | the #1065 sector counters |
| `23-ncu-tensor-core-counters` | demo 08 HMMA counters |
| `24-ncu-demo14-counters` | demo 14 memory counters |
| `25-host-dispatch-breakdown` | per-execution API cost by differencing |
| `26-tensor-core-datatypes` | demo 16, all five operand types |
| `27-profiler-metrics` | **the consolidated, chart-ready CSVs** |

### Failures — kept deliberately

`results/failures/` records what was attempted and did not work, with commands
and exact errors. A failure with a log is evidence; a missing measurement is
not.

---

## 2. How to load it

**Chart-ready, start here:**

```bash
results/raw/27-profiler-metrics/comparison.csv   # 52 paired rows, ratios precomputed
results/raw/27-profiler-metrics/metrics.csv      # 172 rows, long format
python3 scripts/build-metrics-csv.py             # regenerate both from raw captures
```

`metrics.csv` is long format (`demo, kernel, implementation, metric, unit,
value, launches, source, file`) so it pivots however a question needs. **Every
row carries the `file` it came from** — any number traces back to its capture.

**Querying the manifest:**

```bash
python3 -c "
import json; d=json.load(open('results/nvidia-meeting/manifest.json'))
for r in d['results']:
    print(r['id'], r['status'], '|', r.get('conclusion','')[:90])"
```

**Nsight Systems traces:** export to SQLite, then query.

```bash
nsys export --type sqlite -o t.sqlite trace.nsys-rep
# kernel times
SELECT s.value, COUNT(*), AVG(k.end-k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN StringIds s ON s.id=k.demangledName GROUP BY s.value;
# transfer histogram — this is how buffer reuse was verified
SELECT bytes, copyKind, COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY bytes, copyKind;
```

**Nsight Compute CSVs** have preamble before the real header — skip to the line
starting `"ID"` before parsing, and strip thousands separators from values.

---

## 3. Six ways to draw a wrong conclusion

Each of these produced a wrong answer in this repo before being caught. They are
the reason most claims here carry a caveat.

**1. Mixing `ncu` and `nsys` — including their *ratios*.** `ncu` serialises
launches, flushes caches and disallows clock boost. The same demo 15 kernels at
identical verified geometry give memory-bound ratios of **1.02–1.04 under `ncu`**
and **1.24–1.31 under `nsys`**. Quote `nsys` for performance, `ncu` for
counters. Never one axis, never a cross-mode ratio.
→ `results/nvidia-meeting/measurement-mode/`

**2. Comparing unequal launch geometry.** Demo 15 pins both sides to
block=256/grid=16384 — verify from `launch__block_size` in the trace, not from a
source comment. **Demo 12 and demo 01 do not**: TornadoVM's default worker grid
picks 1024-thread blocks there, which changes occupancy independently of code
quality.

**3. Dividing a total by iteration count.** Start-up cost does not amortise out
of a single trace. Per-execution cost must come from **differencing two
execution counts**. Dividing overstated it ~12× once.
→ `results/raw/25-host-dispatch-breakdown/MANIFEST.md`

**4. Differencing without dropping one-shot APIs.** Differencing cancels call
*counts* cleanly but not one-time *durations* — `cuCtxCreate_v2` alone runs to
~110 ms and its variance swamps a few-µs signal. Drop APIs whose delta
call-count is ~0.

**5. Trusting a tiny kernel's timing.** Demos 08 and 16 compute 128–256 output
elements on one warp. They are **code-generation validation**. No timing claim
is made from them and none should be.

**6. Reading a wall-clock number that includes process start-up.** Demo 12 runs
two plans in one JVM; the first absorbs `cuCtxCreate` and all initialisation.
Its wall clock cannot compare fused against unfused — its kernel times can.
→ `results/nvidia-meeting/task-F-fused-gemm/wallclock-inversion-resolved.md`

---

## 4. The findings, by question

### Is TornadoVM's generated code competitive?

**Yes, once layout and specialisation are controlled.** Demo 15 under `nsys`,
matched geometry, sm_89: elementwise **1.31**, stencil **1.24**, polynomial
**0.88** (TornadoVM faster). Both deltas are attributed below, and neither is
arithmetic quality.

### Why are memory-bound kernels slower?

**Data layout, not code generation.** `FloatArray`'s payload sits 16 bytes into
the allocation; a warp request is 128 contiguous bytes and the L1↔L2 path is
sector-addressed at 32 bytes, so every access straddles a fifth sector — **5.00
vs 4.00 sectors/request**, measured on four demos and six kernels.

Proven by sweeping *only the base-pointer offset of one compiled binary*: if
code generation mattered, that could not reproduce it.
→ `results/nvidia-meeting/AlignmentSweep.cu`

**32-byte alignment is sufficient**, not 128 — any 32B-aligned payload gives
4.00. That is now [PR #1066](https://github.com/beehive-lab/TornadoVM/pull/1066).

### Why is the compute-bound kernel faster?

**JIT specialisation, visible in SASS.** `degree=256` is a task argument, so
Graal compiles after it is bound: **288 SASS instructions, 256 FFMA, 1 branch** —
fully unrolled — against **13 branch-class instructions** in the nvcc
runtime-scalar build. Give nvcc a template constant and it reaches parity. The
advantage is *when* compilation happens.

TornadoVM writes every cubin to
`$TORNADOVM_HOME/var/cuda-codecache/device-0-0/` by default, so `cuobjdump -sass`
reads it directly.

### Where does the wall-clock time actually go?

**Host dispatch.** ~8.3 µs per execution of CUDA driver API for a one-kernel
graph, of which a 24-byte kernel-argument stack frame is re-uploaded every launch
and three `cuStreamSynchronize` are issued where one would do. Both are
[#1028](https://github.com/beehive-lab/TornadoVM/issues/1028) findings 1 and 2.

**CUDA Graph replay removes the whole sequence**: 6 launches + 7 H2D + 42 event
calls → 1 `cuGraphLaunch` + 3 event calls, **5.4× less host API time**.

### Does library-task integration cost anything?

**No.** Both implementations invoke the same CUTLASS kernel at the same launch
configuration and it performs identically — **1.00–1.04×**. And in a
`JIT → cuDNN → JIT → cuDNN` graph the input crosses once, the output once per
execution, and the intermediates never: device buffers are reused with no host
round trip.

### What is the tensor-core story?

All five operand combinations the emitter can produce are exercised and
validated at **max abs err 0.00000**, with counters matching emitted PTX
instruction for instruction. The ceiling is `MMAShape = {M16N8K16, M16N8K32}` —
sm_89-class; `wgmma` and `tcgen05` are not reachable by extending that enum.

### Can we prototype CUDA Tile?

**Not yet.** CUDA 12.6.85: no header, no qualifiers, no library, no flags, no
NVRTC symbols. CUDA 13.0.88: `crt/cuda_tile.h` (60 lines, one entity) and three
`__tile*__` qualifiers — but still **no library, no flags, no NVRTC symbols**.
That is *plumbing*, not a usable API.

---

## 5. Comparing across architectures

**Never merge or average two architectures.** Side by side in a table is fine;
one number covering both is not. The sm_120 bundle is separate for this reason.

Selected side-by-side (full table in `results/nvidia-meeting-sm120/summary.md`):

| measurement | sm_89 | sm_120 |
|---|---|---|
| elementwise TVM/CUDA | 1.31 | **1.019** |
| stencil TVM/CUDA | 1.24 | **1.005** |
| alignment wall-clock cost | +28.2% | **+6.8%** |
| block 256→1024 (TVM / CUDA) | 1.131 / 1.159 | **1.295 / 1.364** |
| CUDA Graph host-API speedup | 5.4× | 5.29× |
| instruction count, TVM vs nvcc | 9,437,184 vs 7,864,320 | **equal** |

Two readings that need care:

- **The memory-bound gap largely closes on Blackwell.** Real, and it weakens the
  practical urgency of #1066 on newer hardware — worth saying before that PR
  merges.
- **The instruction-count parity is not a TornadoVM improvement.** nvcc 13.0's
  count *rose* to meet TornadoVM's. Attribute it correctly.

Counter names differ: **FP8 issues as QMMA on sm_120, HMMA on sm_89**, so
`op_hmma + op_imma` no longer sums to the tensor-pipe total there. A
cross-architecture FP8 comparison needs different metrics per side.

---

## 6. Corrections — do not resurrect these

Retracted claims, kept visible so they are not reintroduced from an older file.

| Claim | Status |
|---|---|
| "demo 12's 1.20–1.23× JIT ratio corroborates the alignment finding" | **retracted** — launch geometry differed (1024 vs 256 block) |
| "~1.075× is the defensible codegen number, replacing 1.20–1.31×" | **retracted** — mixed `ncu` and `nsys` modes, and swept in demo 15, which was never confounded |
| "demo 01's occupancy shows a codegen difference" | **retracted** — launch-config artefact |
| "TornadoVM-side SASS is not capturable" | **wrong** — the cubin is on disk by default |
| "the fused path is slower, cause unknown" | **resolved** — execution-ordering artefact |
| "a Blackwell card will take the PTX fallback" | **wrong** — only when the toolkit predates the GPU |

The valid figures are the `nsys`, matched-geometry ones: **1.31 / 1.24 / 0.88**.

---

## 7. Upstream issues found

| Issue | Summary |
|---|---|
| [#1063](https://github.com/beehive-lab/TornadoVM/issues/1063) | `CuDnn.sdpaForward` launches no kernel, returns all-zero |
| [#1064](https://github.com/beehive-lab/TornadoVM/issues/1064) | CUDA lowering crash on a ternary before an allocation |
| [#1065](https://github.com/beehive-lab/TornadoVM/issues/1065) | `FloatArray` header misaligns coalescing — PR [#1066](https://github.com/beehive-lab/TornadoVM/pull/1066) |
| [#1067](https://github.com/beehive-lab/TornadoVM/issues/1067) | a `KernelContext` kernel that fails to compile silently returns **wrong results** |
| [#1071](https://github.com/beehive-lab/TornadoVM/issues/1071) | zero dispatch and copy-out timers in the CUDA profiler |
| [#1072](https://github.com/beehive-lab/TornadoVM/pull/1072) | assertions carried no messages, so failures read `[REASON] null` |

Plus PR [#1022](https://github.com/beehive-lab/TornadoVM/pull/1022) — per-launch
dispatch cost, verified against a same-session `develop` baseline.

**#1067 is a silent data-corruption path and is arguably the most serious.**

---

## 8. Verifying a claim yourself

1. Find it in `results/nvidia-meeting/summary.md` or `manifest.json`.
2. Follow the `raw` / `file` field to the capture.
3. Check the caveat attached to that entry — most carry one.
4. Re-run it: the exact command is in the task `README.md` or the plan document.
5. On different hardware, follow
   [`REPRODUCE-ON-ANOTHER-GPU.md`](REPRODUCE-ON-ANOTHER-GPU.md) and write to a
   **new** `results/nvidia-meeting-<arch>/`.

If a claim has no file behind it, it is not in this repo and should not be said.
