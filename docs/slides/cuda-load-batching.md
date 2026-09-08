---
marp: true
theme: default
paginate: true
title: "Closing the register-tiled gap: a CUDA code-generator fix"
---

<!--
Slide source for the load-batching finding. Renders as-is:

    npx @marp-team/marp-cli@latest docs/slides/cuda-load-batching.md -o deck.pdf
    npx @marp-team/marp-cli@latest docs/slides/cuda-load-batching.md -o deck.pptx

Each `---` starts a slide. Each slide has a `Notes:` block at the end — that is what
to say, not what to put on the screen. Numbers are machine-extractable from
docs/findings/cuda-load-batching.yaml; if the two ever disagree, the YAML and the CSVs
under results/raw/31-load-batching-reorder/ win.

Every number: RTX 4090 (sm_89), CUDA 12.6, JDK 25.0.2, TornadoVM built from
beehive-lab/TornadoVM develop + PR #1079. One machine, one architecture, 2026-09-08.
-->

# Closing the register-tiled gap

### A 1.65x CUDA code-generator fix in TornadoVM

**upstream PR [#1079](https://github.com/beehive-lab/TornadoVM/pull/1079) · issue [#1078](https://github.com/beehive-lab/TornadoVM/issues/1078)**

RTX 4090 (sm_89) · CUDA 12.6 · one machine, one architecture

<!--
The story is diagnosis, not the patch. The patch is 200 lines. Getting to *why* took ruling out four plausible causes that were all wrong.
-->

---

## The setup: a ladder, not a benchmark

Six implementations of the same 1024x1024 sgemm, same GPU, same session:

| rung | what |
|---|---|
| 1 | naive `@Parallel` |
| 2 | KernelContext, shared-memory tiled |
| **3** | **KernelContext, register-tiled (4x4 micro-tile)** |
| 4 | CUTLASS |
| 5 | cuBLAS sgemm |
| 6 | cuBLAS TF32 |

Rung 3 has a **hand-written CUDA twin**: same algorithm, same tile sizes, same launch geometry.

<!--
The ladder exists so every rung is comparable to the one below it. Rung 3 is the interesting one because it is the last rung a user writes themselves — above it you are calling a vendor library.
-->

---

## The observation

Rung 3, `nsys` per-launch kernel time, n=1024:

| | median |
|---|---|
| TornadoVM `kcRegisterTiled` | **112.9 µs** |
| hand-written CUDA `registerTiled` | **67.5 µs** |

### TornadoVM is 1.67x slower running the same algorithm

<!--
Kernel time, not wall-clock — no host dispatch in these numbers. Same block and grid dimensions, verified from the profiler rather than from source comments.
-->

---

## What it was **not**

Four hypotheses, all measured, all wrong:

| hypothesis | probe | result |
|---|---|---|
| worse arithmetic | FFMA count | **identical** — 1,073,741,824 both sides |
| register pressure / spills | `launch__registers_per_thread` | **identical** — 72, zero spills |
| shared-memory bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu` | TornadoVM **fewer** |
| pointer aliasing | add `const` / `__restrict__` | **1.00x** — no change |

<!--
This slide is the honest part of the talk. Each of these took a run to eliminate, and the aliasing one is the interesting failure — it is *nearly* the right answer, and it still measured nothing.
-->

---

## What it was: memory latency

Same kernel, same launch, `ncu`:

| counter | TornadoVM | hand-written |
|---|---|---|
| `long_scoreboard` stalls per issue | **12.10** | 1.45 |
| issue rate (% of peak sustained) | **20.8%** | 50.9% |

### Same work, same occupancy, no spills — the kernel was simply waiting

<!--
`long_scoreboard` is the stall reason for a warp waiting on a global load. Twelve cycles per issue means the machine is idle almost all the time. Issue rate is the same statement from the other side.
-->

---

## The generated code

`tornado --printKernel`, the k-tile staging loop:

```c
ul_107  =  ul_0 + l_106;
f_108   =  *(( float *) ul_107);   // global load
adf_5[i_97]  =  f_108;             // shared store consumes it immediately
i_109   =  i_92 + i_102;           // only now the next address
l_110   =  (long long) i_109;
ul_113  =  ul_0 + ((l_110 + 4L) << 2);
f_114   =  *(( float *) ul_113);   // cannot start until the first landed
adf_5[i_88]  =  f_114;
```

### One global load in flight at a time

<!--
Eight of these pairs per tile iteration. The address arithmetic for load *n+1* sits *after* the store of load *n*, so the chain is fully serial.
-->

---

## The SASS says it in three lines

`L` = global load, `S` = shared store, in the staging loop:

```
TornadoVM, before          L S L S L S L S L S L S L S L S      generic LD
TornadoVM, after           L L L L L L L L S S S S S S S S      generic LD
hand-written CUDA          L L L L L L L L S S S S S S S S      LDG
```

<!--
This is the whole finding on one slide. Both TornadoVM rows use the *same* memory instruction; only the schedule differs. The hand-written row uses a *different* instruction — and that is the cause.
-->

---

## Root cause: generic `LD` vs `LDG`

**Hand-written CUDA** loads through `const float *`
→ ptxas lowers it to **`LDG`**, a load *proven* to target global memory
→ proven not to alias the shared store beside it
→ **ptxas batches the loads itself**

**TornadoVM** casts an integer address to a plain pointer
→ ptxas lowers it to a **generic `LD`**, which *may* target shared memory
→ must stay ordered against every shared store
→ **the staging chain serialises**

<!--
This is why `__restrict__` did nothing. Restrict tells the compiler two pointers do not alias each other. It does not tell it which address space a generic pointer points into.
-->

---

## The negative result that proved it

Can the slow schedule be written in CUDA C at all? Three source orders of one kernel:

1. interleaved: load, store, load, store
2. interleaved + `asm volatile("" ::: "memory")` between pairs
3. batched: all loads, then all stores

### All three compile to **byte-identical SASS**

448 instructions, `LDG`x8 then `STS`x8, every time.

<!--
I built this probe to reproduce the slow schedule and it refused to produce it. That failure is the useful result: ptxas picks the schedule, not the source, and the asm barrier constrains only the NVVM frontend. You cannot write this bug in CUDA C.
-->

---

## The fix

TornadoVM knows the address space at **LIR level** (`CUDAMemorySpace`) even though its emitted C does not encode it. So do the batching in the code generator.

Within a straight-line run of *{non-local load, register arithmetic, local/shared store}*:

### sink the stores to the end of the run

Sink stores rather than hoist loads — hoisting a load means hoisting its whole address-arithmetic chain; sinking a store moves **one instruction**.

<!--
`CUDAGlobalLoadBatching`, run from `CUDACompilationResultBuilder.emitBlock`. The run ends at the first instruction outside that set, so barriers, shared loads, global stores, atomics and MMA statements all terminate it with no special case.
-->

---

## Why it is safe

1. **Disjoint address spaces** — writes go to shared/private, reads come from global/constant
2. **No reader in the run** — a shared/private load is *not* eligible, so it ends the run
3. **Store order preserved** — two writes to one slot still land in order
4. **Dependences hold** — a store only moves *later*, so its operands are still defined first
5. **No clobber** — checked explicitly that nothing jumping ahead of a store defines a value it reads
6. **Visibility unchanged** — the run ends before any barrier

If any one store fails the check, the **whole run** is left untouched.

<!--
Point 5 is the one the address-space argument does not cover — it would matter if the LIR ever reused a variable. It is tested rather than assumed.
-->

---

## The result

n=1024, **100 launches per arm**, `nsys`. One build, flag toggled at run time.

| arm | median | stddev | |
|---|---|---|---|
| before | 112.895 µs | 0.152 | — |
| **after** | **68.416 µs** | 0.541 | **1.650x** |
| hand-written CUDA | 67.455 µs | 0.388 | |

### Gap to hand-written CUDA: 1.67x → **1.4%**

| counter | before | after |
|---|---|---|
| `long_scoreboard` per issue | 12.10 | **1.45** |
| issue rate | 20.8% | **50.9%** |

<!--
One build with a system property toggled, so the two arms differ only in emitted instruction order — not compiler, driver, machine or data. Stddev under 0.8% of median in every arm.
-->

---

## How we know it is safe

**Full device suite twice on the same build**, flag off and on — 1199 tests each:

### zero regressions

11 failures identical in both runs, pre-existing on `develop`.

**New no-GPU codegen tests** — 15 targeted cases, mostly "must NOT reorder", plus a property test over **4,000 random instruction sequences** checking five invariants. Runs in under a second, on every push, no GPU.

**New device tests** — 7 kernels where a wrong schedule changes the **answer**, not the speed. Integer arithmetic, exact expected values.

<!--
A codegen reordering bug does not fail loudly — it returns a slightly wrong number. That is why the property test exists and why the device kernels are built so a mis-schedule is a wrong answer. Two tests failed only in the baseline run; those look like flakes and I am not claiming the fix repaired them.
-->

---

## What is left

**Still on the table for this kernel:**

- TornadoVM still emits generic `LD` where nvcc emits `LDG`
  → emitting `__ldg()` or a global-qualified pointer would fix the schedule *at the source*
- The 16-byte `FloatArray` header still costs **5 sectors per warp access instead of 4**

Both are separate from this fix, and together they are most of the residual 1.4%.

<!--
The `LDG` change is the more interesting follow-up — it is a change to address lowering, it would help beyond this pattern, and it would make this peephole redundant for the cases ptxas can then handle itself.
-->

---

## Reproduce it

```bash
git clone <demo-repo> && source scripts/setup-env.sh
cd results/raw/31-load-batching-reorder
./collect.sh "$TORNADOVM_HOME" 1024 100
```

Writes `per-launch-durations.csv` (every launch, all 3 arms), `kernel-summary.csv`,
`ncu-counters.csv`, `sass-schedules.txt`, `pure-cuda-probe.csv`.

**Caveats:** one GPU, one kernel shape. A kernel that does not stage through shared memory is unaffected. `nsys` and `ncu` disagree on ratios for this family — counters are for *why*, timings for *how much*, never on one axis.

<!--
Needs a build containing the fix; without it both arms give the "before" numbers. `ncu` is optional, the script skips the counter table if it is missing.
-->
