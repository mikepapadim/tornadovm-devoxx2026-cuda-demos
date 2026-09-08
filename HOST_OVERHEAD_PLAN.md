# Host and runtime overhead — investigation and fix plan

Follow-on campaign to upstream issue
[#1028](https://github.com/beehive-lab/TornadoVM/issues/1028). That issue closed
the first chapter: it found four host-side costs, shipped fixes for three of
them, and left a ranked list of what remains. This plan picks that list up,
re-ranks it against fresh measurements, and adds the workstreams that only
became worth doing once kernel time came down.

**Method is not restated here.** Every recipe, baseline and gotcha lives in the
`tornadovm-perf-campaign` skill (`TornadoVM/.claude/skills/tornadovm-perf-campaign/`):
`references/measurement.md` for JFR and nsys, `references/probes.md` for the probe
harness, `references/catalogue.md` for the pattern catalogue and the shares already
established. Read the catalogue before starting a workstream — the most common way
to waste a campaign is optimising a path that is already 0.1% of wall.

---

## 1. Why now

Kernel-side work has been landing (PR
[#1066](https://github.com/beehive-lab/TornadoVM/pull/1066) payload alignment,
PR [#1079](https://github.com/beehive-lab/TornadoVM/pull/1079) global-load
batching). Every microsecond removed from a kernel **raises** the host share of
wall time. That is the Amdahl consequence and it is measurable on demo 17's
register-tiled rung, n=1024, 30 executions:

| | kernel time | wall clock | non-kernel | non-kernel share |
|---|---|---|---|---|
| before #1079 | 112.9 µs | 429 µs | 316 µs | **73.7%** |
| after #1079 | 68.4 µs | 391 µs | 323 µs | **82.5%** |

The absolute non-kernel cost is unchanged (316 vs 323 µs — same number twice,
which is the point). Its **share went up by nine points from one kernel fix.**
Further kernel work has a ceiling of about 1.2x on this workload's wall clock;
the host side has the other 4x.

Caveat that Phase 0 exists to remove: "non-kernel" above is wall minus kernel,
so it bundles genuine device transfer with host cost. A first orientation
capture over the whole 6-rung ladder (180 executions) splits as:

| | value | note |
|---|---|---|
| `cuStreamSynchronize` | 4,124 calls, 137.7 ms | ~23 per execution; median 285 ns, mean 33 µs — the mean is the blocking waits |
| D2H copies | 1,972, 58.6 ms | ~11 per execution, median 2.7 µs |
| H2D copies | 1,810, 8.4 ms | median ~1.0 µs at the API |
| `cuCtxCreate_v2` | **1 call, 102 ms** | start-up |
| `cuMemHostRegister_v2` | 21 calls, 21.5 ms | start-up pinning |
| `cuLibraryLoadData` | 4 calls, 8.8 ms | code install |

Two things fall out immediately. **Start-up is ~140 ms of one-off cost** against
a decode/steady-state budget measured in microseconds — for anything short-lived
it dominates everything else in this plan. And **23 stream synchronisations per
execution** is far above the 1 that issue #1028's work got `saxpy` down to, so
the multi-graph plan shape is paying a cost the single-graph benchmark does not.

---

## 2. What is already done — do not redo it

**Landed on `develop`:**

| PR | What |
|---|---|
| [#1022](https://github.com/beehive-lab/TornadoVM/pull/1022) | Skip the per-launch kernel stack-frame upload when unchanged (2.03x on short graphs) |
| [#1025](https://github.com/beehive-lab/TornadoVM/pull/1025) | Propagate CUDA driver failures to Java instead of logging them |
| [#1027](https://github.com/beehive-lab/TornadoVM/pull/1027) | Clear only the written prefix of each wait-list row (1.57x deps-on decode) |
| [#1029](https://github.com/beehive-lab/TornadoVM/pull/1029) | Read non-terminal copy-outs asynchronously (1.27x on an 8-output graph) |
| [#1002](https://github.com/beehive-lab/TornadoVM/pull/1002) | Allocate wait-event lists on demand |
| [#1035](https://github.com/beehive-lab/TornadoVM/pull/1035), [#1049](https://github.com/beehive-lab/TornadoVM/pull/1049), [#1004](https://github.com/beehive-lab/TornadoVM/pull/1004) | Correctness prerequisites for trusting any of the above |
| [#1040](https://github.com/beehive-lab/TornadoVM/pull/1040) | Interpreter bytecode dispatch through named handlers |

**Open, in flight — coordinate, do not duplicate:**

| PR | What | Bearing on this plan |
|---|---|---|
| [#1024](https://github.com/beehive-lab/TornadoVM/pull/1024) | Deferred profiler timestamp harvest | W4 builds on it |
| [#1030](https://github.com/beehive-lab/TornadoVM/pull/1030) | `withDeferredOutputs()` — overlap read-back | **W2 is largely this**; measure its ceiling, don't re-derive it |
| [#1031](https://github.com/beehive-lab/TornadoVM/pull/1031) | Coalesce on-demand copy-outs (2.6x for 8 objects) | W6 |
| [#1032](https://github.com/beehive-lab/TornadoVM/pull/1032) | Copy-out contract tests | Prerequisite for W6 |
| [#1051](https://github.com/beehive-lab/TornadoVM/pull/1051) | Cache the Graal compilation result per device | **W1b** — part of start-up |
| [#986](https://github.com/beehive-lab/TornadoVM/pull/986), [#982](https://github.com/beehive-lab/TornadoVM/pull/982) | `executeAsync()`; stream routing | Adjacent; may subsume parts of W3 |

**Closed without merging — the win is unclaimed:**
[#1023](https://github.com/beehive-lab/TornadoVM/pull/1023) (3 → 1
`cuStreamSynchronize` per execution). W3 starts by finding out *why* it was
closed before writing any code.

**Open bugs that gate work here:** [#1071](https://github.com/beehive-lab/TornadoVM/issues/1071)
(dispatch timers structurally 0, copy-out time never recorded — you cannot use
TornadoVM's own profiler to validate W1–W4 until this is fixed) and
[#1006](https://github.com/beehive-lab/TornadoVM/issues/1006) (CUDA graph replay
fails silently when buffers are re-pointed — a category-1 silent-wrong-answer bug
that blocks W7).

---

## 3. Non-negotiables

Lifted from the skill because breaking them invalidates results silently:

- `-Dtornado.recover.bailout=False` on **every** measurement, or you are timing
  sequential Java.
- **Never measure with `-Dtornado.profiler=True`.** It perturbs a short graph by
  ~3x and misreports transfer timers by up to 20x. Use JFR for host, nsys for
  driver.
- **A/B in the same build family**, 3+ runs, medians, and always a **GPU-bound
  control that must not move** (`saxpy 2000 16777216`, `nbody 500 16384`). If the
  control moves, the measurement is wrong.
- **Separate displaced work from removed work.** Repeat every A/B at two
  iteration counts. A gap that shrinks with more iterations is a cost you moved.
- **Call counts are more robust than totals** in nsys; totals include start-up.
- One mechanism = one number = one PR.
- Report start-up and steady state **separately**. A decode win is invisible in
  whole-run wall on a short run.

---

## 4. Phase 0 — attribution before any change

Nothing gets optimised until the split is known. Deliverable: a committed
baseline table under `results/raw/32-host-overhead/`.

**0.1 Split wall clock properly for three shapes.** Dispatch-bound
(`saxpy 100000 512`), the demo-17 ladder (multi-graph, realistic), and a
GPU-bound control. For each, capture kernel time, H2D/D2H time, driver API time
and JFR host samples, and reconcile them to wall clock. The reconciliation is the
point: if kernel + transfer + attributable host ≠ wall, the residue is the thing
to chase.

```bash
nsys profile -t cuda --sample=none --cpuctxsw=none -f true -o base <cmd>
nsys stats --force-export=true \
  --report cuda_api_sum --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum base.nsys-rep
```

**0.2 Separate start-up from steady state.** Same workload at two execution
counts (e.g. 10 and 100). Fixed cost falls out as the intercept. Expect start-up
to be ~140 ms; confirm and attribute it.

**0.3 JFR leaf-frame ranking per phase**, with `--stack-depth 96` and the phase
filter (`initializeTornadoVMPlan` for start-up). Note the two documented gotchas:
`--stack-depth` is mandatory, and do not split the print output on `jdk.`.

**0.4 Explain the 23 `cuStreamSynchronize` per execution** on the ladder against
`saxpy`'s 1. This is the largest unexplained number in §1 and may be the whole
of W3.

**Exit criterion:** a table where every workload's wall clock is accounted for to
within 10%, and the top three host costs are named with their shares.

---

## 5. Workstreams, ranked

Ranked by (measured ceiling × confidence) ÷ cost. Each states its hypothesis as a
number, per the skill — "dispatch feels slow" is not a hypothesis.

### W1 — Start-up (largest untouched absolute number)

~140 ms per process, against steady-state costs in microseconds. Dominates every
short run, every CI job and every LLM run under ~20 s.

- **W1a. Does the cubin cache actually skip NVRTC on a second run?**
  `tornado.cuda.codecache.enable` defaults to `True` and cubins land in
  `$TORNADOVM_HOME/var/cuda-codecache/device-0-0/`. *Hypothesis to test first:
  a second run of the same kernel does zero NVRTC work.* If it does not, that is
  the single biggest start-up win available. Known related defect worth fixing
  regardless: `resolveDirectory` concatenates onto `$TORNADOVM_HOME`, so an
  absolute `-Dtornado.cuda.codecache.dir` is silently not honoured.
- **W1b. Graal front-end caching** — PR #1051 is open; measure it, don't rebuild it.
- **W1c. `cuMemHostRegister` — 21 calls, ~1 ms each.** Why 21? Hypothesis: host
  buffers are pinned eagerly whether or not they are ever used for an async copy.
  Probe: count registrations against buffers actually used asynchronously.
- **W1d. `cuCtxCreate` at 102 ms** is probably irreducible driver init, but
  establish whether it is on the critical path or can overlap class loading and
  Graal warm-up. Cheap to answer, and it bounds everything else in W1.

**Ceiling:** if W1a and W1c pay, a short run improves by tens of milliseconds —
larger in absolute terms than everything in W2–W6 combined.

### W2 — The terminal blocking read-back

Issue #1028 measured it at 65% of remaining host time on a dispatch-bound graph;
the catalogue puts the *recoverable* fraction at ~17%, because much of that sync
overlaps work already enqueued. **PR #1030 (`withDeferredOutputs()`) is the fix
and is already open.** This workstream is therefore: measure its ceiling
honestly on the three Phase-0 shapes, and determine whether it should become the
default rather than opt-in. Do not re-derive the mechanism.

### W3 — Per-execution driver call count on multi-graph plans

`saxpy` is down to 1 `cuStreamSynchronize` per execution; the ladder shows ~23.
**Start by reading why #1023 was closed unmerged** — that PR took 3 → 1 and was
not accepted, and the reason will shape whatever replaces it. Then attribute the
23 (per graph? per task? per transfer?) and remove the ones that wait on an
already-empty stream.

**Hypothesis:** a plan of N task graphs pays O(N) synchronisations per
`execute()` where O(1) would do.

### W4 — Profiler: correctness first, then cost

Two problems, and the correctness one blocks the campaign's own instrumentation:

- **#1071**: dispatch times are structurally 0 and copy-out time is never
  recorded. Until this is fixed, TornadoVM's own profiler cannot validate any
  result in this plan. **Highest priority in W4.**
- Residual ~2.2x profiler-on overhead is now the timing events themselves
  (`cuEventRecord` 615 ns with `CU_EVENT_DEFAULT` vs 137 ns with
  `CU_EVENT_DISABLE_TIMING`, six per iteration). Route: opt-in timing granularity
  (kernels only). Reusing operation N−1's end event as N's start would halve it
  but is incompatible with #1024's deferred harvesting — do not attempt both.

### W5 — Wait-list sizing and knob consolidation

Small, cheap, low-risk; good first tasks.

- A wait-list row is still allocated at `tornado.max.events` (32768 ints =
  128 KB) on first write, even though #1027 removed the *clear* cost. Grow on
  demand.
- `tornado.eventpool.maxwaitevents` vs `tornado.max.events` — two knobs, similar
  names, very different cost profiles, and only the former is exposed by
  front-ends such as `llama-tornado`. Consolidate, or at minimum document.

### W6 — Transfer-API consolidation

`read`/`enqueueRead`, `streamOut`/`streamOutBlocking`, and ~96 near-identical
device-context overloads differing by one boolean. This is where two of the
catalogue's anti-patterns live ("a blocking/async pair that is not
substitutable"), and it has already produced silent wrong results. **#1032's
contract tests must land first** — they pin behaviour before the refactor.
Correctness workstream with a performance side effect, not the reverse.

### W7 — CUDA graphs as the default for repeated plans

The catalogue measures 72 → 103 tok/s (1.42x) on LLM decode from batching a whole
pipeline into one submission, and calls the feature "existing, under-used".
**Blocked on #1006** — replay fails with `CUresult=700` when a captured graph's
buffers are re-pointed, *silently*. That is a category-1 bug and outranks the
speed-up. Fix #1006, then evaluate making graph capture automatic for plans that
execute unchanged more than N times.

---

## 6. Explicitly not worth working on

Measured, and already near zero. Recorded so they are not re-attempted:

- **The Java bytecode interpreter.** ~0.9% of wall on a dispatch-bound graph
  (118 `ExecutionSample` vs 13,545 `NativeMethodSample`, and most Java samples
  were the benchmark harness's own `Arrays.sort`).
- **Per-token activation graphs** — 0.0% of GPU time, ~0.1% of wall. Structurally
  ugly, not worth optimising.
- **Three dead ends from the previous campaign**, per the catalogue: an existing
  "non-blocking stream-out" flag that measured *slower*; an ordering-barrier
  theory; double-buffered outputs (a single in-order queue already orders copy N
  before kernel N+1).

---

## 7. Risks

- **Measuring the wrong thing.** The most likely failure. Phase 0's
  reconciliation requirement exists to catch it: if the parts do not sum to wall
  clock, stop.
- **A "win" that is displaced work.** Mitigated by the two-iteration-count rule.
- **Correctness regressions that look like speed-ups.** #1025 exists because a
  cubin that loaded but could not launch reported a 35x speed-up produced
  entirely by no kernel running. Any result better than the stated ceiling should
  be treated as a bug report first.
- **Aggregate test counts are per-machine.** On the reference box CUDA
  `--quickPass` fails 85 and OpenCL 271 on unmodified `develop`. OpenCL claims
  must be per test class, same device, before/after.
- **One machine is one data point.** RTX 4090 only. Say so in every PR.

---

## 8. Tracking

- Keep **#1028 as the umbrella issue**: one comment per result, including
  disproven hypotheses. It is the only shared memory across PRs and sessions.
- Evidence under `results/raw/32-host-overhead/`, following the batch-31
  convention: raw CSVs, a `collect.sh` that regenerates them, and a README that
  states the caveats.
- **Write a resume pointer to memory** when a session ends mid-campaign: branch,
  baseline failure counts, last measured table, next hypothesis. Re-deriving a
  baseline costs two builds.
- One PR per mechanism, in the shape the skill specifies: cost, cause, change,
  numbers with a control row, testing against a same-machine baseline, backends
  tested and not tested, and negatives. **A PR that only lists wins reads as
  unmeasured.**

## 9. Suggested order

1. **Phase 0** — nothing else is trustworthy without it.
2. **W4's #1071** — needed to instrument the rest.
3. **W1a** — one measurement (does the cubin cache work?) with the largest
   absolute payoff behind it.
4. **W3** — the 23-synchronisations anomaly, which Phase 0 may already have
   explained.
5. **W5** — cheap, low-risk, good while a build runs.
6. **W2** — measure #1030's ceiling; decide opt-in vs default.
7. **W6**, then **W7** — both gated on correctness work landing first.
