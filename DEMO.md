# NVIDIA demo — run sheet

**Verified end to end on 2026-09-09**, RTX 4090, driver 565.57.01, CUDA 12.6, JDK 25.0.2.

Everything lives in `~/nvidia-demo`. Nothing depends on `/tmp`. The SDK is a private
copy, so a reboot, a `/tmp` sweep, or rebuilding TornadoVM elsewhere cannot break it.

The build is TornadoVM `develop` plus the load-batching fix
([PR #1079](https://github.com/beehive-lab/TornadoVM/pull/1079)) — i.e. what the CUDA
backend looks like once that lands. There is no before/after toggling; the demo shows
where the backend stands.

Total runtime: **under 4 minutes** for all three.

---

## Step 0 — setup (once, ~5 s)

```bash
source ~/nvidia-demo/env.sh
```

This sets `JAVA_HOME`, points `TORNADOVM_HOME` at `~/nvidia-demo/tornadovm-patched`,
puts `tornado` on `PATH`, sets `$DEMO=~/nvidia-demo`, and **leaves you in
`~/nvidia-demo/classes`** — which is where all three commands must be run from.
Both the Java and the CUDA sides are already compiled; there is no build step.

Confirm before you start:

```bash
pwd                 # -> /home/michalis/nvidia-demo/classes
tornado --devices   # -> NVIDIA GeForce RTX 4090
```

---

## Step 1 — where the backend stands  (~90 s)

**Run from:** `~/nvidia-demo/classes`

```bash
bash $DEMO/1-ladder.sh
```

Six implementations of the same 2048×2048 sgemm. Prints `measuring n=2048, 10
executions (about 90 seconds)...` then sits silent while nsys profiles both sides —
**that pause is expected.**

### Expected

```
rung                                  TornadoVM   hand-written     ratio
--------------------------------------------------------------------------
1. naive @Parallel                    3403.4 us      3387.7 us    1.005x
2. KernelContext tiled                2596.6 us      2590.3 us    1.002x
3. KernelContext register-tiled        499.0 us       486.0 us    1.027x
```

Numbers move ±1% run to run. Ratios are stable.

### What to say

Same algorithm, same tile sizes, same launch geometry, same arithmetic — the Java
side JIT-compiled at run time, **within 3% of hand-written CUDA at every rung**. The
Java rungs use `KernelContext`, TornadoVM's explicit shared-memory and thread-index
API, so rungs 2 and 3 are hand-written CUDA transliterated into Java.

### Do not say

**Do not quote the wall clock** the ladder prints if anyone runs it directly.
TornadoVM's wall clock includes host dispatch and three 16 MB transfers; the CUDA
binary reports kernel time only. Comparing them shows a fake ~3.3× gap. This script
measures kernel time via nsys on both sides precisely to avoid that.

---

## Step 2 — the CUDA it generates  (~40 s)

**Run from:** `~/nvidia-demo/classes`

```bash
bash $DEMO/2-printkernel.sh
```

### Expected (abridged — 40 lines total)

```
== signature

    __global__ void kcRegisterTiled(long long *_kernel_context, unsigned char *_constant_region, unsigne

== the k-tile staging loop: eight global loads, then eight shared stores

    f_108  =  *(( float *) ul_107);             <-- global load
    f_114  =  *(( float *) ul_113);             <-- global load
    ...                                          (eight loads)
    adf_5[i_97]  =  f_108;                      <-- shared store
    adf_5[i_88]  =  f_114;                      <-- shared store
    ...                                          (eight stores)

== the inner product: shared read-backs feeding a register micro-tile

    f_161  =  adf_5[i_160];                     <-- shared load
    f_179  =  fma(f_170, f_178, f_177);         <-- FMA into the accumulator
```

### What to say

418 lines of CUDA C produced from a Java method at run time and handed straight to
NVRTC. Point at the two blocks: eight global loads issued together, then eight shared
stores — that is what keeps several loads in flight.

No bounds checks anywhere: `CUDAHighTier` appends an `ExceptionSuppression` phase that
deletes every guard and condition before code generation.

---

## Step 3 — the one difference that is left  (~40 s)

**Run from:** `~/nvidia-demo/classes`

```bash
bash $DEMO/3-sass.sh
```

### Expected

```
                         staging schedule   memory ops
  ---------------------- ------------------ ----------------------
  TornadoVM              LLLLLLLLSSSSSSSS   STS x8  LD x8
  hand-written CUDA      LLLLLLLLSSSSSSSS   STS x8  LDG x8
```

`L` = global load, `S` = shared store (`STS`), in the k-tile staging loop.

### What to say

The **schedule** is identical — that is the 3% in step 1. The **instruction** is not.

TornadoVM casts an integer address to a plain pointer, and the emitted C carries no
address space (`GLOBAL_MEM_MODIFIER` is the empty string), so ptxas lowers it to a
**generic `LD`**. A generic load may target shared memory, so ptxas must keep it
ordered against every shared store — it cannot batch these itself, which is why
TornadoVM has to do it in the code generator, where the address space *is* known.

**End on the open question:** emitting `__ldg()` or a global-qualified pointer would
hand ptxas the address space directly and fix the schedule at its source. That is the
thing to ask them about.

---

## If something misbehaves

| symptom | fix |
|---|---|
| `tornado: command not found` or no device | re-run `source ~/nvidia-demo/env.sh`; check `nvidia-smi` |
| you are in the wrong directory | `cd ~/nvidia-demo/classes` — all three need it |
| a Java stack trace from `medianOf` | the ladder needs ≥2 executions; the scripts pass 3 or 10, so only happens if you edit the args |
| step 3 prints "no cubin cached" | `tornado.cuda.codecache.enable` must be on (default); cache is `$TORNADOVM_HOME/var/cuda-codecache/device-0-0/` |
| nsys not found / permission denied | steps 2 and 3 do not need nsys — run those; use the committed CSVs for step 1's numbers |

**Full fallback if the GPU is unavailable:** every number is committed in the demos
repo under `results/raw/31-load-batching-reorder/` — `ladder-kernel-times.csv`,
`per-launch-durations.csv` (303 launches), `sass-schedules.txt`, `ncu-counters.csv`.
The talk survives with no hardware.

---

## Numbers you may be asked for

| | |
|---|---|
| speedup from the fix, rung 3 | 695.7 → 499.4 µs at n=2048 (**1.39×**); 112.9 → 68.4 µs at n=1024 (**1.65×**) |
| gap to hand-written CUDA, rung 3 | was **1.42×**, now **1.02×** |
| `long_scoreboard` stalls per issue | 12.10 → **1.45** |
| issue rate | 20.8% → **50.9%** |
| FFMA count, registers, spills | unchanged: 1,073,741,824 / 72 / 0 |
| correctness | full device suite 1206 tests, **zero regressions** |

## Backing material

- upstream PR: <https://github.com/beehive-lab/TornadoVM/pull/1079>
- evidence + raw CSVs: `results/raw/31-load-batching-reorder/`
- slides: `docs/slides/cuda-load-batching.md` (Marp → `npx @marp-team/marp-cli@latest <file> -o deck.pdf`)
- structured numbers: `docs/findings/cuda-load-batching.yaml`
