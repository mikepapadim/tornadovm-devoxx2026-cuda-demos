# NVIDIA demo — run sheet

**Verified end to end on 2026-09-09**, RTX 4090, driver 565.57.01, CUDA 12.6, JDK 25.0.2.

Everything lives in `~/nvidia-demo`. Nothing depends on `/tmp`. The SDK is a private
copy, so a reboot, a `/tmp` sweep, or rebuilding TornadoVM elsewhere cannot break it.

The build is TornadoVM `develop` plus the load-batching fix
([PR #1079](https://github.com/beehive-lab/TornadoVM/pull/1079)) — i.e. what the CUDA
backend looks like once that lands. There is no before/after toggling; the demo shows
where the backend stands.

Total runtime: **under 6 minutes** for all four. Steps 2-4 are the comparison
against hand-written CUDA; step 1 stands alone and is the natural opener.

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

## Step 1 — one kernel, end to end  (~70 s)

**Run from:** `~/nvidia-demo/classes`

```bash
bash $DEMO/0-tiled.sh
```

The simplest thing that is still a real GPU kernel: a 16x16 shared-memory tiled
matmul, written once in Java. Runs it, profiles it, and prints the complete generated kernel.
Takes `[n] [executions]`, default `1024 10`.

### Expected

```
 1/3  RUN -- a shared-memory tiled matmul written in Java
tiled matmul  n=1024  tile=16x16  grid=64x64 blocks of 16x16 threads
first execution (JIT compile + run): 100582 us
steady-state wall clock (n=9):      median 1103 us  (min 1096, max 1156)
validation: c[512][341] = 491.000, expected 491.000 -> PASSED

 2/3  PROFILE -- what the GPU actually did (Nsight Systems)
   kernel
      tiled                    333.5 us   x10 launches
   transfers
      [CUDA memcpy Host-to-Devic   485.5 us per execution
      [CUDA memcpy Device-to-Hos   271.1 us per execution

   kernel 333 us  +  transfers 757 us  =  1090 us
   which is essentially the wall clock printed in step 1.

 3/3  THE GENERATED CUDA -- what TornadoVM handed to NVRTC
===============================================================
__global__ void tiled(long long *_kernel_context, ..., unsigned char *arg3, int arg4)
{
  ...declarations...
  __shared__ float adf_3[256];
  __shared__ float adf_4[256];
  i_5  =  (threadIdx.x);
  i_7  =  (threadIdx.y);
  i_39  =  (blockIdx.x);
  ...
  for(;i_48 < 1024;)                     <- the k-tile loop
  {
    f_54  =  *(( float *) ul_53);        <- global load
    adf_3[i_38]  =  f_54;                <- shared store
    f_62  =  *(( float *) ul_61);
    adf_4[i_38]  =  f_62;
    __syncthreads();
    f_63  =  adf_3[i_8];                 <- 32 shared read-backs, fully unrolled
    f_64  =  adf_4[i_5];
    ...
    __syncthreads();
    f_95  =  fma(f_63, f_64, f_47);      <- 16 FMAs, one per k
    f_96  =  fma(f_65, f_66, f_95);
    ...
    f_110  =  fma(f_93, f_94, f_109);
  }  // B1
  *(( float *) ul_116)  =  f_47;         <- the single global store
  return;
}

   ^ 140 lines of CUDA C, generated from one Java method at run time.
```

### What to say

Three things, in order.

**It is a real kernel, printed in full.** `__shared__` tiles come from
`ctx.allocateFloatLocalArray`, both `__syncthreads()` from `ctx.localBarrier()`,
`threadIdx`/`blockIdx` from the `KernelContext` fields, and it ends in one global
store. 140 lines from one Java method, compiled at run time and handed to NVRTC.
It validates against a CPU reference.

Worth pointing at in the listing: **Graal fully unrolled the 16-iteration inner
loop** into 32 shared read-backs and 16 back-to-back `fma()` calls. The k-tile loop
survives as a loop; the inner one does not. That unrolling is a JIT advantage --
`TILE` is a compile-time constant to TornadoVM at the moment it generates code.

**The first execution costs 100 ms and the rest cost 1.1 ms.** That is the JIT
compiling the kernel. Worth saying out loud before anyone asks.

**The profile explains the wall clock exactly**: 333 us of kernel plus 757 us of
transfers is the 1.1 ms. This is why every comparison in steps 2-4 uses kernel time
-- wall clock on this workload is mostly PCIe.

### Or run it by hand

The script is only a wrapper. Every part is one command, from
`~/nvidia-demo/classes`:

```bash
# 1/3  run it
tornado --classpath . TiledMM 1024 10

# 2/3  profile it
nsys profile -t cuda --force-overwrite=true -o /tmp/t \
     tornado --classpath . TiledMM 1024 10
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv /tmp/t.nsys-rep
nsys stats --force-export=true --report cuda_gpu_mem_time_sum --format csv /tmp/t.nsys-rep

# 3/3  the generated CUDA, in full
tornado --printKernel --classpath . TiledMM 1024 3
```

`--force-export=true` matters: `nsys stats` refuses to run if a `.sqlite` from an
earlier export of the same name is still lying around, which happens the moment you
re-run the profile. Without it the second attempt fails with a bare usage message.

`--printKernel` prints every kernel the plan compiles; here there is only one.
Pipe to `less` if you want to scroll it live.

**Without the launcher**, if someone asks what `tornado` is doing — it is a plain
JVM plus an argfile, and the flag is an ordinary system property:

```bash
java @$TORNADOVM_HOME/tornado-argfile -Dtornado.printKernel=True -cp . TiledMM 1024 3
```

Same output. Useful for showing there is no magic in the launcher.

### Note

The transfer counts (534 and 522 copies) include TornadoVM's small internal copies,
not just the three 4 MB arrays; the per-execution *time* is the number that matters
and is what is printed.

---

## Step 2 — where the backend stands  (~90 s)

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

### Or run it by hand

From `~/nvidia-demo/classes`:

```bash
# TornadoVM side
nsys profile -t cuda --force-overwrite=true -o /tmp/tv \
     tornado --classpath . MatMulLadder 2048 10
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv /tmp/tv.nsys-rep

# hand-written CUDA side
nsys profile -t cuda --force-overwrite=true -o /tmp/cu \
     $DEMO/matmul_ladder_cuda 2048 10
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv /tmp/cu.nsys-rep
```

Compare the `Med (ns)` column: `kcRegisterTiled` against `registerTiled`,
`kcTiled` against `tiled`, `naive` against `naive`. The script just does that
arithmetic for you.

### Do not say

**Do not quote the wall clock** the ladder prints if anyone runs it directly.
TornadoVM's wall clock includes host dispatch and three 16 MB transfers; the CUDA
binary reports kernel time only. Comparing them shows a fake ~3.3× gap. This script
measures kernel time via nsys on both sides precisely to avoid that -- and step 1
shows the arithmetic behind it.

---

## Step 3 — the CUDA it generates  (~40 s)

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

### Or run it by hand

From `~/nvidia-demo/classes`:

```bash
tornado --printKernel --classpath . MatMulLadder 512 3 | less
```

Search inside `less` for `kcRegisterTiled` (`/kcRegisterTiled`) to jump to the
register-tiled kernel, then look at the run of `*(( float *) ...)` loads followed by
the run of `adf_*[...] =` stores.

Or pull just that kernel out:

```bash
tornado --printKernel --classpath . MatMulLadder 512 3 2>/dev/null \
  | awk '/__global__ void kcRegisterTiled/,/^\}/'
```

---

## Step 4 — the one difference that is left  (~40 s)

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

### Or run it by hand

From `~/nvidia-demo/classes`:

```bash
# TornadoVM's own cubin -- the code cache writes it on every run
tornado --classpath . MatMulLadder 512 3
ls $TORNADOVM_HOME/var/cuda-codecache/device-0-0/
cuobjdump -sass $TORNADOVM_HOME/var/cuda-codecache/device-0-0/kcRegisterTiled-*.cubin \
  | grep -oE '\b(LD|LDG|STS)\b' | tr -d '\n'; echo

# the hand-written one
nvcc -arch=sm_89 -O3 -std=c++17 -cubin -o /tmp/hand.cubin $DEMO/MatMulLadder.cu
cuobjdump -sass /tmp/hand.cubin | grep -oE '\b(LD|LDG|STS)\b' | tr -d '\n'; echo
```

The first prints `LDLDLDLD...STSSTS...`, the second `LDGLDGLDG...STSSTS...` — same
schedule, different load instruction. That the cubin is simply sitting on disk is
worth mentioning: `tornado.cuda.codecache.enable` is on by default, so the compiled
kernel is always inspectable after a run.

---

## If something misbehaves

| symptom | fix |
|---|---|
| `tornado: command not found` or no device | re-run `source ~/nvidia-demo/env.sh`; check `nvidia-smi` |
| you are in the wrong directory | `cd ~/nvidia-demo/classes` — all three need it |
| a Java stack trace from `medianOf` | the ladder needs ≥2 executions; the scripts pass 3 or 10, so only happens if you edit the args |
| step 3 prints "no cubin cached" | `tornado.cuda.codecache.enable` must be on (default); cache is `$TORNADOVM_HOME/var/cuda-codecache/device-0-0/` |
| nsys not found / permission denied | steps 3 and 4 do not need nsys — run those; use the committed CSVs for step 2's numbers |

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
