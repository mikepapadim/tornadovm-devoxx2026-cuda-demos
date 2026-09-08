# NVIDIA demo — three commands

Self-contained in `~/nvidia-demo`. Nothing depends on `/tmp`, and the SDK is a copy,
so a reboot or a tmp sweep cannot break it.

The SDK is `develop` plus the load-batching fix
([PR #1079](https://github.com/beehive-lab/TornadoVM/pull/1079)) — i.e. what
TornadoVM's CUDA backend looks like once that lands. No before/after toggling: the
demo shows where the backend stands.

## Setup — once, before you start

```bash
source ~/nvidia-demo/env.sh
```

Sets `JAVA_HOME`, points `TORNADOVM_HOME` at the SDK, and drops you in
`~/nvidia-demo/classes` where both sides are already compiled. No build step in
front of the audience.

Sanity check: `tornado --devices` prints the RTX 4090.

---

## 1. Where the backend stands  (~90 s)

```bash
bash $DEMO/1-ladder.sh
```

Six implementations of the same 2048×2048 sgemm, **kernel time via nsys**:

```
rung                                  TornadoVM   hand-written     ratio
1. naive @Parallel                    3408.5 us      3387.9 us    1.006x
2. KernelContext tiled                2599.7 us      2594.8 us    1.002x
3. KernelContext register-tiled        500.7 us       487.6 us    1.027x
```

**Say:** same algorithm, same tile sizes, same launch geometry — JIT-compiled from
Java, within 3% of hand-written CUDA at every rung. The Java is written with
`KernelContext`, TornadoVM's explicit shared-memory and thread-index API.

**Do not quote the wall clock the ladder prints.** TornadoVM's includes host
dispatch and three 16 MB transfers; the CUDA binary's does not. This script
compares kernel time precisely to avoid that.

## 2. The CUDA it generates  (~40 s)

```bash
bash $DEMO/2-printkernel.sh
```

The generated `kcRegisterTiled`: the k-tile staging loop (eight global loads, then
eight shared stores), the shared read-backs, and the FMA chain.

**Say:** 418 lines of CUDA C produced from a Java method at run time and handed
straight to NVRTC. No bounds checks — `CUDAHighTier` runs an `ExceptionSuppression`
phase that deletes every guard before code generation.

## 3. The one difference that is left  (~40 s)

```bash
bash $DEMO/3-sass.sh
```

```
                       staging schedule   memory ops
TornadoVM              LLLLLLLLSSSSSSSS   STS x8  LD  x8
hand-written CUDA      LLLLLLLLSSSSSSSS   STS x8  LDG x8
```

**Say:** the *schedule* is identical — that is why the two are within 3%. The
*instruction* is not. TornadoVM casts an integer address to a plain pointer and the
emitted C carries no address space (`GLOBAL_MEM_MODIFIER` is the empty string), so
ptxas lowers it to a generic `LD`. A generic load may target shared memory, so
ptxas must keep it ordered against every shared store and cannot batch these
itself — TornadoVM has to do it in the code generator, where the address space is
known.

**The open question for them:** emitting `__ldg()` or a global-qualified pointer
would hand ptxas the address space directly and fix the schedule at its source.

---

## If something misbehaves

- **`tornado --devices` finds nothing** — re-`source ~/nvidia-demo/env.sh`, check
  `nvidia-smi`.
- **A Java stack trace** — the ladder needs at least 2 executions (`medianOf`
  divides by `executions - 1`). The scripts pass 3 or 10.
- **Demo 3 finds no cubin** — `tornado.cuda.codecache.enable` must be on (it is by
  default); cache lives at `$TORNADOVM_HOME/var/cuda-codecache/device-0-0/`.
- **Fallback** — every number is committed under
  `results/raw/31-load-batching-reorder/`, so the talk survives a dead GPU.

## Backing material

- upstream PR: https://github.com/beehive-lab/TornadoVM/pull/1079
- evidence + raw CSVs: `results/raw/31-load-batching-reorder/`
- slides: `docs/slides/cuda-load-batching.md` (Marp → PDF/PPTX)
- structured numbers: `docs/findings/cuda-load-batching.yaml`
