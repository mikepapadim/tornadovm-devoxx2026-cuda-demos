# NVIDIA demo — three commands

Everything is self-contained in `~/nvidia-demo`. Nothing depends on `/tmp`, and the
patched SDK is a copy, so a reboot or a tmp sweep will not break it.

## Setup — once, before you start

```bash
source ~/nvidia-demo/env.sh
```

Sets `JAVA_HOME`, points `TORNADOVM_HOME` at the patched SDK, and drops you in
`~/nvidia-demo/classes` where the demo classes already are. Both sides are
pre-compiled — no build step in front of the audience.

Sanity check: `tornado --devices` should print the RTX 4090.

---

## 1. The result  (~2 min)

```bash
bash $DEMO/1-ladder.sh
```

Six implementations of the same 2048x2048 sgemm. **Kernel time via nsys**, three
arms: TornadoVM with the fix off, with it on, and hand-written CUDA.

```
rung                  TornadoVM before  TornadoVM after  hand-written  after vs CUDA
3. register-tiled             694.2 us         499.2 us      486.0 us         1.027x
```

**Say:** rungs 1 and 2 are already at 1.005x and 1.001x of hand-written CUDA — the
generated arithmetic was never the problem. Rung 3, the one doing register
blocking, was 1.43x off. It is now 1.027x.

**Do not quote the wall clock the demo prints.** TornadoVM's includes host dispatch
and three 16 MB transfers; the CUDA binary's does not. That comparison is
apples-to-oranges and this script deliberately does not make it.

## 2. What changed  (~1 min)

```bash
bash $DEMO/2-printkernel.sh
```

The emitted CUDA C for the staging loop, before and after.

```
BEFORE:  f_108 = *((float *) ul_107);   <-- global load
         adf_5[i_97] = f_108;           <-- shared STORE      one load in flight
AFTER:   eight loads, then eight stores
```

**Say:** same 16 statements both sides, 14 in a different position. Nothing added,
removed or rewritten — only reordered.

## 3. Why ptxas could not do it  (~1 min)

```bash
bash $DEMO/3-sass.sh
```

```
TornadoVM, batching off  LSLSLSLSLSLSLSLS  generic LD
TornadoVM, batching on   LLLLLLLLSSSSSSSS  generic LD
hand-written CUDA        LLLLLLLLSSSSSSSS  LDG
```

**Say:** both TornadoVM rows use the *same* instruction and differ only in
schedule. The hand-written row uses a *different* instruction, and that is the
cause — a generic `LD` may target shared memory, so ptxas must keep it ordered
against every shared store. `LDG` is proven global, so ptxas batches those itself.

This is the slide for a compiler audience: TornadoVM knows the address space at LIR
level even though its emitted C does not encode it, which is why the fix belongs in
the code generator. The follow-up they will ask about is emitting `__ldg()` or a
global-qualified pointer, which would fix the schedule at its source.

---

## If something misbehaves

- **`tornado --devices` finds nothing** — re-`source ~/nvidia-demo/env.sh`; check
  `nvidia-smi`.
- **A script prints a Java stack trace** — the ladder needs at least 2 executions
  (`medianOf` divides by `executions - 1`). The scripts already pass 3 or 10.
- **Demo 3 says no cubin was cached** — `tornado.cuda.codecache.enable` must be on;
  it is by default. The cache lives at
  `$TORNADOVM_HOME/var/cuda-codecache/device-0-0/`.
- **Fallback**: every number above is committed in the demos repo under
  `results/raw/31-load-batching-reorder/` with the raw CSVs, so the talk survives a
  dead GPU.

## Backing material

- upstream PR: https://github.com/beehive-lab/TornadoVM/pull/1079
- evidence + raw CSVs: `results/raw/31-load-batching-reorder/`
- slides: `docs/slides/cuda-load-batching.md` (Marp, renders to PDF/PPTX)
- structured numbers: `docs/findings/cuda-load-batching.yaml`
