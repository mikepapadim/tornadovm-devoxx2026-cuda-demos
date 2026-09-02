# Talk 1 — TornadoVM Hybrid API: Java + NVIDIA CUDA Libraries

Draft content. Every technical claim below carries an evidence tag
(**Observed** / **Source-backed** / **Documented** / **Hypothesis** /
**Blocked**) per `PLAN.md` §6 — cross-reference `docs/claims.md` for the full
evidence map. Live sequence and exact commands: `docs/demo-runbook.md`.
System under test: TornadoVM `6.0.0-jdk22plus-cuda` (SDKMAN release),
JDK 25.0.2, CUDA backend only, one RTX 4090. Numbers re-measured on 6.0.0 are
in `README.md` and `results/raw/18-tornadovm-6-migration/`; where this draft
still quotes a figure from the earlier 5.2.1 source-built pin
(`99549c9862eda8d584e35e99924f9c865501eb3a`) it says so.

## Opening (2–3 min)

Java developers who want GPU acceleration today usually face a binary
choice: hand-write JNI bindings to a vendor library, or give up and stay on
the CPU. TornadoVM's Hybrid API removes that binary choice — a single
`TaskGraph` can mix JIT-compiled Java kernels (`@Parallel` methods TornadoVM
compiles to real CUDA at runtime) with calls into vendor-optimized native
libraries (cuBLAS, cuFFT, cuDNN, cuSPARSE, CUTLASS), sharing GPU-resident
buffers between them with no host round-trip — all from plain Java, on
whatever JDK you're already running.

**Live, first thing:** `demos/00-hello-gpu` with `--enableProfiler console`.
One `@Parallel` method, no JNI, and a JSON block that says `"BACKEND": "CUDA"`.
That's the entire promise of the talk in one command — everything after this
is "how far does that promise go."

## Narrative arc

1. **A Java kernel is a real CUDA kernel.** `--printKernel` shows the literal
   generated `extern "C" __global__ void` source (**Observed**,
   `demos/01-first-cuda-kernel`). Not a black box, not a DSL that "sort of"
   maps to CUDA — inspectable generated code.

2. **CUDA-runtime behaviour is exposed on `TornadoExecutionPlan`, not as raw
   bindings.** There is no `cudaMalloc`/`cudaMemcpy`-shaped API for user code
   (**Source-backed**, `docs/hybrid-api-inventory.md` §6). Instead:
   `withCUDAGraph()` (graph capture/replay), `withIntraPlanConcurrency()`
   (multi-stream execution), `withStagedTransfers()` (pinned-memory
   staging) — all three explicitly CUDA-backend-only per the API's own
   Javadoc, no-op on other backends. This is a deliberate design choice:
   expose CUDA's real mechanisms as typed, composable execution-plan
   options instead of a leaky low-level binding.

3. **Vendor libraries ride the same `TaskGraph`.** `libraryTask()` sits
   alongside `.task()` on the same builder — a cuBLAS call and a hand-written
   Java kernel are structurally the same kind of graph node
   (**Source-backed + Observed**, `TaskGraph.java:737-895`,
   `docs/hybrid-api-inventory.md` §1). Six providers exist in the pinned
   tree — cuBLAS, cuBLASLt, cuFFT, cuDNN, cuSPARSE, CUTLASS — and 71/71
   upstream provider unit tests pass on this GPU (**Observed**).
   **Live:** `demos/04-cublas-hybrid` — JIT `scale` → cuBLAS `sgemv` → JIT
   `bias`, one graph, shared buffers, `--enableProfiler console` shows every
   stage on the same device. Then `demos/05-cufft-hybrid` — a 4-stage
   forward-FFT → JIT filter → inverse-FFT → JIT normalize pipeline, same
   pattern, a different library.

4. **The CUDA-specific mechanisms are real, not marketing.**
   - Graph capture/replay: `demos/02-cuda-runtime-api` shows correctness
     (8 replays, mutated inputs, each checked against a CPU reference).
     `demos/07-cuda-graph-benefit` then *quantifies* it — same 6-stage graph,
     `nograph` vs. `graph`, steady-state speedup 6.47x–7.02x across three
     independent runs (**Observed, this-run/this-workload only** — say that
     out loud, don't generalize it).
   - Multi-stream concurrency: `demos/06-cuda-streams`, Nsight Systems
     timeline evidence of genuinely overlapping kernel windows across a
     4-stream pool vs. one sequential stream (**Observed**). Honest caveat,
     stated in the talk, not hidden: concurrency is not a universal win —
     the same mechanism inside `demos/11-integrated-showcase` gave
     0.78x–1.21x on a smaller, launch-overhead-bound workload. The lesson is
     "measure your own workload," not "TornadoVM streams are always faster."
   - Combining both: `withCUDAGraph()` + `withIntraPlanConcurrency()` on one
     plan is **not documented or tested anywhere upstream** — this repo
     tried it (`demos/11-integrated-showcase`, `combined` mode) and it works
     correctly, but didn't beat `graph` alone at this workload size. Present
     this as original probing with an honest negative result, not a
     discovered feature to oversell.

5. **Tensor Cores from Java.** `KernelContext` exposes `mmaFragment`/`mma`/
   `mmaStore` intrinsics that lower to real inline `mma.sync.aligned` PTX
   (**Observed**, `demos/08-tensor-core-mma`, generated-code comparison
   against a zero-`mma.sync` scalar kernel from the same compile). No
   hardware-counter (occupancy, tensor-pipe activity %) number is available
   on this presentation machine — **Blocked**, `ERR_NVGPUCTRPERM`, a driver
   permission default, not a TornadoVM limitation
   (`results/failures/08-nsight-compute-permission.md`). Say this plainly if
   asked; the generated-code evidence stands on its own.

## Live-coding sequence

See `docs/demo-runbook.md` "Talk 1" for the exact commands, expected output,
and per-step fallback. Order: 00 → 01 → 02 → 04 → 05 → 06 → 07 → 08 → 11.
Cut 02 and 05 first if short on time; never cut 00 (opening) or 11 (closer).

## Technical explanation (for the "how does this actually work" segment)

- One native library handle/context per `(device, execution plan)`, created
  by `TornadoLibraryProvider.createContext()`, bound to that plan's CUDA
  stream (**Source-backed**, `docs/hybrid-api-inventory.md` §5).
- `prepare()` runs before every launch region *and* before CUDA-graph
  capture starts, specifically so a library can pre-allocate
  capture-unsafe memory (cuFFT plans, cuDNN workspaces) outside the
  captured region (**Source-backed**, `TornadoLibraryProvider` Javadoc;
  **Observed** working via passing `testXxxWithCudaGraph` suites for all 5
  capturable providers).
- `Access[]` (`READ_ONLY`/`WRITE_ONLY`/`READ_WRITE`) on library-task
  parameters drives the same data-transfer/consistency tracking as JIT task
  parameters — one consistency model for both kinds of graph node
  (**Source-backed**, usage pattern in `CuBlas.java`/`CuFft.java`).

## Conclusion

The Hybrid API's real contribution isn't "Java can call cuBLAS" (JNI could
always do that) — it's that a hand-written kernel and a vendor library call
become the *same kind of thing* inside one execution graph: same buffer
model, same consistency tracking, same CUDA-graph capturability, same
profiler output. That composability is what turns "GPU acceleration" from an
all-or-nothing rewrite into an incremental one, without leaving the JDK.

## What to do if a demo fails live

`docs/demo-runbook.md` has a named pre-captured log for every single command
in this talk. Recovery rule: check `nvidia-smi` and `tornado --devices`
first (broken environment vs. broken demo), then fall back to the log and
keep narrating — never invent a number that wasn't actually captured.
