# Open questions for NVIDIA

Grouped by what each answer would unblock. Each is stated with the fact from
this host that motivates it, so the question can be answered precisely.

Host facts referenced throughout: RTX 4090 (sm_89), driver 565.57.01, CUDA/NVRTC
12.6.85. Inventory: `tile-feasibility/inventory.txt`.

## 1. Runtime compilation path

TornadoVM compiles per task graph, at execution time, from Java bytecode through
Graal IR to **CUDA C source**, then NVRTC. Anything C cannot express is emitted
as inline PTX `asm volatile` in that generated C (verified: MMA and `cp.async`).

NVRTC 12.6 on this host exports `nvrtcGetCUBIN`, `nvrtcGetPTX`,
**`nvrtcGetNVVM`** and **`nvrtcGetLTOIR`**.

- Given a JIT that already has a typed IR (Graal), is emitting **C text** the
  path you would recommend, or is **NVVM IR via libNVVM** the better ingestion
  point for a compiler front end?
- `nvrtcGetNVVM` returns NVVM IR *from* NVRTC. Is there a supported path to feed
  NVVM IR *into* the toolchain at runtime with comparable guarantees, and is
  libNVVM's IR version contract stable enough for a JIT to target across toolkit
  releases?
- What is the intended long-term status of `nvrtcGetLTOIR` for a JIT that wants
  link-time optimisation across a task graph's kernels?

## 2. CUDA Tile — C++ surface versus Tile IR

**On this host there is no CUDA Tile support of any kind**: zero toolkit headers,
zero libraries, zero `nvcc` flags and zero `libnvrtc` symbols matching *tile*
(counts recorded in the inventory). This is a statement about CUDA 12.6.85 on
this machine only.

- Which toolkit release first ships a usable CUDA Tile path on Linux x86_64, and
  what exactly does it install (headers, library, compiler entry point)?
- Is the intended integration surface for a **non-C++ front end** the Tile C++
  templates, or a **Tile IR** that a compiler can emit directly? A JIT with its
  own IR would rather target the latter; going through C++ templates means text
  generation and a full C++ parse per compilation.
- Is Tile reachable through a **runtime compilation** API at all, or is it
  AOT-only today? TornadoVM has no AOT step — everything is compiled after
  argument values are known, which is where its measured compute-bound advantage
  comes from (see `summary.md`, task C).
- What are the tile shape/dtype constraints, and how do they map onto the
  fragment-level model we currently emit (`M16N8K16`, `M16N8K32`)?

## 3. Launch and execution semantics

Measured on this host: per-execution CUDA driver API overhead of ~8.3 µs for a
one-kernel task graph, of which a 24-byte kernel-argument stack frame is
re-uploaded on **every** launch and three `cuStreamSynchronize` are issued where
one would do (`summary.md`, task E).

- For a runtime that dispatches many short kernels, what is the recommended
  modern launch path — `cuLaunchKernel`, graphs, or something else?
- Does a Tile-compiled kernel launch through the same driver entry points and
  the same argument-passing mechanism, or does it impose its own launch
  semantics that a runtime would have to special-case?
- Are CUDA Graphs the intended answer to per-dispatch overhead at this scale, and
  are there constraints we should expect when capturing graphs that contain both
  JIT-generated and vendor-library kernels?

## 4. Hopper and Blackwell

Three independent ceilings on this host, all verified:

| Layer | Ceiling | Source |
|---|---|---|
| GPU | sm_89 | `nvidia-smi` compute cap 8.9 |
| NVRTC 12.6 toolkit | **sm_90** | `nvrtcGetSupportedArchs` — 14 archs, sm_50…sm_90 |
| TornadoVM MMA emitter | sm_89-class | `MMAShape` = {M16N8K16, M16N8K32}, 5 operand suffixes, all `.row.col` |

- `wgmma` (sm_90) and `tcgen05` (sm_100) are not reachable by adding entries to
  that enum — the fragment and scheduling models differ. For a JIT emitting at
  fragment granularity today, what is the recommended migration target: keep
  emitting `mma.sync` for older arches and add a separate path, or move the whole
  MMA surface to a tile-level abstraction?
- Is there a supported way to **query** an architecture's available MMA
  shapes/dtypes at runtime, rather than hard-coding a table per arch?
- Hardware access: sm_90 is within this toolkit's reach but not this GPU's. What
  is the realistic route to Hopper/Blackwell access for prototyping — DGX Cloud
  credits, a loaner, or a remote CI target?

## 5. Recommended counters and methodology

We currently use `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_{ld,st}.ratio`
for coalescing, `sm__inst_executed_pipe_tensor_op_{hmma,imma}.sum` for tensor
pipes, `dram__throughput.avg.pct_of_peak_sustained_elapsed` for bandwidth
saturation, and execution-count differencing under `nsys` for host cost.

- For judging **generated code quality** specifically, which counters would you
  put first, and which of the ones above would you consider misleading?
- ~~Is there a supported way to obtain SASS for an NVRTC-compiled cubin held in
  process memory?~~ **Withdrawn.** TornadoVM already writes every compiled cubin
  to an on-disk module cache by default (`tornado.cuda.codecache.enable=True`),
  so `cuobjdump -sass` works directly and task C now has SASS from both sides.
  The question rested on a false premise on our side.
- For tensor-core work, is `sm__pipe_tensor_cycles_active` the right utilisation
  proxy at small tile counts, or does it mislead below a certain occupancy?

## 6. Collaboration scope

- Would a **GEMM + fused epilogue** kernel class be a reasonable first scoped
  prototype? It is already exercised here through CUTLASS as a library task.
- What would NVIDIA want to see from a prototype before it is considered worth
  supporting — correctness parity, a specific kernel class, or coverage across
  arches?

## Explicitly not claimed

No CUDA Tile performance claim is made anywhere in this bundle. Task G is a host
feasibility inventory. No Tile code was written, compiled or run.
