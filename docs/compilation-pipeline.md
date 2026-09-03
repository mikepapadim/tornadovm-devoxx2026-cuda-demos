# The CUDA compilation pipeline, class by class

How a Java method becomes a running CUDA kernel in TornadoVM 6.0.0, with the
real class names, and where a second code emitter would have to plug in.

**Provenance.** Every class, method and string named here was verified against
the SDK this repo pins — `6.0.0-jdk22plus-cuda`, installed from SDKMAN — by
`javap` and by reading the class constant pools directly. It is not read from
an older source tree, and it is not read from upstream `develop`. Where a
behaviour could not be established from the artefacts, it is marked as such
rather than guessed.

## The path

```
Java bytecode
      │
      │  JVMCI
      ▼
  Graal IR ──────────────────────────────────────────────────────────┐
      │                                                              │
      │  TornadoVM's own tiers, replacing Graal's back end:          │
      │    CUDAHighTier → CUDAMidTier → CUDALowTier                  │
      │    + CUDA-specific phases (see below)                        │
      ▼                                                              │
  CUDA LIR         CUDALIRGenerator, CUDANodeLIRBuilder              │
      │            CUDALIRStmt.* statements                          │
      │                                                              │
      │  CUDABackend.emitCode(...) →                                 │
      │  CUDACompilationResultBuilder.emit(LIR)                      │
      ▼                                                              │
  CUDA C source    CUDAAssembler  ← string emission, not an IR ──────┘
      │            (inline PTX asm for MMA / cp.async)
      │
      │  NVRTC, at runtime, through the Java FFM API
      ▼
  cubin  ──────  nvrtcGetCUBIN, --gpu-architecture=sm_<cc>
      │
      │  (fallback) nvrtcGetPTX, --gpu-architecture=compute_<cc>
      ▼          then the driver JIT-compiles the PTX
  cuModuleLoadDataEx → CUDAKernel → cuLaunchKernel
```

`tornado --printKernel` prints the CUDA C at the third stage, so every claim
below can be checked in one command against any demo in this repo.

## Front end — Graal, with TornadoVM's tiers

`uk.ac.manchester.tornado.drivers.cuda.graal.compiler` holds the compiler
proper: `CUDACompiler`, `CUDACompilerConfiguration`, and the three tiers
`CUDAHighTier` / `CUDAMidTier` / `CUDALowTier`. Lowering runs through
`CUDALIRGenerationPhase` into `CUDALIRGenerator` and `CUDANodeLIRBuilder`,
with `CUDANodeMatchRules` for pattern-matched instruction selection.

The CUDA-specific phases in `...cuda.graal.phases` are where the backend's
opinions live:

| Phase | What it is for |
|---|---|
| `CUDATensorCoreSupportPhase` | recognises the MMA intrinsics and lowers them |
| `TornadoCUDAIntrinsicsReplacements` | maps `KernelContext` calls to CUDA nodes |
| `CUDAFP16SupportPhase`, `CUDAFP64SupportPhase` | half and double support |
| `CUDAFMAPhase` | multiply-add contraction |
| `TornadoParallelScheduler` | maps `@Parallel` loops onto the thread grid |
| `TornadoTaskSpecialisation` | **specialises on runtime argument values** |
| `TornadoAtomicsParametersPhase`, `TornadoAtomicsScheduling` | atomics |
| `TornadoFixedArrayCopyPhase`, `TornadoFloatingReadReplacement` | memory |
| `InfinityReplacementPhase`, `InverseSquareRootPhase` | math lowering |

`TornadoTaskSpecialisation` is the one that produces demo 15's compute-bound
win: because compilation happens after the task's arguments are bound, a loop
count that is a task argument becomes a compile-time constant and the FMA chain
unrolls. nvcc cannot do this ahead of time without a template parameter.

## Back end — string emission, and why that matters

`CUDABackend.emitCode(...)` drives `CUDACompilationResultBuilder.emit(LIR)`,
which walks blocks (`emitBlock`, `emitLoopBlock`, `emitRelocatedInstructions`)
and hands each LIR statement to `CUDAAssembler`
(`...cuda.graal.asm.CUDAAssembler`, with `CUDAAssemblerConstants`,
`CUDAVariablePrefix`, `CUDAConstantValue`).

**The output is CUDA C text.** Anything C cannot express is emitted as inline
PTX `asm volatile` inside that text. Two examples verified in the 6.0.0 jars:

- `CUDALIRStmt$MMAComputeStmt` carries the literal
  `asm volatile("mma.sync.aligned.`, completed by `MMAShape.getPtxName()`
  (`m16n8k16` / `m16n8k32`) and one of five operand suffixes in `MMAOperand`:
  `.row.col.f32.f16.f16.f32`, `.row.col.s32.s8.s8.s32`,
  `.row.col.f32.bf16.bf16.f32`, `.row.col.f32.e4m3.e4m3.f32`,
  `.row.col.f32.e5m2.e5m2.f32`.
- `CUDALIRStmt$CpAsyncCopyStmt` carries `cp.async.ca.shared.global`.

This is the single most consequential design fact in the backend. Instruction
selection for tensor cores and async copy is **string concatenation in a LIR
statement**, not a lowering pass over typed operations. It works — demo 08
proves it emits exactly one HMMA, by hardware counter — but it means:

- the set of reachable instructions is the set someone wrote a statement class
  for, which is why `MMAShape` has exactly two entries and stops at sm_89;
- there is no verification between "Graal IR said multiply-accumulate" and
  "this PTX string is the right instruction for the operand types";
- adding an instruction family means adding statements and enum entries, not
  teaching a lowering pass a new target.

## NVRTC — through the Java FFM API, not JNI

`uk.ac.manchester.tornado.drivers.cuda.ffm` reaches NVRTC and the driver API
through `java.lang.foreign` (Panama): `NVRTCAPI`, `CUDADriverAPI`, `NVTXAPI`,
plus `CUDAHandles` for typed pointers (`Program`, `Kernel`, `Context`, `Queue`,
`Event`, `Device`).

`NVRTCAPI` binds the real entry points with `SymbolLookup` and passes
`MemorySegment`s: `nvrtcCreateProgram`, `nvrtcCompileProgram`,
`nvrtcGetProgramLog`, `nvrtcGetPTX`, **`nvrtcGetCUBIN`**, `nvrtcVersion`,
`nvrtcGetSupportedArchs`. `toolkitRoots()` and `isAvailable()` discover the
installed toolkit at runtime.

`ffm.CUDACompiler` is the driver of that: `build(...)` → `compile(...)` →
`compileOnce(...)` returning an `NvrtcResult`, then `loadModule(...)`.

**Two output paths, and the distinction matters:**

| flag | NVRTC call | result |
|---|---|---|
| `--gpu-architecture=sm_<cc>` | `nvrtcGetCUBIN` | cubin directly — no separate `ptxas` step |
| `--gpu-architecture=compute_<cc>` | `nvrtcGetPTX` | PTX, JIT-compiled by the driver at load |

So the common path is **CUDA C → cubin in one NVRTC call**; PTX is the fallback,
with `warnAboutPtxFallbackOnce(...)` and `explainLoadFailure(...)` covering the
case where the installed toolkit cannot target the GPU. `CUDACompiler` also
carries a `HeaderMode` with `canCompileHeader(...)` / `probeCompile(...)` and
`--include-path=` options, i.e. it probes what the local NVRTC will accept
rather than assuming.

The class also carries a specific, well-written diagnostic for a real
compatibility trap: CUDA 13 removed `sm_50..sm_72`, so a new toolkit on an old
GPU fails, and it tells the user to install CUDA 12.8 or use Turing or later.

### A legacy seam worth knowing about

`CUDAProgram` — the class that delegates into `ffm.CUDACompiler` — exposes
`clBuildProgram`, `clReleaseProgram`, `clGetProgramInfo`. `CUDACodeCache` has
fields `OPENCL_CACHE_ENABLE`, `OPENCL_DUMP_BINS`, `OPENCL_DUMP_SOURCE`,
`OPENCL_CACHE_DIR` alongside `CUDA_CODE_CACHE_ENABLE`. The CUDA backend was
derived from the OpenCL one and the naming was never renamed through. Harmless,
but it will confuse anyone reading the code for the first time, and it is worth
saying out loud before someone else notices it.

## Runtime — dispatch

`CUDADeviceContext`, `CUDACommandQueue` / `CUDACommandQueueTable`,
`CUDAEventPool`, `CUDACodeCache` (a `ConcurrentHashMap<String,
CUDAInstalledCode>`), `CUDAKernel`, `CUDAGridInfo`, `CUDAStreamType`.

The per-execution cost of this layer is measured in
`results/raw/25-host-dispatch-breakdown/`: ~8.3 µs of CUDA driver API per
execution, of which a 24-byte kernel-argument stack frame is re-uploaded on
every launch and three `cuStreamSynchronize` are issued where one would do.
Both are diagnosed upstream in
[TornadoVM#1028](https://github.com/beehive-lab/TornadoVM/issues/1028) with an
open PR; 6.0.0 shipped without it.

## Where a second emitter would plug in

The question this document exists to answer: if the target were **cuTile**, or
NVVM IR via libNVVM, rather than CUDA C text — what changes?

The seam is narrow and well-defined, which is the good news:

1. **Everything above `CUDALIRGenerationPhase` is reusable as is.** Bytecode →
   Graal IR → parallelisation, memory-space assignment and task specialisation
   are target-independent. That is the majority of the work and none of it
   assumes C.

2. **`CUDAAssembler` plus the `CUDALIRStmt` family is the emitter**, and it is
   the thing that would be duplicated rather than modified. It is a string
   sink: `CUDABackend.createAssembler()` returns it, and
   `CUDACompilationResultBuilder.emit(LIR)` drives it. A second implementation
   emitting a different target from the same LIR is a contained change — the
   LIR is already typed and already carries the parallel structure.

3. **`ffm.CUDACompiler` is where the toolchain call would change** —
   `compileOnce(...)` currently calls `nvrtcCompileProgram` and takes
   `nvrtcGetCUBIN`. A different lowering path would substitute its own
   compiler invocation and return the same `NvrtcResult`-shaped artefact to
   `loadModule(...)`, which already handles both cubin and PTX.

4. **`MMAShape` and `MMAOperand` would stop being the ceiling.** Today the
   reachable tensor-core instruction set is bounded by two enums. A tile-level
   IR would move that decision out of TornadoVM entirely, which is the main
   argument for doing it at all: `wgmma` and `tcgen05` are not reachable by
   adding enum entries, because the fragment and scheduling model differs.

The honest caveat: what is *not* established here is how much of the parallel
structure survives the LIR. The LIR carries the thread-index arithmetic already
lowered to scalar operations, so a tile-level target would likely want to
intercept **higher** — at or above `CUDALowTier` — rather than at the assembler.
Determining that is exactly the design-review conversation this repo's brief
asks for, and it is not answerable from the artefacts alone.

## See also

- [`NVIDIA-BRIEF.md`](NVIDIA-BRIEF.md) — the measurements and the asks
- [`claims.md`](claims.md) — every claim in this repo mapped to its evidence
- demo 08 (`ctx.mma`) and demo 14 (`cp.async`) for the inline-PTX paths
  end to end, with `--printKernel` output in each README
