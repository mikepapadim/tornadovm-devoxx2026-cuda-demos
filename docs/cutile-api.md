# The TornadoVM CUDA Tile (cuTile) API

Everything below was read from the pinned TornadoVM source or measured on this machine.
Nothing here is inferred from documentation alone.

```text
TORNADO_SHA = 8d592d6fbaafe42d66e1e22c27057cb1cdcf9097   (beehive-lab/TornadoVM, develop)
Checkout    = vendor/tornadovm   (gitignored; profile `develop` in env/sdk/)
Supersedes  = the CUTILE_* feat/cutile pin in env/versions.env -- PR #1083 is merged,
              so the tile demos no longer need a feature branch, JDK 21 or --enable-preview
Measured on = RTX 5070 Ti (sm_120), driver 580.142, CUDA 13.4.92 userspace toolkit
```

> **It is not called "CuTile" in Java.** Grepping the tree for a `CuTile` class finds nothing.
> The public API is `uk.ac.manchester.tornado.api.tile.TileContext`; the literal string
> `cutile` survives only in a launch-hint attribute, a temp-directory prefix and branch names.

## What it is

A tile kernel describes what **one tile block** does over tiles of a tensor. The tile compiler
decides how many threads back it, which tensor-core instruction to issue, and how tiles move
through shared memory. Nothing in the API names a thread, a warp or a fragment.

A task becomes a tile task when its kernel method takes a `TileContext` as its **first
parameter** — the same dispatch rule by which a leading `KernelContext` selects the
kernel-parallel API. The path is CUDA-only and additive: a tile task shares a `TaskGraph`,
a stream and its device buffers with ordinary JIT tasks and with native library tasks.

## Public API surface

`vendor/tornadovm/tornado-api/src/main/java/uk/ac/manchester/tornado/api/tile/`

| Class | Lines | Role |
| --- | --- | --- |
| `TileContext.java` | 1328 | the entry point; ~150 methods, each with a JVM fallback so a tile kernel also runs as plain Java |
| `PartitionView.java` | 434 | a tiled view of a buffer: loads, stores, masked forms, atomics |
| `TensorView.java` | 177 | wraps `ct::tensor_span` |
| `Tile.java` | 97 | a value tile |
| `DType.java` | 166 | the element types |

Plus `uk.ac.manchester.tornado.api.exceptions.TornadoDeviceTileNotSupported`.

### `DType` (`DType.java:31-47`)

| Constant | C++ type | Bytes |
| --- | --- | --- |
| `F16` | `__half` | 2 |
| `BF16` | `__nv_bfloat16` | 2 |
| `F32` | `float` | 4 |
| `F64` | `double` | 8 |
| `TF32` | `__nv_tf32` | 4 |
| `FP8_E4M3` | `__nv_fp8_e4m3` | 1 |
| `FP8_E5M2` | `__nv_fp8_e5m2` | 1 |
| `S8` | `signed char` | 1 |
| `S32` | `int` | 4 |
| `PRED` | `bool` | 1 |

### `TileContext`, by group (file:line from `TileContext.java`)

| Group | Methods |
| --- | --- |
| Block index | `bidX`/`bidY`/`bidZ` (`:89`,`:93`,`:97`), `numBlocksX/Y/Z` (`:104`-`:112`), `setBlockIndex` (`:125`), `setBlockCount` (`:135`) |
| Views | `view(...)` at rank 1/2/3 for `FloatArray`, `HalfFloatArray`, `BFloat16Array`, `IntArray`, `Int8Array`, `ByteArray`, `DoubleArray` (`:145`-`:268`); `view(FP8Array, DType, ...)` (`:228`,`:232`) |
| Strided views | `viewStrided(...)` (`:290`-`:348`), incl. `viewStrided(ByteArray, DType, ...)` (`:336`) for a raw buffer at a chosen element type |
| Partition | `partition(view, tileExtent)` (`:363`), `(view, tileRows, tileColumns)` (`:368`), rank-3 (`:374`) |
| Creation | `zeros` (`:385`,`:390`), `full` (`:396`,`:402`), `ones` (`:409`,`:413`), `iota` (`:417`,`:430`) |
| Elementwise | `add` (`:442`), `sub`, `mul`, `div`, `maximum`, `minimum`, `scale` (`:601`), `fma` (`:822`) |
| Predicates | `lessThan`/`lessOrEqual`/`greaterThan`/`greaterOrEqual`/`equalTo`/`notEqualTo` (`:495`-`:540`, tile-tile and tile-scalar), `logicalAnd`/`logicalOr`/`logicalNot`, `select` (`:561`) |
| Matmul | `mma(a, b, acc)` (`:625`), `matmul(a, b)` (`:673`) |
| Reductions | `sum` (`:709`), `max`, `min`, `prod`, `allOf`, `anyOf`, `reduceBitAnd/Or/Xor` (`:770`-`:813`) — all `(Tile, int axis)` |
| Scans | `prefixSum` (`:1132`), `prefixProduct` (`:1137`) |
| Math | `exp`, `sin`, `cos`, `sqrt`, `log`, `log2`, `exp2`, `tanh`, `abs`, `ceil`, `floor`, `tan`, `sinh`, `cosh`, `rsqrt`, `atan2`, `pow`, `remainder`, `floorDiv`, `ceilDiv`, `mulhi`, `isNaN`, `isInfinite` (`:827`-`:994`) |
| Bitwise | `bitwiseAnd/Or/Xor/Not`, `shiftLeft`, `shiftRight` (`:963`-`:986`) |
| Shape/type | `broadcast` (`:1012`,`:1016`), `reshape` (`:1049`-`:1057`), `extract` (`:1088`,`:1092`), `bitcast` (`:1170`), `concat` (`:1192`), `transpose` (`:1217`), `cast` (`:1237`) |

### `PartitionView` (`PartitionView.java`)

`getDType` (`:48`), `getRank` (`:52`), `getTileDimension` (`:56`), `getBlockCount` (`:63`);
`load`/`loadMasked` and `store`/`storeMasked` at rank 1, 2 and 3 (`:67`-`:151`); atomics
`atomicAdd/Sub/Min/Max/And/Or/Xor/Exchange/Load/Store` at rank 1 and 2 (`:285`-`:384`).

## The three rules

1. **Tile shapes are compile-time constants and powers of two.** A shape is part of the
   kernel's type, so a differently shaped variant is a differently compiled kernel. Passing a
   shape as a method parameter is rejected at sketch time with a message naming the argument.
   Write a literal or a `static final int` — a constant passed *into a helper* does not count,
   because inlining happens after parsing.
2. **Extents are not constrained.** Views take runtime extents, so problem size does not
   specialise the kernel; only the tile shape does.
3. **The worker grid counts tile blocks**, and the block is pinned to `1x1x1` by
   `CUDATileScheduler`. Do not set local work yourself.

On rule 3, the scheduler is explicit about why
(`tornado-drivers/cuda/.../scheduler/CUDATileScheduler.java:68-86`): a grid with no explicit
local work reaches the launch path as `null`, picks up `DEFAULT_BLOCK_SIZE`, and the kernel
"would run with a 256-thread block and silently compute the wrong answer" — so `submit`
forces `1,1,1`, and `calculateGlobalWork` treats the domain cardinality as a **block** count.

## Backend lowering

All under `vendor/tornadovm/tornado-drivers/cuda/src/main/java/uk/ac/manchester/tornado/drivers/cuda/`:

| File | Role |
| --- | --- |
| `graal/phases/CUDATileSupportPhase.java` | capability gate; rejects non-constant tile shapes at sketch time; rejects an unconsumed `view` |
| `graal/compiler/plugins/CUDATileGraphBuilderPlugins.java` | intrinsifies every `TileContext`/`PartitionView` method |
| `graal/lir/CUDATileStmt.java` | emits the CUDA Tile C++ — `ct::partition_view`, `ct::tensor_span`, `ct::layout_strided_mapping`, `ct::full`, `ct::iota`, `ct::mma`, `ct::matmul`, `ct::element_cast`, `ct::cat`, `ct::bid()`, `ct::num_blocks()`, `ct::shape<…>`, `ct::tile<…>` |
| `graal/nodes/` | 19 `CUDATile*Node` classes |
| `graal/backend/CUDAPreamble.java:86-89` | `#include "cuda_tile.h"`, `namespace ct = cuda::tiles;`, `using namespace ct::literals;` |
| `graal/asm/CUDAAssemblerConstants.java:34` | kernel modifier `extern "C" __tile_global__` |
| `scheduler/CUDATileScheduler.java` | pins the block, counts tile blocks |
| `CUDATileCompiler.java` | drives **nvcc** to a cubin |

### The compile path is nvcc, not NVRTC

`CUDATileCompiler` detects a tile kernel by the presence of `__tile_global__` (`:80`) and
shells out (`:148-155`):

```
nvcc -tilecubin --tile-only -std=c++20 -arch=sm_<cc> [extra] -o <out>.cubin <in>.cu
```

in a `tornado-cutile-` temp directory, with a compile timeout. `-tilecubin` emits a cubin;
`--tile-only` skips SIMT codegen, which the generated translation unit has none of.

**`nvcc` spawns `tileiras` by bare name**, so `tileiras` must be on `PATH` or the compile
fails late with `sh: 1: tileiras: not found`. `scripts/setup-env.sh` handles this.

## Requirements, and what happens below them

| | |
| --- | --- |
| CUDA Toolkit | **13.3+** — `CUDATileCompiler.MINIMUM_TOOLKIT = 13003` (`:59`) |
| Driver | **R580+** to load a CUDA 13 cubin |
| GPU | compute capability **8.0+** |

The gate is on **nvcc**, deliberately not on NVRTC: the javadoc at `:107-112` notes that "a
system CUDA 12.6 with a userspace 13.3 nvcc can compile tile kernels perfectly well", so
gating on NVRTC "asks the wrong question". `locateNvcc` (`:205-243`) honours
`-Dtornado.cuda.nvcc`, then `$CUDA_PATH/bin/nvcc`, then **`site-packages/nvidia/cu13/bin/nvcc`**
(the pip-wheel location, searched on purpose), then `/usr/local/cuda/bin/nvcc`, then bare
`nvcc` — accepting the first whose version is ≥ 13.3.

Below any requirement the task throws `TornadoDeviceTileNotSupported` naming the missing one.

### A userspace toolkit is enough

This machine's system toolkits are CUDA **12.0** (`/usr/bin/nvcc`, first on `PATH`) and
**13.0.88** (`/usr/local/cuda`) — both below the gate. What unblocked it, with no root:

```bash
pip install --user nvidia-cuda-nvcc 'cuda-tile[tileiras]'
```

That lands `nvcc` **13.4.92** and `tileiras` at exactly the path `locateNvcc` probes, and
brings the real 5,493-line `cuda::tiles` header. (The stub at
`/usr/local/cuda-13.0/targets/x86_64-linux/include/crt/cuda_tile.h` is 50 lines and declares
only `cuda::cutile::print` — CUDA 13.0 cannot compile tile code.)

## The silent-host-fallback hazard

`TornadoOptions.RECOVER_BAILOUT` defaults to **true**, and every `TileContext` method has a
JVM implementation. A tile task that fails to compile therefore **falls back to the host,
computes the correct answer, and prints whatever verdict the program prints** — with the GPU
untouched.

Any demo, test or benchmark that claims to exercise cuTile must run with:

```
-Dtornado.recover.bailout=False
```

A `correct` verdict is not, by itself, evidence that cuTile executed. Demand the codegen
(`--printKernel` showing `__tile_global__` and `ct::…`) or the cubin's SASS.

## Launch hints

`-Dtornado.cuda.tile.hints=occupancy=2,num_cta_in_cga=1` puts
`[[ using cutile : hint(0, …) ]]` on every generated tile kernel
(`graal/backend/CUDABackend.java:245-280`). **The keys are not validated** — an invented key
compiles silently and does nothing.

## Upstream tests and examples

- **Tests**: `tornado-unittests/src/main/java/uk/ac/manchester/tornado/unittests/tile/` —
  21 files. `TileSupport.java` is the capability probe; it reports `[UNSUPPORTED]` rather than
  failing, and its javadoc (`:34-46`) warns that **repeating a failed tile launch has brought
  the JVM down with a SIGSEGV** — probe once.
- **Examples**: `tornado-examples/src/main/java/uk/ac/manchester/tornado/examples/tile/` —
  `TileVectorAdd`, `TileMatrixMultiply`, `TileSoftmax`, `TileAttention`, `TileTransformerBlock`,
  `TileQuantizedProjection`, `TileQwenNormKernels`, `TileExamples`.
- **Docs**: `docs/source/tile-api.rst` (771 lines).

## Measured on sm_120 — 2026-09-18

Full tile unit-test sweep against this pin. Evidence:
[`results/raw/35-develop-cutile-baseline/`](../results/raw/35-develop-cutile-baseline/).

**212 of 219 pass. All 7 failures are FP8 arithmetic**, in `TestTileOpLevel`:
`testFp8Add`, `testFp8Sub`, `testFp8Mul`, `testFp8Div`, `testFp8Maximum`, `testFp8Exp`,
`testFp8Sum`. Every other tile test class is clean:

| Class | Ran | Failed | Class | Ran | Failed |
| --- | --- | --- | --- | --- | --- |
| `TestTileOpLevel` | 106 | **7** | `TestTileRowKernels` | 20 | 0 |
| `TestTileMatmul` | 7 | 0 | `TestTileMasking` | 10 | 0 |
| `TestTileGemmVariants` | 9 | 0 | `TestTileElementwise` | 8 | 0 |
| `TestTileAttention` | 7 | 0 | `TestTileChaining` | 8 | 0 |
| `TestTileAttentionVariants` | 8 | 0 | `TestTileDTypes` | 8 | 0 |
| `TestTileAtomics` | 6 | 0 | `TestTileMathOps` | 6 | 0 |
| `TestTileMixedPipelines` | 5 | 0 | `TestTileRank3` | 4 | 0 |
| `TestTileLlmKernels` | 4 | 0 | `TestTileShapes` | 4 | 0 |
| `TestTileRecurrence` | 3 | 0 | `TestTileStridedViews` | 3 | 0 |
| `TestTileRawBuffers` | 2 | 0 | | | |

### Root cause of the FP8 failures

Not the documented "fp8 below CC 9.0" limitation — this GPU is CC 12.0. **CUDA Tile C++ 13.4
defines no arithmetic operators for FP8 tile element types**, and `CUDATileStmt` emits the
operator form directly. Reduced to pure C++
([`results/raw/35-develop-cutile-baseline/fp8-probe/`](../results/raw/35-develop-cutile-baseline/fp8-probe/)):

```cpp
cv.store(av.load(ct::bid().x) + bv.load(ct::bid().x), ct::bid().x);   // __nv_fp8_e4m3
```
```
error: no operator "+" matches these operands
note: concept "cuda::tiles::__1::tile_like" not satisfied for "<<error-type>>"
```

Converting for the arithmetic compiles cleanly:

```cpp
auto x = ct::element_cast<float>(av.load(ct::bid().x));
auto y = ct::element_cast<float>(bv.load(ct::bid().x));
cv.store(ct::element_cast<__nv_fp8_e4m3>(x + y), ct::bid().x);        // exit 0
```

So there are two defensible fixes upstream: emit the `element_cast` round-trip for FP8
elementwise/reduction ops, or gate them in `CUDATileSupportPhase` so they report
`[UNSUPPORTED]` instead of failing. Either way the tests currently assert behaviour the
toolchain does not provide.

## Documented limitations (from `tile-api.rst:587-630`)

Against CUDA 13.3's 84 `__tile_builtin__` operations, the API covers **71**. The 13 not
covered: the ten `_masked` atomic variants, `atomic_compare_exchange` in both forms, and
`permute`. Beyond that:

- no `layout_right_padded`, no non-unit column stride, no strided rank-3 view;
- fp8 tiles are rejected below CC 9.0 by `tileiras` (`unsupported type 'f8E4M3FN'`);
- fp64 `mma` is correct but lowers to scalar `DADD`/`DMUL`, not `DMMA`, on Ada;
- `DType.TF32` has no buffer type — a tf32 operand must come from a cast;
- `view`/`partition`/loads/stores/`reshape` reach rank 3; tile creation, atomics, `broadcast`,
  `extract`, `mma`, the reductions and `transpose` stop at rank 2;
- `withBatch` is rejected for tile tasks, `@Reduce` cannot share the graph, and a reflectively
  resolved tile kernel fails.

## Demos in this repository

| Demo | Concept | Tile API it exercises |
| --- | --- | --- |
| [19-cutile-matmul](../demos/19-cutile-matmul/) | the same GEMM with threads, then with tiles | `view`, `partition`, `zeros`, `load`, `store`, `mma` |
| [20-cutile-hybrid](../demos/20-cutile-hybrid/) | a JIT kernel, a tile kernel and cuBLAS in one graph | as 19, inside a mixed `TaskGraph` |
| [21-cutile-flash-attention](../demos/21-cutile-flash-attention/) | flash attention through CUDA Tile | `+ exp`, `max`, `maximum`, `sub`, `div`, `mul`, `scale`, `sum`, `cast`, `transpose`, `full` |
| [22-matmul-ladder-fp16-tile](../demos/22-matmul-ladder-fp16-tile/) | the FP16 ladder with a tile rung, measured | as 19, next to hand-written `mma.sync` |
| **[23-cutile-row-scan](../demos/23-cutile-row-scan/)** | per-row prefix sum over a **ragged** extent | **`prefixSum`**, **`loadMasked`/`storeMasked`**, loop-carried `[1,1]` carry, implicit broadcast |
| **[24-cutile-histogram](../demos/24-cutile-histogram/)** | value histogram, every block into every bin | **`atomicAdd`**, **`select`**, **`greaterOrEqual`/`lessThan`/`logicalAnd`**, `full`, `sum` |

Demos 23 and 24 were chosen against that table: before them, nothing in the repo used a
scan, a masked load or store, a loop-carried reduction, any predicate, or any of the ten
atomic operations.
