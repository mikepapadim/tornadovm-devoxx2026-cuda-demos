#!/usr/bin/env bash
# Compile and run the hand-written CUDA equivalent of every Track A demo.
#
#   bash scripts/run-all-cuda.sh [output-dir]
#
# Needs only the CUDA toolkit (nvcc, cuBLAS, cuFFT, cuDNN) -- no JDK, no
# TornadoVM. Exits non-zero if any .cu fails to compile or run.
#
# Demo 12 additionally needs CUTLASS, which is header-only and not vendored
# here. Fetch it once and point CUTLASS_DIR at it, or that demo is skipped:
#   git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git
#   export CUTLASS_DIR=$PWD/cutlass
#
# Demos 19-24 are CUDA Tile kernels and need toolkit 13.3+ with tileiras on PATH.
# A userspace install is enough and needs no root:
#   pip install --user nvidia-cuda-nvcc 'cuda-tile[tileiras]' nvidia-cuda-cccl
#   export PATH="$HOME/.local/lib/python3.11/site-packages/nvidia/cu13/bin:$PATH"
# They build with --enable-tile, which mixes tile kernels and host code into one
# executable; TornadoVM instead drives `nvcc -tilecubin --tile-only` to a bare cubin.
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$(mktemp -d)}"
mkdir -p "$out"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[ERROR] nvcc not found on PATH -- install the CUDA toolkit." >&2
  exit 1
fi

# Compute capability of the first visible GPU, e.g. 8.9 -> sm_89.
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
arch="sm_${cc:-89}"
echo "Building for $arch"

fail=0
pass=0
skip=0

check() { # label logfile
  # Match real failure indicators, not the bare substring "error": a correct run that
  # reports "max abs error 0.0002" must not be scored a failure. Matching diagnostics
  # ("cuda err", "error:") and explicit verdicts instead.
  if grep -qE 'WRONG|FAILED|INCORRECT|error:|cuda err|cublas err|cudnn err|Segmentation fault' "$2"; then
    echo "FAIL $1 (see $2)"; fail=$((fail + 1))
  elif grep -qE 'correct|PASSED|out: \[' "$2"; then
    echo "OK   $1"; pass=$((pass + 1))
  else
    echo "FAIL $1 -- no verdict in output (see $2)"; fail=$((fail + 1))
  fi
}

# Fields: demo-directory : source file : extra nvcc flags : run arguments
while IFS='|' read -r d src flags args; do
  [ -n "$d" ] || continue
  bin="$out/$(basename "$d")"

  if [ "$d" = "12-cutlass-fused-epilogue" ]; then
    if [ -z "${CUTLASS_DIR:-}" ] || [ ! -d "${CUTLASS_DIR:-}/include" ]; then
      echo "SKIP $d -- set CUTLASS_DIR to a CUTLASS checkout to build this one"
      skip=$((skip + 1)); continue
    fi
    flags="$flags -std=c++17 -I$CUTLASS_DIR/include -I$CUTLASS_DIR/tools/util/include"
  fi

  # shellcheck disable=SC2086  -- $flags must word-split into separate nvcc flags
  if ! nvcc -arch="$arch" $flags -o "$bin" "$repo_root/demos/$d/$src" \
        > "$out/$(basename "$d").build.log" 2>&1; then
    echo "FAIL $d -- compile (see $out/$(basename "$d").build.log)"
    fail=$((fail + 1)); continue
  fi
  echo "OK   $d -- compile"; pass=$((pass + 1))

  # shellcheck disable=SC2086
  "$bin" $args > "$out/$(basename "$d").run.log" 2>&1
  check "$d -- run" "$out/$(basename "$d").run.log"
done <<'DEMOS'
00-hello-gpu|Hello.cu||
01-first-cuda-kernel|VectorAddKernel.cu||
02-cuda-runtime-api|CudaGraphReplay.cu||
04-cublas-hybrid|CuBlasSgemvHybrid.cu|-lcublas|
05-cufft-hybrid|CuFftLowPassHybrid.cu|-lcufft|
06-cuda-streams|CudaStreamsOverlap.cu||8 32768 65536 8 both
07-cuda-graph-benefit|CudaGraphBenefit.cu||4096 6 50 both
08-tensor-core-mma|TensorCoreMMA.cu||
11-integrated-showcase|IntegratedShowcase.cu|-lcublas|6 8 8 20 all
12-cutlass-fused-epilogue|CutlassFusedEpilogue.cu||512 512 512 5
13-cudnn-jit-convblock|CuDnnConvBlockHybrid.cu|-I/usr/include/x86_64-linux-gnu -lcudnn|4 16 32 32 16 5
14-warp-async-shared|WarpAsyncSharedReduce.cu||2048 512 5
15-kernel-time-comparison|KernelTimeComparison.cu||1048576 128 5
16-tensor-core-datatypes|TensorCoreDataTypes.cu||
17-matmul-ladder|MatMulLadder.cu|-lcublas|256 3
18-matmul-ladder-fp16|MatMulLadderFP16.cu|-lcublas|256 3
19-cutile-matmul|TileMatMul.cu|--enable-tile -std=c++20 -O3|256 10
20-cutile-hybrid|TileHybridPipeline.cu|--enable-tile -std=c++20 -O3 -lcublas|256 20 both
21-cutile-flash-attention|TileFlashAttention.cu|--enable-tile -std=c++20 -O3|128 256 20
22-matmul-ladder-fp16-tile|MatMulLadderFP16Tile.cu|--enable-tile -std=c++20 -O3 -lcublas|256 20
23-cutile-row-scan|CuTileRowScan.cu|--enable-tile -std=c++20 -O3|4096 1000 20
24-cutile-histogram|CuTileHistogram.cu|--enable-tile -std=c++20 -O3|1048576 256 20
DEMOS

# Demo 15's two diagnostic probes: they attribute the kernel-time differences
# rather than validating a result, so they are checked for a clean build+run only.
for probe in ProbeHeaderAlignment ProbeJitSpecialisation; do
  src="$repo_root/demos/15-kernel-time-comparison/$probe.cu"
  [ -f "$src" ] || continue
  if nvcc -arch="$arch" -o "$out/$probe" "$src" > "$out/$probe.build.log" 2>&1 \
     && "$out/$probe" > "$out/$probe.run.log" 2>&1; then
    echo "OK   15-kernel-time-comparison/$probe"; pass=$((pass + 1))
  else
    echo "FAIL 15-kernel-time-comparison/$probe (see $out/$probe.*.log)"; fail=$((fail + 1))
  fi
done

echo
echo "== Summary: $pass passed, $fail failed, $skip skipped =="
echo "   logs: $out"
[ "$fail" -eq 0 ] || exit 1
