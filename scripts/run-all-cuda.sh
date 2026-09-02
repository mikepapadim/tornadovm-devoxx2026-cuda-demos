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
  if grep -qE 'WRONG|FAILED|INCORRECT|error' "$2"; then
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
DEMOS

echo
echo "== Summary: $pass passed, $fail failed, $skip skipped =="
echo "   logs: $out"
[ "$fail" -eq 0 ] || exit 1
