#!/usr/bin/env bash
# Compile and run every Track A demo on the active TornadoVM SDK profile,
# BOTH ways: the `tornado` launcher and the `java @argfile` reproducibility path.
# Requires a CUDA GPU. Exits non-zero if any demo fails to compile or run.
#
#   source scripts/setup-env.sh && bash scripts/run-all-demos.sh [output-dir]
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$(mktemp -d)}"
mkdir -p "$out"

if [ -z "${TORNADOVM_HOME:-}" ] || [ ! -d "$TORNADOVM_HOME/share/java/tornado" ]; then
  echo "[ERROR] TORNADOVM_HOME is not set to a TornadoVM SDK." >&2
  echo "        run: source scripts/setup-env.sh" >&2
  exit 1
fi
argfile="$TORNADOVM_HOME/tornado-argfile"
[ -f "$argfile" ] || tornado --generate-argfile >/dev/null 2>&1

cp=$(ls "$TORNADOVM_HOME"/share/java/tornado/*.jar | tr '\n' ':')
fail=0
pass=0
skip=0

echo "== profile: ${TORNADO_SDK_PROFILE:-<unset>} (tile API: ${TORNADO_HAS_TILE_API:-0})"

# A tile task silently falls back to the JVM when it cannot compile, because
# TornadoOptions.RECOVER_BAILOUT defaults true and every TileContext method has a
# host implementation. That fallback computes the right answer and prints the same
# "correct" verdict this script greps for, so a wholly blocked GPU path would score
# as a pass. Disabling bailout turns that silent fallback into a visible error.
jvmflags="-Dtornado.recover.bailout=False ${TORNADO_JVM_FLAGS:-}"

# Does the active profile satisfy a demo's `requires` field?
have_requirement() {
  case "$1" in
    "")   return 0 ;;
    tile) [ "${TORNADO_HAS_TILE_API:-0}" = "1" ] ;;
    *)    return 1 ;;
  esac
}

check() { # label logfile
  # Every demo prints an explicit correctness verdict; a run that printed a
  # failure verdict, or printed none at all, is a failure.
  if grep -q '\[UNSUPPORTED\]' "$2"; then
    echo "SKIP $1 -- requirement not met (see $2)"; skip=$((skip + 1))
  elif grep -qE 'WRONG|FAILED|Exception|Error occurred' "$2"; then
    echo "FAIL $1 (see $2)"; fail=$((fail + 1))
  elif grep -qE 'correct|PASSED|out: \[' "$2"; then
    echo "OK   $1"; pass=$((pass + 1))
  else
    echo "FAIL $1 -- no verdict in output (see $2)"; fail=$((fail + 1))
  fi
}

# Fields: demo-directory : main class : demo arguments
while IFS=: read -r d m a req; do
  [ -n "$d" ] || continue
  echo "== $d"

  if ! have_requirement "$req"; then
    echo "SKIP $d -- profile ${TORNADO_SDK_PROFILE:-<unset>} lacks: $req"; skip=$((skip + 1)); continue
  fi

  mkdir -p "$out/$d"

  # shellcheck disable=SC2086  -- TORNADO_JAVAC_FLAGS must word-split
  if ! "$JAVA_HOME/bin/javac" ${TORNADO_JAVAC_FLAGS:-} -proc:none -cp "$cp" -d "$out/$d" "$repo_root/demos/$d"/*.java \
        > "$out/$d/javac.log" 2>&1; then
    echo "FAIL $d -- compile (see $out/$d/javac.log)"; fail=$((fail + 1)); continue
  fi
  echo "OK   $d -- compile"; pass=$((pass + 1))

  # shellcheck disable=SC2086  -- $a must word-split into separate demo arguments
  ( cd "$out/$d" && tornado --jvm="$jvmflags" --classpath . "$m" $a ) > "$out/$d/tornado.log" 2>&1
  check "$d -- tornado launcher" "$out/$d/tornado.log"

  # shellcheck disable=SC2086
  # shellcheck disable=SC2086
  ( cd "$out/$d" && java "@$argfile" $jvmflags -cp . "$m" $a ) > "$out/$d/javaargfile.log" 2>&1
  check "$d -- java @argfile" "$out/$d/javaargfile.log"
done <<'DEMOS'
00-hello-gpu:Hello::
01-first-cuda-kernel:VectorAddKernel::
02-cuda-runtime-api:CudaGraphReplay::
04-cublas-hybrid:CuBlasSgemvHybrid::
05-cufft-hybrid:CuFftLowPassHybrid::
06-cuda-streams:CudaStreamsOverlap:8 32768 65536 8 both:
07-cuda-graph-benefit:CudaGraphBenefit:4096 6 50 both:
08-tensor-core-mma:TensorCoreMMA::
11-integrated-showcase:IntegratedShowcase:6 8 8 20 all:
12-cutlass-fused-epilogue:CutlassFusedEpilogue:512 512 512 5:
13-cudnn-jit-convblock:CuDnnConvBlockHybrid:4 16 32 32 16 5:
14-warp-async-shared:WarpAsyncSharedReduce:2048 512 5:
15-kernel-time-comparison:KernelTimeComparison:1048576 128 5:
16-tensor-core-datatypes:TensorCoreDataTypes::
17-matmul-ladder:MatMulLadder:256 3:
18-matmul-ladder-fp16:MatMulLadderFP16:256 3:
19-cutile-matmul:TileMatMul:256 10:tile
20-cutile-hybrid:TileHybridPipeline:256 20 both:tile
21-cutile-flash-attention:TileFlashAttention:128 256 20:tile
22-matmul-ladder-fp16-tile:MatMulLadderFP16Tile:256 5:tile
23-cutile-row-scan:CuTileRowScan:4096 1000 20:tile
24-cutile-histogram:CuTileHistogram:1048576 256 20:tile
DEMOS

echo
echo "== Summary: $pass passed, $fail failed, $skip skipped =="
echo "   logs: $out"
[ "$fail" -eq 0 ] || exit 1
