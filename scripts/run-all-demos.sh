#!/usr/bin/env bash
# Compile and run every Track A demo on the pinned TornadoVM 6.0.0 CUDA SDK,
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

check() { # label logfile
  # Every demo prints an explicit correctness verdict; a run that printed a
  # failure verdict, or printed none at all, is a failure.
  if grep -qE 'WRONG|FAILED|Exception|Error occurred' "$2"; then
    echo "FAIL $1 (see $2)"; fail=$((fail + 1))
  elif grep -qE 'correct|PASSED|out: \[' "$2"; then
    echo "OK   $1"; pass=$((pass + 1))
  else
    echo "FAIL $1 -- no verdict in output (see $2)"; fail=$((fail + 1))
  fi
}

# Fields: demo-directory : main class : demo arguments
while IFS=: read -r d m a; do
  [ -n "$d" ] || continue
  echo "== $d"
  mkdir -p "$out/$d"

  if ! "$JAVA_HOME/bin/javac" -cp "$cp" -d "$out/$d" "$repo_root/demos/$d"/*.java \
        > "$out/$d/javac.log" 2>&1; then
    echo "FAIL $d -- compile (see $out/$d/javac.log)"; fail=$((fail + 1)); continue
  fi
  echo "OK   $d -- compile"; pass=$((pass + 1))

  # shellcheck disable=SC2086  -- $a must word-split into separate demo arguments
  ( cd "$out/$d" && tornado --classpath . "$m" $a ) > "$out/$d/tornado.log" 2>&1
  check "$d -- tornado launcher" "$out/$d/tornado.log"

  # shellcheck disable=SC2086
  ( cd "$out/$d" && java "@$argfile" -cp . "$m" $a ) > "$out/$d/javaargfile.log" 2>&1
  check "$d -- java @argfile" "$out/$d/javaargfile.log"
done <<'DEMOS'
00-hello-gpu:Hello:
01-first-cuda-kernel:VectorAddKernel:
02-cuda-runtime-api:CudaGraphReplay:
04-cublas-hybrid:CuBlasSgemvHybrid:
05-cufft-hybrid:CuFftLowPassHybrid:
06-cuda-streams:CudaStreamsOverlap:8 32768 65536 8 both
07-cuda-graph-benefit:CudaGraphBenefit:4096 6 50 both
08-tensor-core-mma:TensorCoreMMA:
11-integrated-showcase:IntegratedShowcase:6 8 8 20 all
12-cutlass-fused-epilogue:CutlassFusedEpilogue:512 512 512 5
13-cudnn-jit-convblock:CuDnnConvBlockHybrid:4 16 32 32 16 5
14-warp-async-shared:WarpAsyncSharedReduce:2048 512 5
DEMOS

echo
echo "== Summary: $pass passed, $fail failed =="
echo "   logs: $out"
[ "$fail" -eq 0 ] || exit 1
