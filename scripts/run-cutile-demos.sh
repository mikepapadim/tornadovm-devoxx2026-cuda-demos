#!/usr/bin/env bash
# LEGACY. Kept only for the `feat/cutile` feature-branch SDK, a jdk21-dev build whose
# tornado-api classes need --release 21 --enable-preview.
#
# PREFER scripts/run-all-demos.sh. PR #1083 is merged into upstream develop, so the tile
# demos now run on a jdk22plus build with no preview flags, on the same JDK as every other
# demo, through the same runner. Which SDK is active is one line -- TORNADO_SDK_PROFILE in
# env/versions.env -- and a demo the active SDK cannot run is skipped rather than failed:
#
#   source scripts/setup-env.sh && bash scripts/run-all-demos.sh
#
# This script also does NOT pass -Dtornado.recover.bailout=False, so a tile kernel that
# fails to compile here falls back to the JVM and still prints "correct". run-all-demos.sh
# passes it.
#
# Compile and run the CUDA Tile demos (19, 20, 21, 22) BOTH ways: the `tornado` launcher and
# the `java @argfile` reproducibility path. Exits non-zero if any demo fails or reports a
# wrong result.
#
#   TORNADOVM_HOME=<cutile SDK> JAVA_HOME=<jdk21> bash scripts/run-cutile-demos.sh [out-dir]
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$(mktemp -d)}"
mkdir -p "$out"

if [ -z "${TORNADOVM_HOME:-}" ] || [ ! -d "$TORNADOVM_HOME/share/java/tornado" ]; then
  echo "[ERROR] TORNADOVM_HOME is not set to a TornadoVM SDK with the tile API." >&2
  exit 1
fi
if ! ls "$TORNADOVM_HOME"/share/java/tornado/tornado-api-*.jar >/dev/null 2>&1; then
  echo "[ERROR] no tornado-api jar under $TORNADOVM_HOME" >&2
  exit 1
fi

argfile="$TORNADOVM_HOME/tornado-argfile"
[ -f "$argfile" ] || tornado --generate-argfile >/dev/null 2>&1

cp=$(ls "$TORNADOVM_HOME"/share/java/tornado/*.jar | tr '\n' ':')
fail=0
pass=0

check() { # label logfile
  if grep -qE 'WRONG|FAILED|Exception|Error occurred' "$2"; then
    echo "FAIL $1 (see $2)"; fail=$((fail + 1))
  elif grep -qE 'correct|PASSED' "$2"; then
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

  # --enable-preview: the cuTile branch is a jdk21-dev build, whose tornado-api classes use
  # JDK 21 preview features. The released 6.0.0-jdk22plus-cuda SDK does not need this.
  if ! "$JAVA_HOME/bin/javac" --release 21 --enable-preview -proc:none -cp "$cp" -d "$out/$d" \
        "$repo_root/demos/$d"/*.java > "$out/$d/javac.log" 2>&1; then
    echo "FAIL $d -- compile (see $out/$d/javac.log)"; fail=$((fail + 1)); continue
  fi
  echo "OK   $d -- compile"; pass=$((pass + 1))

  # shellcheck disable=SC2086  -- $a must word-split into separate demo arguments
  ( cd "$out/$d" && tornado --classpath . "$m" $a ) > "$out/$d/tornado.log" 2>&1
  check "$d -- tornado launcher" "$out/$d/tornado.log"

  # shellcheck disable=SC2086
  ( cd "$out/$d" && java "@$argfile" --enable-preview -cp . "$m" $a ) > "$out/$d/javaargfile.log" 2>&1
  check "$d -- java @argfile" "$out/$d/javaargfile.log"
done <<'DEMOS'
19-cutile-matmul:TileMatMul:256 10
20-cutile-hybrid:TileHybridPipeline:256 20 both
21-cutile-flash-attention:TileFlashAttention:128 256 20
22-matmul-ladder-fp16-tile:MatMulLadderFP16Tile:256 5
DEMOS

echo
echo "== Summary: $pass passed, $fail failed =="
echo "   logs: $out"
[ "$fail" -eq 0 ] || exit 1
