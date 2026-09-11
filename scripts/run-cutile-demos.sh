#!/usr/bin/env bash
# Compile and run the CUDA Tile demos (19, 20, 21) BOTH ways: the `tornado` launcher and
# the `java @argfile` reproducibility path. Exits non-zero if any demo fails or reports a
# wrong result.
#
# These demos are NOT part of scripts/run-all-demos.sh, because they do not run on the
# pinned TornadoVM 6.0.0 SDK: TileContext does not exist there. They need a TornadoVM built
# from the cuTile branch, plus CUDA Toolkit 13.3+ and driver R580+ (see
# env/versions.env, section "CUDA Tile").
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
  elif grep -qE 'correct' "$2"; then
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
DEMOS

echo
echo "== Summary: $pass passed, $fail failed =="
echo "   logs: $out"
[ "$fail" -eq 0 ] || exit 1
