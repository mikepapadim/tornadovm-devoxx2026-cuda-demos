#!/usr/bin/env bash
# Demo 29: the Hybrid API, live. One TaskGraph: NVIDIA cuFFT library tasks and Java kernels JIT-compiled by TornadoVM,
# on shared device buffers; a low-pass filter sweeps over a noisy signal, one execution per frame.
#
#   bash demos/29-hybrid-api-live/run.sh                     the full-screen dashboard (~20 s; terminal >= 124 x 40)
#   bash demos/29-hybrid-api-live/run.sh plain [tornado|java] the same program without the dashboard (its protocol
#                                                            lines and the verdict), via tornado or java @argfile
# Needs: source scripts/setup-env.sh (TornadoVM 7.0.0 with the CUDA backend; cuFFT ships with it).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
[ -n "${TORNADOVM_HOME:-}" ] || { echo "run.sh: TORNADOVM_HOME is not set (source scripts/setup-env.sh)" >&2; exit 1; }
export TORNADOVM_HOME JAVA_HOME
if [ "${1:-}" != plain ]; then
    exec python3 "$here/dashboard.py" "$here/build/dashboard"
fi
mode="${2:-tornado}"
classes="$here/build/classes"; mkdir -p "$classes"
"$JAVA_HOME/bin/javac" -cp "$(ls "$TORNADOVM_HOME"/share/java/tornado/*.jar | tr '\n' ':')" -d "$classes" "$here/HybridLive.java"
case "$mode" in
    tornado) exec tornado --classpath "$classes" HybridLive 110 ;;
    java)    [ -f "$TORNADOVM_HOME/tornado-argfile" ] || tornado --generate-argfile >/dev/null
             exec "$JAVA_HOME/bin/java" "@$TORNADOVM_HOME/tornado-argfile" -cp "$classes" HybridLive 110 ;;
    *)       sed -n '2,8p' "$0"; exit 1 ;;
esac
