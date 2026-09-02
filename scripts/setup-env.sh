#!/usr/bin/env bash
# Set up the pinned TornadoVM 6.0.0 CUDA environment for this repo.
#
#   source scripts/setup-env.sh
#
# Sets JAVA_HOME / TORNADOVM_HOME / PATH from env/versions.env and regenerates
# the JDK-specific argfile at $TORNADOVM_HOME/tornado-argfile.
#
# Install prerequisites once (see README.md "Quick install"):
#   sdk install java 25.0.2-open
#   sdk install tornadovm 6.0.0-jdk22plus-cuda

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
[ -s "$HOME/.sdkman/bin/sdkman-init.sh" ] && source "$HOME/.sdkman/bin/sdkman-init.sh"

_jdk=$(grep '^JDK_SDKMAN_CANDIDATE=' "$_repo_root/env/versions.env" | cut -d= -f2)
_tvm=$(grep '^TORNADO_SDKMAN_CANDIDATE=' "$_repo_root/env/versions.env" | cut -d= -f2)

export JAVA_HOME="$HOME/.sdkman/candidates/java/$_jdk"
export TORNADOVM_HOME="$HOME/.sdkman/candidates/tornadovm/$_tvm"

if [ ! -x "$JAVA_HOME/bin/java" ]; then
  echo "[ERROR] JDK not installed: $JAVA_HOME" >&2
  echo "        run: sdk install java $_jdk" >&2
  return 1 2>/dev/null || exit 1
fi
if [ ! -d "$TORNADOVM_HOME/share/java/tornado" ]; then
  echo "[ERROR] TornadoVM SDK not installed: $TORNADOVM_HOME" >&2
  echo "        run: sdk install tornadovm $_tvm" >&2
  return 1 2>/dev/null || exit 1
fi

# TORNADO_SDK is the pre-6.0.0 variable name; 6.0.0 reads TORNADOVM_HOME.
# Unset the old one so a stale value cannot select a different SDK.
unset TORNADO_SDK

export PATH="$JAVA_HOME/bin:$TORNADOVM_HOME/bin:$PATH"
export TORNADO_ARGFILE="$TORNADOVM_HOME/tornado-argfile"

# The argfile encodes JDK-specific flags (EnableJVMCI is required on JDK <= 26 and
# fatal on 27+), so regenerate it for whichever JDK is active now.
tornado --generate-argfile >/dev/null 2>&1

echo "JAVA_HOME       = $JAVA_HOME"
echo "TORNADOVM_HOME  = $TORNADOVM_HOME"
echo "TORNADO_ARGFILE = $TORNADO_ARGFILE"
