#!/usr/bin/env bash
# Set up the TornadoVM CUDA environment for this repo.
#
#   source scripts/setup-env.sh                    # the pinned profile
#
#   export TORNADO_SDK_PROFILE=sdkman-6.0.0        # a one-off override -- EXPORT it first:
#   source scripts/setup-env.sh                    # `VAR=x source ...` does not persist,
#                                                  # so the name would be lost afterwards
#
# Which SDK is used is decided by ONE line -- TORNADO_SDK_PROFILE in
# env/versions.env -- which names a file in env/sdk/. See env/sdk/README.md.
#
# Sets JAVA_HOME / TORNADOVM_HOME / PATH and regenerates the JDK-specific argfile
# at $TORNADOVM_HOME/tornado-argfile. Also exports the profile's capability flags
# (TORNADO_HAS_TILE_API, TORNADO_MIN_CUDA_TOOLKIT, TORNADO_NVCC) so scripts can
# skip demos a profile cannot run instead of failing them.

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
[ -s "$HOME/.sdkman/bin/sdkman-init.sh" ] && source "$HOME/.sdkman/bin/sdkman-init.sh"

# The caller's TORNADO_SDK_PROFILE wins over the pinned one, so a single demo run
# can be checked against another SDK without editing a tracked file.
if [ -z "${TORNADO_SDK_PROFILE:-}" ]; then
  TORNADO_SDK_PROFILE=$(grep '^TORNADO_SDK_PROFILE=' "$_repo_root/env/versions.env" | cut -d= -f2)
fi
_profile="$_repo_root/env/sdk/$TORNADO_SDK_PROFILE.env"

if [ ! -f "$_profile" ]; then
  echo "[ERROR] no such SDK profile: $TORNADO_SDK_PROFILE" >&2
  echo "        available: $(cd "$_repo_root/env/sdk" && ls *.env | sed 's/\.env$//' | tr '\n' ' ')" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$_profile"
export TORNADO_SDK_PROFILE

export JAVA_HOME="$HOME/.sdkman/candidates/java/$JDK_SDKMAN_CANDIDATE"

case "$TORNADO_SDK_KIND" in
  sdkman)
    export TORNADOVM_HOME="$HOME/.sdkman/candidates/tornadovm/$TORNADO_SDKMAN_CANDIDATE"
    _install_hint="sdk install tornadovm $TORNADO_SDKMAN_CANDIDATE"
    ;;
  local-build)
    # A source build lands in dist/<name>-linux-amd64/<name>/; the version is in the
    # directory name, so glob for it rather than restating the version in two places.
    _dist=$(ls -d "$_repo_root/$TORNADO_BUILD_DIR"/dist/*-"$TORNADO_BACKEND"-linux-amd64/*/ 2>/dev/null | tail -1)
    export TORNADOVM_HOME="${_dist%/}"
    _install_hint="cd $TORNADO_BUILD_DIR && make BACKEND=$TORNADO_BACKEND"
    ;;
  *)
    echo "[ERROR] profile $TORNADO_SDK_PROFILE has an unknown TORNADO_SDK_KIND: ${TORNADO_SDK_KIND:-<unset>}" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

if [ ! -x "$JAVA_HOME/bin/java" ]; then
  echo "[ERROR] JDK not installed: $JAVA_HOME" >&2
  echo "        run: sdk install java $JDK_SDKMAN_CANDIDATE" >&2
  return 1 2>/dev/null || exit 1
fi
if [ -z "$TORNADOVM_HOME" ] || [ ! -d "$TORNADOVM_HOME/share/java/tornado" ]; then
  echo "[ERROR] TornadoVM SDK not found for profile $TORNADO_SDK_PROFILE: ${TORNADOVM_HOME:-<unresolved>}" >&2
  echo "        run: $_install_hint" >&2
  return 1 2>/dev/null || exit 1
fi

# TORNADO_SDK is the pre-6.0.0 variable name; 6.0.0+ reads TORNADOVM_HOME and
# 6.1.1 warns on it. Unset it so a stale value cannot select a different SDK.
unset TORNADO_SDK

export PATH="$JAVA_HOME/bin:$TORNADOVM_HOME/bin:$PATH"
export TORNADO_ARGFILE="$TORNADOVM_HOME/tornado-argfile"

# Capability flags, exported for run-all-demos.sh and any demo wrapper.
export TORNADO_HAS_TILE_API="${TORNADO_HAS_TILE_API:-0}"
export TORNADO_MIN_CUDA_TOOLKIT="${TORNADO_MIN_CUDA_TOOLKIT:-0}"
export TORNADO_NVCC="$(eval echo "${TORNADO_NVCC:-}")"
export TORNADO_JAVAC_FLAGS="${TORNADO_JAVAC_FLAGS:-}"
export TORNADO_JVM_FLAGS="${TORNADO_JVM_FLAGS:-}"

# nvcc spawns `tileiras` (the tile IR assembler) by bare name, so it only works
# when tileiras is on PATH. Both ship in the same wheel's bin directory, so
# putting the pinned nvcc's directory on PATH satisfies the lookup. Without this
# a tile compile fails late with `sh: 1: tileiras: not found`.
if [ -n "$TORNADO_NVCC" ] && [ -x "$TORNADO_NVCC" ]; then
  export PATH="$(dirname "$TORNADO_NVCC"):$PATH"
fi

# The argfile encodes JDK-specific flags (EnableJVMCI is required on JDK <= 26 and
# fatal on 27+), so regenerate it for whichever JDK is active now.
tornado --generate-argfile >/dev/null 2>&1

echo "TORNADO_SDK_PROFILE = $TORNADO_SDK_PROFILE ($TORNADO_SDK_KIND, $TORNADO_VERSION)"
echo "JAVA_HOME           = $JAVA_HOME"
echo "TORNADOVM_HOME      = $TORNADOVM_HOME"
echo "TORNADO_ARGFILE     = $TORNADO_ARGFILE"
if [ "$TORNADO_HAS_TILE_API" = "1" ]; then
  echo "tile API            = yes (needs nvcc >= $TORNADO_MIN_CUDA_TOOLKIT, using ${TORNADO_NVCC:-auto-located})"
else
  echo "tile API            = no (demos requiring it will report SKIPPED_REQUIREMENT)"
fi
