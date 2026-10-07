# Shared settings for the Devoxx demo scripts. Sourced by each demo*.sh; nothing to run by hand.
# shellcheck shell=bash

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Where things live on this machine
DEMOS_REPO="${DEMOS_REPO:-$(cd "$DEMO_ROOT/.." && pwd)}"   # this repo: the TornadoVM 7.2.0 demos (Hybrid API, CUDA Tile, JVector)
JITLLM_DIR="${JITLLM_DIR:-$DEMO_ROOT/jitllm}"                       # latest jitLLM (main), cloned and built by setup.sh
CUVS_SDK="${CUVS_SDK:-$HOME/.sdkman/candidates/tornadovm/7.2.0-jdk22plus-cuda}"  # TornadoVM 7.2.0 (released; ships tornado-cuvs)
LLAMA_BENCH="${LLAMA_BENCH:-$HOME/llama.cpp-ref/build/bin/llama-bench}"
ADA_100K="${ADA_100K:-$HOME/jvector-work/datasets/public/ada_002_100k_base_99287.fvecs}"
SHOWCASE_DATA="${SHOWCASE_DATA:-$HOME/jvector-work/showcase-data}"     # demo 27: ada-002 100k + 1M, JVector's CPU graph, 4 segments
JVECTOR_LOCAL="${JVECTOR_LOCAL:-$HOME/jvector-work/jvector}"           # JVector clone with feat/gpu-build-pq-compaction (not pushed)
BUILD="$DEMO_ROOT/build"

# Models (see models.sh for everything on this machine)
MODEL_BENCH="${MODEL_BENCH:-$HOME/models/Qwen3-0.6B-F16.gguf}"            # jitLLM vs llama.cpp
MODEL_CHAT="${MODEL_CHAT:-$HOME/models/Llama-3.2-1B-Instruct-f16.gguf}"    # live generation
MODEL_CODE="${MODEL_CODE:-$HOME/jcon/models/Qwen3-4B-f16.gguf}"             # writes the GPU kernel (fancyJitllm.sh)

# --- presentation helpers -------------------------------------------------------------------
step()  { printf '\n\033[1;35m==> %s\033[0m\n' "$*"; }
note()  { printf '\033[0;36m    %s\033[0m\n' "$*"; }
show()  { printf '\033[1;32m$\033[0m %s\n' "$*"; }
run()   { show "$*"; eval "$@"; }
pause() { [ -n "${NO_PAUSE:-}" ] || read -r -p $'\033[2m    [enter] to continue\033[0m ' _; }

# --- environments ---------------------------------------------------------------------------

# TornadoVM 7.2.0 (released, SDKMAN) for the Hybrid API and CUDA Tile demos
use_tornadovm_7() {
    # SDKMAN's init script (sourced by setup-env.sh) is not safe under `set -u`
    set +eu
    # shellcheck disable=SC1091
    source "$DEMOS_REPO/scripts/setup-env.sh" >/dev/null 2>&1
    set -eu
    TORNADO_CP=$(ls "$TORNADOVM_HOME"/share/java/tornado/*.jar | tr '\n' ':')
}

# Compile one demo of the demos repo into build/<demo> (a second or two)
compile_demo() {
    local d="$1"
    mkdir -p "$BUILD/$d"
    "$JAVA_HOME/bin/javac" -proc:none -cp "$TORNADO_CP" -d "$BUILD/$d" "$DEMOS_REPO/demos/$d"/*.java
}

# The TornadoVM develop build jitLLM main is built against (JDK 21)
use_jitllm() {
    export JAVA_HOME="$HOME/.sdkman/candidates/java/21.0.2-open"
    export PATH="$JAVA_HOME/bin:$PATH"
    eval "$(cd "$JITLLM_DIR" && scripts/tornadovm-dev.sh env 2>/dev/null)"
}

# TornadoVM 7.2.0 (ships tornado-cuvs) for JVector + cuVS (JDK 25)
use_cuvs_sdk() {
    export TORNADOVM_HOME="$CUVS_SDK"
    export JAVA_HOME="$HOME/.sdkman/candidates/java/25.0.2-open"
    export PATH="$TORNADOVM_HOME/bin:$JAVA_HOME/bin:$PATH"
    [ -f "$TORNADOVM_HOME/tornado-argfile" ] || tornado --generate-argfile >/dev/null 2>&1
}
