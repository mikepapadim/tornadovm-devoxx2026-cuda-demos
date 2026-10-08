#!/usr/bin/env bash
# One-time setup (needs the network): clone and build the latest jitLLM.
# Re-run to update jitLLM to the latest main.
set -euo pipefail
source "$(dirname "$0")/env.sh"

step "jitLLM: latest main"
if [ -d "$JITLLM_DIR/.git" ]; then git -C "$JITLLM_DIR" pull -q --ff-only; else git clone -q https://github.com/beehive-lab/jitllm.git "$JITLLM_DIR"; fi
git -C "$JITLLM_DIR" log --oneline -1
export JAVA_HOME="$HOME/.sdkman/candidates/java/21.0.2-open" PATH="$HOME/.sdkman/candidates/java/21.0.2-open/bin:$PATH"
(cd "$JITLLM_DIR" && { scripts/tornadovm-dev.sh status >/dev/null 2>&1 || scripts/tornadovm-dev.sh setup --backend cuda --jdk 21; } \
    && scripts/tornadovm-dev.sh build clean package -DskipTests -q)

step "done: bash check.sh"
