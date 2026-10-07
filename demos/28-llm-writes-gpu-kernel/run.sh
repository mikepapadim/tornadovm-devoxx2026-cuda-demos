#!/usr/bin/env bash
# Demo 28: an LLM, running in pure Java on the GPU (jitLLM), writes a GPU kernel in Java; TornadoVM compiles it to CUDA
# and runs it; the same Java method runs on the CPU as the reference; the result is drawn in the terminal. ~25 s.
#
#   bash demos/28-llm-writes-gpu-kernel/run.sh [tornado|java]               generate live with jitLLM, then run
#   bash demos/28-llm-writes-gpu-kernel/run.sh [tornado|java] --reference   skip jitLLM: run the kernel the model
#                                                                             wrote for this prompt in rehearsal
#   bash demos/28-llm-writes-gpu-kernel/run.sh dashboard                     the full-screen live version (~30 s): every
#       component lit while it works, nvidia-smi showing both processes on the one GPU, a live zoom (dashboard/)
# Needs: source scripts/setup-env.sh (TornadoVM 7.2.0 runs the kernel). Live generation also needs a built jitLLM:
#   JITLLM_DIR=<clone of beehive-lab/jitllm, built>  JITLLM_JAVA_HOME=<its JDK 21>  MODEL=<Qwen3-4B-f16.gguf>
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
if [ "${1:-}" = dashboard ]; then
    [ -n "${TORNADOVM_HOME:-}" ] || { echo "run.sh: TORNADOVM_HOME is not set (source scripts/setup-env.sh)" >&2; exit 1; }
    : "${JITLLM_DIR:?set JITLLM_DIR, JITLLM_JAVA_HOME and MODEL (see README.md)}" "${JITLLM_JAVA_HOME:?}" "${MODEL:?}"
    exec python3 "$here/dashboard/dashboard.py" "$here/build/dashboard"
fi
mode=tornado
case "${1:-}" in tornado|java) mode="$1"; shift ;; esac
reference=false
[ "${1:-}" = --reference ] && reference=true
[ -n "${TORNADOVM_HOME:-}" ] || { echo "run.sh: TORNADOVM_HOME is not set (source scripts/setup-env.sh)" >&2; exit 1; }
build="$here/build"; rm -rf "$build"; mkdir -p "$build"
render="python3 $here/render.py"
bold() { printf '\n    \033[1m%s\033[0m\n' "$*"; }

# 1. the kernel: written live by the model, or the rehearsal one
printf '\n    \033[1;36m❯\033[0m \033[1m%s\033[0m\n\n' "$(fold -s -w 104 "$here/prompt.txt" | sed '2,$s/^/      /')"
if $reference; then
    bold "the kernel Qwen3-4B wrote for this prompt in rehearsal (--reference)"
    cp "$here/reference.kernel" "$build/kernel.java"
else
    : "${JITLLM_DIR:?set JITLLM_DIR to a built clone of beehive-lab/jitllm, or use --reference}"
    : "${JITLLM_JAVA_HOME:?set JITLLM_JAVA_HOME to the JDK 21 jitLLM uses}"
    : "${MODEL:?set MODEL to a GGUF model, e.g. Qwen3-4B-f16.gguf}"
    # greedy decoding (temperature 0): the same prompt and model give the same kernel
    (
        export JAVA_HOME="$JITLLM_JAVA_HOME" PATH="$JITLLM_JAVA_HOME/bin:$PATH"
        cd "$JITLLM_DIR"
        eval "$(scripts/tornadovm-dev.sh env 2>/dev/null)"
        ./jitllm run --gpu --cuda-graphs --model "$MODEL" -c 2048 --max-new-tokens 500 --temperature 0 \
            -sp "$(cat "$here/system-prompt.txt")" --prompt "$(cat "$here/prompt.txt") /no_think" 2>"$build/generation.err"
    ) | tee "$build/generation.out" | $render stream
    printf '    \033[2mjitLLM: %s (prompt + answer)\033[0m\n' "$(grep -o 'achieved tok/s.*' "$build/generation.err" || echo 'no metrics line')"
    awk '/```java/{f=1;next} /```/{f=0} f' "$build/generation.out" > "$build/kernel.java"
    bold "the kernel, as written by the model"
fi
$render code "$build/kernel.java"

# 2. the harness: the kernel inside a class that runs it on the GPU, then on the CPU, and compares
insert() {
    python3 - "$1" "$here/Harness.template" "$build/Harness.java" <<'PY'
import sys
kernel, template, out = sys.argv[1:]
body = open(kernel).read().rstrip().replace("\n", "\n    ")
open(out, "w").write(open(template).read().replace("/*KERNEL*/", body))
PY
}
compile() { "$JAVA_HOME/bin/javac" -cp "$(ls "$TORNADOVM_HOME"/share/java/tornado/*.jar | tr '\n' ':')" -d "$build" "$build/Harness.java"; }
insert "$build/kernel.java"
if ! compile 2> "$build/javac.log"; then
    printf '\n    \033[1;31mthe kernel did not compile:\033[0m\n'; sed 's/^/      /' "$build/javac.log" | head -12
    $reference && exit 1
    printf '\n    \033[2mfalling back to the kernel the same model wrote for this prompt in rehearsal (reference.kernel)\033[0m\n'
    insert "$here/reference.kernel"
    compile
fi

# 3. run: TornadoVM JIT-compiles the Java method to CUDA (printed), runs it on the GPU, then the CPU runs the same method
printf '\n    \033[1;36m⠿\033[0m running on the GPU (TornadoVM) and on one CPU thread ...\n'
if [ "$mode" = java ]; then
    [ -f "$TORNADOVM_HOME/tornado-argfile" ] || tornado --generate-argfile >/dev/null
    "$JAVA_HOME/bin/java" "@$TORNADOVM_HOME/tornado-argfile" -Dtornado.printKernel=True -cp "$build" Harness > "$build/run.log" 2>&1
else
    tornado --printKernel --classpath "$build" Harness > "$build/run.log" 2>&1
fi
echo
$render cuda "$build/run.log" 14
$render fractal "$build/run.log"
grep -E "^LlmGpuKernel: " "$build/run.log"
