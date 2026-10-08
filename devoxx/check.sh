#!/usr/bin/env bash
# Pre-talk check: GPU, every demo once in its fastest form, pass/fail per demo. About a minute.
set -uo pipefail
cd "$(dirname "$0")"
export NO_PAUSE=1
log="$(mktemp -d)"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
check() { # name, command, success pattern
    if eval "$2" > "$log/$1.log" 2>&1 && grep -qE "$3" "$log/$1.log"; then echo "OK    $1"; else echo "FAIL  $1   (see $log/$1.log)"; fi
}
check demoHybrid  "bash demoHybrid.sh"        "All iterations correct"
check demoTile    "bash demoTile.sh"          "All rungs produced the same, correct result"
check demoJitllm  "bash demoJitllm.sh"        "pp512"
check fancyJitllmCode "bash fancyJitllm.sh code" "PASSED"
check fancyJitllmLive "bash fancyJitllmLive.sh" "LiveDashboard: PASSED"
echo "logs: $log"
