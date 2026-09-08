#!/usr/bin/env bash
# Collects every number behind the CUDA global-load batching fix, as raw CSV.
#
# Three questions, three arms:
#   1. Does the fix change TornadoVM's kernel time?   TornadoVM, flag off vs on.
#   2. How close does it get to hand-written CUDA?    the ladder's own CUDA registerTiled.
#   3. Is the penalty TornadoVM's fault at all?       ReorderProbe.cu, one kernel, three schedules.
#
# Everything comes from ONE TornadoVM build with the flag toggled at run time, so the two
# TornadoVM arms differ only in the emitted instruction order.
#
# Usage:  ./collect.sh <tornadovm-sdk-dir> [n] [executions]
#
# Writes, next to this script:
#   per-launch-durations.csv   every individual kernel launch (plot this)
#   kernel-summary.csv         median/mean/stddev per arm
#   pure-cuda-probe.csv        ReorderProbe.cu, three schedules
#   ncu-counters.csv           stall + issue counters, flag off vs on
set -euo pipefail

SDK="${1:?usage: collect.sh <tornadovm-sdk-dir> [n] [executions]}"
N="${2:-1024}"
EXEC="${3:-100}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DEMO="$HERE/../../../demos/17-matmul-ladder"
WORK="$(mktemp -d)"
NCU="${NCU:-/opt/nvidia/nsight-compute/2024.3.2/ncu}"

export TORNADOVM_HOME="$SDK"
export PATH="$SDK/bin:$PATH"

echo "== building both sides (n=$N, executions=$EXEC)"
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d "$WORK" "$DEMO/MatMulLadder.java" 2>/dev/null
nvcc -arch=sm_89 -O3 -std=c++17 -lcublas -o "$WORK/matmul_ladder" "$DEMO/MatMulLadder.cu"
nvcc -arch=sm_89 -O3 -std=c++17 -o "$WORK/reorder_probe" "$HERE/ReorderProbe.cu"

# Every individual launch duration, so the distribution can be plotted rather than just its median.
trace() {   # trace <arm> <nsys-rep> <kernel-name-substring>
    nsys stats --report cuda_gpu_trace --format csv "$2" 2>/dev/null \
      | awk -F, -v arm="$1" -v k="$3" 'NR>1 && $0 ~ k {print arm "," $2}'
}

echo "arm,duration_ns" > "$HERE/per-launch-durations.csv"

echo "== TornadoVM, batching OFF"
( cd "$WORK" && nsys profile --trace=cuda --force-overwrite=true -o tv_off \
    tornado --jvm=-Dtornado.cuda.batchGlobalLoads=False --classpath . MatMulLadder "$N" "$EXEC" >/dev/null 2>&1 )
trace tornadovm_off "$WORK/tv_off.nsys-rep" kcRegisterTiled >> "$HERE/per-launch-durations.csv"

echo "== TornadoVM, batching ON"
( cd "$WORK" && nsys profile --trace=cuda --force-overwrite=true -o tv_on \
    tornado --classpath . MatMulLadder "$N" "$EXEC" >/dev/null 2>&1 )
trace tornadovm_on "$WORK/tv_on.nsys-rep" kcRegisterTiled >> "$HERE/per-launch-durations.csv"

echo "== hand-written CUDA"
( cd "$WORK" && nsys profile --trace=cuda --force-overwrite=true -o cuda ./matmul_ladder "$N" "$EXEC" >/dev/null 2>&1 )
trace handwritten_cuda "$WORK/cuda.nsys-rep" registerTiled >> "$HERE/per-launch-durations.csv"

echo "== summarising"
python3 - "$HERE/per-launch-durations.csv" "$HERE/kernel-summary.csv" <<'PY'
import csv, statistics, sys
rows = {}
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        rows.setdefault(r["arm"], []).append(float(r["duration_ns"]) / 1000.0)
order = ["tornadovm_off", "tornadovm_on", "handwritten_cuda"]
base = statistics.median(rows["tornadovm_off"]) if "tornadovm_off" in rows else None
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "launches", "median_us", "mean_us", "stddev_us", "min_us", "p5_us", "p95_us", "max_us", "speedup_vs_off"])
    for arm in order:
        v = sorted(rows.get(arm, []))
        if not v:
            continue
        p = lambda q: v[min(len(v) - 1, int(q * len(v)))]
        w.writerow([arm, len(v), f"{statistics.median(v):.3f}", f"{statistics.mean(v):.3f}",
                    f"{statistics.stdev(v):.3f}" if len(v) > 1 else "0",
                    f"{v[0]:.3f}", f"{p(0.05):.3f}", f"{p(0.95):.3f}", f"{v[-1]:.3f}",
                    f"{base / statistics.median(v):.3f}" if base else ""])
print(open(sys.argv[2]).read())
PY

echo "== pure-CUDA schedule probe"
"$WORK/reorder_probe" "$N" "$EXEC" > "$HERE/pure-cuda-probe.csv"
cat "$HERE/pure-cuda-probe.csv"

echo "== Nsight Compute counters"
if [ -x "$NCU" ]; then
    M=smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
    M=$M,smsp__issue_active.avg.pct_of_peak_sustained_active
    M=$M,sm__sass_thread_inst_executed_op_ffma_pred_on.sum
    M=$M,launch__registers_per_thread
    M=$M,l1tex__data_bank_conflicts_pipe_lsu.sum
    echo "arm,metric,unit,value" > "$HERE/ncu-counters.csv"
    for mode in off on; do
        [ "$mode" = off ] && J=--jvm=-Dtornado.cuda.batchGlobalLoads=False || J=""
        ( cd "$WORK" && "$NCU" --csv --metrics "$M" --kernel-name kcRegisterTiled --launch-count 1 \
            tornado $J --classpath . MatMulLadder "$N" "$EXEC" 2>/dev/null ) \
          | grep '"kcRegisterTiled"' \
          | awk -F'","' -v arm="tornadovm_$mode" '{n=NF; gsub(/"/,"",$(n-2)); gsub(/"/,"",$(n-1)); gsub(/"/,"",$n); gsub(/,/,"",$n); print arm "," $(n-2) "," $(n-1) "," $n}' \
          >> "$HERE/ncu-counters.csv"
    done
    cat "$HERE/ncu-counters.csv"
else
    echo "   skipped: no ncu at $NCU (set NCU=/path/to/ncu)"
fi

rm -rf "$WORK"
echo "== done, CSVs written to $HERE"
