#!/usr/bin/env bash
# Steps 2, 3 and 5 — everything that needs ncu performance counters.
# Run after RmProfilingAdminOnly reads 0.
#
# Differences from the doc, deliberate:
#   - all output goes to results/nvidia-meeting-sm120/, never into the sm_89
#     baseline dir (the doc's Step 3 would overwrite four committed baseline CSVs)
#   - /usr/local/cuda-13.0/bin is prepended, or nvcc resolves to the distro 12.0
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1
REPO=$PWD
export PATH=/usr/local/cuda-13.0/bin:$PATH
source scripts/setup-env.sh >/dev/null || exit 1
D=$REPO/results/nvidia-meeting-sm120
NCU=$(command -v ncu)

if ! grep -q 'RmProfilingAdminOnly: 0' /proc/driver/nvidia/params; then
  echo "ABORT: RmProfilingAdminOnly is not 0 — counters still admin-only."; exit 2
fi
echo "counters unlocked; using $NCU"

echo "=== Step 2 — alignment sweep ==="
nvcc -arch=sm_120 -O3 -o /tmp/alignment_sweep results/nvidia-meeting/AlignmentSweep.cu || exit 1
$NCU --csv --metrics \
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
dram__bytes_op_read.sum,dram__bytes_op_write.sum,gpu__time_duration.sum \
 /tmp/alignment_sweep --counters > "$D/alignment-sweep-ncu.csv" 2>"$D/alignment-sweep-ncu.err"
/tmp/alignment_sweep > "$D/alignment-sweep-timing.log" 2>&1
echo "  -> alignment-sweep-ncu.csv"

echo "=== Step 3 — geometry-controlled 2x2 ==="
mkdir -p "$D/task-B2"
cd "$REPO/results/nvidia-meeting/task-B2-geometry-controlled" || exit 1
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . GeometryControlled.java || exit 1
nvcc -arch=sm_120 -O3 -o geom GeometryControlled.cu || exit 1
M='l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__block_size,launch__registers_per_thread,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.sum'
for b in 256 1024; do
  $NCU --csv --target-processes all --metrics "$M" \
    "$JAVA_HOME/bin/java" @"$TORNADOVM_HOME/tornado-argfile" -cp . GeometryControlled 16777216 $b 2 > "$D/task-B2/tvm_$b.csv"  2>"$D/task-B2/tvm_$b.err"
  $NCU --csv --metrics "$M" ./geom 16777216 $b 2 > "$D/task-B2/cuda_$b.csv" 2>"$D/task-B2/cuda_$b.err"
  echo "  -> task-B2/{tvm,cuda}_$b.csv"
done

echo "=== Step 5 — tensor cores, instruction counts ==="
mkdir -p "$D/task-D"
cd "$REPO/demos/16-tensor-core-datatypes" || exit 1
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . TensorCoreDataTypes.java || exit 1
tornado --printKernel --classpath . TensorCoreDataTypes > "$D/task-D/d16-printkernel.log" 2>&1
$NCU --csv --target-processes all --metrics \
sm__inst_executed_pipe_tensor_subpipe_hmma_op_hmma.sum,sm__inst_executed_pipe_tensor_subpipe_imma_op_imma.sum,\
sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_tensor_subpipe_hmma.sum,\
sm__inst_executed_pipe_tensor_subpipe_imma.sum,sm__pipe_tensor_cycles_active.sum \
 "$JAVA_HOME/bin/java" @"$TORNADOVM_HOME/tornado-argfile" -cp . TensorCoreDataTypes > "$D/task-D/d16-ncu.csv" 2>"$D/task-D/d16-ncu.err"
grep -oE 'mma\.sync\.aligned\.[a-z0-9.]+' "$D/task-D/d16-printkernel.log" | sort | uniq -c > "$D/task-D/mma-shapes.txt"
# ncu does NOT fail on an unknown metric: it writes the row with a value of "n/a",
# says nothing on stderr and exits 0. Checking $? or the .err file catches nothing —
# the CSV itself is the only place the rejection shows. Always record what IS
# available so a renamed metric is diagnosable from the bundle alone.
$NCU --query-metrics 2>/dev/null | grep -iE 'tensor|hmma|imma' > "$D/task-D/available-tensor-metrics.txt"
for f in "$D/alignment-sweep-ncu.csv" "$D/task-D/d16-ncu.csv"; do
  if grep -q ',"n/a"' "$f" 2>/dev/null || grep -q ',n/a' "$f" 2>/dev/null; then
    echo "  WARNING: $f contains n/a values — a metric name was rejected silently."
    echo "           Check the names against $D/task-D/available-tensor-metrics.txt"
  fi
done
echo "  -> task-D/"
echo "ALL BLOCKED STEPS ATTEMPTED"
