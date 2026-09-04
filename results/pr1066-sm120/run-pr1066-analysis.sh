#!/usr/bin/env bash
# PR #1066 (align native array payload to 32 bytes) re-measured on sm_120.
#
# Method mirrors the sm_89 analysis in the PR body, deliberately:
#   - ONE build supplies both arms. -Dtornado.cuda.payloadAlignment=1 restores the
#     pre-patch layout exactly, so "before" and "after" differ in one property and
#     nothing else - not two builds, not two checkouts.
#   - kernel time only, from `nsys stats --report cuda_gpu_kern_sum`, never wall clock
#   - sector counts from `ncu`; they are the robust half of the measurement
#   - hand-written CUDA built at -arch=sm_120 as the reference arm
#
# Usage: run-pr1066-analysis.sh <tornadovm-sdk-home>
set -o pipefail
SDK=${1:?usage: run-pr1066-analysis.sh <tornadovm-sdk-home>}
cd "$(dirname "$0")/../.." || exit 1
REPO=$PWD
D=$REPO/results/pr1066-sm120
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_PATH=/usr/local/cuda-13.0
export JAVA_HOME=$HOME/.sdkman/candidates/java/25.0.2-open
export TORNADOVM_HOME=$SDK
export PATH=$JAVA_HOME/bin:$TORNADOVM_HOME/bin:$PATH
ARGFILE=$TORNADOVM_HOME/tornado-argfile
tornado --generate-argfile >/dev/null 2>&1

mkdir -p "$D/raw"
BEFORE="-Dtornado.cuda.payloadAlignment=1"   # pre-patch layout: payload at base+16
AFTER=""                                     # patch default: 32-byte aligned payload

echo "SDK      = $TORNADOVM_HOME"
echo "argfile  = $ARGFILE"

run_tvm() {  # run_tvm <outfile> <extra-jvm-props> <class> <args...>
  local out=$1; shift
  local props=$1; shift
  # shellcheck disable=SC2086
  "$JAVA_HOME/bin/java" @"$ARGFILE" $props -cp . "$@" > "$out" 2>&1
}

################################################################################
echo "=== 0. provenance ==="
{
  echo "date            : $(date -Is)"
  echo "gpu             : $(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader)"
  echo "nvcc            : $(nvcc --version | tail -1)"
  echo "ncu             : $(ncu --version | sed -n 's/^Version //p')"
  echo "nsys            : $(nsys --version | tail -1)"
  echo "jdk             : $("$JAVA_HOME/bin/java" -version 2>&1 | head -1)"
  echo "tornado sdk     : $TORNADOVM_HOME"
  echo "tornado commit  : $(git -C "$HOME/TornadoVM" rev-parse --short HEAD) ($(git -C "$HOME/TornadoVM" rev-parse --abbrev-ref HEAD))"
  echo "demos commit    : $(git -C "$REPO" rev-parse --short HEAD)"
} > "$D/provenance.txt"
cat "$D/provenance.txt"

################################################################################
echo "=== 1. generated code must be byte-identical between the two arms ==="
cd "$REPO/results/nvidia-meeting/task-B2-geometry-controlled" || exit 1
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . GeometryControlled.java || exit 1
for arm in before after; do
  p=$([ $arm = before ] && echo "$BEFORE" || echo "$AFTER")
  run_tvm "$D/raw/printkernel-$arm.log" "$p --printKernel" GeometryControlled 4194304 256 1
  grep -oE '\+ [0-9]+L' "$D/raw/printkernel-$arm.log" | sort | uniq -c > "$D/raw/offsets-$arm.txt"
done
if diff -q "$D/raw/offsets-before.txt" "$D/raw/offsets-after.txt" >/dev/null; then
  echo "  OK - identical index offsets in both arms"
else
  echo "  DIFFER - the patch changed generated code, which it must not:"
  diff "$D/raw/offsets-before.txt" "$D/raw/offsets-after.txt"
fi

################################################################################
echo "=== 2. sectors per request, before vs after (the robust measurement) ==="
M='l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,gpu__time_duration.sum,launch__block_size,launch__registers_per_thread,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.sum'
nvcc -arch=sm_120 -O3 -o geom GeometryControlled.cu || exit 1
for arm in before after; do
  p=$([ $arm = before ] && echo "$BEFORE" || echo "$AFTER")
  # shellcheck disable=SC2086
  ncu --csv --target-processes all --metrics "$M" \
    "$JAVA_HOME/bin/java" @"$ARGFILE" $p -cp . GeometryControlled 16777216 256 2 \
    > "$D/raw/ncu-tvm-$arm.csv" 2>"$D/raw/ncu-tvm-$arm.err"
  echo "  -> ncu-tvm-$arm.csv"
done
ncu --csv --metrics "$M" ./geom 16777216 256 2 > "$D/raw/ncu-cuda.csv" 2>"$D/raw/ncu-cuda.err"
echo "  -> ncu-cuda.csv (hand-written reference)"

################################################################################
echo "=== 3. kernel time, three kernels, before vs after vs hand-written CUDA ==="
cd "$REPO/demos/15-kernel-time-comparison" || exit 1
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . KernelTimeComparison.java || exit 1
nvcc -arch=sm_120 -O3 -o kernel_time_comparison KernelTimeComparison.cu || exit 1
for arm in before after; do
  p=$([ $arm = before ] && echo "$BEFORE" || echo "$AFTER")
  for rep in 1 2 3; do
    nsys profile --trace=cuda --force-overwrite=true -o "$D/raw/d15-$arm-$rep" \
      "$JAVA_HOME/bin/java" @"$ARGFILE" $p -cp . KernelTimeComparison 4194304 256 20 \
      > "$D/raw/d15-$arm-$rep.log" 2>&1
    nsys stats --report cuda_gpu_kern_sum --format csv \
      -o "$D/raw/d15-$arm-$rep" "$D/raw/d15-$arm-$rep.nsys-rep" >/dev/null 2>&1
  done
  echo "  -> d15-$arm-{1,2,3}"
done
for rep in 1 2 3; do
  nsys profile --trace=cuda --force-overwrite=true -o "$D/raw/d15-cuda-$rep" \
    ./kernel_time_comparison 4194304 256 20 > "$D/raw/d15-cuda-$rep.log" 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv \
    -o "$D/raw/d15-cuda-$rep" "$D/raw/d15-cuda-$rep.nsys-rep" >/dev/null 2>&1
done
echo "  -> d15-cuda-{1,2,3}"

################################################################################
echo "=== 4. buffer-size sweep, before vs after (the PR's headline table) ==="
cd "$REPO/results/nvidia-meeting/task-B2-geometry-controlled" || exit 1
#      1MB      4MB       16MB      64MB       256MB
for n in 262144 1048576 4194304 16777216 67108864; do
  for arm in before after; do
    p=$([ $arm = before ] && echo "$BEFORE" || echo "$AFTER")
    nsys profile --trace=cuda --force-overwrite=true -o "$D/raw/sweep-$arm-$n" \
      "$JAVA_HOME/bin/java" @"$ARGFILE" $p -cp . GeometryControlled "$n" 256 20 \
      > "$D/raw/sweep-$arm-$n.log" 2>&1
    nsys stats --report cuda_gpu_kern_sum --format csv \
      -o "$D/raw/sweep-$arm-$n" "$D/raw/sweep-$arm-$n.nsys-rep" >/dev/null 2>&1
  done
  echo "  -> sweep n=$n"
done

echo "ALL MEASUREMENTS DONE"
