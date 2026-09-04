#!/usr/bin/env bash
# Repeat of the buffer-size sweep with the two arms INTERLEAVED and repeated.
#
# The first pass ran all of "before" then all of "after", one run per point, which
# cannot separate a real effect from thermal drift or run-to-run variance. Here the
# arms alternate within each repetition and each point is measured REPS times, so
# drift affects both arms equally and the spread is visible.
set -o pipefail
SDK=${1:?usage: run-sweep-repeated.sh <tornadovm-sdk-home>}
REPS=${2:-5}
cd "$(dirname "$0")/../.." || exit 1
REPO=$PWD
D=$REPO/results/pr1066-sm120
export PATH=/usr/local/cuda-13.0/bin:$PATH
export JAVA_HOME=$HOME/.sdkman/candidates/java/25.0.2-open
export TORNADOVM_HOME=$SDK
export PATH=$JAVA_HOME/bin:$TORNADOVM_HOME/bin:$PATH
ARGFILE=$TORNADOVM_HOME/tornado-argfile
mkdir -p "$D/raw-repeat"
cd "$REPO/results/nvidia-meeting/task-B2-geometry-controlled" || exit 1

for rep in $(seq 1 "$REPS"); do
  for n in 262144 1048576 4194304 16777216 67108864; do
    for arm in before after; do
      p=$([ "$arm" = before ] && echo "-Dtornado.cuda.payloadAlignment=1" || echo "")
      tag="$D/raw-repeat/n${n}-${arm}-r${rep}"
      # shellcheck disable=SC2086
      nsys profile --trace=cuda --force-overwrite=true -o "$tag" \
        "$JAVA_HOME/bin/java" @"$ARGFILE" $p -cp . GeometryControlled "$n" 256 20 \
        > "$tag.log" 2>&1
      nsys stats --report cuda_gpu_kern_sum --format csv -o "$tag" "$tag.nsys-rep" >/dev/null 2>&1
    done
  done
  echo "rep $rep done ($(date +%T))"
done
echo "REPEATED SWEEP DONE"
