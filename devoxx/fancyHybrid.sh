#!/usr/bin/env bash
# Hybrid API, the fancy version of demoHybrid.sh (slides 14-16): Java kernels and cuBLAS in one task graph, then a
# CUDA graph. Same programs (demos 04 and 07), drawn as pipelines and bar charts. About 10 seconds.
#   bash fancyHybrid.sh          NO_PAUSE=1 to skip the [enter] between acts
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$DEMO_ROOT/fancy.sh"
use_tornadovm_7

$FANCY banner "Hybrid API · Java kernels + NVIDIA libraries, one task graph" "TornadoVM 7.2.0 · RTX 4090 · every result checked against the CPU"

$FANCY act 1 "Java → cuBLAS → Java" "two Java kernels JIT-compiled to CUDA, cublasSgemv in between, sharing device buffers   (slide 15)"
compile_demo 04-cublas-hybrid
(cd "$BUILD/04-cublas-hybrid" && spin "running the task graph 3 times, profiler on" "$FANCY_LOGS/hybrid.log" \
    tornado --enableProfiler console --classpath . CuBlasSgemvHybrid 8 8 3)
$FANCY hybrid-tasks "$FANCY_LOGS/hybrid.log"
fancy_pause

$FANCY act 2 "Capture once, replay" "plan.withCUDAGraph(): the first run captures the graph, every later run is one cuGraphLaunch   (slide 16)"
compile_demo 07-cuda-graph-benefit
(cd "$BUILD/07-cuda-graph-benefit" && spin "50 executions, plain and as a CUDA graph" "$FANCY_LOGS/graph.log" \
    tornado --classpath . CudaGraphBenefit 4096 6 50 both)
$FANCY hybrid-graph "$FANCY_LOGS/graph.log"

$FANCY scoreboard "Hybrid API on the RTX 4090"
