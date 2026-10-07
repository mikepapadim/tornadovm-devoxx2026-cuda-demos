#!/usr/bin/env bash
# CUDA Tile from Java, the fancy version of demoTile.sh (slides 17-19): threads, tiles and cuBLAS in one CUDA graph,
# then the FP16 GEMM ladder. Same programs (demos 20 and 25), drawn as a pipeline and a bar chart. About 15 seconds.
#   bash fancyTile.sh            NO_PAUSE=1 to skip the [enter] between acts
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$DEMO_ROOT/fancy.sh"
use_tornadovm_7
# a tile kernel that fails to compile would otherwise fall back to the CPU and still print "correct"
BAILOUT="-Dtornado.recover.bailout=False"

$FANCY banner "CUDA Tile from Java · TileContext" "TornadoVM 7.2.0 · RTX 4090 · no CPU fallback: a tile kernel that fails to compile fails the demo"

$FANCY act 1 "Threads, tiles, cuBLAS: one CUDA graph" "a KernelContext kernel, a TileContext GEMM, cublasSgemv and a @Parallel loop, captured once   (slide 18)"
compile_demo 20-cutile-hybrid
(cd "$BUILD/20-cutile-hybrid" && spin "capturing and replaying the pipeline" "$FANCY_LOGS/pipeline.log" \
    tornado --printBytecodes --jvm="$BAILOUT" --classpath . TileHybridPipeline 256 2 graph)
$FANCY tile-pipeline "$FANCY_LOGS/pipeline.log"
fancy_pause

$FANCY act 2 "Ten lines of tiles vs a hand-tuned kernel" "FP16 GEMM, n = 2048: KernelContext, four tile shapes, a launch hint and cuBLAS, each checked   (slide 19)"
compile_demo 25-tile-ladder
(cd "$BUILD/25-tile-ladder" && spin "8 rungs x 30 executions" "$FANCY_LOGS/ladder.log" \
    tornado --jvm="$BAILOUT" --classpath . TileLadder 2048 30)
$FANCY tile-ladder "$FANCY_LOGS/ladder.log"

$FANCY scoreboard "CUDA Tile from Java on the RTX 4090"
