#!/usr/bin/env bash
#
# Collect kernel-level numbers for a matmul ladder demo and print the speedups.
#
#   bash scripts/compare-ladder.sh 17 [n] [executions]    # FP32 ladder
#   bash scripts/compare-ladder.sh 18 [n] [executions]    # FP16 ladder
#   bash scripts/compare-ladder.sh 22 [n] [executions]    # FP16 ladder + CUDA Tile rung
#
# Demo 22 needs TORNADOVM_HOME pointing at a cuTile-branch build, not the pinned
# 6.0.0 SDK; that build is jdk21-dev, hence the extra preview flags below.
#
# Wall clock on these demos is dominated by host dispatch -- at small n a slower
# kernel can look faster. This runs each ladder under Nsight Systems for
# per-kernel GPU time and under Nsight Compute for the counters that explain the
# ranking, then prints both.
#
# The two are reported SEPARATELY and must not be compared with each other: ncu
# serialises launches, flushes caches and disallows clock boost, so its absolute
# times run several times higher and its ratios differ from nsys ratios on the
# same kernels. Use nsys for speed, ncu for why.
set -uo pipefail

DEMO="${1:-17}"
N="${2:-2048}"
EXECUTIONS="${3:-20}"

case "$DEMO" in
    17) DIR="demos/17-matmul-ladder";      MAIN="MatMulLadder";      LABEL="FP32" ;;
    18) DIR="demos/18-matmul-ladder-fp16"; MAIN="MatMulLadderFP16";  LABEL="FP16" ;;
    22) DIR="demos/22-matmul-ladder-fp16-tile"; MAIN="MatMulLadderFP16Tile"; LABEL="FP16+tile"
        JAVAC_FLAGS="--release 21 --enable-preview -proc:none"; JAVA_FLAGS="--enable-preview" ;;
    *)  echo "[ERROR] demo must be 17 (FP32), 18 (FP16) or 22 (FP16 + CUDA Tile); got '$DEMO'" >&2; exit 2 ;;
esac
JAVAC_FLAGS="${JAVAC_FLAGS:-}"
JAVA_FLAGS="${JAVA_FLAGS:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${TORNADOVM_HOME:-}" ] || [ -z "${JAVA_HOME:-}" ]; then
    echo "[ERROR] run 'source scripts/setup-env.sh' from the repo root first." >&2
    exit 1
fi

# Nsight Compute: the one on PATH may be newer than the driver and unable to
# connect. Prefer a version that actually works over whichever is first.
NCU=""
for candidate in /opt/nvidia/nsight-compute/*/ncu "$(command -v ncu 2>/dev/null)"; do
    [ -x "$candidate" ] || continue
    if "$candidate" --metrics smsp__cycles_elapsed.avg /bin/true >/dev/null 2>&1; then
        NCU="$candidate"; break
    fi
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "== $LABEL matmul ladder (demo $DEMO), n=$N, $EXECUTIONS executions"
echo "   GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo unknown)"
echo

echo "-- building"
# shellcheck disable=SC2086  -- JAVAC_FLAGS must word-split
javac $JAVAC_FLAGS -cp "$TORNADOVM_HOME/share/java/tornado/*" -d "$WORK" "$DIR/$MAIN.java" || exit 1

echo "-- Nsight Systems: per-kernel GPU time"
# The tornado launcher resolves a different JDK under nsys; call java directly.
( cd "$WORK" && nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o ladder \
    "$JAVA_HOME/bin/java" @"$TORNADOVM_HOME/tornado-argfile" $JAVA_FLAGS -cp . "$MAIN" "$N" "$EXECUTIONS" \
    > run.log 2>&1 )
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "$WORK/ladder.nsys-rep" 2>/dev/null \
    | grep -E '^[0-9]' > "$WORK/kern.csv"

if [ ! -s "$WORK/kern.csv" ]; then
    echo "[ERROR] no kernel data captured; see $WORK/run.log" >&2
    sed -n '/Exception\|Error/p' "$WORK/run.log" | head -5 >&2
    exit 1
fi

N_VAL="$N" python3 - "$WORK/kern.csv" <<'PY'
import csv, os, re, sys
n = int(os.environ["N_VAL"])
gflop = 2.0 * n * n * n / 1e9
H = "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name".split(",")
rows = []
for r in csv.reader(open(sys.argv[1])):
    d = dict(zip(H, r))
    name = re.sub(r"\(.*", "", d["Name"]).strip().strip('"')
    # CUTLASS and cuBLAS kernels carry long templated names; keep them readable.
    if "cutlass" in name and "tensorop" in name: short = "cuBLAS/CUTLASS tensorop"
    elif "cutlass" in name and "simt" in name:   short = "cuBLAS/CUTLASS simt"
    elif "cutlass" in name:                       short = "CUTLASS"
    elif len(name) > 34:                          short = name[:31] + "..."
    else:                                         short = name
    rows.append((short, float(d["Avg (ns)"])))
if not rows:
    sys.exit("no rows")
rows.sort(key=lambda x: -x[1])
slowest = rows[0][1]
print()
print(f"   {'kernel':<34}{'GPU avg':>12}{'GFLOP/s':>12}{'vs slowest':>12}")
print(f"   {'-'*34}{'-'*12}{'-'*12}{'-'*12}")
for name, avg in rows:
    print(f"   {name:<34}{avg/1000:>10.1f}us{gflop/(avg/1e9):>12.0f}{slowest/avg:>11.1f}x")
print()
print("   nsys steady-state kernel time. Wall clock is higher: it includes host dispatch.")
PY

if [ -n "$NCU" ]; then
    echo
    echo "-- Nsight Compute: why (counters, NOT comparable with the times above)"
    M='gpu__time_duration.sum,smsp__inst_executed.sum,launch__registers_per_thread'
    M="$M,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu.sum"
    M="$M,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
    M="$M,smsp__issue_active.avg.pct_of_peak_sustained_active"
    ( cd "$WORK" && "$NCU" --csv --target-processes all --launch-count 12 --metrics "$M" \
        "$JAVA_HOME/bin/java" @"$TORNADOVM_HOME/tornado-argfile" $JAVA_FLAGS -cp . "$MAIN" 256 2 2>/dev/null ) \
        > "$WORK/ncu.csv"
    python3 - "$WORK/ncu.csv" <<'PY'
import csv, io, sys, collections, re
lines = open(sys.argv[1], errors="replace").read().splitlines()
i = next((j for j, l in enumerate(lines) if l.startswith('"ID"')), None)
if i is None:
    sys.exit("   (no counter data captured)")
d = collections.OrderedDict()
for r in csv.DictReader(io.StringIO("\n".join(lines[i:]))):
    k = re.sub(r"\(.*", "", r["Kernel Name"]).strip()
    d.setdefault(k, {})[r["Metric Name"]] = r["Metric Value"].replace(",", "")
SHORT = {
    "smsp__inst_executed.sum": "instrs",
    "launch__registers_per_thread": "regs",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": "spill",
    "l1tex__data_bank_conflicts_pipe_lsu.sum": "bank-conf",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "gmem-stall",
    "smsp__issue_active.avg.pct_of_peak_sustained_active": "issue%",
}
order = list(SHORT)
print(f"   {'kernel':<26}" + "".join(f"{SHORT[m]:>12}" for m in order))
print(f"   {'-'*26}" + "-" * (12 * len(order)))
for k, m in d.items():
    name = k if len(k) <= 26 else k[:23] + "..."
    print(f"   {name:<26}" + "".join(f"{m.get(x,'-'):>12}" for x in order))
print()
print("   spill=0 means private arrays stayed in registers. gmem-stall is the")
print("   global-memory dependency stall per issue-active; issue% is the issue")
print("   rate. A high gmem-stall with a low issue% is latency that is not hidden.")
PY
else
    echo
    echo "-- Nsight Compute: skipped (no ncu able to connect to this driver)"
fi
