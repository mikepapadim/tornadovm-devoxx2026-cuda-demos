# Reproducing this evidence pack on another GPU

Written for an RTX 5070 (Blackwell), but architecture-agnostic. Follow it on any
CUDA GPU to produce a **parallel** bundle that can be compared against the sm_89
baseline.

## The one rule that must not be broken

**Never merge the new results into `results/nvidia-meeting/`, and never average
across architectures.** Write to `results/nvidia-meeting-<arch>/`, e.g.
`results/nvidia-meeting-sm120/`. Every figure stays labelled with its own GPU.
Two architectures side by side in a table is fine; one number covering both is
not.

Baseline for comparison: **sm_89 / RTX 4090, driver 565.57.01, CUDA 12.6.85,
JDK 25.0.2, TornadoVM 6.0.0-jdk22plus-cuda**, in
[`results/nvidia-meeting/summary.md`](../results/nvidia-meeting/summary.md).

---

## Step 0 — version gates, before anything else

Three checks decide whether the run is even possible. Do them first.

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
nvcc --version | tail -2
nvcc --list-gpu-arch | tail -4          # highest arch the toolkit can target
```

**Gate 1 — does the toolkit know your GPU's arch?** On the sm_89 host,
`nvcc --list-gpu-arch` tops out at `compute_90` (CUDA 12.6.85). **A Blackwell
consumer card is newer than that.** If your `compute_cap` is above the toolkit's
maximum you need a newer CUDA (12.8+ or 13.x) for the *hand-written CUDA* side —
`nvcc -arch=sm_<cc>` will simply fail otherwise.

Confirm what NVRTC itself will accept, which is what TornadoVM asks:

```bash
cat > /tmp/archq.c <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <nvrtc.h>
int main(void){int M,m;nvrtcVersion(&M,&m);printf("nvrtc %d.%d\n",M,m);
int n=0;if(nvrtcGetNumSupportedArchs(&n)!=NVRTC_SUCCESS){puts("no arch query");return 1;}
int*a=malloc(n*sizeof(int));nvrtcGetSupportedArchs(a);
printf("archs(%d): ",n);for(int i=0;i<n;i++)printf("sm_%d%s",a[i],i+1<n?", ":"\n");return 0;}
EOF
gcc -o /tmp/archq /tmp/archq.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvrtc
LD_LIBRARY_PATH=/usr/local/cuda/lib64 /tmp/archq
```

**Gate 2 — Nsight Compute must match the driver.** On the baseline host the
`ncu` on `PATH` (2026.2.1.0) cannot connect to driver 565.57.01 at all, and its
error message blames a "stub libcuda.so", which is misleading. Find one that
works:

```bash
for n in /opt/nvidia/nsight-compute/*/ncu; do
  echo "== $n"; "$n" --version | head -2
  "$n" --metrics smsp__cycles_elapsed.avg /bin/true 2>&1 | grep -E "ERROR|Connected" | head -2
done
```

**Gate 3 — counter permission.**

```bash
grep RmProfilingAdminOnly /proc/driver/nvidia/params   # want 0
# if 1:
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u && sudo reboot
```

## Step 0b — provenance

Nothing is worth collecting without this.

```bash
D=results/nvidia-meeting-<arch>; mkdir -p $D/env
{ date -u +%Y-%m-%dT%H:%M:%SZ
  git -C . log -1 --format='demo-repo=%H'
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.memory,power.limit,persistence_mode --format=csv
  nvcc --version | tail -2; . /etc/os-release; echo "$PRETTY_NAME"; uname -srmo
  java -version 2>&1 | head -1; echo "TORNADOVM_HOME=$TORNADOVM_HOME"
  grep RmProfilingAdminOnly /proc/driver/nvidia/params
} > $D/env/provenance.txt 2>&1
```

## Setup

```bash
sdk install java 25.0.2-open
sdk install tornadovm 6.0.0-jdk22plus-cuda
git clone https://github.com/mikepapadim/tornadovm-devoxx2026-cuda-demos
cd tornadovm-devoxx2026-cuda-demos
source scripts/setup-env.sh      # MUST be run from the repo root — see traps
tornado --devices
```

---

## Step 1 — correctness first (cheapest signal, most likely to find bugs)

```bash
bash scripts/run-all-demos.sh   2>&1 | tee $D/run-all-demos.log   # Java side
bash scripts/run-all-cuda.sh    2>&1 | tee $D/run-all-cuda.log    # CUDA side, no JDK needed
```

**Record every failure with its log. Do not skip past them** — on a new
architecture a failure here is the most valuable result in the whole run.

Demo 12's CUDA side needs CUTLASS, which is not vendored:

```bash
git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git
export CUTLASS_DIR=$PWD/cutlass
```

## Step 2 — alignment sweep (task B)

The highest-value single measurement, and it is self-contained CUDA.

```bash
nvcc -arch=sm_<cc> -O3 -o alignment_sweep results/nvidia-meeting/AlignmentSweep.cu
NCU=<working ncu>
$NCU --csv --metrics \
 l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
 ./alignment_sweep --counters > $D/alignment-sweep-ncu.csv
./alignment_sweep > $D/alignment-sweep-timing.log
```

**sm_89 baseline:** 0/32/64/128 B → 4.00 sectors/request; **16 B → 5.00**.

**What to look for:** the sector size is a hardware property. If Blackwell's
L1↔L2 sector granularity differs, the offset at which the penalty appears will
move, and **the 32-byte conclusion behind
[PR #1066](https://github.com/beehive-lab/TornadoVM/pull/1066) would need
revisiting for that architecture.** This sweep answers it empirically — do not
assume 32.

## Step 3 — geometry-controlled 2×2 (task B2)

Separates the layout effect from the launch-configuration effect.

```bash
cd results/nvidia-meeting/task-B2-geometry-controlled
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . GeometryControlled.java
nvcc -arch=sm_<cc> -O3 -o geom GeometryControlled.cu
M='l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__block_size,launch__registers_per_thread,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.sum'
for b in 256 1024; do
  $NCU --csv --target-processes all --metrics "$M" \
    $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . GeometryControlled 16777216 $b 2 > tvm_$b.csv
  $NCU --csv --metrics "$M" ./geom 16777216 $b 2 > cuda_$b.csv
done
```

**sm_89 baseline:** 1.075 (alignment) × 1.131 (block size) = 1.216, and the
product matches the uncontrolled ratio exactly. Occupancy 78→51% on both sides.
Blackwell has a different SM budget, so the block-size term will differ — that
is expected, and is precisely why it must be measured, not carried over.

## Step 4 — kernel-time baseline under nsys (task A)

```bash
cd demos/15-kernel-time-comparison
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . KernelTimeComparison.java
nvcc -arch=sm_<cc> -O3 -o cuda15 KernelTimeComparison.cu
nsys profile --trace=cuda -o tvm15 $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . KernelTimeComparison 4194304 256 20
nsys profile --trace=cuda -o cuda15 ./cuda15 4194304 256 20
nsys stats --report cuda_gpu_kern_sum --format csv tvm15.nsys-rep
```

**Verify geometry from the trace, not from the source:** both sides must report
`block=256, grid=16384`. If they do not, the comparison is void.

**sm_89 baseline (nsys):** elementwise 1.31, stencil 1.24, polynomial 0.88.

## Step 5 — tensor cores, all five operand types (task D)

```bash
cd demos/16-tensor-core-datatypes
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . TensorCoreDataTypes.java
tornado --printKernel --classpath . TensorCoreDataTypes > d16-printkernel.log
$NCU --csv --target-processes all --metrics \
 sm__inst_executed_pipe_tensor_op_hmma.sum,sm__inst_executed_pipe_tensor_op_imma.sum,\
sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.sum \
 $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . TensorCoreDataTypes > d16-ncu.csv
grep -oE 'mma\.sync\.aligned\.[a-z0-9.]+' d16-printkernel.log | sort | uniq -c
```

**sm_89 baseline:** all four PASS at max abs err 0.00000; BF16 4 HMMA, int8
2 **IMMA**, FP8 e4m3/e5m2 2 HMMA each. Plus demo 08: 1 HMMA / 16 cycles, scalar
control 0.

**These counter names may not exist on Blackwell.** If `ncu` rejects a metric,
list what is available and record the substitution:

```bash
$NCU --query-metrics 2>/dev/null | grep -iE "tensor|hmma|imma" | head -20
```

## Step 6 — host dispatch and CUDA Graph (task E)

Per-execution cost **must** be obtained by differencing two execution counts.
Dividing a total by iterations overstates it by ~12× — that mistake is recorded
in `results/raw/25-host-dispatch-breakdown/MANIFEST.md`.

```bash
cd demos/07-cuda-graph-benefit
javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . CudaGraphBenefit.java
for m in nograph graph; do for e in 10 100; do
  nsys profile --trace=cuda --sample=none --cpuctxsw=none -o d07_${m}_$e \
    $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . CudaGraphBenefit 4096 6 $e $m
  nsys stats --report cuda_api_sum --format csv d07_${m}_$e.nsys-rep > api_${m}_$e.csv
done; done
```

Buffer reuse across `JIT → libraryTask → JIT` (demo 13): export the trace to
SQLite and histogram `CUPTI_ACTIVITY_KIND_MEMCPY` by `bytes`. Intermediates must
not appear.

**sm_89 baseline:** nograph 6 launches + 7 H2D + 42 event calls per execution
(33,774 ns host API excl. sync); graph 1 `cuGraphLaunch` + 3 event calls
(6,283 ns) — **5.4× less**.

## Step 7 — CUDA Tile inventory (task G)

**Do not implement CUDA Tile. Inventory only, and make no performance claim.**

```bash
find /usr/local/cuda*/include -maxdepth 2 -iname '*tile*' | wc -l
find /usr/local/cuda*/lib64 /usr/local/cuda*/bin -maxdepth 1 -iname '*tile*' | wc -l
nvcc --help 2>/dev/null | grep -ci tile
nm -D /usr/local/cuda/lib64/libnvrtc.so.12 2>/dev/null | grep -ci tile
nm -D /usr/local/cuda/lib64/libnvrtc.so.* | awk '$2=="T"{print $3}' | sort
```

**sm_89 baseline (CUDA 12.6.85): all four counts were 0.** NVRTC did export
`nvrtcGetNVVM` and `nvrtcGetLTOIR`.

**This is the highest-value delta if the 5070 host has a newer toolkit.** CUDA
12.8+ or 13.x may ship a Tile path that 12.6 does not. A non-zero count here
changes the whole feasibility conversation — capture the full file list.

## Step 8 — SASS (task C)

TornadoVM writes every compiled cubin to disk **by default**:

```bash
ls "$TORNADOVM_HOME/var/cuda-codecache/device-0-0/"
cuobjdump -sass "$TORNADOVM_HOME/var/cuda-codecache/device-0-0/polynomial-<key>.cubin" > $D/tvm-polynomial-sass.txt
```

**sm_89 baseline:** `polynomial` with `degree=256` → 288 instructions, **256
FFMA, 1 branch** (fully unrolled). nvcc runtime-scalar variant: 13 branches.

**If the toolkit had to fall back to PTX** (Gate 1), the cache may hold no cubin
for that arch — record that as a finding, it is exactly the fallback path
described below.

---

## Known traps — each of these cost time on the baseline host

1. **`source scripts/setup-env.sh` must be run from the repo root.** Anywhere
   else it silently leaves `JAVA_HOME` at SDKMAN's `current` (JDK 21 here) and
   every later command dies with `UnsupportedClassVersionError`.
2. **Under `nsys` and `ncu`, do not use the `tornado` launcher.** It resolves a
   different JDK. Use
   `$JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . <Main>`.
3. **`ncu` on `PATH` may be too new for the driver** — see Gate 2.
4. **`tornado.cuda.codecache.dir` is concatenated onto `$TORNADOVM_HOME`**, not
   used as an absolute path. `-Dtornado.cuda.codecache.dir=/tmp/cc` writes to
   `$TORNADOVM_HOME/tmp/cc/device-0-0`.
5. **`ncu` and `nsys` disagree on *ratios*, not just absolute times.** Same
   kernels, same geometry: 1.02–1.04 under `ncu` vs 1.24–1.31 under `nsys`.
   Never mix them, and never put them on one axis.
6. **The TornadoVM profiler perturbs its own measurement.** Use it for the
   cold/warm compile split, not for absolute steady-state timings.
7. **Demo 12's wall clock cannot compare fused vs unfused** — both plans run in
   one JVM and the first absorbs process start-up.

---

## Where bugs are most likely on a newer architecture

Concrete, checkable hypotheses. Each is a real finding whether it holds or not.

### 1. PTX fallback instead of cubin — *most likely, and benign if it works*

`ffm/CUDACompiler` asks NVRTC which archs it supports. If your GPU's arch is not
among them it does **not** fail: it emits `--gpu-architecture=compute_<best
known>` and lets the driver JIT the PTX, warning once. So a Blackwell card on
CUDA 12.6 is expected to **run via PTX fallback**, not to break.

- Confirm from the log which path was taken (look for the fallback warning).
- Then confirm the **MMA demos still validate** — the inline `mma.sync` PTX is
  embedded in the generated CUDA C and must be legal for the *virtual* arch it
  is compiled against. This is the most plausible place for a genuine failure.
- Compare load time and first-execution latency against the cubin path.

### 2. Tensor-core counter names

Blackwell may rename or drop `sm__inst_executed_pipe_tensor_op_{hmma,imma}`.
If a metric is rejected, that is a methodology finding, not a failed run —
record the substitution.

### 3. Sector granularity

The alignment sweep assumes nothing; it measures. If the penalty appears at a
different offset, **PR #1066's 32-byte default needs revisiting for that arch**.

### 4. FP8 / MMA gating

`CUDATensorCoreSupportPhase` requires compute capability ≥ 8.9 and NVRTC ≥ 12.4
for FP8. Both should pass on Blackwell — if FP8 is *rejected*, the gate logic is
wrong for newer arches and that is a bug worth filing.

### 5. `MMAShape` ceiling

`{M16N8K16, M16N8K32}` are Ampere/Ada-era shapes. They remain valid on newer
hardware, but `wgmma` (sm_90) and `tcgen05` (sm_100) are unreachable. Expect
correct-but-not-optimal tensor-core use — worth quantifying, not filing.

### 6. Already-known bugs — should reproduce anywhere

| Issue | Expected on any arch |
|---|---|
| [#1067](https://github.com/beehive-lab/TornadoVM/issues/1067) | a `KernelContext` kernel that fails to compile silently returns **wrong results**; reproducer in the issue |
| [#1063](https://github.com/beehive-lab/TornadoVM/issues/1063) | `CuDnn.sdpaForward` launches no kernel |
| [#1064](https://github.com/beehive-lab/TornadoVM/issues/1064) | CUDA lowering crash on a ternary before an allocation |

If any of these **does not** reproduce, that is itself informative.

---

## What to send back

```
results/nvidia-meeting-<arch>/
├── env/provenance.txt          ← without this nothing else is usable
├── run-all-demos.log, run-all-cuda.log
├── alignment-sweep-{ncu.csv,timing.log}
├── task-B2/{tvm,cuda}_{256,1024}.csv
├── task-A/*.nsys-rep + kern_sum CSVs
├── task-D/d16-{printkernel.log,ncu.csv}
├── task-E/api_{nograph,graph}_{10,100}.csv + demo13 trace
├── tile-feasibility/inventory.txt
├── task-C/tvm-polynomial-sass.txt
└── failures/                   ← every non-zero exit, with its log
```

Then produce a **side-by-side** table per metric, one row per architecture,
never a merged average. Populate the sm_89 column from
`results/nvidia-meeting/summary.md` and `manifest.json`.

A failure with a log is worth more than a missing measurement. Record it.
