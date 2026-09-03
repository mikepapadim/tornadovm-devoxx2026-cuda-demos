#!/usr/bin/env python3
"""
Consolidate every Nsight Compute / Nsight Systems capture under results/raw/
into one tidy, chart-ready CSV.

Output: results/raw/27-profiler-metrics/metrics.csv, long format:

    demo,kernel,implementation,metric,unit,value

Long format on purpose: one row per measurement, so a plotting tool can pivot
on whatever axis a given figure needs without the file being reshaped first.

Usage:  python3 scripts/build-metrics-csv.py
"""
import csv, io, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
OUT = os.path.join(RAW, "27-profiler-metrics", "metrics.csv")

# Short, slide-friendly names for the ncu metrics we actually chart.
METRIC_LABEL = {
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio": ("sectors_per_request_load", "sectors"),
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio": ("sectors_per_request_store", "sectors"),
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": ("global_load_sectors", "sectors"),
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": ("global_store_sectors", "sectors"),
    "l1tex__data_bank_conflicts_pipe_lsu.sum": ("bank_conflicts", "count"),
    "smsp__inst_executed.sum": ("instructions_executed", "count"),
    "gpu__time_duration.sum": ("kernel_time_ncu", "ns"),
    "dram__bytes_read.sum": ("dram_bytes_read", "bytes"),
    "dram__bytes_write.sum": ("dram_bytes_write", "bytes"),
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": ("dram_throughput_pct_peak", "percent"),
    "sm__warps_active.avg.pct_of_peak_sustained_active": ("achieved_occupancy", "percent"),
    "sm__inst_executed_pipe_tensor_op_hmma.sum": ("hmma_instructions", "count"),
    "sm__inst_executed_pipe_tensor_op_imma.sum": ("imma_instructions", "count"),
    "sm__inst_executed_pipe_tensor.sum": ("tensor_instructions", "count"),
    "sm__pipe_tensor_cycles_active.sum": ("tensor_pipe_cycles", "cycles"),
    "sm__pipe_tensor_op_hmma_cycles_active.sum": ("tensor_pipe_cycles", "cycles"),
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct": ("bytes_used_per_sector", "percent"),
}

# ncu CSV captures: (path, demo, implementation)
NCU_FILES = [
    ("27-profiler-metrics/ncu-tornado01.csv", "01-vector-add", "TornadoVM"),
    ("27-profiler-metrics/ncu-cuda01.csv", "01-vector-add", "CUDA"),
    ("23-ncu-tensor-core-counters/ncu-tornado08.csv", "08-tensor-core", "TornadoVM"),
    ("23-ncu-tensor-core-counters/ncu-cuda08.csv", "08-tensor-core", "CUDA"),
    ("24-ncu-demo14-counters/ncu-tornado14.csv", "14-warp-async-shared", "TornadoVM"),
    ("24-ncu-demo14-counters/ncu-cuda14.csv", "14-warp-async-shared", "CUDA"),
    ("22-ncu-alignment-counters/ncu-tornado15.csv", "15-kernel-time", "TornadoVM"),
    ("22-ncu-alignment-counters/ncu-cuda15.csv", "15-kernel-time", "CUDA"),
    ("22-ncu-alignment-counters/ncu-probe-offsets.csv", "15-alignment-probe", "CUDA"),
    ("26-tensor-core-datatypes/16-datatypes-ncu.csv", "16-tensor-core-datatypes", "TornadoVM"),
]

# nsys cuda_gpu_kern_sum captures: (path, demo, implementation)
NSYS_KERN = [
    ("25-host-dispatch-breakdown/warp14-cuda_gpu_kern_sum.csv", "14-warp-async-shared", "TornadoVM"),
    ("25-host-dispatch-breakdown/cuda14-cuda_gpu_kern_sum.csv", "14-warp-async-shared", "CUDA"),
    ("21-kernel-time-comparison/tornado-nsys-kernsum.csv", "15-kernel-time", "TornadoVM"),
    ("21-kernel-time-comparison/cuda-nsys-kernsum.csv", "15-kernel-time", "CUDA"),
]

# nsys cuda_api_sum captures, for host-side dispatch cost
NSYS_API = [
    ("25-host-dispatch-breakdown/warp14-cuda_api_sum.csv", "14-warp-async-shared", "TornadoVM"),
    ("25-host-dispatch-breakdown/cuda14-cuda_api_sum.csv", "14-warp-async-shared", "CUDA"),
]


def num(s):
    s = (s or "").strip().strip('"').replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def read_csv_after_header(path, first_col):
    """ncu and nsys both prepend noise before the real CSV header."""
    if not os.path.exists(path):
        return []
    lines = open(path, errors="replace").read().splitlines()
    idx = next((i for i, l in enumerate(lines) if l.lstrip().startswith(first_col)), None)
    if idx is None:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(lines[idx:]))))


def clean_kernel(name):
    return re.sub(r"\(.*", "", (name or "").strip()).strip() or "unknown"


rows = []

for rel, demo, impl in NCU_FILES:
    for r in read_csv_after_header(os.path.join(RAW, rel), '"ID"'):
        label = METRIC_LABEL.get(r.get("Metric Name", ""))
        if not label:
            continue
        v = num(r.get("Metric Value"))
        if v is None:
            continue
        name, unit = label
        rows.append({
            "demo": demo, "kernel": clean_kernel(r.get("Kernel Name")),
            "implementation": impl, "metric": name, "unit": unit,
            "value": v, "source": "ncu", "file": rel,
        })

for rel, demo, impl in NSYS_KERN:
    for r in read_csv_after_header(os.path.join(RAW, rel), "Time (%)"):
        v = num(r.get("Avg (ns)"))
        if v is None:
            continue
        rows.append({
            "demo": demo, "kernel": clean_kernel(r.get("Name")),
            "implementation": impl, "metric": "kernel_time_nsys", "unit": "ns",
            "value": v, "source": "nsys", "file": rel,
        })

for rel, demo, impl in NSYS_API:
    for r in read_csv_after_header(os.path.join(RAW, rel), "Time (%)"):
        calls, total = num(r.get("Num Calls")), num(r.get("Total Time (ns)"))
        if calls is None or total is None:
            continue
        for metric, unit, value in (("api_calls", "count", calls),
                                    ("api_total_time", "ns", total)):
            rows.append({
                "demo": demo, "kernel": (r.get("Name") or "").strip(),
                "implementation": impl, "metric": metric, "unit": unit,
                "value": value, "source": "nsys-api", "file": rel,
            })

# Deduplicate repeated launches of the same kernel by averaging: ncu emits one
# row per launch, and a chart wants one number per (demo, kernel, impl, metric).
agg = {}
for r in rows:
    k = (r["demo"], r["kernel"], r["implementation"], r["metric"])
    agg.setdefault(k, {"row": r, "vals": []})["vals"].append(r["value"])

out = []
for (demo, kernel, impl, metric), d in sorted(agg.items()):
    r = d["row"]
    vals = d["vals"]
    out.append({
        "demo": demo, "kernel": kernel, "implementation": impl,
        "metric": metric, "unit": r["unit"],
        "value": round(sum(vals) / len(vals), 4),
        "launches": len(vals), "source": r["source"], "file": r["file"],
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["demo", "kernel", "implementation", "metric",
                                       "unit", "value", "launches", "source", "file"])
    w.writeheader()
    w.writerows(out)

# Second output: paired TornadoVM-vs-CUDA rows with the ratio precomputed.
# metrics.csv is the full record; this is the file a bar chart reads directly.
COMPARISON = os.path.join(RAW, "27-profiler-metrics", "comparison.csv")
pairs = {}
for r in out:
    k = (r["demo"], r["kernel"], r["metric"], r["unit"])
    pairs.setdefault(k, {})[r["implementation"]] = r["value"]

comp = []
for (demo, kernel, metric, unit), v in sorted(pairs.items()):
    if "TornadoVM" not in v or "CUDA" not in v:
        continue
    t, c = v["TornadoVM"], v["CUDA"]
    comp.append({
        "demo": demo, "kernel": kernel, "metric": metric, "unit": unit,
        "tornadovm": t, "cuda": c,
        # >1 means TornadoVM's number is larger. For time/sectors/instructions
        # that is worse; for throughput and occupancy it is better. The README
        # says which is which -- do not read the column as "slowdown".
        "ratio_tornadovm_over_cuda": round(t / c, 4) if c else "",
    })

with open(COMPARISON, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["demo", "kernel", "metric", "unit",
                                       "tornadovm", "cuda", "ratio_tornadovm_over_cuda"])
    w.writeheader()
    w.writerows(comp)

print(f"wrote {OUT}: {len(out)} rows")
print(f"wrote {COMPARISON}: {len(comp)} paired rows")
demos = sorted({r['demo'] for r in out})
print("demos:", ", ".join(demos))
print("metrics:", ", ".join(sorted({r['metric'] for r in out})))
