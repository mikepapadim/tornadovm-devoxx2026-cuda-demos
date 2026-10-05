#!/usr/bin/env python3
"""A live dashboard for TornadoVM's Hybrid API: one TaskGraph mixing NVIDIA cuFFT library tasks with Java kernels
JIT-compiled by TornadoVM, on shared device buffers. It runs HybridLive.java and draws, frame by frame:

  * the pipeline, host -> [cuFFT forward] -> [lowPass, Java] -> [cuFFT inverse] -> [normalize, Java] -> host, each
    task colored by kind (NVIDIA library / Java JIT) with its GPU time from TornadoVM's profiler;
  * the signal: the noisy input, the spectrum after the Java filter (kept and removed bins), the clean output;
  * what crossed PCIe (bytes in / out per execution), and every frame's check against the exact answer;
  * at the end, the same graph replayed as a CUDA graph.

    dashboard.py <workdir>     run by run.sh; NO_PAUSE=1 exits at the end without waiting for Enter

Needs TORNADOVM_HOME and JAVA_HOME (scripts/setup-env.sh).

The numbers come from the program's output (TornadoVM's profiler and timers); nothing is replayed. Standard library.
"""
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = sys.argv[1]
os.makedirs(WORK, exist_ok=True)

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, BLUE, YELLOW, CYAN, MAGENTA, RED, GREY, WHITE = (
    "\033[1;38;5;83m", "\033[1;38;5;75m", "\033[1;38;5;221m", "\033[1;38;5;51m", "\033[1;38;5;207m",
    "\033[1;38;5;196m", "\033[38;5;240m", "\033[1;38;5;231m")
LIBRARY, JAVA = GREEN, BLUE


def fit(s, width):
    """Pad or cut a string with color codes to exactly `width` visible columns."""
    out, n, i = [], 0, 0
    while i < len(s) and n < width:
        m = re.match(r"\033\[[0-9;]*m", s[i:])
        if m:
            out.append(m.group(0))
            i += len(m.group(0))
            continue
        out.append(s[i])
        n += 1
        i += 1
    return "".join(out) + RESET + " " * (width - n)


# --- state, filled from the program's output --------------------------------------------------------------------
TASKS = [  # (profiler key, label, kind)
    ("hybrid.forward", "cuFFT forward", "NVIDIA cuFFT library"),
    ("hybrid.lowPass", "lowPass", "Java @Parallel, JIT"),
    ("hybrid.inverse", "cuFFT inverse", "NVIDIA cuFFT library"),
    ("hybrid.normalize", "normalize", "Java @Parallel, JIT"),
]
lock = threading.Lock()
S = {"input": [], "output": [], "spec": [], "cutoff": None, "frame": -1, "wall": [], "correct": 0, "wrong": 0,
     "tasks": {}, "task_hist": {k: [] for k, _, _ in TASKS}, "copy_in": 0, "copy_out": 0, "max_err": 0.0,
     "graph": None, "phase": "starting", "started": time.time(), "message": "", "pulse": 0.0}
done = threading.Event()


# --- drawing helpers ------------------------------------------------------------------------------------------
def braille_plot(values, width, height, color, lo=-2.5, hi=2.5):
    """A line plot in braille dots: 2 x 4 dots per character."""
    if not values:
        return [" " * width] * height
    w, h = width * 2, height * 4
    grid = [[0] * width for _ in range(height)]
    bits = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]]
    prev = None
    for x in range(w):
        v = values[min(len(values) - 1, x * len(values) // w)]
        y = int((hi - max(lo, min(hi, v))) / (hi - lo) * (h - 1))
        for yy in (range(min(prev, y), max(prev, y) + 1) if prev is not None else [y]):
            grid[yy // 4][x // 2] |= bits[yy % 4][x % 2]
        prev = y
    zero = int((hi - 0) / (hi - lo) * (h - 1)) // 4
    lines = []
    for r, row in enumerate(grid):
        chars = "".join(chr(0x2800 + c) if c else ("·" if r == zero and x % 4 == 0 else " ") for x, c in enumerate(row))
        lines.append(f"{color}{chars}{RESET}")
    return lines


def spectrum_plot(mags, cutoff, width, height, ghost=None):
    """Bars per frequency band on a log axis (bin 1 .. 1200): what the Java filter kept in green; what it removed as
    grey ghosts, from the unfiltered spectrum of the first frame."""
    if not mags:
        return [" " * width] * height
    cols = []
    for c in range(width):
        a = int(round(math.exp(math.log(1) + (math.log(len(mags)) - math.log(1)) * c / width)))
        b = max(a + 1, int(round(math.exp(math.log(len(mags)) * (c + 1) / width))))
        removed = cutoff is not None and a >= cutoff
        cols.append((max((ghost if removed and ghost else mags)[a:b] or [0]), a))
    top = max([m for m, _ in cols] + [max(ghost) if ghost else 0]) or 1
    blocks = " ▁▂▃▄▅▆▇█"
    lines = []
    for r in range(height):
        row = []
        for m, a in cols:
            level = (m / top) ** 0.5 * height * 8 - (height - 1 - r) * 8
            ch = blocks[max(0, min(8, int(level)))] if level > 0 else " "
            removed = cutoff is not None and a >= cutoff
            row.append(f"{GREY if removed else GREEN}{ch}{RESET}")
        lines.append("".join(row))
    # the cutoff marker on the axis
    axis = []
    for c, (_, a) in enumerate(cols):
        if cutoff is not None and a >= cutoff and (c == 0 or cols[c - 1][1] < cutoff):
            axis.append(f"{YELLOW}▲{RESET}")
        else:
            axis.append(f"{GREY}─{RESET}")
    return lines + ["".join(axis)]


def bar(value, top, width, color):
    n = max(1 if value > 0 else 0, round(value / top * width)) if top else 0
    return f"{color}{'█' * n}{RESET}{GREY}{'·' * (width - n)}{RESET}"


# --- the screen -----------------------------------------------------------------------------------------------
def pipeline(width):
    """host -> [4 tasks inside a GPU frame] -> host, each task with its kind and this frame's GPU time."""
    with lock:
        tasks, pulse, phase = dict(S["tasks"]), S["pulse"], S["phase"]
    box_w = 22
    active = int(pulse) % 6 if phase == "sweep" else -1  # the flow of one execution: in, 4 tasks, out
    boxes = []
    for i, (key, label, kind) in enumerate(TASKS):
        color = LIBRARY if "NVIDIA" in kind else JAVA
        lit = active == i + 1
        us = tasks.get(key)
        edge, (tl, tr, bl, br), side = ("━", "┏┓┗┛", "┃") if lit else ("─", "┌┐└┘", "│")
        timing = f"{us:7.1f} µs on GPU" if us is not None else "      …"
        boxes.append([f"{color}{tl}{edge * box_w}{tr}{RESET}",
                      f"{color}{side}{RESET}{fit(f' {BOLD}{i + 1} {label}', box_w)}{color}{side}{RESET}",
                      f"{color}{side}{RESET}{fit(f' {DIM}{kind}', box_w)}{color}{side}{RESET}",
                      f"{color}{side}{RESET}{fit(f' {color}{timing}', box_w)}{color}{side}{RESET}",
                      f"{color}{bl}{edge * box_w}{br}{RESET}"])
    host_in = [f"{WHITE}host{RESET}", f"{DIM}signal{RESET}", "", f"{YELLOW}{'━━▶' if active == 0 else '──▷'}{RESET}", ""]
    host_out = ["", f"{WHITE}host{RESET}", f"{DIM}output{RESET}", f"{YELLOW}{'━━▶' if active == 5 else '──▷'}{RESET}", ""]
    with lock:
        cin, cout = S["copy_in"], S["copy_out"]
    inner = 4 * (box_w + 2) + 3 * 3 + 2  # four boxes, three arrows, a space each side
    label = " on the GPU · one TaskGraph · the tasks share device buffers: nothing returns to the host in between "
    lines = [fit(f"{' ' * 8}{GREY}┏━{RESET}{WHITE}{label}{RESET}{GREY}{'━' * (inner - len(label) - 1)}┓{RESET}", width)]
    for r in range(5):
        joint = f"{GREY} ▶ {RESET}" if r == 2 else "   "
        row = f"{fit(host_in[r], 7)} {GREY}┃{RESET} " + joint.join(b[r] for b in boxes) + f" {GREY}┃{RESET} {fit(host_out[r], 8)}"
        lines.append(fit(row, width))
    lines.append(fit(f"{' ' * 8}{GREY}┗{'━' * inner}┛{RESET}", width))
    lines.append(fit(f"        {YELLOW}▶ copy in{RESET} {cin / 1024:.0f} KB per execution {DIM}(signal, cutoff){RESET}     "
                     f"{YELLOW}copy out ▶{RESET} {cout / 1024:.0f} KB {DIM}(output + spectrum, for this display){RESET}     "
                     f"{DIM}GPU times: TornadoVM's profiler{RESET}", width))
    return lines


def signal_panel(width, height):
    with lock:
        inp, out, spec, cut = list(S["input"]), list(S["output"]), list(S["spec"]), S["cutoff"]
    plot_h = max(4, (height - 7) // 3)
    lines = [f"{YELLOW}{BOLD}input{RESET}{DIM} · 2 tones + 8 noise tones (the same every frame){RESET}"]
    lines += braille_plot(inp, width, plot_h, YELLOW)
    cut_txt = f" · Java lowPass zeroed bins ≥ {cut}" if cut is not None else ""
    lines.append(f"{GREEN}{BOLD}spectrum{RESET}{DIM} (cuFFT forward, log axis){cut_txt}: {RESET}{GREEN}kept{RESET}{DIM} / {RESET}"
                 f"{GREY}removed{RESET}")
    with lock:
        ghost = S.get("ghost")
    lines += spectrum_plot(spec, cut, width, plot_h, ghost)
    lines.append(f"{CYAN}{BOLD}output{RESET}{DIM} · after cuFFT inverse + normalize: what comes back to the host{RESET}")
    lines += braille_plot(out, width, plot_h, CYAN)
    return [fit(l, width) for l in lines][:height] + [" " * width] * max(0, height - len(lines))


def stats_panel(width, height):
    with lock:
        tasks, hist, frame, wall, correct, wrong, err, graph, phase = (
            dict(S["tasks"]), {k: list(v) for k, v in S["task_hist"].items()}, S["frame"], list(S["wall"]),
            S["correct"], S["wrong"], S["max_err"], S["graph"], S["phase"])
    out = [f"{BOLD}GPU time per task{RESET}{DIM} · median of the last 15 executions{RESET}", ""]
    recent = {key: statistics.median(hist[key][-15:]) if hist[key] else tasks.get(key, 0) for key, _, _ in TASKS}
    top = max(list(recent.values()) + [1])
    for key, label, kind in TASKS:
        color = LIBRARY if "NVIDIA" in kind else JAVA
        med = recent[key]
        out.append(f"{color}{label:<14}{RESET}{bar(med, top, width - 30, color)} {med:6.1f} µs")
        out.append(f"{DIM}{'':14}this execution {tasks.get(key, 0):.1f} µs · {len(hist[key])} so far{RESET}")
    out.append("")
    out.append(f"{GREEN}■{RESET} NVIDIA library   {BLUE}■{RESET} Java kernel (JIT)")
    out.append("")
    out.append(f"{BOLD}every frame checked{RESET}{DIM} against the exact answer{RESET}")
    out.append(f"{GREEN}✔ {correct} correct{RESET}" + (f"   {RED}✘ {wrong} wrong{RESET}" if wrong else "")
               + f"{DIM}   max error {err:.1e}{RESET}")
    out.append(f"{DIM}(a sum of tones: a low-pass keeps exactly{RESET}")
    out.append(f"{DIM} the tones below the cutoff){RESET}")
    out.append("")
    if wall:
        out.append(f"{DIM}frame {frame + 1} · wall {wall[-1] / 1000:.2f} ms incl. profiling{RESET}")
    if graph:
        plain, cg, ok = graph
        out.append("")
        out.append(f"{MAGENTA}{BOLD}then: the same graph, as a CUDA graph{RESET}")
        out.append(f"{DIM}median of 300 executions each, no profiler{RESET}")
        gtop = max(plain, cg)
        out.append(f"{'plain execute()':<17}{bar(plain, gtop, width - 32, YELLOW)} {plain:6.1f} µs")
        out.append(f"{'withCUDAGraph()':<17}{bar(cg, gtop, width - 32, MAGENTA)} {cg:6.1f} µs")
        out.append(f"{MAGENTA}▶ {plain / cg:.1f}x faster per execution{RESET}" + (f"  {GREEN}✔ correct{RESET}" if ok else f"  {RED}✘{RESET}"))
    elif phase == "graph":
        out.append("")
        out.append(f"{MAGENTA}⠿ replaying the same graph as a CUDA graph …{RESET}")
    return [fit(l, width) for l in out][:height] + [" " * width] * max(0, height - len(out))


def render():
    cols, rows = shutil.get_terminal_size((150, 46))
    width = max(124, cols)
    title = (f"{BOLD} TornadoVM Hybrid API{RESET}{DIM} · NVIDIA libraries and Java kernels in one TaskGraph, on one GPU"
             f" — {RESET}{GREEN}■ library task{RESET}{DIM} and {RESET}{BLUE}■ Java @Parallel task{RESET}")
    lines = [fit(title, width), ""] + pipeline(width) + [""]
    body = max(16, rows - len(lines) - 2)
    sw = 56
    lw = width - sw - 5
    left, right = signal_panel(lw, body), stats_panel(sw, body)
    for i in range(body):
        lines.append(f" {left[i]}  {GREY}│{RESET} {right[i]}")
    with lock:
        msg = S["message"]
    lines.append(fit(f" {msg}", width))
    sys.stdout.write("\033[H" + "\n".join(fit(l, width) for l in lines[: rows - 1]) + "\033[J")
    sys.stdout.flush()


def renderer():
    while not done.is_set():
        with lock:
            S["pulse"] += 0.6  # the flow animation: one execution's order (copy in, 4 tasks, copy out)
        render()
        time.sleep(0.1)
    render()


# --- running the program --------------------------------------------------------------------------------------
def run():
    tornado_env = 'CP=$(ls "$TORNADOVM_HOME"/share/java/tornado/*.jar | tr "\\n" ":"); export PATH="$TORNADOVM_HOME/bin:$PATH"'
    build = subprocess.run(["bash", "-c", f'{tornado_env}; "$JAVA_HOME/bin/javac" -cp "$CP" -d "{WORK}" "{HERE}/HybridLive.java"'],
                           capture_output=True, text=True)
    if build.returncode:
        with lock:
            S["message"] = f"{RED}javac failed: {build.stderr.strip()[:200]}{RESET}"
        return False
    p = subprocess.Popen(["bash", "-c", f'{tornado_env}; exec tornado --classpath "{WORK}" HybridLive 110 wait'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    log = open(os.path.join(WORK, "run.log"), "w")
    with lock:
        S["message"] = f"{DIM}TornadoVM is JIT-compiling the two Java kernels to CUDA; cuFFT plans are created on first use …{RESET}"
    for line in p.stdout:
        log.write(line)
        kind, _, rest = line.rstrip("\n").partition(" ")
        if kind == "IN":
            with lock:
                S["input"] = [float(v) for v in rest.split()]
        elif kind == "FRAME":
            f, cut, us, verdict, err = rest.split()
            with lock:
                S["frame"], S["cutoff"], S["phase"] = int(f), int(cut), "sweep"
                if int(f) > 0:  # frame 0 includes the JIT compilation
                    S["wall"].append(float(us))
                S["correct" if verdict == "correct" else "wrong"] += 1
                S["max_err"] = max(S["max_err"], float(err))
                S["message"] = f"{DIM}sweeping the cutoff: one execution of the whole graph per frame, ~10 frames/s{RESET}"
        elif kind == "PROFILE":
            try:
                prof = json.loads(rest)["hybrid"]
            except (ValueError, KeyError):
                continue
            with lock:
                S["copy_in"] = int(prof.get("TOTAL_COPY_IN_SIZE_BYTES", 0))
                S["copy_out"] = int(prof.get("TOTAL_COPY_OUT_SIZE_BYTES", 0))
                for key, _, _ in TASKS:
                    if key in prof and "TASK_KERNEL_TIME" in prof[key]:
                        us = int(prof[key]["TASK_KERNEL_TIME"]) / 1000
                        S["tasks"][key] = us
                        if S["frame"] > 0:
                            S["task_hist"][key].append(us)
        elif kind == "OUT":
            with lock:
                S["output"] = [float(v) for v in rest.split()]
        elif kind == "SPEC":
            with lock:
                S["spec"] = [float(v) for v in rest.split()]
                if S["frame"] == 0:
                    S["ghost"] = list(S["spec"])  # frame 0's cutoff is above every tone: the unfiltered spectrum
            if S["frame"] == 0:
                time.sleep(3)  # the first frame on screen: the pipeline and the noisy signal, before the sweep
                p.stdin.write("go\n")
                p.stdin.flush()
        elif kind == "GRAPH":
            plain, cg, verdict = rest.split()
            with lock:
                S["graph"] = (float(plain), float(cg), verdict == "correct")
        elif kind == "END":
            with lock:
                S["phase"] = "done"
        if kind == "SPEC" and S["frame"] >= 0 and S["phase"] == "sweep" and S["frame"] == 109:
            with lock:
                S["phase"] = "graph"
    p.wait()
    log.close()
    return p.returncode == 0


def main():
    sys.stdout.write("\033[?1049h\033[?25l")
    painter = threading.Thread(target=renderer, daemon=True)
    painter.start()
    ok = False
    try:
        ran = run()
        ok = ran and S["wrong"] == 0 and S["correct"] > 0 and S["graph"] is not None and S["graph"][2]
        with lock:
            S["message"] = (f"{GREEN}{BOLD}PASSED{RESET}{DIM} · {S['correct']} executions correct · library and Java tasks on the same buffers{RESET}"
                            if ok else f"{RED}{BOLD}FAILED{RESET}{DIM} (see {WORK}/run.log){RESET}") + \
                (f"{DIM}   [enter] to exit{RESET}" if not os.environ.get("NO_PAUSE") else "")
        time.sleep(1)
        if not os.environ.get("NO_PAUSE"):
            input()
    finally:
        done.set()
        painter.join(timeout=1)
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()
    meds = {label: statistics.median(S["task_hist"][key]) for key, label, _ in TASKS if S["task_hist"][key]}
    print("  per task (median GPU time): " + ", ".join(f"{k} {v:.1f} µs" for k, v in meds.items()))
    print(f"  frames: {S['correct']} correct, {S['wrong']} wrong, max error {S['max_err']:.1e}; "
          f"copy in {S['copy_in']} B, copy out {S['copy_out']} B per execution")
    if S["graph"]:
        print(f"  CUDA graph: {S['graph'][0]:.1f} -> {S['graph'][1]:.1f} µs per execution ({S['graph'][0] / S['graph'][1]:.1f}x)")
    print(f"  logs: {WORK}")
    print("HybridDashboard: PASSED" if ok else "HybridDashboard: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
