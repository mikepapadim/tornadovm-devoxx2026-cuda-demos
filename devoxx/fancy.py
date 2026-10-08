#!/usr/bin/env python3
"""Terminal rendering for the fancy*.sh demos: banners, act headers, bar charts and a scoreboard, drawn from the
real output of each demo (the logs the scripts capture). Standard library only. NO_COLOR=1 disables colors.

    fancy.py banner  TITLE SUBTITLE
    fancy.py act     N TITLE SUBTITLE
    fancy.py hybrid-tasks LOG          demo 04: per-task kernel times of the Java + cuBLAS task graph
    fancy.py hybrid-graph LOG          demo 07: plain execute vs CUDA graph replay
    fancy.py tile-pipeline LOG         demo 20: the tasks captured into one CUDA graph
    fancy.py tile-ladder LOG           demo 25: the FP16 GEMM ladder
    fancy.py llm-chat LOG              jitLLM generation: tokens per second
    fancy.py llm-bench PP TG LLAMA     jitLLM prefill/decode vs llama.cpp
    fancy.py stream                    stdin -> terminal as tokens arrive: drops <think></think> and ``` fences
    fancy.py code FILE                 a Java source with line numbers and syntax colors
    fancy.py cuda LOG [LINES]          the start of the CUDA kernel TornadoVM generated (from --printKernel)
    fancy.py fractal LOG               the generated kernel's result: the image, GPU vs CPU time, match check
    fancy.py scoreboard TITLE          every row the renders above recorded (in $FANCY_STATE), with checks

Each render appends its headline row and its check to $FANCY_STATE, so the script can end with a scoreboard.
"""
import os
import re
import statistics
import sys

COLOR = os.environ.get("NO_COLOR") is None
STATE = os.environ.get("FANCY_STATE")
WIDTH = 46

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, YELLOW, CYAN, MAGENTA, BLUE, RED, WHITE = (
    "\033[1;32m", "\033[1;33m", "\033[1;36m", "\033[1;35m", "\033[1;34m", "\033[1;31m", "\033[1;37m")


def paint(color, s):
    return f"{color}{s}{RESET}" if COLOR else s


def visible(s):
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def pad(s, width):
    return s + " " * max(0, width - visible(s))


def read(path):
    with open(path, errors="replace") as f:
        return re.sub(r"\033\[[0-9;]*m", "", f.read())


def record(row, value, ok=True, check=None):
    """Remember a scoreboard row and its check for `scoreboard`."""
    if STATE:
        with open(STATE, "a") as f:
            f.write(f"{row}\t{value}\t{'PASS' if ok else 'FAIL'}\t{check or ''}\n")


# --- building blocks ------------------------------------------------------------------------------------------

def box(lines, color=CYAN, width=72):
    print(paint(color, "  ╔" + "═" * width + "╗"))
    for line in lines:
        print(paint(color, "  ║") + pad(line, width) + paint(color, "║"))
    print(paint(color, "  ╚" + "═" * width + "╝"))


def banner(title, subtitle):
    print()
    box([paint(BOLD, "   " + title), paint(DIM, "   " + subtitle)])


def act(number, title, subtitle):
    print()
    head = f"  ━━ {number} · {title} "
    print(paint(MAGENTA, head + "━" * max(3, 76 - len(head))))
    print("    " + subtitle)
    print()


def bars(title, rows, unit, fmt="{:,.1f}", lower_is_better=True):
    """rows: (label, value, color). One scale: the largest value fills the width. The best value gets a star."""
    top = max(v for _, v, _ in rows)
    best = (min if lower_is_better else max)(v for _, v, _ in rows)
    print("    " + paint(BOLD, title))
    for label, value, color in rows:
        n = max(1, round(value / top * WIDTH))
        star = paint(GREEN, " ★") if value == best else ""
        print(f"    {pad(label, 34)} {paint(color, '█' * n)}{' ' * (WIDTH - n)} {fmt.format(value):>9} {unit}{star}")
    print()


def pipeline(stages, frame=None):
    """stages: (name, kind, color) drawn as boxes joined by arrows; frame labels an enclosing CUDA graph."""
    tops, mids, bots = [], [], []
    for name, kind, color in stages:
        w = max(len(name), len(kind)) + 2
        tops.append(paint(color, "┌" + "─" * w + "┐"))
        mids.append(paint(color, "│") + paint(BOLD, name.center(w)) + paint(color, "│"))
        bots.append(paint(color, "└") + paint(DIM, kind.center(w, "─")) + paint(color, "┘"))
    joint = lambda parts, sep: sep.join(parts)
    lines = [joint(tops, "   "), joint(mids, paint(WHITE, " ▶ ")), joint(bots, "   ")]
    if frame:
        width = visible(lines[0]) + 2
        print("    " + paint(DIM, "┏╸" + frame + "╺" + "━" * max(0, width - len(frame) - 4) + "┓"))
        for line in lines:
            print("    " + paint(DIM, "┃ ") + line + paint(DIM, " ┃"))
        print("    " + paint(DIM, "┗" + "━" * (width - 2) + "┛"))
    else:
        for line in lines:
            print("    " + line)
    print()


def big(text, color=GREEN):
    print(f"    {paint(color, '▶ ' + text)}")
    print()


# --- demo renders ---------------------------------------------------------------------------------------------

def hybrid_tasks(log):
    text = read(log)
    names = {"CuBlasSgemvHybrid.scale": ("scale", "Java kernel", BLUE), "cublasSgemv": ("cublasSgemv", "NVIDIA cuBLAS", GREEN),
             "CuBlasSgemvHybrid.bias": ("bias", "Java kernel", BLUE)}
    pipeline([names[k] for k in ("CuBlasSgemvHybrid.scale", "cublasSgemv", "CuBlasSgemvHybrid.bias")])
    # per iteration: the task kernel times printed before each "iteration i:" line
    iterations, current = [], {}
    for line in text.splitlines():
        m = re.search(r'"METHOD":\s*"([^"]+)"', line)
        if m:
            method = m.group(1)
        m = re.search(r'"TASK_KERNEL_TIME":\s*"?([0-9.]+)', line)
        if m:
            current[method] = float(m.group(1)) / 1000
        m = re.match(r"iteration (\d+): (\w+)", line)
        if m:
            iterations.append((int(m.group(1)), m.group(2), dict(current)))
            current = {}
    print("    " + paint(BOLD, "kernel time per task, every iteration on the GPU") + paint(DIM, "   (µs)"))
    for i, status, times in iterations:
        cells = []
        for key in ("CuBlasSgemvHybrid.scale", "cublasSgemv", "CuBlasSgemvHybrid.bias"):
            t = times.get(key)
            cells.append(f"{names[key][0]:>11} {t:9.1f}" if t is not None else f"{names[key][0]:>11} {'-':>9}")
        mark = paint(GREEN, "✔ correct") if status == "correct" else paint(RED, "✘ " + status)
        note = paint(DIM, "  first call: cuBLAS initialisation") if i == 0 else ""
        print(f"    iteration {i}  " + "  ".join(cells) + "   " + mark + note)
    print()
    ok = "All iterations correct" in text
    steady = [sum(t.values()) for i, _, t in iterations if i > 0]
    if steady:
        big(f"Java kernels and cuBLAS share device buffers: {statistics.median(steady):.0f} µs of kernel time per pass, no host round trip")
    record("Hybrid: Java → cuBLAS → Java", f"{len(iterations)} iterations, all correct" if ok else "INCORRECT", ok,
           "hybrid: every iteration correct")


def hybrid_graph(log):
    text = read(log)
    m = re.search(r"nograph=([0-9.]+) us, graph=([0-9.]+) us, speedup=([0-9.]+)x", text)
    first = re.findall(r"first execution[^:]*: (\d+) us", text)
    if not m:
        print(paint(RED, "    no steady-state summary in the log"))
        record("CUDA graph replay", "missing", False, "graph: summary present")
        return
    plain, graph, speedup = float(m.group(1)), float(m.group(2)), float(m.group(3))
    pipeline([("capture", "first run", YELLOW), ("cuGraphLaunch", "every later run", GREEN)])
    bars("steady-state wall clock per execution (median of 49)",
         [("plain execute()", plain, YELLOW), ("withCUDAGraph() replay", graph, GREEN)], "µs")
    if len(first) == 2:
        print(paint(DIM, f"    first executions (JIT, capture): plain {int(first[0]) / 1000:.0f} ms, graph {int(first[1]) / 1000:.0f} ms"))
        print()
    big(f"{speedup:.1f}x faster per execution with one launch for the whole graph")
    ok = text.count("All executions correct") == 2
    record("CUDA graph replay", f"{plain:.0f} µs → {graph:.0f} µs  {speedup:.1f}x", ok, "graph: every execution correct")


def tile_pipeline(log):
    text = read(log)
    capture = text.split("EXECUTION_GRAPH_BEGIN_CAPTURE", 1)[-1].split("EXECUTION_GRAPH_END_CAPTURE", 1)[0]
    kinds = {"scale": ("scale", "KernelContext", BLUE), "gemm": ("gemm", "CUDA Tile", MAGENTA),
             "project[cublasSgemv]": ("cublasSgemv", "NVIDIA cuBLAS", GREEN), "biasRelu": ("biasRelu", "@Parallel", BLUE)}
    stages = []
    for m in re.finditer(r"LAUNCH\s+task\s+(?:hybrid\.\S+ - )?(?:- hybrid\.)?(\S+)", capture):
        task = m.group(1)
        stages.append(kinds.get(task, (task, "task", WHITE)))
    pipeline(stages, frame=" one CUDA graph ")
    replays = text.count("EXECUTION_GRAPH_LAUNCH")
    print(f"    captured once, launched {paint(BOLD, str(replays))} times: each execution is a single cuGraphLaunch")
    print()
    ok = len(stages) == 4 and replays >= 1
    record("Threads + tiles + cuBLAS", f"{len(stages)} tasks, 1 CUDA graph, {replays} launches", ok,
           "tile pipeline: 4 tasks inside one captured graph")


def tile_ladder(log):
    text = read(log)
    rows = re.findall(r"^(\d)\. (.+?)\s{2,}\S+\s+(\d+)\s+(\d+)\s*$", text, re.M)
    if not rows:
        print(paint(RED, "    no ladder summary in the log"))
        record("Tile GEMM ladder", "missing", False, "ladder: summary present")
        return
    color = lambda name: GREEN if "cuBLAS" in name else MAGENTA if "Tile" in name else BLUE
    bars("FP16 GEMM 2048³, median wall clock per call   " + paint(BLUE, "■ ") + paint(DIM, "KernelContext  ") + paint(MAGENTA, "■ ")
         + paint(DIM, "TileContext  ") + paint(GREEN, "■ ") + paint(DIM, "cuBLAS"),
         [(f"{n}. {name}", float(us), color(name)) for n, name, us, _ in rows], "µs", fmt="{:,.0f}")
    by = {name: float(us) for _, name, us, _ in rows}
    tile32, tile_best, hinted, opt = (by.get("TileContext 32x32x32"), by.get("TileContext 128x128x64"),
                                      by.get("TileContext 128x128x64 +hint"), by.get("KernelContext, optimised"))
    if tile32 and tile_best:
        print(f"    tile shape alone: {tile32:,.0f} → {tile_best:,.0f} µs ({tile32 / tile_best:.1f}x), same ten lines of Java")
    if hinted and opt:
        verdict = "beats" if hinted < opt else "trails"
        print(f"    one launch hint: the tile kernel {paint(BOLD, verdict)} the hand-optimised KernelContext ({hinted:,.0f} vs {opt:,.0f} µs)")
    print(paint(DIM, "    wall clock includes host dispatch; the slide's TFLOP/s are kernel time from nsys"))
    print()
    ok = "All rungs produced the same, correct result" in text
    record("Tile GEMM ladder (2048³)", f"tile {hinted:,.0f} µs vs hand-tuned {opt:,.0f}" if hinted and opt else "-", ok,
           "ladder: every rung correct")


def llm_chat(log, row="jitLLM generation"):
    text = read(log)
    m = re.search(r"achieved tok/s: ([0-9.]+)\. Tokens: (\d+), seconds: ([0-9.]+)", text)
    if not m:
        print(paint(RED, "    no performance line in the log"))
        record(row, "missing", False, "chat: tokens generated")
        return
    print()
    # jitLLM's own metric: prompt and answer tokens together, over the whole request
    big(f"{float(m.group(1)):.0f} tokens/s · {m.group(2)} tokens (prompt + answer) in {float(m.group(3)):.2f} s · pure Java on the GPU")
    record(row, f"{float(m.group(1)):.0f} tok/s", int(m.group(2)) > 0, f"{row}: tokens generated")


def llm_bench(pp_log, tg_log, llama_log):
    pp = re.search(r"pp512 b512\s+([0-9.]+) ± ([0-9.]+)", read(pp_log))
    tg = re.search(r"tg128\s+([0-9.]+) ± ([0-9.]+)", read(tg_log))
    llama = read(llama_log)
    lpp = re.search(r"pp512 \|\s+([0-9.]+) ± ([0-9.]+)", llama)
    ltg = re.search(r"tg128 \|\s+([0-9.]+) ± ([0-9.]+)", llama)
    if not (pp and tg and lpp and ltg):
        print(paint(RED, "    a benchmark line is missing (see the logs)"))
        record("jitLLM vs llama.cpp", "missing", False, "bench: all four numbers")
        return
    f = lambda m: (float(m.group(1)), float(m.group(2)))
    (jp, jpe), (jt, jte), (lp, lpe), (lt, lte) = f(pp), f(tg), f(lpp), f(ltg)
    bars("prefill: 512-token prompt, batch 512  (tokens/s, higher is better)",
         [(f"jitLLM (Java + cuBLAS)    ±{jpe:,.0f}", jp, CYAN), (f"llama.cpp (C++/CUDA)      ±{lpe:,.0f}", lp, YELLOW)],
         "t/s", fmt="{:,.0f}", lower_is_better=False)
    bars("decode: 128 tokens, one at a time  (tokens/s, higher is better)",
         [(f"jitLLM (Java, CUDA graph) ±{jte:,.0f}", jt, CYAN), (f"llama.cpp (C++/CUDA)      ±{lte:,.0f}", lt, YELLOW)],
         "t/s", fmt="{:,.0f}", lower_is_better=False)
    print(f"    prefill {paint(BOLD, f'{jp / lp:.2f}x')} llama.cpp (llama.cpp varies ±{lpe:,.0f} between runs) · "
          f"decode {paint(BOLD, f'{jt / lt:.2f}x')}: same GPU, same GGUF")
    print()
    record("jitLLM vs llama.cpp, prefill", f"{jp:,.0f} vs {lp:,.0f} t/s  {jp / lp:.2f}x", True)
    record("jitLLM vs llama.cpp, decode", f"{jt:,.0f} vs {lt:,.0f} t/s  {jt / lt:.2f}x", True, "bench: all four numbers measured")


def highlight_java(line):
    """Keywords, annotations, numbers, strings and comments in color."""
    if line.strip().startswith("//"):
        return paint(DIM, line)
    out = []
    for token in re.split(r"(@\w+|\"[^\"]*\"|\b\d[\d_.]*[fFL]?\b|\b\w+\b)", line):
        if token.startswith("@"):
            out.append(paint("\033[1;38;5;208m", token))
        elif token.startswith('"'):
            out.append(paint(GREEN, token))
        elif re.fullmatch(r"\d[\d_.]*[fFL]?", token):
            out.append(paint("\033[0;38;5;141m", token))
        elif token in JAVA_KEYWORDS:
            out.append(paint("\033[1;38;5;75m", token))
        elif token in ("IntArray", "FloatArray", "Math"):
            out.append(paint("\033[0;38;5;43m", token))
        else:
            out.append(token)
    return "".join(out)


def stream():
    """Copy stdin to the terminal as it arrives, indented, without the model's empty <think></think> or the fences."""
    hidden = {"<think>", "</think>", "```java", "```"}
    pending, at_line_start, started = "", True, False
    while True:
        ch = sys.stdin.read(1)
        if not ch:
            break
        pending += ch
        # hold back text that may be the start of a hidden marker, until it is decided
        if any(h.startswith(pending.strip()) for h in hidden) and pending.strip() and ch != "\n":
            continue
        if pending.strip() in hidden:
            pending = ""
            continue
        for c in pending:
            if not started and c in "\n ":
                continue  # nothing shown yet: skip the blank lines the hidden markers leave
            started = True
            if at_line_start and c != "\n":
                sys.stdout.write("    " + paint(DIM, "│ "))
                at_line_start = False
            sys.stdout.write(paint("\033[0;38;5;152m", c) if c != "\n" else c)
            at_line_start = c == "\n"
        sys.stdout.flush()
        pending = ""
    print()


def code(path):
    lines = open(path).read().rstrip("\n").split("\n")
    for n, line in enumerate(lines, 1):
        print(f"    {paint(DIM, f'{n:3d} │')} {highlight_java(line)}")
    print()


def cuda(log, count="16"):
    text = read(log)
    start = text.find('extern "C" __global__')
    if start < 0:
        print(paint(DIM, "    (no CUDA source in the log)"))
        return
    lines = text[start:].split("\n")[: int(count)]
    print("    " + paint(BOLD, "...which TornadoVM JIT-compiled to CUDA") + paint(DIM, f"  (first {count} lines, --printKernel)"))
    for line in lines:
        line = line if len(line) <= 108 else line[:105] + "..."
        print("    " + paint(DIM, "┊ ") + paint("\033[0;38;5;109m", line))
    print()


def fractal(log):
    text = read(log)
    gpu = re.search(r"GPU_MS ([0-9.]+)", text)
    cpu = re.search(r"CPU_MS ([0-9.]+)", text)
    match = re.search(r"MATCH (\d+) (\d+)", text)
    rows = [list(map(int, line.split()[1:])) for line in text.splitlines() if line.startswith("ROW ")]
    if not (gpu and cpu and match and rows):
        print(paint(RED, "    the kernel did not run (see the log)"))
        record("LLM-written GPU kernel", "did not run", False, "kernel: runs and matches the CPU")
        return
    top = max(max(r) for r in rows) or 1
    palette = [17, 18, 19, 20, 26, 32, 38, 44, 50, 86, 122, 158, 194, 229, 223, 217, 211, 205, 199, 163]
    for row in rows:
        cells = []
        for v in row:
            if v >= top:
                cells.append("\033[48;5;16m " if COLOR else "#")
            else:
                level = palette[min(len(palette) - 1, int((v / top) ** 0.5 * len(palette)))]
                cells.append(f"\033[48;5;{level}m " if COLOR else " .:-=+*%@"[min(8, v * 9 // top)])
        print("    " + "".join(cells) + (RESET if COLOR else ""))
    print()
    g, c = float(gpu.group(1)), float(cpu.group(1))
    same, total = int(match.group(1)), int(match.group(2))
    bars("4096 × 3072 pixels, up to 256 iterations each",
         [("same method, Java, 1 CPU thread", c, YELLOW), ("same method, GPU via TornadoVM", g, GREEN)], "ms", fmt="{:,.1f}")
    print(f"    {paint(BOLD, f'{100 * same / total:.2f}%')} of {total:,} pixels identical to the CPU run"
          + paint(DIM, "  (the rest: float rounding at the set's edge; the GPU fuses multiply-adds)"))
    print()
    ok = same / total >= 0.99
    record("LLM-written GPU kernel", f"{g:.1f} ms GPU vs {c:,.0f} ms 1 CPU", ok,
           f"kernel: compiles, runs on the GPU, {100 * same / total:.2f}% of pixels match the CPU (>= 99%)")


def scoreboard(title):
    rows = []
    if STATE and os.path.exists(STATE):
        with open(STATE) as f:
            rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    print()
    box([paint(BOLD, f"  {title}"), paint(DIM, "  " + "─" * 68)] + [f"  {r[0]:<34}{r[1][:36]}" for r in rows])
    checks = [(r[2], r[3]) for r in rows if r[3]]
    for verdict, check in checks:
        print("    " + (paint(GREEN, "✔ ") if verdict == "PASS" else paint(RED, "✘ ")) + check)
    ok = all(r[2] == "PASS" for r in rows) and rows
    print()
    print("PASSED -- every check holds" if ok else "FAILED -- a check did not hold (see above)")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    command, args = argv[1], argv[2:]
    handlers = {"banner": banner, "act": act, "hybrid-tasks": hybrid_tasks, "hybrid-graph": hybrid_graph,
                "tile-pipeline": tile_pipeline, "tile-ladder": tile_ladder, "llm-chat": llm_chat,
                "llm-bench": llm_bench, "scoreboard": scoreboard, "stream": stream, "code": code,
                "cuda": cuda, "fractal": fractal}
    handlers[command](*args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
