# Devoxx Belgium 2026: demos for "Tap into the NVIDIA ecosystem from Java"

Every demo of the talk that runs on this machine, one command each. (Apache Flink + cuDF runs on the other machine.)

```bash
git clone git@github.com:mikepapadim/tornadovm-devoxx2026-cuda-demos.git
cd tornadovm-devoxx2026-cuda-demos/devoxx    # the demos these scripts run are in ../demos
bash check.sh            # before the talk: runs everything once, prints OK/FAIL per demo (~1 min)

bash demoHybrid.sh       # slides 14-16  Java kernel -> cuBLAS -> Java kernel in one task graph; CUDA graph replay   (~3 s)
bash demoTile.sh         # slides 17-19  threads + CUDA Tile + cuBLAS in one CUDA graph; the tile GEMM ladder      (~6 s)
bash demoJitllm.sh       # live demo 1   jitLLM generates on the GPU, then jitLLM vs llama.cpp                      (~20 s)
```

Each script prints the command before running it and waits for **Enter** between steps (`NO_PAUSE=1` to skip).

**Fancy versions**, same programs, drawn for the big screen: live spinners, pipeline diagrams, bar charts and a
scoreboard with PASSED/FAILED checks, all rendered from the demos' real output by `fancy.py`.

```bash
bash fancyHybrid.sh      # Java -> cuBLAS -> Java pipeline, kernel time per task; CUDA graph 8-9x as bars              (~10 s)
bash fancyTile.sh        # threads + tiles + cuBLAS inside one CUDA graph; the FP16 GEMM ladder as a bar chart      (~20 s)
bash fancyJitllm.sh      # chat · the LLM writes a GPU kernel, TornadoVM runs it, a fractal · vs llama.cpp  [chat|code|bench] (~35 s)
bash fancyJitllmLive.sh  # full-screen live dashboard: the Java LLM writes a kernel, it runs on the SAME GPU, every part lit  (~30 s)
```

## How to run the fancy demos

**1. Once per machine** (needs the network; clones and builds jitLLM):

```bash
cd tornadovm-devoxx2026-cuda-demos/devoxx
bash setup.sh            # re-run any time to update jitLLM to the latest main
```

Paths to the TornadoVM SDKs, the demos repo, llama.cpp and the models are all in `env.sh`; each one
can be overridden with an environment variable of the same name (e.g. `DEMOS_REPO=... bash fancyHybrid.sh`).

**2. Before the talk:**

```bash
bash check.sh            # every demo once, OK/FAIL per demo (~1 min)
bash demoJitllm.sh chat  # warms jitLLM's kernel JIT so the first fancy run is not slow
nvidia-smi               # the GPU should be idle: no stray java processes
```

**3. Terminal:** a dark background, a font with Unicode block and braille characters, 256 colours, and a window of at
least **120 × 40** (required by `fancyJitllmLive.sh`; the others are fine at 100 columns). Python 3 (standard library
only) renders everything.

**4. Run** (from the repo root; each script runs from anywhere):

| command | what you see | time |
|---|---|---|
| `bash fancyHybrid.sh` | Java → cuBLAS → Java pipeline, per-task kernel time; CUDA graph speed-up as bars | ~10 s |
| `bash fancyTile.sh` | threads + tiles + cuBLAS in one CUDA graph; FP16 GEMM ladder | ~20 s |
| `bash fancyJitllm.sh` | all three acts: `chat`, `code` (the LLM writes a GPU kernel), `bench` (vs llama.cpp) | ~35 s–1.5 min |
| `bash fancyJitllm.sh chat` / `code` / `bench` | one act only | |
| `bash fancyJitllmLive.sh` | full-screen live dashboard (see below); `[enter]` leaves it | ~30 s |

Each ends with a scoreboard that says `PASSED` only if every check behind it held (see the table further down).

**Knobs** (environment variables):

| variable | effect |
|---|---|
| `NO_PAUSE=1` | no `[enter]` between acts (and the live dashboard exits at the end) |
| `NO_COLOR=1` | plain output from the `fancy*.sh` renderer (not the live dashboard) |
| `FANCY_LOGS=<dir>` | keep the raw logs of each step there (default: a fresh temp dir, printed on failure) |
| `MODEL_CHAT`, `MODEL_CODE`, `MODEL_BENCH` | another GGUF for chat / kernel writing / the benchmark (`bash models.sh` lists them) |
| `ASSEMBLE_SECONDS=<s>` | how long the live dashboard holds the javac step (default 4) |

If a step fails, its spinner turns into a red `✘` with the path of its log.

## The demos

| script | variants | what to point at |
|---|---|---|
| `demoHybrid.sh` | | Per task: the Java kernels (`scale`, `bias`) and `cublasSgemv` all on CUDA, every iteration `correct`. The first iteration's cuBLAS time is its one-time initialisation. Then `withCUDAGraph()`: about 9× faster steady state (≈310 → 35 µs). |
| `demoTile.sh` | | The capture: `@Parallel` scale, a `TileContext` GEMM, `cublasSgemv` and a Java `biasRelu` between `BEGIN_CAPTURE` and `END_CAPTURE`, then replays. Then the ladder: tile shape alone takes the tile GEMM from 2.1 to 1.2 ms, and one launch hint passes the hand-optimised kernel. Wall clock includes dispatch; slide 19's TFLOP/s are kernel time from nsys. |
| `demoJitllm.sh` | `chat`, `bench` | Generation in pure Java on the GPU (~190 tok/s, Llama-3.2-1B F16). Then Qwen3-0.6B F16: prefill with cuBLAS (`--with-native-libraries`), decode with Java kernels replayed as a CUDA graph, and llama.cpp on the same GPU and GGUF. |

Other models: `bash models.sh` lists every GGUF on the machine, about 30, among them Qwen3-1.7B F16
(`~/jcon/models/Qwen3-1.7B-f16.gguf`, the slide's second prefill model), Qwen3-4B/8B F16, Llama-3.2-1B/3B,
Llama-3-8B, gemma-4-E2B, Phi-3, Mistral-7B, DeepSeek-R1-Distill-Qwen-1.5B/7B and Qwen3.8-27B Q4_0. Use one with
`MODEL_CHAT=<path> bash demoJitllm.sh chat` or `MODEL_BENCH=<path> bash demoJitllm.sh bench`.

### The live dashboard: `fancyJitllmLive.sh`

The point it makes visible: **one GPU, two Java workloads**. A Java LLM engine runs on the GPU and writes Java,
and that Java then runs on the same GPU. Full screen (at least 120 × 40), redrawn ten times a second, about 30 s.

![dashboard at the end of a run](live/screenshot-final.png)

* **The pipeline row** has five components, each lit (heavy border, spinner, elapsed time) while it works and ticked
  with its time when done. The arrows light up as work flows.
  1. the jitLLM engine running Qwen3-4B;
  2. the generated Java;
  3. javac;
  4. TornadoVM's JIT to CUDA;
  5. the kernel running.
* **The GPU panel** polls `nvidia-smi`. It shows utilization, memory, and **which process holds the GPU**, labelled
  and colored by the component that started it (matched through `/proc` parent PIDs). The utilization timeline
  covers the whole run, each sample in the color of the active component. A finished run shows the magenta LLM
  burst, then the cyan kernel load, on one device. "Seen on this GPU during the run" lists both processes: jitLLM
  about 8.4 GB, the kernel about 0.5 GB.
* **The content panel** follows the work:
  * **the prompt:** typed out in a magenta chat bubble at the start, labelled "sent to Qwen3-4B running in jitLLM
    on the GPU", with a note on what the system prompt contains. It stays above the code as the code streams in,
    and as a one-line `❯ prompt:` header in every later view;
  * the code streaming from the model;
  * **how the answer becomes a program** (the javac step, held 4 s after javac so it can be read; `ASSEMBLE_SECONDS`):
    1. extract the ```` ```java ```` block from the model's raw output;
    2. insert it at the `/*KERNEL*/` marker of `live/Harness.template`, giving `Harness.java`. The view shows that
       file with the model's lines marked by a yellow bar and the fixed harness in grey. Arrows point at
       `.task("mandelbrot", Harness::mandelbrot, …)` (on the GPU) and at the plain `mandelbrot(…)` call (the same
       method on the CPU);
    3. `javac` against the TornadoVM jars, then the size of `Harness.class` and javac's real time.

    ![the javac step](live/screenshot-harness.png)
  * the CUDA TornadoVM generated from it;
  * the kernel's output: a zoom into Seahorse Valley, 100 frames at 7680 × 4320, one kernel execution per frame
    (~17 ms each), paced at ~12 frames/s so the load shows on `nvidia-smi`. It is drawn with half-blocks, two
    pixels per cell, with histogram colors.
* **The footer and the check:** jitLLM's metrics, the GPU-vs-CPU check of the same method (99.87% of pixels), the
  frame times, and `PASSED`.

Everything comes from the processes it starts and from `nvidia-smi`; nothing is replayed. The model's prompt is
`live/prompt.txt`. Because of temperature 0 the kernel is the same every run. If it ever fails to compile, the
dashboard marks javac red and runs `live/reference.kernel` (the model's rehearsal output), labelled. Logs land in
a temp directory printed at the end. `[enter]` leaves the dashboard; `NO_PAUSE=1` exits straight away.

### What the fancy versions show (and check)

| script | acts | checks behind PASSED |
|---|---|---|
| `fancyHybrid.sh` | 1. the `scale ▶ cublasSgemv ▶ bias` pipeline, and every iteration's kernel time per task. Iteration 0 is labelled as cuBLAS initialisation (~38 ms). 2. capture ▶ cuGraphLaunch, then plain vs graph bars: 307–313 → 35–36 µs, 8.5–8.8× | every iteration correct; every execution correct |
| `fancyTile.sh` | 1. the four captured tasks (KernelContext, CUDA Tile, cuBLAS, @Parallel) drawn inside one CUDA-graph frame, with the launch count. 2. the 8-rung ladder colored by kind, the tile-shape gain (~1.7–1.8×), and whether the hinted tile kernel beats the hand-tuned one | 4 tasks in the captured graph; every rung correct |
| `fancyJitllm.sh` | 1. **Chat:** the prompt, the tokens streaming live, then the tok/s (~180–190). 2. **The model writes a GPU kernel:** Qwen3-4B F16, run by jitLLM on the GPU at ~67 tok/s, writes a Mandelbrot kernel in Java (`@Parallel` loops), shown streaming and then syntax-highlighted. A harness (`codegen/Harness.java`) compiles it, TornadoVM JIT-compiles it to CUDA (the first lines are shown, from `--printKernel`) and runs it on 4096 × 3072 pixels. The *same* method then runs as plain Java on one CPU thread, and the fractal is drawn in the terminal: 3.1 ms vs ~1,180 ms, 99.79% of pixels identical (the rest is float rounding at the set's edge: the GPU fuses multiply-adds). 3. **Bench:** three spinners, then prefill and decode bars with each side's ± | tokens generated (twice); the kernel compiles, runs and matches the CPU on ≥ 99% of pixels; all four bench numbers measured |

The numbers are this run's, not constants: the scoreboard shows what was measured. Two of them vary between runs,
and the fancy output shows that honestly.
* **Tile ladder:** the hinted tile kernel and the hand-tuned KernelContext are within 1–3% in wall clock. At 10
  executions their order flipped between runs, so `fancyTile.sh` uses 30 per rung, and the line says *beats* or
  *trails*.
* **The generated kernel** comes from greedy decoding (temperature 0), so the same prompt and model give the same
  kernel; rehearsal shows what the stage will show. If the live kernel ever fails to compile, the act shows the
  compiler error, then runs `codegen/mandelbrot.reference.java` (the kernel the same model wrote for the same prompt
  in rehearsal), labelled as such. Smaller models do worse: Llama-3.2-3B's kernel did not compile, and whole programs
  (not just the kernel) failed with both models. That is why the model writes only the kernel and the harness does
  the rest. `MODEL_CODE=<gguf>` picks another model.
* **jitLLM vs llama.cpp prefill:** within llama.cpp's own ±9k run-to-run spread (0.99–1.02×). Decode is about 0.78×
  (400 vs 516 t/s).

## Measured on this machine (RTX 4090, 2026-10-01)

| | result |
|---|---|
| Hybrid: CUDA graph vs plain execute | 34.7 vs 312 µs, 8.98× |
| Tile ladder, n = 2048 (wall clock) | KernelContext optimised 1176 µs · TileContext 128×128×64 + hint 1123 µs · cuBLAS 1060 µs |
| jitLLM Qwen3-0.6B F16, prefill pp512 b512 | jitLLM 64,818–65,340 t/s · llama.cpp 60,923–64,545 t/s (± up to 9k between its runs) |
| jitLLM Qwen3-0.6B F16, decode tg128 | jitLLM 399 t/s (CUDA graphs) · llama.cpp 515 t/s |

jitLLM is the latest `main` (`d875f083`, 2026-09-30). llama.cpp is the local build `f172be756`.

## What is behind each script

| demo | uses |
|---|---|
| `demoHybrid.sh`, `demoTile.sh` | `../demos` (demos 04, 07, 20, 25) on the released TornadoVM 7.2.0 (SDKMAN, JDK 25) |
| `demoJitllm.sh` | `./jitllm` (beehive-lab/jitllm `main`), built against its TornadoVM develop build (JDK 21); `~/llama.cpp-ref/build/bin/llama-bench` |

`fancy.sh` (spinner, logs) and `fancy.py` (rendering, standard-library Python 3) are shared by the fancy scripts;
each run keeps its raw logs in a temp directory (`FANCY_LOGS=<dir>` to choose it). All paths are in `env.sh`. `bash setup.sh` (network) updates jitLLM to the latest `main` and rebuilds it.

## If something fails on stage

* **`check.sh` says FAIL:** open the log it names; most failures are a missing environment (`env.sh` paths).
* **jitLLM is slow on the first run:** the first run JIT-compiles the kernels. Run `bash demoJitllm.sh chat` once
  before the talk.
* **GPU busy:** check `nvidia-smi` for stray Java processes from an interrupted demo.
