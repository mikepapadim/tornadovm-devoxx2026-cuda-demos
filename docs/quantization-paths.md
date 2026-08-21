# GPULlama3.java quantized LLM paths on TornadoVM CUDA (Track B3)

Status: **Observed** unless marked otherwise. Environment: this repo's pinned TornadoVM CUDA build
(`env/versions.env`, `TORNADO_SHA=99549c9862eda8d584e35e99924f9c865501eb3a`), RTX 4090, driver
565.57.01, CUDA 12.6.85, JDK 21.0.2. GPULlama3.java pinned at
`GPULLAMA3_SHA=bbe42fdc8cd475bb6104cefa42118dd6e068538b` (`main`, 2026-08-15), built as documented
in `docs/gpullama3-reproduction.md` (`mvn clean install -Dtornadovm.base.version=5.2.1
-Djdk.version.suffix=-jdk21-dev`). All runs use the same model family (Llama-3.2-1B-Instruct) so the
three paths are directly comparable; GPU confirmed idle (`nvidia-smi`, 0%/4 MiB) before each run.

## 1. Source-level dispatch (source-backed, read before running anything)

TornadoVM-backed weight loading and forward-pass dispatch both gate explicitly on `GGMLType`,
independently, in two different places:

- `vendor/GPULlama3.java/src/main/java/org/beehive/gpullama3/model/loader/AbstractModelLoader.java:41-48`
  (`getModelQuantization`) maps the GGUF `general.file_type` metadata int to a supported string:
  `1→FP16`, `7→Q8_0`, `{14,15}→Q8_0` (Q4_K_S/M — K-quants dequantized to Q8_0), `{16,17}→Q8_0`
  (Q5_K_S/M), `18→Q8_0` (Q6_K). **Every other file_type value, including `2` (legacy `Q4_0`), falls
  through to `default -> throw new UnsupportedOperationException("Unsupported quantization format:
  " + modelQuantizationAsInt + " (as int).")`.**
- `vendor/GPULlama3.java/src/main/java/org/beehive/gpullama3/model/loader/LlamaModelLoader.java:109-114`
  (`createTornadoVMWeights`) independently re-validates: `if (ggmlType != GGMLType.F16 && ggmlType
  != GGMLType.Q8_0) throw new UnsupportedOperationException(...)`.
- `vendor/GPULlama3.java/src/main/java/org/beehive/gpullama3/tornadovm/plan/ForwardPlanFactory.java:78-84`
  (`create`) dispatches the forward-pass execution plan: `case F16 -> ...`, `case Q8_0 -> ...`,
  `case F32 -> throw new UnsupportedOperationException("F32 plans not yet implemented")`, `case Q4_0
  -> throw new UnsupportedOperationException("Q4_0 plans not yet implemented")`.
- Dedicated per-model-family TornadoVM execution-plan classes exist under
  `tornadovm/plan/components/{fp16,q8_0}/` and `tornadovm/layers/type/q8_0/` (grep-counted: 8 Q8_0
  plan-component classes, one per supported model family) but there is **no** `tornadovm/plan/components/q4_0/`
  or `.../q4_k/` directory anywhere in the tree.

Conclusion from source alone, before running anything: on the current pinned SHA, TornadoVM
execution supports exactly two weight formats — **FP16** and **Q8_0** (the latter also serving as
the dequantization target for K-quants Q4_K/Q5_K/Q6_K) — and explicitly rejects legacy **Q4_0**
(and F32) at two independent checkpoints (model load, forward-plan dispatch).

## 2. FP16 — working (Observed, reproduced 4x total: 3x in task 10, 1x here)

```bash
source vendor/tornadovm/setvars.sh   # this repo's pinned TornadoVM env
source vendor/GPULlama3.java/set_paths
./llama-tornado --gpu --cuda \
  --model /home/michalis/test_install/GPULlama3.java/beehive-llama-3.2-1b-instruct-fp16.gguf \
  --prompt "Explain what a GPU kernel is in one sentence." --max-tokens 64 --seed 42
```

Output: coherent, correct completion ("A GPU kernel is a small, self-contained piece of code that
is executed by a Graphics Processing Unit (GPU) to perform a specific task, such as matrix
multiplication or convolution."), 164.20 tok/s (this run/this GPU, not a general claim). Exit 0.
Log: `results/raw/11-quantization/fp16-inference-run.log`. See `docs/gpullama3-reproduction.md` for
the original 3-run reproduction (build blocker, workaround, profiler-confirmed CUDA execution).

## 3. Q8_0 — working (Observed, new for this task)

```bash
./llama-tornado --gpu --cuda \
  --model /home/michalis/llama3.java-tornadovm/Llama-3.2-1B-Instruct-Q8_0.gguf \
  --prompt "Explain what a GPU kernel is in one sentence." --max-tokens 64 --seed 42
```

Output: coherent, correct completion (same sentence as the FP16 run above, expected — Q8_0 is a
high-fidelity quantization of the same model), 186.47 tok/s (this run/this GPU). Exit 0. Log:
`results/raw/11-quantization/q8_0-inference-run.log`.

**CUDA execution confirmed by profiler JSON**, not just a claimed flag — re-ran with `--profiler`
and a second prompt/seed:

```bash
./llama-tornado --gpu --cuda \
  --model /home/michalis/llama3.java-tornadovm/Llama-3.2-1B-Instruct-Q8_0.gguf \
  --prompt "What is the capital of Italy?" --max-tokens 32 --seed 7 --profiler
```

Every TornadoVM stage reports `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"` (91.27
tok/s under profiler — extra per-token JSON-dump overhead, same direction as prior demos'
profiler-overhead notes, e.g. task 10). Log:
`results/raw/11-quantization/q8_0-inference-run-profiler.log`.

Model requirement: any GGUF with `general.file_type == 7` (native Q8_0), or `14/15/16/17/18`
(K-quants, transparently dequantized to Q8_0 at load — see §1); same runtime prerequisites as FP16
(§2 of `docs/gpullama3-reproduction.md`: TornadoVM SDK pinned to this repo's 5.2.1-jdk21-dev build,
CUDA backend, `--gpu --cuda`).

## 4. Q4_0 — blocked (Observed failure, matches source-level prediction from §1)

```bash
./llama-tornado --gpu --cuda \
  --model /home/michalis/llama3.java-tornadovm/Llama-3.2-1B-Instruct-Q4_0.gguf \
  --prompt "What is the capital of Italy?" --max-tokens 32 --seed 7
```

Fails deterministically, exit 1, **before any GPU/TornadoVM work starts** (fails during GGUF
metadata parsing, at model-config-creation time):

```
Exception in thread "main" java.lang.UnsupportedOperationException: Unsupported quantization format: 2 (as int).
	at org.beehive.gpullama3.model.loader.AbstractModelLoader.getModelQuantization(AbstractModelLoader.java:48)
	at org.beehive.gpullama3.model.loader.LlamaModelLoader.createConfiguration(LlamaModelLoader.java:51)
	at org.beehive.gpullama3.model.loader.LlamaModelLoader.createConfiguration(LlamaModelLoader.java:29)
	at org.beehive.gpullama3.model.loader.AbstractModelLoader.loadModel(AbstractModelLoader.java:94)
	at org.beehive.gpullama3.model.ModelType$1.loadModel(ModelType.java:31)
	at org.beehive.gpullama3.model.loader.ModelLoader.loadModel(ModelLoader.java:105)
	at org.beehive.gpullama3.LlamaApp.main(LlamaApp.java:61)
```

Log: `results/raw/11-quantization/q4_0-inference-run.log`. `general.file_type == 2` is the GGUF
tag for legacy `Q4_0` — this is the exact code path predicted by §1's source read
(`AbstractModelLoader.java:48`'s `default` branch), reached before `ForwardPlanFactory`'s own
`Q4_0 -> throw ...` branch (§1) would even be evaluated. Two independent guards in the source agree
Q4_0 is unsupported; the runtime failure confirms it end-to-end with the real jar and real GGUF
file, not just static reading.

**Classification: blocked, not a bug** — `ForwardPlanFactory.java:84`'s message ("Q4_0 plans not
yet implemented") indicates this is planned/future work upstream, not a permanent limitation.

## 5. Q4_K / Q5_K / Q6_K (K-quants) — documented, not independently reproduced

Source (`AbstractModelLoader.java:41-48`, `effectiveGpuWeightType`) states these are accepted at
metadata-parse time and dequantized to `Q8_0` before TornadoVM weight loading, i.e. they should
work via the same Q8_0 execution path as §3. **Not run**: no `Q4_K`/`Q5_K`/`Q6_K` GGUF file was
found on this machine (checked `/home/michalis`, depth 5, common model directories used by task 10 —
`~/jcon/models/`, `~/llama3.java-tornadovm/`, `~/test_install/GPULlama3.java/`, `~/.langchain4j/`;
none present). Classified **documented** (source-backed, not observed), per `PLAN.md` §6 — do not
claim this works until reproduced with an actual K-quant file.

## 6. Summary table

| Path | Classification | Evidence |
|---|---|---|
| FP16 | **working** | Observed, 4 runs total (3 in task 10 + 1 here), profiler-confirmed CUDA |
| Q8_0 | **working** | Observed, 2 runs here, profiler-confirmed CUDA, 186.47 tok/s (this run) |
| Q4_K / Q5_K / Q6_K | **documented** | Source-backed (dequant-to-Q8_0 path exists) — no test file available, not reproduced |
| Q4_0 (legacy) | **blocked** | Observed failure, `UnsupportedOperationException`, matches source prediction exactly; upstream marks it "not yet implemented" |
| F32 | **blocked** | Source-backed only (`ForwardPlanFactory.java:81`, same `throw` pattern as Q4_0) — not independently run, no F32 GGUF on hand, out of this task's requested scope (FP16/Q8/Q4 only) |
