# Batch 51 — migration to TornadoVM 7.2.0 (2026-10-07)

SDK: `sdk install tornadovm 7.2.0-jdk22plus-cuda` (tag v7.2.0 = 87a465f), JDK 25.0.2-open, RTX 4090, driver 610.57.04.

| File | What |
| --- | --- |
| `run-all-demos.log`, `logs/` | `scripts/run-all-demos.sh`: 69 passed, 0 failed, 0 skipped (demos 00-25, compile + tornado + java @argfile) |
| `setup-26.log`, `setup-27.log` | demos 26/27 `setup.sh`: JVector + jvector-gpu rebuilt against tornado-api/tornado-cuvs 7.2.0 |
| `demo26-*.log` … `demo29-*.log` | demos 26-29 under both run paths, all `PASSED` (27: `race` act; 28: `--reference`; 29: `plain`) |
| `devoxx-check.log`, `devoxx-check-logs/` | `devoxx/check.sh`: 7/7 OK (jitLLM from `JITLLM_DIR=~/demoDevoxx/jitllm`) |

Redraw-heavy logs (fancyJitllmLive, demo29) are trimmed to their last 40 KB.
