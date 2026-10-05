# Batch 50: demo 29, the Hybrid API live

Captured 2026-10-05 on the sm_89 host: RTX 4090, driver 610.57.04, JDK 25.0.2, TornadoVM 7.0.0
(`7.0.0-jdk22plus-cuda`).

| log | run | executions correct | CUDA graph (plain → graph, median of 300) | verdict | wall |
|---|---|---|---|---|---|
| `run-plain-tornado.log.gz` | plain, tornado | 110/110 | 63.3 → 38.9 µs | PASSED | 11.0 s |
| `run-plain-java-argfile.log.gz` | plain, java @argfile | 110/110 | 62.1 → 38.9 µs | PASSED | 10.9 s |
| `dashboard-screen.log.gz` and `dashboard-run.log.gz` | the dashboard (NO_PAUSE=1) | 110/110 | in the log | PASSED | 15.1 s |

* **Median GPU time per task**, from TornadoVM's profiler (plain tornado run, executions 2–110): forward=96.5 lowPass=75.8 inverse=53.0 normalize=39.6 µs.
* **Copies per execution:** 32,804 B in and 65,576 B out (`TOTAL_COPY_IN/OUT_SIZE_BYTES`).
* **Frames are paced** at about 90 ms, so single execution times vary as the GPU clocks down between them.
* **The screenshots** in `demos/29-hybrid-api-live/screenshots/` are frames of a dashboard run on this machine,
  rendered from the ANSI.
