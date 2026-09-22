# Batch 44 — Every demo on TornadoVM 7.0.0, after merging the CUDA Tile work

Captured 2026-09-22 on the merge of two lines of work: the 7.0.0 migration and
re-measurement (batches 40–43, this machine) and the CUDA Tile demos 19–24 with the
SDK-profile system (batches 33–39, done in parallel on the sm_120 machine).

RTX 4090 (sm_89), driver 610.57.04, JDK 25.0.2, profile `sdkman-7.0.0`
(`7.0.0-jdk22plus-cuda`, commit 65eb834), CUDA toolkit 12.6.85 plus the pip-wheel
CUDA 13.3 `nvcc`/`tileiras` (cuda-tile 1.5.0) for the tile path and a CUDA 13 runtime
for the CUTLASS bridge (`setup-env.log`).

## Why 7.0.0 became the default profile

The tile demos had a `develop` profile only because no release had the tile API.
7.0.0 does: PR #1083 is an ancestor of tag `v7.0.0`, and every class demos 19–24 import
(`uk.ac.manchester.tornado.api.tile.{Tile,TileContext,DType,PartitionView}`,
`TornadoDeviceTileNotSupported`) is in `tornado-api-7.0.0.jar`. `env/sdk/README.md`
already prescribed the step: add `sdkman-<version>.env` with `TORNADO_HAS_TILE_API=1`
and change the selector.

## Result

- **`scripts/run-all-demos.sh`: 66 passed, 0 failed, 0 skipped** (`run-all-demos.log`) —
  22 demos × compile, `tornado` launcher, `java @argfile`. The runner passes
  `-Dtornado.recover.bailout=False`, so a tile kernel that failed to compile would error
  rather than fall back to the host.
- **Tile kernels really run on sm_89.** `nsys` on demo 19 (`19-nsys-kernels.txt`): the
  `tiles` kernel launches 11 times at 2.6 µs, next to `naive` (237.0 µs) and
  `tiledByHand` (46.8 µs).
- **`scripts/run-all-cuda.sh`: 43 passed, 2 failed** (`run-all-cuda.log`, CUTLASS 3.5.1).
  Upstream recorded 44 passed, 0 failed, 1 skipped on sm_120. The two failures here are
  toolchain effects, neither caused by the merge:
  - **05** (`cuda-05-cufft-run.log`, `cuFFT error 5`): with a tile profile active,
    `setup-env.sh` puts the pip-wheel `nvcc` 13.3 first on `PATH`; that wheel has no
    cuFFT, so the binary links the system's `libcufft.so.10`. Rebuilt with the toolkit's
    `nvcc` 12.6 it passes, with or without the CUDA 13 runtime on `LD_LIBRARY_PATH`.
  - **24** (`cuda-24-histogram-build.log`): does not compile against this box's CUDA Tile
    headers — `partition_view` has no member `atomic_add`. Compiles on sm_120.

## Files

| File | What it is |
|---|---|
| `setup-env.log` | `source scripts/setup-env.sh` on the merged tree, default profile |
| `run-all-demos.log` | the Java suite, 66/66 |
| `run-all-cuda.log` | the hand-written CUDA suite, 43/45 |
| `cuda-05-cufft-run.log`, `cuda-24-histogram-build.log` | the two CUDA-side failures |
| `19-nsys-kernels.txt` | demo 19 kernels under `nsys` on sm_89 |
