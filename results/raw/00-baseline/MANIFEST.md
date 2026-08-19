# Machine manifest — batch 00 baseline

Run date: 2026-08-19 (host `storm`)

## Evidence classification: Observed

All values below were captured directly from command output on this machine; see the accompanying raw log files in this directory.

## Source

- TornadoVM: `beehive-lab/TornadoVM`, branch `develop`, SHA `99549c9862eda8d584e35e99924f9c865501eb3a`.
- Checked out at `vendor/tornadovm` (excluded from git via `.gitignore`; not committed).
- `git ls-remote --heads origin` confirmed `develop` and `master` pointed at the same commit at clone time.

## GPU / driver

See `nvidia-smi-q.log` and `nvidia-smi-gpu.csv`.

- GPU: NVIDIA GeForce RTX 4090
- Driver: 565.57.01
- Compute capability: 8.9
- Memory: 24564 MiB
- Persistence mode: Enabled
- `nvidia-smi`-reported CUDA version: 12.7 (driver-supported API level, not the installed toolkit)

## CUDA toolkit (as installed / used by the build)

- `nvcc`: release 12.6, V12.6.85 (build cuda_12.6.r12.6/compiler.35059454_0)
- `ptxas`: release 12.6, V12.6.85

## JDK / build tools

- JDK: OpenJDK 21.0.2 (2024-01-16), via sdkman `21.0.2-open`
- Maven: 3.6.3
- gcc: 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04)
- cmake: 3.22.1
- OS: Ubuntu 22.04.5 LTS, kernel 6.8.0-58-generic

## Build command

```
cd vendor/tornadovm
make BACKEND=cuda
```

Full build log: `tornadovm-build-cuda.log`. Result: `BUILD SUCCESS` for all modules; `TornadoVM build success`, `Backend : CUDA`, `Commit : 99549c9`.

## Device exposure check

```
source vendor/tornadovm/setvars.sh
tornado --devices
```

Output: `tornado-devices.log` — one CUDA driver, one device (`0:0`, DEFAULT), `NVIDIA GeForce RTX 4090`, 23.5 GB global memory.

## Smoke example

`uk.ac.manchester.tornado.examples.VectorAddInt` (size=256), run via:

```
tornado --enableProfiler console -cp <tornado-examples jar> uk.ac.manchester.tornado.examples.VectorAddInt 256
```

Output: `vectoradd-int-cuda-run.log` — 10/10 iterations report `Result is correct`, profiler JSON shows `"BACKEND": "CUDA"`, `"DEVICE": "NVIDIA GeForce RTX 4090"`, `"DEVICE_ID": "0:0"` for every iteration.
