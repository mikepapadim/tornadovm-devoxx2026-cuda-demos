# RESOLVED (2026-09-03) — Nsight Compute (`ncu`) hardware counter access denied

Date: 2026-08-20. Task: `auto/tasks/08.md`.
**Resolved 2026-09-03.** Both blockers below were real and independent; the fix
for each is in "Resolution" at the end. The measurement this document was
written to record is now captured in
`results/raw/23-ncu-tensor-core-counters/`, and the same unblocking produced
`results/raw/22-ncu-alignment-counters/`. The analysis below is kept because
the diagnosis — in particular that the "stub libcuda.so" message is a version
mismatch, not a library-path problem — is what made the fix findable.

## What was attempted

Capture Nsight Compute Tensor Core activity/instruction-count metrics (e.g.
`sm__pipe_tensor_op_hmma_cycles_active_v2.sum`,
`sm__inst_executed_pipe_tensor_op_hmma_v2.sum`) for
`demos/08-tensor-core-mma/TensorCoreMMA.java`'s `gemmMMASingleTile` kernel, to
supplement the generated-code evidence with a hardware-counter measurement.

## Commands and exact errors

1. Default `ncu` on `PATH` resolves to `/usr/local/cuda-12.6/bin/ncu`, which
   is a version-selector wrapper that always launches the newest installed
   Nsight Compute toolkit — on this machine that is `2026.2.1.0`
   (`/opt/nvidia/nsight-compute/2026.2.1`):

   ```
   ncu --target-processes all -k "regex:^gemmMMA$" --launch-count 1 \
     --metrics sm__pipe_tensor_op_hmma_cycles_active_v2.sum,... \
     tornado -m tornado.examples/.../MatrixMultiplicationMMA 512 512 512
   ```
   ```
   ==ERROR== Nsight Compute failed to connect to the CUDA driver (stub libcuda.so[.1] on path?).
   ==ERROR== The application returned an error code (1).
   ```
   Reproduced on a trivial standalone CUDA C program (`/tmp/t.cu`, one
   `__global__` kernel, compiled with the pinned `nvcc` 12.6.85) with the
   same wrapper — confirms this is not TornadoVM/JNI-specific. `LD_LIBRARY_PATH`
   is empty and the real driver library is present and otherwise functional
   (`/usr/lib/x86_64-linux-gnu/libcuda.so.1`, driver `565.57.01`, `nvidia-smi`
   and normal CUDA execution — `nvcc`-compiled binaries, `tornado`, `nsys` —
   all work). Root cause: `2026.2.1.0` (published far ahead of this driver's
   Oct 2024 release) cannot establish its profiling connection against
   driver `565.57.01`.

2. Found an older Nsight Compute install matching the pinned CUDA 12.6
   toolkit era, `/opt/nvidia/nsight-compute/2024.3.2/ncu`
   (`Version 2024.3.2.0`), invoked directly (bypassing the version-selector
   wrapper):

   ```
   /opt/nvidia/nsight-compute/2024.3.2/ncu --target-processes all -k k \
     --launch-count 1 --metrics gpu__time_duration.sum /tmp/t
   ```
   This connects to the driver successfully (`==PROF== Connected to process
   ...`) but fails with:
   ```
   ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
   NVIDIA GPU Performance Counters on the target device 0. For instructions
   on enabling permissions and to get more information see
   https://developer.nvidia.com/ERR_NVGPUCTRPERM
   ```
   This is the standard NVIDIA driver restriction
   (`NVreg_RestrictProfilingToAdminUsers=1`, the default) that limits GPU
   performance-counter access to root/admin unless the kernel module
   parameter is changed (requires a `modprobe.d` config change and a driver
   reload or reboot) or the profiling process runs with elevated privileges.

3. Checked for non-interactive privilege escalation: `sudo -n true` →
   `sudo: a password is required`. No passwordless sudo is configured for
   this user on this machine, and this is an unattended autonomous run with
   no human available to supply a password — did not attempt to prompt for
   one, and did not modify the NVIDIA kernel module parameters (a
   system-wide, hard-to-reverse change requiring a reboot, out of scope for
   this task and outside the "reversible local action" bar for autonomous
   changes).

## Environment

- GPU: NVIDIA GeForce RTX 4090, driver `565.57.01`.
- CUDA toolkit: `nvcc`/`ptxas` 12.6.85 (`/usr/local/cuda-12.6`).
- Nsight Compute installs present: `2023.1.1`, `2024.3.2`, `2026.2.1`
  (under `/opt/nvidia/nsight-compute/`); the `ncu` on `PATH` always resolves
  to the newest (`2026.2.1.0`, `build 38283040`, `public-release`).
- Nsight Systems (`nsys`) is unaffected — it does not require GPU
  performance-counter access and was already used successfully for demo 06
  (`results/raw/06-cuda-streams/nsys-*.nsys-rep`).

## What this means for task 08's acceptance criterion

The acceptance criterion is "Only call it Tensor Core/MMA accelerated when
**profiler or generated-code evidence** supports that claim" (disjunctive).
Nsight Compute hardware-counter evidence is blocked for the reason above.
Generated-code evidence is captured instead and is sufficient on its own:
`results/raw/08-tensor-core-mma/tensorcoremma-printkernel.log` and
`results/raw/08-tensor-core-mma/mma-printkernel-512.log` show the CUDA
backend's JIT compiler emitting a real
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` PTX instruction (plus
`ldmatrix.sync.aligned` operand loads) for the MMA kernels, and **zero**
`mma.sync` instructions for the scalar reference kernel compiled from the
same run — a direct, source-backed comparison, not an inference from
performance numbers alone.

## Next action (if a future invocation has GPU counter access)

- Either run as root/with `sudo`, or set
  `options nvidia NVreg_RestrictProfilingToAdminUsers=0` in
  `/etc/modprobe.d/`, reload the `nvidia` kernel module (or reboot), then:
  ```
  /opt/nvidia/nsight-compute/2024.3.2/ncu --target-processes all \
    -k "regex:^gemmMMASingleTile$" --launch-count 1 \
    --metrics sm__pipe_tensor_op_hmma_cycles_active_v2.sum,sm__inst_executed_pipe_tensor_op_hmma_v2.sum \
    tornado --classpath demos/08-tensor-core-mma TensorCoreMMA
  ```
  (use the `2024.3.2` install directly, not the bare `ncu` on `PATH`, since
  that resolves to `2026.2.1.0` which cannot connect to this machine's
  driver at all — a separate, more fundamental blocker than the permission
  error).


## Resolution (2026-09-03)

Both blockers were fixed, in the order they were diagnosed.

**Blocker 1 — wrong `ncu`.** `/usr/local/cuda/bin/ncu` selects `2026.2.1.0`,
which cannot connect to driver `565.57.01` at all. Its error message
("failed to connect to the CUDA driver (stub libcuda.so on path?)") is
misleading: `LD_LIBRARY_PATH` was empty and `libcuda.so.1` resolved correctly
to `/lib/x86_64-linux-gnu/libcuda.so.1` throughout. No library-path change was
ever needed. **Fix: invoke `/opt/nvidia/nsight-compute/2024.3.2/ncu` by
absolute path.**

**Blocker 2 — counter permission.** `RmProfilingAdminOnly: 1` in
`/proc/driver/nvidia/params` → `ERR_NVGPUCTRPERM`. Fixed system-wide:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
  | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u
sudo reboot
```

Verified after reboot — `RmProfilingAdminOnly: 0`. This required an
interactive `sudo` password (`sudo -n true` still reports a password is
required), which is why the original autonomous run could not do it.

```
$ /opt/nvidia/nsight-compute/2024.3.2/ncu --metrics smsp__cycles_elapsed.avg ./01-first-cuda-kernel
==PROF== Connected to process 3931
  vectorAdd(const float *, const float *, float *, int) ... CC 8.9
    smsp__cycles_elapsed.avg    cycle    5,028.50
```

**Note for a fresh machine:** the metric names in the "Next action" command
above use the `_v2` suffix, which `2024.3.2` does not accept. The working
names on this install are `sm__inst_executed_pipe_tensor_op_hmma.sum` and
`sm__pipe_tensor_op_hmma_cycles_active.sum` (no suffix).
