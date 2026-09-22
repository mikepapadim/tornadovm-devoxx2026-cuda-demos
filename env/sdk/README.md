# SDK profiles

Each file here pins **one** TornadoVM SDK the demos can run against. Exactly one is
active at a time, selected by `TORNADO_SDK_PROFILE` in `../versions.env`.

    source scripts/setup-env.sh                    # the pinned profile

    export TORNADO_SDK_PROFILE=sdkman-6.0.0        # a one-off override
    source scripts/setup-env.sh

Export the override *before* sourcing. `TORNADO_SDK_PROFILE=x source scripts/setup-env.sh`
selects the right SDK, but bash restores the variable when the command finishes, so later
scripts in the same shell (`run-all-demos.sh`) would no longer see the name.

## Why this exists

The tile demos (19-24) need a TornadoVM with `TileContext`. Until 7.0.0 no release had it,
so the default was a source build of `develop`; since 2026-09-22 the default is the released
`sdkman-7.0.0`, which has it. That used to mean a second runner script and a hand-set `TORNADOVM_HOME`.
Now it is one line, and a demo that the active SDK cannot run reports `SKIPPED_REQUIREMENT`
rather than failing.

## Adding a profile

Copy an existing file and set every variable below — `scripts/setup-env.sh` and
`scripts/run-all-demos.sh` read these and nothing else, so no consumer has to branch on the
profile name.

| Variable | Meaning |
| --- | --- |
| `TORNADO_SDK_KIND` | `sdkman` (an installed SDKMAN candidate) or `local-build` (a `dist/` tree built from source) |
| `TORNADO_SDKMAN_CANDIDATE` | `sdkman` kind only: the candidate name |
| `TORNADO_BUILD_DIR` | `local-build` kind only: the checkout whose `dist/` holds the SDK |
| `JDK_SDKMAN_CANDIDATE` | JDK this SDK requires |
| `TORNADO_VERSION` | version string, for the manifest |
| `TORNADO_SHA` / `TORNADO_RELEASE_COMMIT` | provenance: exact upstream commit |
| `TORNADO_HAS_TILE_API` | `1` if `uk.ac.manchester.tornado.api.tile` is present, else `0` |
| `TORNADO_MIN_CUDA_TOOLKIT` | minimum nvcc for the tile path, as `major*1000+minor` (`0` when not applicable) |
| `TORNADO_NVCC` | nvcc to use for tile compilation, or empty to let TornadoVM locate one |
| `TORNADO_JAVAC_FLAGS` | extra `javac` flags this SDK needs (e.g. a preview-compiled build) |
| `TORNADO_JVM_FLAGS` | extra JVM flags for every run on this SDK |
| `TORNADO_LD_LIBRARY_PATH` | optional: a directory prepended to `LD_LIBRARY_PATH` (7.0.0 needs a CUDA 13 runtime for its CUTLASS bridge) |

When a release ships the tile API, add `sdkman-<version>.env` with
`TORNADO_HAS_TILE_API=1` and change the one selector line.
