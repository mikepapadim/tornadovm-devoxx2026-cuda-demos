// Pure CUDA Tile C++ twin of demo 24 (CuTileHistogram.java).
//
//   Pure cuTile C reference : NVIDIA/TileGym, src/tilegym/ops/tilecpp/moe_align_block.cuh
//                             (stage 1 builds per-block counts and folds them into a shared
//                              array; the same predicate-count-accumulate shape)
//   Reference revision      : ec339c0dbac3efe61e73ac2b782f818d253eaff2 (MIT)
//   Reference source        : https://github.com/NVIDIA/TileGym
//   Adapted, not copied     : reduced to a plain value histogram.
//
// Build (needs CUDA Toolkit >= 13.3; tileiras must be on PATH):
//   nvcc -tilecubin --tile-only -std=c++20 -arch=sm_120 -o CuTileHistogram.cubin CuTileHistogram.cu
//
// -tilecubin emits a cubin, not an executable: a tile kernel is launched by a host that
// cuModuleLoads it, which is exactly what TornadoVM does. Confirm the atomics with:
//   cuobjdump -sass CuTileHistogram.cubin | grep -i ATOM

#include "cuda_tile.h"

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int TILE = 256;

// counts[b] = number of values v with b <= v < b+1.
//
// One tile block per chunk of TILE values. Every block walks every bin, so every add is
// contended -- which is the point. A plain store would keep only the last block's count.
extern "C" __tile_global__ void histogram(float *values, float *counts, int n, int bins) {
    auto vv = ct::partition_view{ct::tensor_span{values, ct::extents{1, n}}, ct::shape<1, TILE>{}};
    // The bin row is partitioned 1x1, so a bin index is a block index.
    auto bv = ct::partition_view{ct::tensor_span{counts, ct::extents{1, bins}}, ct::shape<1, 1>{}};

    auto ones = ct::full<ct::tile<float, ct::shape<1, TILE>>>(1.0f);
    auto zeros = ct::zeros<ct::tile<float, ct::shape<1, TILE>>>();

    auto chunk = vv.load(0, ct::bid().x);
    for (int bin = 0; bin < bins; bin++) {
        // Data-dependent selection without a gather: a predicate over the whole tile.
        auto hit = (chunk >= ct::broadcast<ct::shape<1, TILE>>(float(bin)))
                 && (chunk < ct::broadcast<ct::shape<1, TILE>>(float(bin + 1)));
        auto count = ct::sum(ct::select(hit, ones, zeros), ct::integral_constant<1>{});
        bv.atomic_add(count, ct::memory_order_relaxed_t{}, ct::thread_scope_device_t{}, 0, bin);
    }
}
