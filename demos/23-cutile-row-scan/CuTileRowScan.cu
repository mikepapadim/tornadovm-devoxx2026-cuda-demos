// Pure CUDA Tile C++ twin of demo 23 (CuTileRowScan.java).
//
//   Pure cuTile C reference : NVIDIA/TileGym, src/tilegym/ops/tilecpp/moe_align_block.cuh
//                             (stage 2 is the ct::partial_sum scan; stage 3 carries a
//                              running total across ct::irange, both mirrored here)
//   Reference revision      : ec339c0dbac3efe61e73ac2b782f818d253eaff2 (MIT)
//   Reference source        : https://github.com/NVIDIA/TileGym
//   Adapted, not copied     : reduced to a plain per-row scan over a ragged extent.
//
// Build (needs CUDA Toolkit >= 13.3; tileiras must be on PATH):
//   nvcc -tilecubin --tile-only -std=c++20 -arch=sm_120 -o CuTileRowScan.cubin CuTileRowScan.cu
//
// -tilecubin emits a cubin, not an executable: a tile kernel is launched by a host that
// cuModuleLoads it, which is exactly what TornadoVM does.

#include "cuda_tile.h"

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int TILE = 128;

// out[r][c] = sum of in[r][0..c]; totals[r] = the row's full sum.
// One tile block per row. cols need not be a multiple of TILE: load_masked zero-pads the
// ragged tail and store_masked writes back only the lanes that exist.
extern "C" __tile_global__ void rowScan(float *in, float *out, float *totals, //
                                        int rows, int cols, int colBlocks) {
    auto iv = ct::partition_view{ct::tensor_span{in, ct::extents{rows, cols}}, ct::shape<1, TILE>{}};
    auto ov = ct::partition_view{ct::tensor_span{out, ct::extents{rows, cols}}, ct::shape<1, TILE>{}};
    auto tv = ct::partition_view{ct::tensor_span{totals, ct::extents{rows, 1}}, ct::shape<1, 1>{}};

    int row = ct::bid().x;
    auto carry = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
    for (int block = 0; block < colBlocks; block++) {
        auto values = iv.load_masked(row, block);
        // [1,TILE] + [1,1] broadcasts the carry across the tile.
        ov.store_masked(ct::partial_sum(values, ct::integral_constant<1>{}) + carry, row, block);
        carry = carry + ct::sum(values, ct::integral_constant<1>{});
    }
    tv.store(carry, row, 0);
}
