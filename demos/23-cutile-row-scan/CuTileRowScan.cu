// Hand-written CUDA equivalent of CuTileRowScan.java: the same per-row prefix sum over a
// ragged extent, written directly against NVIDIA's cuda::tiles C++ surface.
//
//   Pure cuTile C reference : NVIDIA/TileGym, src/tilegym/ops/tilecpp/moe_align_block.cuh
//                             (stage 2 is the ct::partial_sum scan; stage 3 carries a
//                              running total across ct::irange, both mirrored here)
//   Reference revision      : ec339c0dbac3efe61e73ac2b782f818d253eaff2 (MIT)
//   Reference source        : https://github.com/NVIDIA/TileGym
//   Adapted, not copied     : reduced to a plain per-row scan over a ragged extent.
//
// Build (CUDA Tile C++ needs toolkit 13.3 or newer; a userspace pip install is enough):
//   pip install --user nvidia-cuda-nvcc 'cuda-tile[tileiras]' nvidia-cuda-cccl
//   nvcc --enable-tile -std=c++20 -arch=sm_120 -O3 -o cutile_row_scan CuTileRowScan.cu
//   ./cutile_row_scan 4096 1000 20
//
// --enable-tile mixes tile kernels and host code in one translation unit, which is what
// lets this be an executable. TornadoVM instead drives `nvcc -tilecubin --tile-only` to a
// bare cubin and loads it itself, because it has no host translation unit to put the
// launch in. The kernel below is otherwise the same one the Java demo generates.

#include <cuda_tile.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

namespace ct = cuda::tiles;
using namespace ct::literals;

#define CK(c) do{cudaError_t e=(c); if(e){printf("cuda err %s\n",cudaGetErrorString(e));exit(1);}}while(0)

constexpr int TILE = 128;

// out[r][c] = sum of in[r][0..c]; totals[r] = the row's full sum.
// One tile block per row. cols need not be a multiple of TILE: load_masked zero-pads the
// ragged tail and store_masked writes back only the lanes that exist.
extern "C" __tile_global__ void rowScan(const float *in, float *out, float *totals, //
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

int main(int argc, char **argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 4096;
    int cols = argc > 2 ? atoi(argv[2]) : 1000;
    int reps = argc > 3 ? atoi(argv[3]) : 20;
    int colBlocks = (cols + TILE - 1) / TILE;

    size_t sz = (size_t) rows * cols;
    std::vector<float> hIn(sz);
    // Small integers, so every prefix is exact in fp32 on both paths and the comparison
    // below can be == rather than a tolerance. Same generator as the Java demo.
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            hIn[(size_t) r * cols + c] = (float) (((r + c) % 7) - 3);

    float *dIn, *dOut, *dTotals;
    CK(cudaMalloc(&dIn, sz * sizeof(float)));
    CK(cudaMalloc(&dOut, sz * sizeof(float)));
    CK(cudaMalloc(&dTotals, (size_t) rows * sizeof(float)));
    CK(cudaMemcpy(dIn, hIn.data(), sz * sizeof(float), cudaMemcpyHostToDevice));

    printf("CUDA Tile row scan (hand-written CUDA): %d rows x %d cols, tile width %d\n", rows, cols, TILE);
    printf("  last tile of each row has %d padded lane(s) handled by load_masked/store_masked\n\n",
           colBlocks * TILE - cols);

    // The grid counts TILE BLOCKS -- one per row -- and the block is 1x1x1: the tile
    // compiler decides how many threads actually back it.
    dim3 grid(rows), block(1, 1, 1);
    for (int i = 0; i < 3; i++) rowScan<<<grid, block>>>(dIn, dOut, dTotals, rows, cols, colBlocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(a);
        rowScan<<<grid, block>>>(dIn, dOut, dTotals, rows, cols, colBlocks);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); t.push_back(ms * 1000.0);
    }
    std::sort(t.begin(), t.end());
    printf("kernel time (CUDA events, median of %d): %.1f us\n\n", reps, t[t.size() / 2]);

    std::vector<float> hOut(sz), hTotals(rows);
    CK(cudaMemcpy(hOut.data(), dOut, sz * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hTotals.data(), dTotals, (size_t) rows * sizeof(float), cudaMemcpyDeviceToHost));

    long scanMismatches = 0, totalMismatches = 0;
    for (int r = 0; r < rows; r++) {
        float running = 0.0f;
        for (int c = 0; c < cols; c++) {
            running += hIn[(size_t) r * cols + c];
            if (hOut[(size_t) r * cols + c] != running) scanMismatches++;
        }
        if (hTotals[r] != running) totalMismatches++;
    }

    printf("scan mismatches %ld/%zu, row-total mismatches %ld/%d\n", scanMismatches, sz, totalMismatches, rows);
    bool ok = scanMismatches == 0 && totalMismatches == 0;
    printf("%s\n", ok ? "result is correct (bit-exact prefix sums and row totals)" : "WRONG");

    cudaFree(dIn); cudaFree(dOut); cudaFree(dTotals);
    return ok ? 0 : 1;
}
