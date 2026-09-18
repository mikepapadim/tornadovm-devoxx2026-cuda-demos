// Hand-written CUDA equivalent of CuTileHistogram.java: the same value histogram, folded
// into shared bins with the CUDA Tile atomics.
//
//   Pure cuTile C reference : NVIDIA/TileGym, src/tilegym/ops/tilecpp/moe_align_block.cuh
//                             (stage 1: per-block counts folded into a shared array)
//   Reference revision      : ec339c0dbac3efe61e73ac2b782f818d253eaff2 (MIT)
//   Reference source        : https://github.com/NVIDIA/TileGym
//   Adapted, not copied     : reduced to a plain value histogram.
//
// Build (CUDA Tile C++ needs toolkit 13.3 or newer; a userspace pip install is enough):
//   pip install --user nvidia-cuda-nvcc 'cuda-tile[tileiras]' nvidia-cuda-cccl
//   nvcc --enable-tile -std=c++20 -arch=sm_120 -O3 -o cutile_histogram CuTileHistogram.cu
//   ./cutile_histogram 1048576 256 20
//
// Confirm the atomics survive to SASS:
//   cuobjdump -sass cutile_histogram | grep -iE 'ATOM|RED'

#include <cuda_tile.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

namespace ct = cuda::tiles;
using namespace ct::literals;

#define CK(c) do{cudaError_t e=(c); if(e){printf("cuda err %s\n",cudaGetErrorString(e));exit(1);}}while(0)

constexpr int TILE = 256;

// counts[b] = number of values v with b <= v < b+1.
//
// One tile block per chunk of TILE values. Every block walks every bin, so every add is
// contended -- which is the point. A plain store would keep only the last block's count.
extern "C" __tile_global__ void histogram(const float *values, float *counts, int n, int bins) {
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
        // The Java API fixes relaxed ordering at device scope; in C++ both are explicit.
        bv.atomic_add(count, ct::memory_order_relaxed_t{}, ct::thread_scope_device_t{}, 0, bin);
    }
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1048576;
    int bins = argc > 2 ? atoi(argv[2]) : 256;
    int reps = argc > 3 ? atoi(argv[3]) : 20;
    if (n % TILE) { printf("n must be a multiple of %d; got %d\n", TILE, n); return 1; }

    std::vector<float> hValues(n);
    // Whole numbers in [0, bins), so each lands in exactly one bin and every count is an
    // exact integer in fp32 however the atomic adds interleave. Same generator as the Java demo.
    for (int i = 0; i < n; i++) hValues[i] = (float) (((long long) i * 31 + 7) % bins);

    float *dValues, *dCounts;
    CK(cudaMalloc(&dValues, (size_t) n * sizeof(float)));
    CK(cudaMalloc(&dCounts, (size_t) bins * sizeof(float)));
    CK(cudaMemcpy(dValues, hValues.data(), (size_t) n * sizeof(float), cudaMemcpyHostToDevice));

    printf("CUDA Tile histogram (hand-written CUDA): %d values into %d bins, tile width %d\n", n, bins, TILE);
    printf("  every one of the %d blocks folds into every one of the %d bins -- %lld contended adds\n\n",
           n / TILE, bins, (long long) (n / TILE) * bins);

    // The grid counts TILE BLOCKS and the block is 1x1x1.
    dim3 grid(n / TILE), block(1, 1, 1);
    // The accumulator must start at zero on every launch, or the bins keep summing.
    auto launch = [&]{ CK(cudaMemset(dCounts, 0, (size_t) bins * sizeof(float)));
                       histogram<<<grid, block>>>(dValues, dCounts, n, bins); };

    for (int i = 0; i < 3; i++) launch();
    CK(cudaDeviceSynchronize());

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(a); launch(); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); t.push_back(ms * 1000.0);
    }
    std::sort(t.begin(), t.end());
    printf("kernel time incl. the zeroing memset (CUDA events, median of %d): %.1f us\n\n", reps, t[t.size() / 2]);

    std::vector<float> hCounts(bins);
    CK(cudaMemcpy(hCounts.data(), dCounts, (size_t) bins * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<long> expected(bins, 0);
    for (int i = 0; i < n; i++) expected[(int) hValues[i]]++;
    long mismatches = 0, total = 0;
    for (int b2 = 0; b2 < bins; b2++) {
        total += (long) hCounts[b2];
        if ((long) hCounts[b2] != expected[b2]) mismatches++;
    }

    printf("bins wrong %ld/%d, counted %ld of %d values\n", mismatches, bins, total, n);
    bool ok = mismatches == 0 && total == n;
    printf("%s\n", ok ? "result is correct (every bin exact, all values counted)" : "WRONG");

    cudaFree(dValues); cudaFree(dCounts);
    return ok ? 0 : 1;
}
