#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
namespace ct = cuda::tiles;
using namespace ct::literals;
#define CK(c) do{cudaError_t e=(c); if(e){printf("cuda err %s\n",cudaGetErrorString(e));exit(1);}}while(0)
constexpr int BM = 128, BN = 128, BK = 64;

// v0: what a tutorial writes -- runtime n, plain pointers
__tile_global__ void v0_runtime(const __half *a, const __half *b, float *c, int n) {
    auto av = ct::partition_view{ct::tensor_span{a, ct::extents{n, n}}, ct::shape<BM, BK>{}};
    auto bv = ct::partition_view{ct::tensor_span{b, ct::extents{n, n}}, ct::shape<BK, BN>{}};
    auto cv = ct::partition_view{ct::tensor_span{c, ct::extents{n, n}}, ct::shape<BM, BN>{}};
    int rb = ct::bid().x, cb = ct::bid().y;
    auto acc = ct::zeros<ct::tile<float, ct::shape<BM, BN>>>();
    for (int s = 0; s < n / BK; s++) acc = ct::mma(av.load(rb, s), bv.load(s, cb), acc);
    cv.store(acc, rb, cb);
}
// v1: + the 16-byte alignment promise TornadoVM emits
__tile_global__ void v1_aligned(const __half *a, const __half *b, float *c, int n) {
    auto av = ct::partition_view{ct::tensor_span{ct::assume_aligned(a, 16_ic), ct::extents{n, n}}, ct::shape<BM, BK>{}};
    auto bv = ct::partition_view{ct::tensor_span{ct::assume_aligned(b, 16_ic), ct::extents{n, n}}, ct::shape<BK, BN>{}};
    auto cv = ct::partition_view{ct::tensor_span{ct::assume_aligned(c, 16_ic), ct::extents{n, n}}, ct::shape<BM, BN>{}};
    int rb = ct::bid().x, cb = ct::bid().y;
    auto acc = ct::zeros<ct::tile<float, ct::shape<BM, BN>>>();
    for (int s = 0; s < n / BK; s++) acc = ct::mma(av.load(rb, s), bv.load(s, cb), acc);
    cv.store(acc, rb, cb);
}
// v2: n known at compile time, as TornadoVM's JIT knows it -- no alignment promise
template <int N>
__tile_global__ void v2_constant(const __half *a, const __half *b, float *c) {
    auto av = ct::partition_view{ct::tensor_span{a, ct::extents{N, N}}, ct::shape<BM, BK>{}};
    auto bv = ct::partition_view{ct::tensor_span{b, ct::extents{N, N}}, ct::shape<BK, BN>{}};
    auto cv = ct::partition_view{ct::tensor_span{c, ct::extents{N, N}}, ct::shape<BM, BN>{}};
    int rb = ct::bid().x, cb = ct::bid().y;
    auto acc = ct::zeros<ct::tile<float, ct::shape<BM, BN>>>();
    for (int s = 0; s < N / BK; s++) acc = ct::mma(av.load(rb, s), bv.load(s, cb), acc);
    cv.store(acc, rb, cb);
}
// v3: both -- the same information TornadoVM's generated kernel carries
template <int N>
__tile_global__ void v3_both(const __half *a, const __half *b, float *c) {
    auto av = ct::partition_view{ct::tensor_span{ct::assume_aligned(a, 16_ic), ct::extents{N, N}}, ct::shape<BM, BK>{}};
    auto bv = ct::partition_view{ct::tensor_span{ct::assume_aligned(b, 16_ic), ct::extents{N, N}}, ct::shape<BK, BN>{}};
    auto cv = ct::partition_view{ct::tensor_span{ct::assume_aligned(c, 16_ic), ct::extents{N, N}}, ct::shape<BM, BN>{}};
    int rb = ct::bid().x, cb = ct::bid().y;
    auto acc = ct::zeros<ct::tile<float, ct::shape<BM, BN>>>();
    for (int s = 0; s < N / BK; s++) acc = ct::mma(av.load(rb, s), bv.load(s, cb), acc);
    cv.store(acc, rb, cb);
}

// v4: v3 + #pragma unroll on the k-loop
template <int N>
__tile_global__ void v4_unroll(const __half *a, const __half *b, float *c) {
    auto av = ct::partition_view{ct::tensor_span{ct::assume_aligned(a, 16_ic), ct::extents{N, N}}, ct::shape<BM, BK>{}};
    auto bv = ct::partition_view{ct::tensor_span{ct::assume_aligned(b, 16_ic), ct::extents{N, N}}, ct::shape<BK, BN>{}};
    auto cv = ct::partition_view{ct::tensor_span{ct::assume_aligned(c, 16_ic), ct::extents{N, N}}, ct::shape<BM, BN>{}};
    int rb = ct::bid().x, cb = ct::bid().y;
    auto acc = ct::zeros<ct::tile<float, ct::shape<BM, BN>>>();
#pragma unroll
    for (int s = 0; s < N / BK; s++) acc = ct::mma(av.load(rb, s), bv.load(s, cb), acc);
    cv.store(acc, rb, cb);
}
// v5: v3 with the k-loop written out as straight-line code (2048/64 = 32 steps), as Graal emits it
__tile_global__ void v5_straight(const __half *a, const __half *b, float *c) {
    constexpr int N = 2048;
    auto av = ct::partition_view{ct::tensor_span{ct::assume_aligned(a, 16_ic), ct::extents{N, N}}, ct::shape<BM, BK>{}};
    auto bv = ct::partition_view{ct::tensor_span{ct::assume_aligned(b, 16_ic), ct::extents{N, N}}, ct::shape<BK, BN>{}};
    auto cv = ct::partition_view{ct::tensor_span{ct::assume_aligned(c, 16_ic), ct::extents{N, N}}, ct::shape<BM, BN>{}};
    int rb = ct::bid().x, cb = ct::bid().y;
    auto acc = ct::zeros<ct::tile<float, ct::shape<BM, BN>>>();
    acc = ct::mma(av.load(rb, 0), bv.load(0, cb), acc);
    acc = ct::mma(av.load(rb, 1), bv.load(1, cb), acc);
    acc = ct::mma(av.load(rb, 2), bv.load(2, cb), acc);
    acc = ct::mma(av.load(rb, 3), bv.load(3, cb), acc);
    acc = ct::mma(av.load(rb, 4), bv.load(4, cb), acc);
    acc = ct::mma(av.load(rb, 5), bv.load(5, cb), acc);
    acc = ct::mma(av.load(rb, 6), bv.load(6, cb), acc);
    acc = ct::mma(av.load(rb, 7), bv.load(7, cb), acc);
    acc = ct::mma(av.load(rb, 8), bv.load(8, cb), acc);
    acc = ct::mma(av.load(rb, 9), bv.load(9, cb), acc);
    acc = ct::mma(av.load(rb, 10), bv.load(10, cb), acc);
    acc = ct::mma(av.load(rb, 11), bv.load(11, cb), acc);
    acc = ct::mma(av.load(rb, 12), bv.load(12, cb), acc);
    acc = ct::mma(av.load(rb, 13), bv.load(13, cb), acc);
    acc = ct::mma(av.load(rb, 14), bv.load(14, cb), acc);
    acc = ct::mma(av.load(rb, 15), bv.load(15, cb), acc);
    acc = ct::mma(av.load(rb, 16), bv.load(16, cb), acc);
    acc = ct::mma(av.load(rb, 17), bv.load(17, cb), acc);
    acc = ct::mma(av.load(rb, 18), bv.load(18, cb), acc);
    acc = ct::mma(av.load(rb, 19), bv.load(19, cb), acc);
    acc = ct::mma(av.load(rb, 20), bv.load(20, cb), acc);
    acc = ct::mma(av.load(rb, 21), bv.load(21, cb), acc);
    acc = ct::mma(av.load(rb, 22), bv.load(22, cb), acc);
    acc = ct::mma(av.load(rb, 23), bv.load(23, cb), acc);
    acc = ct::mma(av.load(rb, 24), bv.load(24, cb), acc);
    acc = ct::mma(av.load(rb, 25), bv.load(25, cb), acc);
    acc = ct::mma(av.load(rb, 26), bv.load(26, cb), acc);
    acc = ct::mma(av.load(rb, 27), bv.load(27, cb), acc);
    acc = ct::mma(av.load(rb, 28), bv.load(28, cb), acc);
    acc = ct::mma(av.load(rb, 29), bv.load(29, cb), acc);
    acc = ct::mma(av.load(rb, 30), bv.load(30, cb), acc);
    acc = ct::mma(av.load(rb, 31), bv.load(31, cb), acc);
    cv.store(acc, rb, cb);
}
// replica: TornadoVM's generated kernel for TileContext 128x128x64, verbatim
#include "replica_kernel.inc"

static bool check(const char *name, float *dC, int n, const std::vector<__half> &hA, const std::vector<__half> &hB) {
    std::vector<float> out((size_t) n * n);
    CK(cudaMemcpy(out.data(), dC, out.size() * 4, cudaMemcpyDeviceToHost));
    std::mt19937 rng(1); int bad = 0; float maxErr = 0;
    for (int t = 0; t < 256; t++) {
        int i = rng() % n, j = rng() % n; float s = 0;
        for (int k = 0; k < n; k++) s += __half2float(hA[(size_t) i * n + k]) * __half2float(hB[(size_t) k * n + j]);
        float e = fabsf(s - out[(size_t) i * n + j]); maxErr = fmaxf(maxErr, e);
        if (e > 0.05f * n / 256.f * fmaxf(1.f, fabsf(s))) bad++;
    }
    printf("%-12s %s maxErr=%.4f bad=%d/256\n", name, bad ? "WRONG" : "correct", maxErr, bad);
    CK(cudaMemset(dC, 0, out.size() * 4));
    return !bad;
}

int main() {
    constexpr int n = 2048; const int reps = 11;
    size_t sz = (size_t) n * n;
    std::vector<__half> hA(sz), hB(sz);
    for (size_t i = 0; i < sz; i++) {
        hA[i] = __float2half(((i * 7 + 3) % 17) * 0.0625f - 0.5f);
        hB[i] = __float2half(((i * 5 + 11) % 13) * 0.0769f - 0.5f);
    }
    __half *dA, *dB; float *dC;
    CK(cudaMalloc(&dA, sz * 2)); CK(cudaMalloc(&dB, sz * 2)); CK(cudaMalloc(&dC, sz * 4));
    CK(cudaMemcpy(dA, hA.data(), sz * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), sz * 2, cudaMemcpyHostToDevice));
    dim3 g(n / BM, n / BN), blk(1, 1, 1);
    for (int r = 0; r < reps; r++) v0_runtime<<<g, blk>>>(dA, dB, dC, n);   CK(cudaDeviceSynchronize()); check("v0_runtime", dC, n, hA, hB);
    for (int r = 0; r < reps; r++) v1_aligned<<<g, blk>>>(dA, dB, dC, n);   CK(cudaDeviceSynchronize()); check("v1_aligned", dC, n, hA, hB);
    for (int r = 0; r < reps; r++) v2_constant<n><<<g, blk>>>(dA, dB, dC);  CK(cudaDeviceSynchronize()); check("v2_constant", dC, n, hA, hB);
    for (int r = 0; r < reps; r++) v3_both<n><<<g, blk>>>(dA, dB, dC);      CK(cudaDeviceSynchronize()); check("v3_both", dC, n, hA, hB);
    for (int r = 0; r < reps; r++) v4_unroll<n><<<g, blk>>>(dA, dB, dC);    CK(cudaDeviceSynchronize()); check("v4_unroll", dC, n, hA, hB);
    for (int r = 0; r < reps; r++) v5_straight<<<g, blk>>>(dA, dB, dC); CK(cudaDeviceSynchronize()); check("v5_straight", dC, n, hA, hB);
    // replica expects TornadoVM's buffer layout: data 16 bytes past the argument pointer,
    // and (after #1066) 32-byte-aligned data. Copy inputs into such buffers.
    unsigned char *rA, *rB, *rC;
    CK(cudaMalloc(&rA, sz * 2 + 64)); CK(cudaMalloc(&rB, sz * 2 + 64)); CK(cudaMalloc(&rC, sz * 4 + 64));
    CK(cudaMemcpy(rA + 32, hA.data(), sz * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(rB + 32, hB.data(), sz * 2, cudaMemcpyHostToDevice));
    for (int r = 0; r < reps; r++) replica<<<g, blk>>>(nullptr, nullptr, nullptr, nullptr, rA + 16, rB + 16, rC + 16, n);
    CK(cudaDeviceSynchronize()); check("replica", (float *) (rC + 32), n, hA, hB);
    return 0;
}
