// Hand-written counterpart of TileLadder.java: the TileContext ladder in native CUDA Tile C++,
// plus cuBLAS, on the same FP16 C = A * B with FP32 accumulate and FP32 output.
//
// Part 1 -- the tile-shape ladder, written the way a CUDA Tile tutorial writes it: `n` is a
// kernel argument. These are the native twins of the Java rungs 3-6, and they are 1.8-4.5x
// SLOWER than the Java rungs at n=2048 on sm_89.
//
// Part 2 -- why. TornadoVM compiles a tile kernel after its arguments are known, and its
// generated source (tornado --printKernel) carries three facts this file's part 1 does not:
//   (a) the extents as constants     ct::extents{2048, 2048}, not {n, n}
//   (b) an alignment promise         ct::assume_aligned(ptr, 16_ic)
//   (c) the k-loop as straight code  32 load/load/mma steps, no loop
// Part 2 adds them to the 128x128x64 kernel one step at a time. With all three it runs at
// exactly TileContext's speed. (c) matters: `#pragma unroll` on the loop does NOT produce it.
// A last variant adds the launch hint occupancy=2, as the Java rung 7 does.
//
// An AOT program has to pick its sizes up front, so part 2 is compiled for n = 256 and
// n = 2048 only and skipped for any other n. That restriction is the point: a JIT gets
// (a)-(c) for every n it is ever called with.
//
// Build (CUDA Tile C++ needs toolkit 13.3 or newer; a userspace pip install is enough):
//   pip install --user nvidia-cuda-nvcc 'cuda-tile[tileiras]' nvidia-cuda-cccl
//   nvcc --enable-tile -std=c++20 -arch=sm_89 -O3 -o tile_ladder TileLadder.cu -lcublas
//   ./tile_ladder 2048 10
// Time kernels with nsys (README.md); the event timings printed here are a convenience.

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

namespace ct = cuda::tiles;
using namespace ct::literals;

#define CK(c) do{cudaError_t e=(c); if(e){printf("cuda err %s\n",cudaGetErrorString(e));exit(1);}}while(0)
#define CB(c) do{cublasStatus_t s=(c); if(s){printf("cublas err %d\n",(int)s);exit(1);}}while(0)

// ------------------------------------------------------------------ part 1: runtime n

template <int BM, int BN, int BK>
__tile_global__ void native_runtime(const __half *a, const __half *b, float *c, int n) {
    auto av = ct::partition_view{ct::tensor_span{a, ct::extents{n, n}}, ct::shape<BM, BK>{}};
    auto bv = ct::partition_view{ct::tensor_span{b, ct::extents{n, n}}, ct::shape<BK, BN>{}};
    auto cv = ct::partition_view{ct::tensor_span{c, ct::extents{n, n}}, ct::shape<BM, BN>{}};
    int rowBlock = ct::bid().x, columnBlock = ct::bid().y;
    auto acc = ct::zeros<ct::tile<float, ct::shape<BM, BN>>>();
    for (int k = 0; k < n / BK; k++) acc = ct::mma(av.load(rowBlock, k), bv.load(k, columnBlock), acc);
    cv.store(acc, rowBlock, columnBlock);
}

// ------------------------------------------------------------------ part 2: 128x128x64

constexpr int BM = 128, BN = 128, BK = 64;

// (a) + (b): constant extents and the alignment promise, loop kept.
template <int N>
__tile_global__ void native_specialised(const __half *a, const __half *b, float *c) {
    auto av = ct::partition_view{ct::tensor_span{ct::assume_aligned(a, 16_ic), ct::extents{N, N}}, ct::shape<BM, BK>{}};
    auto bv = ct::partition_view{ct::tensor_span{ct::assume_aligned(b, 16_ic), ct::extents{N, N}}, ct::shape<BK, BN>{}};
    auto cv = ct::partition_view{ct::tensor_span{ct::assume_aligned(c, 16_ic), ct::extents{N, N}}, ct::shape<BM, BN>{}};
    int rowBlock = ct::bid().x, columnBlock = ct::bid().y;
    auto acc = ct::zeros<ct::tile<float, ct::shape<BM, BN>>>();
    for (int k = 0; k < N / BK; k++) acc = ct::mma(av.load(rowBlock, k), bv.load(k, columnBlock), acc);
    cv.store(acc, rowBlock, columnBlock);
}

// (a) + (b) + (c): the k-loop written out, as TornadoVM's generated kernel has it.
#define STEP(k) acc = ct::mma(av.load(rowBlock, k), bv.load(k, columnBlock), acc);
#define STEPS4(k) STEP(k) STEP(k + 1) STEP(k + 2) STEP(k + 3)
#define STEPS32 STEPS4(0) STEPS4(4) STEPS4(8) STEPS4(12) STEPS4(16) STEPS4(20) STEPS4(24) STEPS4(28)
#define SPECIALISED_PROLOGUE(N)                                                                                        \
    auto av = ct::partition_view{ct::tensor_span{ct::assume_aligned(a, 16_ic), ct::extents{N, N}}, ct::shape<BM, BK>{}}; \
    auto bv = ct::partition_view{ct::tensor_span{ct::assume_aligned(b, 16_ic), ct::extents{N, N}}, ct::shape<BK, BN>{}}; \
    auto cv = ct::partition_view{ct::tensor_span{ct::assume_aligned(c, 16_ic), ct::extents{N, N}}, ct::shape<BM, BN>{}}; \
    int rowBlock = ct::bid().x, columnBlock = ct::bid().y;                                                             \
    auto acc = ct::zeros<ct::tile<float, ct::shape<BM, BN>>>();

__tile_global__ void native_straight_256(const __half *a, const __half *b, float *c) {
    SPECIALISED_PROLOGUE(256) STEPS4(0) cv.store(acc, rowBlock, columnBlock);
}
__tile_global__ void native_straight_2048(const __half *a, const __half *b, float *c) {
    SPECIALISED_PROLOGUE(2048) STEPS32 cv.store(acc, rowBlock, columnBlock);
}

// (a) + (b) + (c) + the launch hint rung 7 of the Java demo uses.
__tile_global__ [[ using cutile : hint(0, occupancy=2) ]] void native_hinted_256(const __half *a, const __half *b, float *c) {
    SPECIALISED_PROLOGUE(256) STEPS4(0) cv.store(acc, rowBlock, columnBlock);
}
__tile_global__ [[ using cutile : hint(0, occupancy=2) ]] void native_hinted_2048(const __half *a, const __half *b, float *c) {
    SPECIALISED_PROLOGUE(2048) STEPS32 cv.store(acc, rowBlock, columnBlock);
}

// ------------------------------------------------------------------ host

template <typename F> static double timeIt(F f, int reps) {
    f();                                   // warm-up
    CK(cudaDeviceSynchronize());
    std::vector<double> t;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(e0); f(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1); t.push_back(ms * 1000.0);
    }
    CK(cudaGetLastError());
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 2048;
    int reps = argc > 2 ? atoi(argv[2]) : 10;
    if (n % 128 != 0) { printf("n must be a multiple of 128; got %d\n", n); return 1; }
    size_t sz = (size_t) n * n;

    std::vector<__half> hA(sz), hB(sz);
    for (size_t i = 0; i < sz; i++) {
        hA[i] = __float2half(((i * 7 + 3) % 17) * 0.0625f - 0.5f);
        hB[i] = __float2half(((i * 5 + 11) % 13) * 0.0769f - 0.5f);
    }
    // Reference over the same FP16-rounded values, as the Java demo computes it.
    std::vector<float> fa(sz), fb(sz), expected(sz, 0.0f);
    for (size_t i = 0; i < sz; i++) { fa[i] = __half2float(hA[i]); fb[i] = __half2float(hB[i]); }
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            float av = fa[(size_t) i * n + k];
            for (int j = 0; j < n; j++) expected[(size_t) i * n + j] += av * fb[(size_t) k * n + j];
        }

    __half *dA, *dB; float *dC;
    CK(cudaMalloc(&dA, sz * sizeof(__half))); CK(cudaMalloc(&dB, sz * sizeof(__half)));
    CK(cudaMalloc(&dC, sz * sizeof(float)));
    CK(cudaMemcpy(dA, hA.data(), sz * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), sz * sizeof(__half), cudaMemcpyHostToDevice));

    const float tolerance = 0.05f * n / 256.0f;
    std::vector<float> out(sz);
    double gflop = 2.0 * n * n * n / 1e9;
    bool ok = true;
    auto report = [&](const char *label, double us) {
        CK(cudaMemcpy(out.data(), dC, sz * sizeof(float), cudaMemcpyDeviceToHost));
        size_t bad = 0; float worst = 0.0f;
        for (size_t i = 0; i < sz; i++) {
            float err = fabsf(expected[i] - out[i]);
            worst = std::max(worst, err);
            if (err > tolerance * std::max(1.0f, fabsf(expected[i]))) bad++;
        }
        printf("%-44s %9.1f %9.0f   max abs err %.4f  %s\n", label, us, gflop / (us / 1e6), worst, bad ? "WRONG" : "correct");
        ok &= bad == 0;
        CK(cudaMemset(dC, 0, sz * sizeof(float)));
    };

    printf("TileLadder (hand-written CUDA Tile + cuBLAS): FP16 C = A * B, %dx%d, FP32 accumulate, %d reps\n\n", n, n, reps);
    printf("%-44s %9s %9s\n", "rung", "median us", "GFLOP/s");

    dim3 one(1, 1, 1);
    printf("-- part 1: the tile-shape ladder, n as a kernel argument\n");
    report("N1. cuTile 32x32x32, runtime n", timeIt([&] { native_runtime<32, 32, 32><<<dim3(n / 32, n / 32), one>>>(dA, dB, dC, n); }, reps));
    report("N2. cuTile 64x64x64, runtime n", timeIt([&] { native_runtime<64, 64, 64><<<dim3(n / 64, n / 64), one>>>(dA, dB, dC, n); }, reps));
    report("N3. cuTile 128x128x32, runtime n", timeIt([&] { native_runtime<128, 128, 32><<<dim3(n / 128, n / 128), one>>>(dA, dB, dC, n); }, reps));
    report("N4. cuTile 128x128x64, runtime n", timeIt([&] { native_runtime<128, 128, 64><<<dim3(n / 128, n / 128), one>>>(dA, dB, dC, n); }, reps));

    printf("-- part 2: 128x128x64 given what TornadoVM's JIT knows\n");
    dim3 g(n / BM, n / BN);
    if (n == 256 || n == 2048) {
        if (n == 256) {
            report("N5. + constant n + assume_aligned", timeIt([&] { native_specialised<256><<<g, one>>>(dA, dB, dC); }, reps));
            report("N6. + k-loop as straight-line code", timeIt([&] { native_straight_256<<<g, one>>>(dA, dB, dC); }, reps));
            report("N7. + launch hint occupancy=2", timeIt([&] { native_hinted_256<<<g, one>>>(dA, dB, dC); }, reps));
        } else {
            report("N5. + constant n + assume_aligned", timeIt([&] { native_specialised<2048><<<g, one>>>(dA, dB, dC); }, reps));
            report("N6. + k-loop as straight-line code", timeIt([&] { native_straight_2048<<<g, one>>>(dA, dB, dC); }, reps));
            report("N7. + launch hint occupancy=2", timeIt([&] { native_hinted_2048<<<g, one>>>(dA, dB, dC); }, reps));
        }
    } else {
        printf("   (skipped: the specialised kernels are compiled for n = 256 and 2048 only --\n"
               "    an AOT build must choose its sizes; TornadoVM's JIT specialises for any n)\n");
    }

    printf("-- reference\n");
    cublasHandle_t h; CB(cublasCreate(&h));
    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major: C^T = B^T * A^T, so pass B first.
    report("R. cuBLAS GemmEx FP16->FP32", timeIt([&] {
        CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, CUDA_R_16F, n, dA, CUDA_R_16F, n,
                        &beta, dC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }, reps));
    CB(cublasDestroy(h));

    printf("\n%s\n", ok ? "All rungs produced the same, correct result" : "At least one rung is WRONG");
    return ok ? 0 : 1;
}
