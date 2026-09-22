// Hand-written CUDA equivalent of MatMulLadderFP16Tile.java: the same FP16 ladder,
// with the CUDA Tile rung.
//
// This is demo 18's ladder plus rung 4. Rungs 1-3 are the SIMT path and are identical to
// MatMulLadderFP16.cu, kept line for line so the only difference between the two files is
// the rung this demo exists for. Rung 4 is the same GEMM against cuda::tiles.
//
// Read rung 3 and rung 4 next to each other. Rung 3 packs fragments by lane, indexes a
// 32-thread warp and names a fixed m16n8k16 shape in inline PTX. Rung 4 says ct::mma and
// nothing else: no lane, no warp, no fragment, no shared memory. Both reach the tensor
// cores; only one of them says how.
//
// Build (CUDA Tile C++ needs toolkit 13.3 or newer; a userspace pip install is enough):
//   pip install --user nvidia-cuda-nvcc 'cuda-tile[tileiras]' nvidia-cuda-cccl
//   nvcc --enable-tile -std=c++20 -arch=sm_120 -O3 -o matmul_ladder_fp16_tile \
//        MatMulLadderFP16Tile.cu -lcublas
//   ./matmul_ladder_fp16_tile 256 20
//
// --enable-tile mixes tile kernels and host code in one translation unit, which is what
// lets this be an executable. TornadoVM's CUDATileCompiler instead drives
// `nvcc -tilecubin --tile-only` to a bare cubin and loads it itself, because it has no host
// translation unit to put the launch in.
//
// CUTLASS (rung 5 in the Java demo) is omitted so this builds with the plain toolkit,
// exactly as demo 18's equivalent omits it.

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

static const int TILE = 16;

// ---- rung 1: one thread per output element
__global__ void naive(const __half *a, const __half *b, __half *c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;
    float sum = 0.0f;
    for (int k = 0; k < n; k++) sum += __half2float(a[i * n + k]) * __half2float(b[k * n + j]);
    c[i * n + j] = __float2half(sum);
}

// ---- rung 2: shared-memory tiles, FP32 accumulate
__global__ void kcTiled(const __half *a, const __half *b, __half *c, int n) {
    __shared__ __half tileA[TILE * TILE];
    __shared__ __half tileB[TILE * TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float sum = 0.0f;
    for (int t = 0; t < n / TILE; t++) {
        tileA[ty * TILE + tx] = a[row * n + t * TILE + tx];
        tileB[ty * TILE + tx] = b[(t * TILE + ty) * n + col];
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            sum += __half2float(tileA[ty * TILE + k]) * __half2float(tileB[k * TILE + tx]);
        __syncthreads();
    }
    c[row * n + col] = __float2half(sum);
}

// ---- rung 3: tensor cores by hand, one warp per 16x16 tile, two m16n8k16 mma calls
__global__ void kcMma(const __half *a, const __half *b, float *c, int n) {
    int warpId = blockIdx.x;
    int lane = threadIdx.x;
    int tilesPerRow = n / 16;
    int tileRow = (warpId / tilesPerRow) * 16;
    int tileCol = (warpId % tilesPerRow) * 16;

    int groupID = lane >> 2, tig = lane & 3;
    float d0[4] = {0,0,0,0}, d1[4] = {0,0,0,0};

    for (int kBase = 0; kBase < n; kBase += 16) {
        __half av[8];
        av[0]=a[(tileRow+groupID)*n+kBase+2*tig];     av[1]=a[(tileRow+groupID)*n+kBase+2*tig+1];
        av[2]=a[(tileRow+groupID+8)*n+kBase+2*tig];   av[3]=a[(tileRow+groupID+8)*n+kBase+2*tig+1];
        av[4]=a[(tileRow+groupID)*n+kBase+2*tig+8];   av[5]=a[(tileRow+groupID)*n+kBase+2*tig+9];
        av[6]=a[(tileRow+groupID+8)*n+kBase+2*tig+8]; av[7]=a[(tileRow+groupID+8)*n+kBase+2*tig+9];
        const uint32_t *A32 = reinterpret_cast<const uint32_t *>(av);

        for (int panel = 0; panel < 2; panel++) {
            __half bv[4];
            int cc = tileCol + 8 * panel + groupID;
            bv[0]=b[(kBase+2*tig)*n+cc];   bv[1]=b[(kBase+2*tig+1)*n+cc];
            bv[2]=b[(kBase+2*tig+8)*n+cc]; bv[3]=b[(kBase+2*tig+9)*n+cc];
            const uint32_t *B32 = reinterpret_cast<const uint32_t *>(bv);
            float *d = panel == 0 ? d0 : d1;
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                : "r"(A32[0]), "r"(A32[1]), "r"(A32[2]), "r"(A32[3]), "r"(B32[0]), "r"(B32[1]));
        }
    }
    for (int panel = 0; panel < 2; panel++) {
        float *d = panel == 0 ? d0 : d1;
        int base = tileCol + 8 * panel;
        c[(tileRow+groupID)*n + base + 2*tig]       = d[0];
        c[(tileRow+groupID)*n + base + 2*tig + 1]   = d[1];
        c[(tileRow+groupID+8)*n + base + 2*tig]     = d[2];
        c[(tileRow+groupID+8)*n + base + 2*tig + 1] = d[3];
    }
}

// ---- rung 4: CUDA Tile. The whole kernel, against all of rung 3 above.
constexpr int TILE_BLOCK = 32;

extern "C" __tile_global__ void tiles(const __half *a, const __half *b, float *c, int n) {
    auto aView = ct::partition_view{ct::tensor_span{a, ct::extents{n, n}}, ct::shape<TILE_BLOCK, TILE_BLOCK>{}};
    auto bView = ct::partition_view{ct::tensor_span{b, ct::extents{n, n}}, ct::shape<TILE_BLOCK, TILE_BLOCK>{}};
    auto cView = ct::partition_view{ct::tensor_span{c, ct::extents{n, n}}, ct::shape<TILE_BLOCK, TILE_BLOCK>{}};

    int rowBlock = ct::bid().x;
    int columnBlock = ct::bid().y;

    auto acc = ct::zeros<ct::tile<float, ct::shape<TILE_BLOCK, TILE_BLOCK>>>();
    for (int step = 0; step < n / TILE_BLOCK; step++) {
        acc = ct::mma(aView.load(rowBlock, step), bView.load(step, columnBlock), acc);
    }
    cView.store(acc, rowBlock, columnBlock);
}

template <typename F> static double timeIt(F f, int reps) {
    for (int i = 0; i < 3; i++) f();
    CK(cudaDeviceSynchronize());
    std::vector<double> t;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(a); f(); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); t.push_back(ms * 1000.0);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// Every rung is checked against the same CPU reference, so a fast wrong rung cannot
// look like a win. The Java demo validates every rung too; demo 18's equivalent does
// not, which is the one way this file is not a straight port of it.
static bool validate(const char *label, const std::vector<float> &expected,
                     const std::vector<float> &actual, int n, float tol) {
    double worst = 0.0;
    for (size_t i = 0; i < expected.size(); i++)
        worst = std::max(worst, (double) fabsf(expected[i] - actual[i]));
    bool ok = worst <= tol;
    printf("   %-28s max abs error %8.4f  %s\n", label, worst, ok ? "correct" : "WRONG");
    return ok;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 256;
    int reps = argc > 2 ? atoi(argv[2]) : 20;
    if (n % 64) { printf("size must be a multiple of 64; got %d\n", n); return 1; }

    size_t sz = (size_t) n * n;
    std::vector<__half> hA(sz), hB(sz);
    for (size_t i = 0; i < sz; i++) {
        hA[i] = __float2half(((i * 7 + 3) % 17) * 0.0625f - 0.5f);
        hB[i] = __float2half(((i * 5 + 11) % 13) * 0.0769f - 0.5f);
    }

    // CPU reference in FP32 from the FP16 inputs, as the Java demo does.
    std::vector<float> expected(sz, 0.0f);
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            float av = __half2float(hA[(size_t) i * n + k]);
            for (int j = 0; j < n; j++)
                expected[(size_t) i * n + j] += av * __half2float(hB[(size_t) k * n + j]);
        }

    __half *dA, *dB, *dC; float *dCf;
    CK(cudaMalloc(&dA, sz * sizeof(__half))); CK(cudaMalloc(&dB, sz * sizeof(__half)));
    CK(cudaMalloc(&dC, sz * sizeof(__half))); CK(cudaMalloc(&dCf, sz * sizeof(float)));
    CK(cudaMemcpy(dA, hA.data(), sz * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), sz * sizeof(__half), cudaMemcpyHostToDevice));

    std::vector<__half> outH(sz);
    std::vector<float> outF(sz);
    auto readHalf = [&]{ CK(cudaMemcpy(outH.data(), dC, sz * sizeof(__half), cudaMemcpyDeviceToHost));
                         for (size_t i = 0; i < sz; i++) outF[i] = __half2float(outH[i]); return outF; };
    auto readFloat = [&]{ CK(cudaMemcpy(outF.data(), dCf, sz * sizeof(float), cudaMemcpyDeviceToHost));
                          return outF; };

    // FP16 storage of the result costs about 3 decimal digits, so the half-output rungs
    // get a looser tolerance than the FP32-output ones. Both are far tighter than a wrong
    // kernel would land.
    const float tolHalf = 0.35f, tolFloat = 0.05f;
    bool ok = true;

    double gflop = 2.0 * n * n * n / 1e9;
    printf("FP16 matmul ladder with a CUDA Tile rung (hand-written CUDA): C = A * B, %dx%d, %d reps\n\n", n, n, reps);
    printf("%-32s %10s %12s\n", "rung", "median us", "GFLOP/s");

    dim3 bn(16, 16), gn((n + 15) / 16, (n + 15) / 16);
    double t1 = timeIt([&]{ naive<<<gn, bn>>>(dA, dB, dC, n); }, reps);
    printf("%-32s %10.1f %12.0f\n", "1. naive", t1, gflop / (t1 / 1e6));
    ok &= validate("naive", expected, readHalf(), n, tolHalf);

    double t2 = timeIt([&]{ kcTiled<<<gn, bn>>>(dA, dB, dC, n); }, reps);
    printf("%-32s %10.1f %12.0f\n", "2. tiled", t2, gflop / (t2 / 1e6));
    ok &= validate("tiled", expected, readHalf(), n, tolHalf);

    int warps = (n / 16) * (n / 16);
    double t3 = timeIt([&]{ kcMma<<<warps, 32>>>(dA, dB, dCf, n); }, reps);
    printf("%-32s %10.1f %12.0f\n", "3. MMA (tensor core, by hand)", t3, gflop / (t3 / 1e6));
    ok &= validate("MMA by hand", expected, readFloat(), n, tolFloat);

    // The tile launch: the grid counts TILE BLOCKS, and the block is 1x1x1 -- the tile
    // compiler decides how many threads actually back it. Same contract as
    // CUDATileScheduler enforces on the TornadoVM side.
    dim3 tileGrid(n / TILE_BLOCK, n / TILE_BLOCK), tileBlock(1, 1, 1);
    double t4 = timeIt([&]{ tiles<<<tileGrid, tileBlock>>>(dA, dB, dCf, n); }, reps);
    printf("%-32s %10.1f %12.0f\n", "4. CUDA Tile (ct::mma)", t4, gflop / (t4 / 1e6));
    ok &= validate("CUDA Tile", expected, readFloat(), n, tolFloat);

    cublasHandle_t h; CB(cublasCreate(&h));
    float alpha = 1.0f, beta = 0.0f;
    double t5 = timeIt([&]{
        CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, CUDA_R_16F, n,
                        dA, CUDA_R_16F, n, &beta, dC, CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }, reps);
    printf("%-32s %10.1f %12.0f\n", "5. cuBLAS GemmEx FP16", t5, gflop / (t5 / 1e6));
    ok &= validate("cuBLAS FP16", expected, readHalf(), n, tolHalf);

    double t6 = timeIt([&]{
        CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, CUDA_R_16F, n,
                        dA, CUDA_R_16F, n, &beta, dCf, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }, reps);
    printf("%-32s %10.1f %12.0f\n", "6. cuBLAS GemmEx FP16->FP32", t6, gflop / (t6 / 1e6));
    ok &= validate("cuBLAS FP16->FP32", expected, readFloat(), n, tolFloat);

    printf("\n(rung 5 of the Java demo, CUTLASS, is omitted here so this builds with the plain toolkit)\n");
    printf("\n%s\n", ok ? "All rungs correct." : "At least one rung is WRONG.");
    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dCf);
    return ok ? 0 : 1;
}
