// Hand-written CUDA equivalent of TileMatMul.java: the same FP16 GEMM three ways.
//
//   1. naive      one thread per output element
//   2. tiled      shared-memory tiles, staged and barriered by hand (the SIMT model)
//   3. tile       CUDA Tile C++: ct::partition_view + ct::mma, no thread indices at all
//
// Read this next to TileMatMul.java. Rungs 1 and 2 are longer in Java only by the
// boilerplate of a TaskGraph; rung 3 is the same kernel in both languages, line for line.
//
// Build (CUDA Tile C++ needs toolkit 13.3 or newer; a userspace pip install is enough):
//   nvcc --enable-tile -std=c++20 -arch=sm_89 -o tilematmul TileMatMul.cu
//   ./tilematmul 256 10

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;
constexpr int BLOCK = 32;

// --- rung 1: one thread per output element --------------------------------------------
__global__ void naive(const __half *a, const __half *b, float *c, int n, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int column = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || column >= n) return;
    float sum = 0.0f;
    for (int inner = 0; inner < k; inner++) {
        sum += __half2float(a[row * k + inner]) * __half2float(b[inner * n + column]);
    }
    c[row * n + column] = sum;
}

// --- rung 2: shared-memory tiles, by hand ---------------------------------------------
__global__ void tiled(const __half *a, const __half *b, float *c, int n, int k) {
    __shared__ float aTile[BLOCK][BLOCK];
    __shared__ float bTile[BLOCK][BLOCK];

    int localRow = threadIdx.x;
    int localColumn = threadIdx.y;
    int row = blockIdx.x * BLOCK + localRow;
    int column = blockIdx.y * BLOCK + localColumn;

    float sum = 0.0f;
    for (int step = 0; step < k / BLOCK; step++) {
        aTile[localRow][localColumn] = __half2float(a[row * k + step * BLOCK + localColumn]);
        bTile[localRow][localColumn] = __half2float(b[(step * BLOCK + localRow) * n + column]);
        __syncthreads();
        for (int inner = 0; inner < BLOCK; inner++) {
            sum += aTile[localRow][inner] * bTile[inner][localColumn];
        }
        __syncthreads();
    }
    c[row * n + column] = sum;
}

// --- rung 3: tiles as the unit of work ------------------------------------------------
// This is the whole kernel. Compare with TileMatMul.tiles in the Java file.
__tile_global__ void tiles(const __half *a, const __half *b, float *c, int m, int n, int k) {
    auto aView = ct::partition_view{ct::tensor_span{a, ct::extents{m, k}}, ct::shape<TILE_M, TILE_K>{}};
    auto bView = ct::partition_view{ct::tensor_span{b, ct::extents{k, n}}, ct::shape<TILE_K, TILE_N>{}};
    auto cView = ct::partition_view{ct::tensor_span{c, ct::extents{m, n}}, ct::shape<TILE_M, TILE_N>{}};

    int rowBlock = ct::bid().x;
    int columnBlock = ct::bid().y;

    auto acc = ct::zeros<ct::tile<float, ct::shape<TILE_M, TILE_N>>>();
    for (auto step : ct::irange(0, k / TILE_K)) {
        acc = ct::mma(aView.load(rowBlock, step), bView.load(step, columnBlock), acc);
    }
    cView.store(acc, rowBlock, columnBlock);
}

// --------------------------------------------------------------------------------------

static float elapsedMs(void (*launch)(const __half *, const __half *, float *, int), const __half *a, const __half *b, float *c, int n, int executions) {
    launch(a, b, c, n);                       // warm up
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < executions; i++) launch(a, b, c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms / executions;
}


static void launchNaive(const __half *a, const __half *b, float *c, int n) {
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);
    naive<<<grid, block>>>(a, b, c, n, n);
}

static void launchTiled(const __half *a, const __half *b, float *c, int n) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid(n / BLOCK, n / BLOCK);
    tiled<<<grid, block>>>(a, b, c, n, n);
}

static void launchTiles(const __half *a, const __half *b, float *c, int n) {
    // Tile launch: the grid counts TILE BLOCKS and the block is always 1x1x1.
    dim3 grid(n / TILE_M, n / TILE_N);
    tiles<<<grid, 1>>>(a, b, c, n, n, n);
}

static void check(const char *label, const std::vector<float> &out, const std::vector<float> &reference, int n, float ms, float baseline) {
    double worst = 0.0;
    for (int i = 0; i < n * n; i++) worst = fmax(worst, fabs(out[i] - reference[i]));
    double tolerance = 0.02 * sqrt((double) n);
    printf("%-42s %14.3f %12.2fx %11.4f   %s\n", label, ms, baseline / ms, worst, worst <= tolerance ? "correct" : "WRONG");
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 256;
    int executions = argc > 2 ? atoi(argv[2]) : 10;
    if (n % TILE_M != 0 || n % BLOCK != 0) {
        printf("n must be a multiple of %d\n", BLOCK > TILE_M ? BLOCK : TILE_M);
        return 1;
    }
    printf("n = %d, %d executions per rung\n\n", n, executions);

    std::vector<__half> a(n * n), b(n * n);
    srand(7);
    for (int i = 0; i < n * n; i++) {
        a[i] = __float2half((float) rand() / RAND_MAX - 0.5f);
        b[i] = __float2half((float) rand() / RAND_MAX - 0.5f);
    }
    std::vector<float> reference(n * n, 0.0f);
    for (int row = 0; row < n; row++)
        for (int column = 0; column < n; column++) {
            float sum = 0.0f;
            for (int inner = 0; inner < n; inner++)
                sum += __half2float(a[row * n + inner]) * __half2float(b[inner * n + column]);
            reference[row * n + column] = sum;
        }

    __half *dA, *dB;
    float *dC;
    cudaMalloc(&dA, n * n * sizeof(__half));
    cudaMalloc(&dB, n * n * sizeof(__half));
    cudaMalloc(&dC, n * n * sizeof(float));
    cudaMemcpy(dA, a.data(), n * n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), n * n * sizeof(__half), cudaMemcpyHostToDevice);

    std::vector<float> out(n * n);
    printf("%-42s %14s %13s %11s   %s\n", "rung", "ms/execution", "vs naive", "max error", "result");

    float naiveMs = elapsedMs(launchNaive, dA, dB, dC, n, executions);
    cudaMemcpy(out.data(), dC, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> naiveOut = out;

    float tiledMs = elapsedMs(launchTiled, dA, dB, dC, n, executions);
    cudaMemcpy(out.data(), dC, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> tiledOut = out;

    float tilesMs = elapsedMs(launchTiles, dA, dB, dC, n, executions);
    cudaMemcpy(out.data(), dC, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    check("1. naive (one thread per element)", naiveOut, reference, n, naiveMs, naiveMs);
    check("2. tiled (shared memory, by hand)", tiledOut, reference, n, tiledMs, naiveMs);
    check("3. CUDA Tile (ct::mma)", out, reference, n, tilesMs, naiveMs);

    printf("\nKernel time only (CUDA events), so these are the numbers the Java demo\n");
    printf("cannot report without nsys.\n");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
