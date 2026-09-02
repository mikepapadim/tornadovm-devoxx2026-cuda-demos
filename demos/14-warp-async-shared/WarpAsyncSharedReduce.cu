// CUDA equivalent of WarpAsyncSharedReduce.java.
//
// The same three optimisations, written the way you would without TornadoVM:
//
//   asynchronous copy  -> inline PTX cp.async.ca.shared.global / commit_group /
//                         wait_group, plus the generic->shared address cast
//   shared memory      -> __shared__ arrays + __syncthreads()
//   warp shuffle       -> __shfl_down_sync
//
// This is almost line-for-line what `tornado --printKernel` emits for the Java
// kernel -- which is the point: KernelContext.asyncCopyToLocal / simdShuffleDown
// are not an abstraction over these instructions, they are these instructions.
//
// Build & run:
//   nvcc -arch=sm_89 -o warp_async_shared WarpAsyncSharedReduce.cu
//   ./warp_async_shared [rows] [rowLen] [executions]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static const int WARP_SIZE = 32;
static const int BLOCK = 128;
static const int WARPS_PER_BLOCK = BLOCK / WARP_SIZE;

// Baseline: one thread per row, straight from global memory.
__global__ void rowSumNaive(const int8_t *data, float *out, int rowLen, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        int sum = 0;
        for (int j = 0; j < rowLen; j++) {
            sum += data[(size_t) row * rowLen + j];
        }
        out[row] = (float) sum;
    }
}

// Optimised: cp.async staging into shared memory + warp-shuffle reduction.
__global__ void rowSumOptimised(const int8_t *data, float *out, int rowLen) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int words = rowLen / 4;

    __shared__ int tile[BLOCK];
    __shared__ float warpPartials[WARPS_PER_BLOCK];

    int total = 0;
    for (int base = 0; base < words; base += BLOCK) {
        int word = base + tid;
        if (word < words) {
            // cp.async: 4 bytes, global -> shared, bypassing registers.
            // The destination must be a shared-window address, not a generic one.
            uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(&tile[tid]));
            const int8_t *src = data + (size_t) row * rowLen + (size_t) word * 4;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                         :: "r"(smem), "l"(src));
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        if (word < words) {
            // Unpack four sign-extended int8 lanes from the staged word.
            int packed = tile[tid];
            total += (packed << 24) >> 24;
            total += (packed << 16) >> 24;
            total += (packed << 8) >> 24;
            total += packed >> 24;
        }
        __syncthreads(); // tile is reused by the next iteration
    }

    // Reduce the 32 lanes of each warp entirely in registers.
    float value = (float) total;
    for (int delta = WARP_SIZE / 2; delta > 0; delta /= 2) {
        value += __shfl_down_sync(0xffffffff, value, delta);
    }

    int lane = tid % WARP_SIZE;
    int warp = tid / WARP_SIZE;
    if (lane == 0) {
        warpPartials[warp] = value;
    }
    __syncthreads();

    if (tid == 0) {
        float blockSum = 0.0f;
        for (int i = 0; i < WARPS_PER_BLOCK; i++) {
            blockSum += warpPartials[i];
        }
        out[row] = blockSum;
    }
}

static bool validate(const char *label, const float *got, const float *ref, int rows) {
    float maxAbs = 0.0f;
    int bad = 0;
    for (int i = 0; i < rows; i++) {
        float diff = fabsf(got[i] - ref[i]);
        maxAbs = fmaxf(maxAbs, diff);
        if (diff > 1e-3f) {
            bad++;
        }
    }
    bool ok = bad == 0;
    printf("  [%-9s] validation %s (max abs err %.5f, %d/%d rows out of tol)\n",
           label, ok ? "PASSED" : "FAILED", maxAbs, bad, rows);
    return ok;
}

int main(int argc, char **argv) {
    const int rows = argc > 1 ? atoi(argv[1]) : 4096;
    const int rowLen = argc > 2 ? atoi(argv[2]) : 1024;
    const int executions = argc > 3 ? atoi(argv[3]) : 20;

    if (rowLen % 4 != 0) {
        printf("rowLen must be a multiple of 4 (got %d).\n", rowLen);
        return 1;
    }

    printf("Warp-shuffle + cp.async + shared-memory row reduction: %d rows x %d int8 values\n",
           rows, rowLen);
    printf("  block = %d threads (%d warps), one block per row\n", BLOCK, WARPS_PER_BLOCK);
    printf("  %d executions per kernel, steady-state median reported (first execution excluded)\n\n",
           executions);

    const size_t dataBytes = (size_t) rows * rowLen;
    const size_t outBytes = (size_t) rows * sizeof(float);

    int8_t *hostData = (int8_t *) malloc(dataBytes);
    for (size_t i = 0; i < dataBytes; i++) {
        hostData[i] = (int8_t) (((long) i * 31 + 1) % 17 - 8);
    }
    std::vector<float> ref(rows);
    for (int row = 0; row < rows; row++) {
        int sum = 0;
        for (int j = 0; j < rowLen; j++) {
            sum += hostData[(size_t) row * rowLen + j];
        }
        ref[row] = (float) sum;
    }

    int8_t *devData;
    float *devNaive, *devOpt;
    CUDA_CHECK(cudaMalloc(&devData, dataBytes));
    CUDA_CHECK(cudaMalloc(&devNaive, outBytes));
    CUDA_CHECK(cudaMalloc(&devOpt, outBytes));
    CUDA_CHECK(cudaMemcpy(devData, hostData, dataBytes, cudaMemcpyHostToDevice));

    std::vector<float> outNaive(rows), outOpt(rows);

    auto timeKernel = [&](const char *label, bool optimised, float *dev, std::vector<float> &host) {
        printf("=== %s ===\n", label);
        std::vector<double> wallUs(executions);
        for (int e = 0; e < executions; e++) {
            auto start = std::chrono::steady_clock::now();
            if (optimised) {
                rowSumOptimised<<<rows, BLOCK>>>(devData, dev, rowLen);
            } else {
                rowSumNaive<<<(rows + 255) / 256, 256>>>(devData, dev, rowLen, rows);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(host.data(), dev, outBytes, cudaMemcpyDeviceToHost));
            auto elapsed = std::chrono::steady_clock::now() - start;
            wallUs[e] = std::chrono::duration<double, std::micro>(elapsed).count();
            if (e == 0) {
                printf("  first execution: %.0f us\n", wallUs[e]);
            }
        }
        std::vector<double> steady(wallUs.begin() + 1, wallUs.end());
        std::sort(steady.begin(), steady.end());
        double median = steady[steady.size() / 2];
        printf("  steady-state median wall-clock (n=%zu): %.0f us\n", steady.size(), median);
        return median;
    };

    double naiveMedian = timeKernel("NAIVE (one thread per row, global memory only)", false, devNaive, outNaive);
    printf("\n");
    double optMedian = timeKernel("OPTIMISED (cp.async + shared + shuffle)", true, devOpt, outOpt);

    printf("\n");
    bool okNaive = validate("naive", outNaive.data(), ref.data(), rows);
    bool okOpt = validate("optimised", outOpt.data(), ref.data(), rows);

    printf("\n=== Summary (steady-state median us, this run/this GPU) ===\n");
    printf("naive     : %.0f\n", naiveMedian);
    printf("optimised : %.0f (%.2fx vs naive)\n", optMedian, naiveMedian / optMedian);
    printf("%s\n", (okNaive && okOpt) ? "Both kernels produce the same, correct result"
                                      : "Result is INCORRECT");

    CUDA_CHECK(cudaFree(devData));
    CUDA_CHECK(cudaFree(devNaive));
    CUDA_CHECK(cudaFree(devOpt));
    free(hostData);
    return 0;
}
