// CUDA equivalent of CudaStreamsOverlap.java.
//
// TornadoExecutionPlan#withIntraPlanConcurrency() is one method call. This is
// the machinery it replaces: create a pool of streams, round-robin the
// independent units across them, and synchronise them all at the end.
//
// The kernel is deliberately small-grid / heavy-inner-loop so that a single
// unit does not saturate the SMs -- the documented precondition for genuine
// kernel overlap to be visible at all.
//
// Build & run:
//   nvcc -arch=sm_89 -o cuda_streams CudaStreamsOverlap.cu
//   ./cuda_streams [units] [unitSize] [innerIterations] [executions] [sequential|concurrent|both]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
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

static const float ALPHA = 0.5f;
// TornadoVM's concurrent mode uses a 4-stream pool; match it for comparability.
static const int STREAM_POOL = 4;

__global__ void computeSmall(const float *x, const float *y, float *result,
                             float alpha, int innerIterations, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float xi = x[i];
        float yi = y[i];
        float val = alpha * xi + yi;
        for (int j = 0; j < innerIterations; j++) {
            val = val * alpha + yi;
        }
        result[i] = val;
    }
}

static float expectedSmall(float x, float y, float alpha, int innerIterations) {
    float val = alpha * x + y;
    for (int j = 0; j < innerIterations; j++) {
        val = val * alpha + y;
    }
    return val;
}

static void runMode(bool concurrent, int units, int unitSize, int innerIterations, int executions) {
    const char *label = concurrent ? "CONCURRENT (4-stream pool)" : "SEQUENTIAL (single stream)";
    printf("\n=== %s ===\n", label);

    const size_t bytes = (size_t) unitSize * sizeof(float);

    std::vector<float *> hostX(units), hostY(units), hostR(units);
    std::vector<float *> devX(units), devY(units), devR(units);
    for (int u = 0; u < units; u++) {
        CUDA_CHECK(cudaMallocHost(&hostX[u], bytes));
        CUDA_CHECK(cudaMallocHost(&hostY[u], bytes));
        CUDA_CHECK(cudaMallocHost(&hostR[u], bytes));
        for (int i = 0; i < unitSize; i++) {
            hostX[u][i] = u + 1.0f;
            hostY[u][i] = u + 2.0f;
        }
        CUDA_CHECK(cudaMalloc(&devX[u], bytes));
        CUDA_CHECK(cudaMalloc(&devY[u], bytes));
        CUDA_CHECK(cudaMalloc(&devR[u], bytes));
    }

    const int nStreams = concurrent ? STREAM_POOL : 1;
    std::vector<cudaStream_t> streams(nStreams);
    for (int s = 0; s < nStreams; s++) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    const int threads = 256;
    const int blocks = (unitSize + threads - 1) / threads;

    std::vector<double> wallUs(executions);
    bool allCorrect = true;
    for (int e = 0; e < executions; e++) {
        auto start = std::chrono::steady_clock::now();
        for (int u = 0; u < units; u++) {
            cudaStream_t s = streams[u % nStreams];
            CUDA_CHECK(cudaMemcpyAsync(devX[u], hostX[u], bytes, cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(devY[u], hostY[u], bytes, cudaMemcpyHostToDevice, s));
            computeSmall<<<blocks, threads, 0, s>>>(devX[u], devY[u], devR[u],
                                                    ALPHA, innerIterations, unitSize);
            CUDA_CHECK(cudaMemcpyAsync(hostR[u], devR[u], bytes, cudaMemcpyDeviceToHost, s));
        }
        for (int s = 0; s < nStreams; s++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        wallUs[e] = std::chrono::duration<double, std::micro>(elapsed).count();

        bool correct = true;
        for (int u = 0; u < units && correct; u++) {
            float expected = expectedSmall(u + 1.0f, u + 2.0f, ALPHA, innerIterations);
            for (int i = 0; i < unitSize; i++) {
                if (fabsf(hostR[u][i] - expected) > 1e-3f * fmaxf(1.0f, fabsf(expected))) {
                    correct = false;
                    break;
                }
            }
        }
        allCorrect &= correct;
        printf("execution %d: %s, wall=%.0f us\n", e, correct ? "correct" : "WRONG", wallUs[e]);
    }

    std::vector<double> sorted = wallUs;
    std::sort(sorted.begin(), sorted.end());
    printf("%s median wall-clock (all %d executions incl. first): %.0f us\n",
           label, executions, sorted[sorted.size() / 2]);
    std::vector<double> steady(wallUs.begin() + 1, wallUs.end());
    std::sort(steady.begin(), steady.end());
    printf("%s median wall-clock (excl. first execution): %.0f us\n",
           label, steady[steady.size() / 2]);
    printf("%s: %s\n", label, allCorrect ? "All executions correct" : "Some executions WRONG");

    for (int s = 0; s < nStreams; s++) {
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
    for (int u = 0; u < units; u++) {
        CUDA_CHECK(cudaFree(devX[u]));
        CUDA_CHECK(cudaFree(devY[u]));
        CUDA_CHECK(cudaFree(devR[u]));
        CUDA_CHECK(cudaFreeHost(hostX[u]));
        CUDA_CHECK(cudaFreeHost(hostY[u]));
        CUDA_CHECK(cudaFreeHost(hostR[u]));
    }
}

int main(int argc, char **argv) {
    int units = argc > 1 ? atoi(argv[1]) : 8;
    int unitSize = argc > 2 ? atoi(argv[2]) : 32 * 1024;
    int innerIterations = argc > 3 ? atoi(argv[3]) : (1 << 16);
    int executions = argc > 4 ? atoi(argv[4]) : 8;
    const char *mode = argc > 5 ? argv[5] : "both";

    printf("units=%d unitSize=%d innerIterations=%d executions=%d mode=%s\n",
           units, unitSize, innerIterations, executions, mode);

    if (strcmp(mode, "sequential") == 0 || strcmp(mode, "both") == 0) {
        runMode(false, units, unitSize, innerIterations, executions);
    }
    if (strcmp(mode, "concurrent") == 0 || strcmp(mode, "both") == 0) {
        runMode(true, units, unitSize, innerIterations, executions);
    }
    return 0;
}
