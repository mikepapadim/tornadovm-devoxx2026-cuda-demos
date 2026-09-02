// CUDA equivalent of CudaGraphBenefit.java.
//
// Same experiment: a 6-stage chain of small kernels, run 50 times, first with a
// plain stream (re-issuing every runtime call each execution) and then with a
// captured CUDA graph replayed by cudaGraphLaunch. Isolates the steady-state
// dispatch-overhead saving that demo 02's correctness demo does not measure.
//
// In Java this is one method call, plan.withCUDAGraph(); here it is
// cudaStreamBeginCapture / cudaStreamEndCapture / cudaGraphInstantiate /
// cudaGraphLaunch plus the pinned-memory requirement.
//
// Build & run:
//   nvcc -arch=sm_89 -o cuda_graph_benefit CudaGraphBenefit.cu
//   ./cuda_graph_benefit [size] [stages] [executions] [nograph|graph|both]

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

// Same coefficients as the Java demo, so both produce identical results.
static const float A[] = { 2.0f, 0.5f, 3.0f, 0.25f, 1.5f, 4.0f };
static const float B[] = { 1.0f, 2.0f, -3.0f, 0.5f, -1.0f, 0.75f };
static const int NCOEF = 6;

__global__ void stage(const float *in, float *out, float a, float b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = a * in[i] + b;
    }
}

static float expectedChain(float x0, int stages) {
    float v = x0;
    for (int s = 0; s < stages; s++) {
        v = A[s % NCOEF] * v + B[s % NCOEF];
    }
    return v;
}

static double runMode(bool useGraph, int size, int stages, int executions) {
    const char *label = useGraph ? "GRAPH (cudaGraphLaunch)" : "NOGRAPH (plain stream)";
    printf("\n=== %s ===\n", label);

    const size_t bytes = (size_t) size * sizeof(float);

    float *hostX, *hostResult;
    CUDA_CHECK(cudaMallocHost(&hostX, bytes));
    CUDA_CHECK(cudaMallocHost(&hostResult, bytes));

    float *devX;
    CUDA_CHECK(cudaMalloc(&devX, bytes));
    std::vector<float *> buf(stages);
    for (int s = 0; s < stages; s++) {
        CUDA_CHECK(cudaMalloc(&buf[s], bytes));
    }
    float *devResult = buf[stages - 1];

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    // Issue the whole pipeline into `stream`. Used both to capture the graph and,
    // in nograph mode, as the work re-issued on every execution.
    auto issue = [&]() {
        CUDA_CHECK(cudaMemcpyAsync(devX, hostX, bytes, cudaMemcpyHostToDevice, stream));
        const float *in = devX;
        for (int s = 0; s < stages; s++) {
            stage<<<blocks, threads, 0, stream>>>(in, buf[s], A[s % NCOEF], B[s % NCOEF], size);
            in = buf[s];
        }
        CUDA_CHECK(cudaMemcpyAsync(hostResult, devResult, bytes, cudaMemcpyDeviceToHost, stream));
    };

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    if (useGraph) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        issue();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    }

    std::vector<double> wallUs(executions);
    bool allCorrect = true;
    for (int e = 0; e < executions; e++) {
        // Mutate the input every execution so a stale replay would be caught.
        for (int i = 0; i < size; i++) {
            hostX[i] = (e + 1) * 0.01f + i * 0.001f;
        }

        auto start = std::chrono::steady_clock::now();
        if (useGraph) {
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        } else {
            issue();
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto elapsed = std::chrono::steady_clock::now() - start;
        wallUs[e] = std::chrono::duration<double, std::micro>(elapsed).count();

        bool correct = true;
        for (int i = 0; i < size; i++) {
            float x0 = (e + 1) * 0.01f + i * 0.001f;
            if (fabsf(hostResult[i] - expectedChain(x0, stages)) > 0.01f) {
                correct = false;
                break;
            }
        }
        allCorrect &= correct;
        if (e < 3 || e == executions - 1) {
            printf("execution %d: %s, wall=%.0f us\n", e, correct ? "correct" : "WRONG", wallUs[e]);
        }
    }

    printf("%s first execution (%s): %.0f us\n", label,
           useGraph ? "graph capture + instantiate" : "warm-up", wallUs[0]);
    std::vector<double> steady(wallUs.begin() + 1, wallUs.end());
    std::sort(steady.begin(), steady.end());
    double median = steady[steady.size() / 2];
    printf("%s steady-state median wall-clock (excl. first execution, n=%zu): %.0f us\n",
           label, steady.size(), median);
    printf("%s: %s\n", label, allCorrect ? "All executions correct" : "Some executions WRONG");

    if (useGraph) {
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(devX));
    for (int s = 0; s < stages; s++) {
        CUDA_CHECK(cudaFree(buf[s]));
    }
    CUDA_CHECK(cudaFreeHost(hostX));
    CUDA_CHECK(cudaFreeHost(hostResult));
    return median;
}

int main(int argc, char **argv) {
    int size = argc > 1 ? atoi(argv[1]) : 4096;
    int stages = argc > 2 ? atoi(argv[2]) : 6;
    int executions = argc > 3 ? atoi(argv[3]) : 50;
    const char *mode = argc > 4 ? argv[4] : "both";

    printf("size=%d stages=%d executions=%d mode=%s\n", size, stages, executions, mode);

    double nographMedian = -1.0;
    double graphMedian = -1.0;
    if (strcmp(mode, "nograph") == 0 || strcmp(mode, "both") == 0) {
        nographMedian = runMode(false, size, stages, executions);
    }
    if (strcmp(mode, "graph") == 0 || strcmp(mode, "both") == 0) {
        graphMedian = runMode(true, size, stages, executions);
    }
    if (nographMedian > 0 && graphMedian > 0) {
        printf("\nsteady-state median: nograph=%.1f us, graph=%.1f us, speedup=%.2fx\n",
               nographMedian, graphMedian, nographMedian / graphMedian);
    }
    return 0;
}
