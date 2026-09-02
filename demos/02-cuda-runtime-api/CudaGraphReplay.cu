// CUDA equivalent of CudaGraphReplay.java.
//
// TornadoExecutionPlan#withCUDAGraph() is one method call. This is what it does
// underneath: create a stream, capture H2D copy + kernel + D2H copy with
// cudaStreamBeginCapture/EndCapture, instantiate the graph, then replay it with
// cudaGraphLaunch instead of re-issuing each runtime call.
//
// The host buffers must be pinned and captured by address, so that mutating
// them between replays changes what the replayed copies actually move -- the
// same property the Java demo relies on to prove the graph is not stale.
//
// Build & run:
//   nvcc -arch=sm_89 -o cuda_graph_replay CudaGraphReplay.cu && ./cuda_graph_replay [size] [replays]

#include <cstdio>
#include <cstdlib>
#include <cmath>

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

__global__ void axpy(const float *x, const float *y, float *result, float alpha, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = alpha * x[i] + y[i];
    }
}

int main(int argc, char **argv) {
    const int size = argc > 1 ? atoi(argv[1]) : 8192;
    const int replays = argc > 2 ? atoi(argv[2]) : 8;
    const size_t bytes = (size_t) size * sizeof(float);

    // Pinned host memory: required for the copies to be capturable in a graph.
    float *hostX, *hostY, *hostResult;
    CUDA_CHECK(cudaMallocHost(&hostX, bytes));
    CUDA_CHECK(cudaMallocHost(&hostY, bytes));
    CUDA_CHECK(cudaMallocHost(&hostResult, bytes));

    float *devX, *devY, *devResult;
    CUDA_CHECK(cudaMalloc(&devX, bytes));
    CUDA_CHECK(cudaMalloc(&devY, bytes));
    CUDA_CHECK(cudaMalloc(&devResult, bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    // --- capture the whole pipeline once -----------------------------------
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CUDA_CHECK(cudaMemcpyAsync(devX, hostX, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(devY, hostY, bytes, cudaMemcpyHostToDevice, stream));
    axpy<<<blocks, threads, 0, stream>>>(devX, devY, devResult, ALPHA, size);
    CUDA_CHECK(cudaMemcpyAsync(hostResult, devResult, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    bool allCorrect = true;
    for (int i = 0; i < replays; i++) {
        float xValue = 1.0f + i;
        float yValue = 2.0f + i;
        for (int j = 0; j < size; j++) {
            hostX[j] = xValue;
            hostY[j] = yValue;
        }

        // One call replaces the copy/launch/copy sequence above.
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float expected = ALPHA * xValue + yValue;
        bool correct = true;
        for (int j = 0; j < size; j++) {
            if (fabsf(hostResult[j] - expected) > 1e-3f) {
                correct = false;
                break;
            }
        }
        allCorrect &= correct;
        printf("replay %d (x=%.1f, y=%.1f): %s, result[0]=%.1f expected=%.1f\n",
               i, xValue, yValue, correct ? "correct" : "WRONG", hostResult[0], expected);
    }
    printf("%s\n", allCorrect ? "All replays correct" : "Some replays WRONG");

    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(devX));
    CUDA_CHECK(cudaFree(devY));
    CUDA_CHECK(cudaFree(devResult));
    CUDA_CHECK(cudaFreeHost(hostX));
    CUDA_CHECK(cudaFreeHost(hostY));
    CUDA_CHECK(cudaFreeHost(hostResult));
    return 0;
}
