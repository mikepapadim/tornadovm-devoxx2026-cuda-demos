// Hand-written CUDA equivalent of TileHybridPipeline.java: a SIMT kernel, a CUDA Tile
// kernel and a cuBLAS call in one stream, then the same sequence captured into a CUDA graph.
//
//   1. scale     __global__ SIMT kernel        (Java: KernelContext task)
//   2. gemm      __tile_global__ tile kernel   (Java: TileContext task)
//   3. sgemv     cublasSgemv                   (Java: libraryTask)
//   4. biasRelu  __global__ SIMT kernel        (Java: @Parallel task)
//
// The interesting line is the launch of stage 2: a tile kernel takes ordinary device
// pointers and an ordinary stream, which is why it mixes with a vendor library and with
// SIMT kernels without any special handling -- the same reason it composes inside a
// TornadoVM TaskGraph.
//
// Build (needs CUDA Toolkit 13.3 or newer for the tile support):
//   nvcc --enable-tile -std=c++20 -arch=sm_89 -lcublas -o tilehybrid TileHybridPipeline.cu
//   ./tilehybrid 256 20

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int TILE = 32;
constexpr float BIAS = 0.5f;

__global__ void scale(float *inOut, int n, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) inOut[i] *= factor;
}

__tile_global__ void gemm(const __half *a, const __half *b, float *c, int m, int n, int k) {
    auto aView = ct::partition_view{ct::tensor_span{a, ct::extents{m, k}}, ct::shape<TILE, TILE>{}};
    auto bView = ct::partition_view{ct::tensor_span{b, ct::extents{k, n}}, ct::shape<TILE, TILE>{}};
    auto cView = ct::partition_view{ct::tensor_span{c, ct::extents{m, n}}, ct::shape<TILE, TILE>{}};

    int rowBlock = ct::bid().x;
    int columnBlock = ct::bid().y;
    auto acc = ct::zeros<ct::tile<float, ct::shape<TILE, TILE>>>();
    for (auto step : ct::irange(0, k / TILE)) {
        acc = ct::mma(aView.load(rowBlock, step), bView.load(step, columnBlock), acc);
    }
    cView.store(acc, rowBlock, columnBlock);
}

__global__ void biasRelu(float *inOut, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float value = inOut[i] + BIAS;
        inOut[i] = value > 0.0f ? value : 0.0f;
    }
}

// One pass of the four stages on a stream. cuBLAS is column-major and the matrix here is
// row-major, so CUBLAS_OP_T computes hidden * x -- the same transposition the Java demo
// passes to CuBlas.cublasSgemv.
static void enqueuePipeline(cublasHandle_t handle, cudaStream_t stream, const __half *a, const __half *b, float *x, float *hidden, float *projected, int n, float factor) {
    scale<<<(n + 63) / 64, 64, 0, stream>>>(x, n, factor);
    gemm<<<dim3(n / TILE, n / TILE), 1, 0, stream>>>(a, b, hidden, n, n, n);
    const float one = 1.0f, zero = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_T, n, n, &one, hidden, n, x, 1, &zero, projected, 1);
    biasRelu<<<(n + 63) / 64, 64, 0, stream>>>(projected, n);
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 256;
    int executions = argc > 2 ? atoi(argv[2]) : 20;
    if (n % TILE != 0) {
        printf("n must be a multiple of %d\n", TILE);
        return 1;
    }
    const float factor = 1.5f;
    printf("n = %d, %d executions\n", n, executions);
    printf("pipeline: SIMT scale -> CUDA Tile gemm -> cuBLAS sgemv -> SIMT biasRelu\n\n");

    std::vector<__half> a(n * n), b(n * n);
    std::vector<float> x(n);
    srand(11);
    for (int i = 0; i < n * n; i++) {
        a[i] = __float2half((float) rand() / RAND_MAX - 0.5f);
        b[i] = __float2half((float) rand() / RAND_MAX - 0.5f);
    }
    for (int i = 0; i < n; i++) x[i] = (float) rand() / RAND_MAX - 0.5f;

    // Reference: the same four stages on the CPU, over the same FP16 values.
    std::vector<float> expected(n);
    for (int row = 0; row < n; row++) {
        double sum = 0.0;
        for (int column = 0; column < n; column++) {
            double hidden = 0.0;
            for (int inner = 0; inner < n; inner++)
                hidden += __half2float(a[row * n + inner]) * __half2float(b[inner * n + column]);
            sum += hidden * (x[column] * factor);
        }
        expected[row] = fmax(0.0, sum + BIAS);
    }

    __half *dA, *dB;
    float *dX, *dHidden, *dProjected;
    cudaMalloc(&dA, n * n * sizeof(__half));
    cudaMalloc(&dB, n * n * sizeof(__half));
    cudaMalloc(&dX, n * sizeof(float));
    cudaMalloc(&dHidden, n * n * sizeof(float));
    cudaMalloc(&dProjected, n * sizeof(float));
    cudaMemcpy(dA, a.data(), n * n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), n * n * sizeof(__half), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    std::vector<float> actual(n);
    auto verify = [&](const char *label, float ms) {
        cudaMemcpy(actual.data(), dProjected, n * sizeof(float), cudaMemcpyDeviceToHost);
        double worst = 0.0, magnitude = 0.0;
        for (int i = 0; i < n; i++) {
            worst = fmax(worst, fabs(actual[i] - expected[i]));
            magnitude = fmax(magnitude, fabs(expected[i]));
        }
        double tolerance = 0.01 * magnitude + 0.01;
        printf("%s  %8.3f ms/execution   max error %.4f of %.4f   %s\n", label, ms, worst, magnitude, worst <= tolerance ? "correct" : "WRONG");
    };

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float streamMs = 0.0f, graphMs = 0.0f;

    // --- stream launches, one submission per stage per execution ---
    cudaMemcpy(dX, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    enqueuePipeline(handle, stream, dA, dB, dX, dHidden, dProjected, n, factor);   // warm up
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < executions; i++) {
        cudaMemcpyAsync(dX, x.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
        enqueuePipeline(handle, stream, dA, dB, dX, dHidden, dProjected, n, factor);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&streamMs, start, stop);
    streamMs /= executions;
    verify("stream launches ", streamMs);

    // --- the same sequence captured once and replayed ---
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    cudaMemcpyAsync(dX, x.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
    enqueuePipeline(handle, stream, dA, dB, dX, dHidden, dProjected, n, factor);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    cudaGraphLaunch(graphExec, stream);            // warm up the replay
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < executions; i++) cudaGraphLaunch(graphExec, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&graphMs, start, stop);
    graphMs /= executions;
    verify("CUDA graph      ", graphMs);

    printf("\nCUDA graph replay: %.3f ms -> %.3f ms per execution, %.2fx faster\n", streamMs, graphMs, streamMs / graphMs);
    printf("Kernel time only (CUDA events). The Java demo measures wall clock, so its\n");
    printf("speedup is larger: it also removes the JVM-side dispatch of four tasks.\n");

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dX);
    cudaFree(dHidden);
    cudaFree(dProjected);
    return 0;
}
