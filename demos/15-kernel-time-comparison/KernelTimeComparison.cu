// CUDA equivalent of KernelTimeComparison.java.
//
// Deliberately written to make the kernel-time comparison as fair as possible:
//
//   * identical kernel names (elementwise / polynomial / stencil), so an Nsight
//     Systems kernel summary from each implementation lines up row by row;
//   * identical block size (256) and grid size, so neither side is advantaged
//     by a different launch configuration;
//   * identical arithmetic, including the bounds checks and the 1/3 constant.
//
// No fast-math, no -use_fast_math: TornadoVM does not enable it either, and
// turning it on here would measure a compiler flag rather than code generation.
//
// Build & run:
//   nvcc -arch=sm_89 -o kernel_time_comparison KernelTimeComparison.cu
//   ./kernel_time_comparison [n] [degree] [executions]

#include <cstdio>
#include <cstdlib>
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

static const int BLOCK = 256;
static const float SCALE = 0.25f;
static const float OFFSET = 0.1f;
static const float COEF = 0.5f;

// Memory-bound: one read, one write, no reuse.
__global__ void elementwise(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * SCALE + OFFSET;
    }
}

// Compute-bound: a dependent chain of `degree` fused multiply-adds.
__global__ void polynomial(const float *in, float *out, int degree, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i];
        float acc = 1.0f;
        for (int d = 0; d < degree; d++) {
            acc = acc * v + COEF;
        }
        out[i] = acc;
    }
}

// Memory-bound with neighbour access: three reads, one write.
__global__ void stencil(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float left = i > 0 ? in[i - 1] : in[0];
        float right = i < n - 1 ? in[i + 1] : in[n - 1];
        out[i] = (left + in[i] + right) * (1.0f / 3.0f);
    }
}

static std::vector<float> reference(const std::vector<float> &input, int n, int degree) {
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; i++) {
        a[i] = input[i] * SCALE + OFFSET;
    }
    for (int i = 0; i < n; i++) {
        float v = a[i];
        float acc = 1.0f;
        for (int d = 0; d < degree; d++) {
            acc = acc * v + COEF;
        }
        b[i] = acc;
    }
    for (int i = 0; i < n; i++) {
        float left = i > 0 ? b[i - 1] : b[0];
        float right = i < n - 1 ? b[i + 1] : b[n - 1];
        c[i] = (left + b[i] + right) * (1.0f / 3.0f);
    }
    return c;
}

int main(int argc, char **argv) {
    const int n = argc > 1 ? atoi(argv[1]) : (1 << 22);
    const int degree = argc > 2 ? atoi(argv[2]) : 256;
    const int executions = argc > 3 ? atoi(argv[3]) : 20;

    printf("Kernel-time comparison: n=%d, polynomial degree=%d, %d executions\n", n, degree, executions);
    printf("  block size %d (identical to the TornadoVM version)\n", BLOCK);
    printf("  kernels: elementwise (memory-bound) -> polynomial (compute-bound) -> stencil (memory-bound)\n");
    printf("  NOTE: the wall-clock below includes host dispatch and transfers.\n");
    printf("        Compare kernel time with nsys -- see this demo's README.\n\n");

    const size_t bytes = (size_t) n * sizeof(float);
    std::vector<float> hostInput(n), hostOutput(n);
    for (int i = 0; i < n; i++) {
        hostInput[i] = (float) (i % 1024) / 1024.0f;
    }
    std::vector<float> ref = reference(hostInput, n, degree);

    float *devInput, *devTmpA, *devTmpB, *devOutput;
    CUDA_CHECK(cudaMalloc(&devInput, bytes));
    CUDA_CHECK(cudaMalloc(&devTmpA, bytes));
    CUDA_CHECK(cudaMalloc(&devTmpB, bytes));
    CUDA_CHECK(cudaMalloc(&devOutput, bytes));
    CUDA_CHECK(cudaMemcpy(devInput, hostInput.data(), bytes, cudaMemcpyHostToDevice));

    const int blocks = (n + BLOCK - 1) / BLOCK;
    std::vector<double> wallUs(executions);

    for (int e = 0; e < executions; e++) {
        auto start = std::chrono::steady_clock::now();
        elementwise<<<blocks, BLOCK>>>(devInput, devTmpA, n);
        polynomial<<<blocks, BLOCK>>>(devTmpA, devTmpB, degree, n);
        stencil<<<blocks, BLOCK>>>(devTmpB, devOutput, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hostOutput.data(), devOutput, bytes, cudaMemcpyDeviceToHost));
        auto elapsed = std::chrono::steady_clock::now() - start;
        wallUs[e] = std::chrono::duration<double, std::micro>(elapsed).count();
    }

    std::vector<double> steady(wallUs.begin() + 1, wallUs.end());
    std::sort(steady.begin(), steady.end());
    printf("first execution: %.0f us\n", wallUs[0]);
    printf("steady-state median wall-clock (n=%zu): %.0f us\n", steady.size(), steady[steady.size() / 2]);

    float maxAbs = 0.0f;
    int bad = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(hostOutput[i] - ref[i]);
        maxAbs = fmaxf(maxAbs, diff);
        if (diff > 1e-4f) {
            bad++;
        }
    }
    printf("validation %s (max abs err %.7f, %d/%d elements out of tol)\n",
           bad == 0 ? "PASSED" : "FAILED", maxAbs, bad, n);
    printf("%s\n", bad == 0 ? "Result is correct" : "Result is INCORRECT");

    CUDA_CHECK(cudaFree(devInput));
    CUDA_CHECK(cudaFree(devTmpA));
    CUDA_CHECK(cudaFree(devTmpB));
    CUDA_CHECK(cudaFree(devOutput));
    return 0;
}
