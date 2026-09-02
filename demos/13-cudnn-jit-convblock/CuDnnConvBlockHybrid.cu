// CUDA equivalent of CuDnnConvBlockHybrid.java.
//
// The same four-stage conv block. In Java it is one TaskGraph with two
// libraryTask calls and two ordinary tasks. Here it is a cuDNN handle, four
// tensor descriptors, a filter descriptor, a convolution descriptor, an
// algorithm choice, a workspace, and your own kernels around them -- all of
// which the hybrid API keeps out of your code.
//
//   1. your kernel  "scale"
//   2. cuDNN        cudnnConvolutionForward   (NCHW, 3x3, pad 1, stride 1)
//   3. your kernel  "addBias"
//   4. cuDNN        cudnnActivationForward    (ReLU)
//
// Build & run:
//   nvcc -arch=sm_89 -lcudnn -o cudnn_conv_block CuDnnConvBlockHybrid.cu
//   ./cudnn_conv_block [N] [C] [H] [W] [K] [executions]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cudnn.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUDNN_CHECK(call)                                                       \
    do {                                                                        \
        cudnnStatus_t st_ = (call);                                             \
        if (st_ != CUDNN_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuDNN error %s at %s:%d\n",                        \
                    cudnnGetErrorString(st_), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static const int R = 3, S = 3, PAD = 1, STRIDE = 1;
static const float SCALE = 0.25f;

__global__ void scaleKernel(const float *in, float *scaled, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        scaled[i] = in[i] * SCALE;
    }
}

__global__ void addBiasKernel(float *conv, const float *bias, int hw, int k, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        conv[i] = conv[i] + bias[(i / hw) % k];
    }
}

static void deterministic(std::vector<float> &arr, int seed, float span) {
    for (size_t i = 0; i < arr.size(); i++) {
        arr[i] = (float) (((long) i * 31 + seed) % 17 - 8) / span;
    }
}

static std::vector<float> reference(const std::vector<float> &in, const std::vector<float> &filter,
                                    const std::vector<float> &bias,
                                    int n, int c, int h, int w, int k) {
    int hw = h * w;
    std::vector<float> out((size_t) n * k * hw);
    for (int img = 0; img < n; img++) {
        for (int oc = 0; oc < k; oc++) {
            for (int oh = 0; oh < h; oh++) {
                for (int ow = 0; ow < w; ow++) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < c; ic++) {
                        for (int r = 0; r < R; r++) {
                            for (int s = 0; s < S; s++) {
                                int ih = oh * STRIDE + r - PAD;
                                int iw = ow * STRIDE + s - PAD;
                                if (ih < 0 || ih >= h || iw < 0 || iw >= w) {
                                    continue;
                                }
                                float x = in[(size_t) ((img * c + ic) * h + ih) * w + iw] * SCALE;
                                float f = filter[(size_t) ((oc * c + ic) * R + r) * S + s];
                                sum += x * f;
                            }
                        }
                    }
                    sum += bias[oc];
                    out[(size_t) ((img * k + oc) * h + oh) * w + ow] = fmaxf(sum, 0.0f);
                }
            }
        }
    }
    return out;
}

int main(int argc, char **argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 4;
    const int c = argc > 2 ? atoi(argv[2]) : 16;
    const int h = argc > 3 ? atoi(argv[3]) : 32;
    const int w = argc > 4 ? atoi(argv[4]) : 32;
    const int k = argc > 5 ? atoi(argv[5]) : 16;
    const int executions = argc > 6 ? atoi(argv[6]) : 10;

    const int hw = h * w;
    const size_t inSize = (size_t) n * c * hw;
    const size_t outSize = (size_t) n * k * hw;
    const size_t filterSize = (size_t) k * c * R * S;

    printf("cuDNN + hand-written kernels conv block: NCHW %dx%dx%dx%d, %d filters %dx%d (pad %d, stride %d)\n",
           n, c, h, w, k, R, S, PAD, STRIDE);
    printf("  pipeline: scale -> cudnnConvolutionForward -> addBias -> cudnnActivationForward(ReLU)\n");
    printf("  %d executions, steady-state median reported (first execution excluded)\n\n", executions);

    std::vector<float> hostInput(inSize), hostFilter(filterSize), hostBias(k), hostOutput(outSize);
    deterministic(hostInput, 1, 8.0f);
    deterministic(hostFilter, 5, 16.0f);
    deterministic(hostBias, 3, 32.0f);
    std::vector<float> ref = reference(hostInput, hostFilter, hostBias, n, c, h, w, k);

    float *devInput, *devScaled, *devFilter, *devBias, *devConv, *devOutput;
    CUDA_CHECK(cudaMalloc(&devInput, inSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devScaled, inSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devFilter, filterSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devBias, k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devConv, outSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devOutput, outSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(devInput, hostInput.data(), inSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devFilter, hostFilter.data(), filterSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devBias, hostBias.data(), k * sizeof(float), cudaMemcpyHostToDevice));

    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t inDesc, outDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnActivationDescriptor_t actDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outDesc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, k, h, w));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, R, S));
    // CROSS_CORRELATION, not CONVOLUTION: no filter flip, matching the reference.
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, PAD, PAD, STRIDE, STRIDE, 1, 1,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU,
                                             CUDNN_NOT_PROPAGATE_NAN, 0.0));

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspaceBytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inDesc, filterDesc, convDesc,
                                                        outDesc, algo, &workspaceBytes));
    void *workspace = nullptr;
    if (workspaceBytes > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspaceBytes));
    }

    const float alpha = 1.0f, beta = 0.0f;
    const int threads = 256;
    std::vector<double> wallUs(executions);

    for (int e = 0; e < executions; e++) {
        auto start = std::chrono::steady_clock::now();

        scaleKernel<<<(inSize + threads - 1) / threads, threads>>>(devInput, devScaled, (int) inSize);
        CUDA_CHECK(cudaGetLastError());

        CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, inDesc, devScaled, filterDesc, devFilter,
                                            convDesc, algo, workspace, workspaceBytes,
                                            &beta, outDesc, devConv));

        addBiasKernel<<<(outSize + threads - 1) / threads, threads>>>(devConv, devBias, hw, k, (int) outSize);
        CUDA_CHECK(cudaGetLastError());

        CUDNN_CHECK(cudnnActivationForward(cudnn, actDesc, &alpha, outDesc, devConv,
                                           &beta, outDesc, devOutput));

        CUDA_CHECK(cudaMemcpy(hostOutput.data(), devOutput, outSize * sizeof(float),
                              cudaMemcpyDeviceToHost));
        auto elapsed = std::chrono::steady_clock::now() - start;
        wallUs[e] = std::chrono::duration<double, std::micro>(elapsed).count();
        if (e == 0) {
            printf("  first execution (cuDNN plan setup): %.0f us\n", wallUs[e]);
        }
    }

    std::vector<double> steady(wallUs.begin() + 1, wallUs.end());
    std::sort(steady.begin(), steady.end());
    printf("  steady-state median wall-clock (n=%zu): %.0f us\n", steady.size(), steady[steady.size() / 2]);

    float maxAbs = 0.0f;
    int bad = 0;
    for (size_t i = 0; i < outSize; i++) {
        float diff = fabsf(hostOutput[i] - ref[i]);
        maxAbs = fmaxf(maxAbs, diff);
        if (diff > 1e-3f) {
            bad++;
        }
    }
    printf("  validation %s (max abs err %.6f, %d/%zu elements out of tol)\n",
           bad == 0 ? "PASSED" : "FAILED", maxAbs, bad, outSize);
    printf("%s\n", bad == 0 ? "Result is correct" : "Result is INCORRECT");

    if (workspace) {
        CUDA_CHECK(cudaFree(workspace));
    }
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(inDesc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
    CUDA_CHECK(cudaFree(devInput));
    CUDA_CHECK(cudaFree(devScaled));
    CUDA_CHECK(cudaFree(devFilter));
    CUDA_CHECK(cudaFree(devBias));
    CUDA_CHECK(cudaFree(devConv));
    CUDA_CHECK(cudaFree(devOutput));
    return 0;
}
