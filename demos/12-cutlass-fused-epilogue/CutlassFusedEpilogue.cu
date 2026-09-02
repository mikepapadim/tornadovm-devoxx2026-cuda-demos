// CUDA equivalent of CutlassFusedEpilogue.java.
//
// The same fused-vs-unfused comparison, written directly against CUTLASS.
//
//   fused   : your "scale" kernel -> cutlass::gemm::device::Gemm with a
//             LinearCombinationRelu epilogue                       (1 kernel)
//   unfused : your "scale" kernel -> the same Gemm with a plain
//             LinearCombination epilogue -> your "biasRelu" kernel (2 kernels)
//
// The Java version selects the epilogue by calling a different method
// (Cutlass::cutlassGemmBiasRelu vs Cutlass::cutlassHgemm). Here it is a
// template parameter, and you also supply the layouts, the tile shapes, the
// architecture tag, the stage count, and the problem-size/leading-dimension
// arguments by hand.
//
// CUTLASS is header-only and NOT vendored in this repo. Fetch it once:
//   git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git
//
// Build & run (CUTLASS_DIR = that checkout):
//   nvcc -arch=sm_89 -std=c++17 -I$CUTLASS_DIR/include -I$CUTLASS_DIR/tools/util/include \
//        -o cutlass_fused_epilogue CutlassFusedEpilogue.cu
//   ./cutlass_fused_epilogue [M] [N] [K] [executions]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/numeric_types.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUTLASS_CHECK(status)                                                   \
    do {                                                                        \
        cutlass::Status s_ = (status);                                          \
        if (s_ != cutlass::Status::kSuccess) {                                  \
            fprintf(stderr, "CUTLASS error %s at %s:%d\n",                      \
                    cutlassGetStatusString(s_), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static const int TILE = 32;
static const float SCALE = 0.5f;

using Element = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;

// Same shapes TornadoVM's CUTLASS provider selects on sm_80+ (visible in the
// kernel name in an nsys trace): 128x128x32 threadblock, 64x64x32 warp,
// 16x8x16 instruction, 3 stages.
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using GemmRelu = cutlass::gemm::device::Gemm<
    Element, Layout, Element, Layout, Element, Layout,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape,
    cutlass::epilogue::thread::LinearCombinationRelu<Element, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 3>;

using GemmPlain = cutlass::gemm::device::Gemm<
    Element, Layout, Element, Layout, Element, Layout,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape,
    cutlass::epilogue::thread::LinearCombination<Element, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 3>;

__global__ void scaleKernel(const Element *a, Element *scaled, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        scaled[i] = Element(float(a[i]) * SCALE);
    }
}

// The epilogue CUTLASS fuses in the "fused" path, as its own pass.
__global__ void biasReluKernel(Element *c, const Element *bias, int n, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float v = float(c[i]) + float(bias[i % n]);
        c[i] = Element(fmaxf(v, 0.0f));
    }
}

// Broadcasts the length-N bias vector to an MxN matrix, so the fused epilogue
// can consume it as its `C` operand with beta = 1.
__global__ void broadcastBias(const Element *bias, Element *biasMatrix, int n, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        biasMatrix[i] = bias[i % n];
    }
}

static void deterministic(std::vector<Element> &arr, int seed, float span) {
    for (size_t i = 0; i < arr.size(); i++) {
        arr[i] = Element((float) (((long) i * 31 + seed) % 17 - 8) / span);
    }
}

static std::vector<float> reference(const std::vector<Element> &a, const std::vector<Element> &b,
                                    const std::vector<Element> &bias, int m, int n, int k) {
    std::vector<float> out((size_t) m * n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += (float(a[(size_t) i * k + p]) * SCALE) * float(b[(size_t) p * n + j]);
            }
            sum += float(bias[j]);
            out[(size_t) i * n + j] = fmaxf(sum, 0.0f);
        }
    }
    return out;
}

static bool validate(const char *label, const std::vector<Element> &got, const std::vector<float> &ref) {
    float maxAbs = 0.0f;
    int bad = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        float diff = fabsf(float(got[i]) - ref[i]);
        maxAbs = fmaxf(maxAbs, diff);
        if (diff > 0.05f) {
            bad++;
        }
    }
    bool ok = bad == 0;
    printf("  [%-7s] validation %s (max abs err %.5f, %d/%zu cells out of tol)\n",
           label, ok ? "PASSED" : "FAILED", maxAbs, bad, ref.size());
    return ok;
}

int main(int argc, char **argv) {
    int m = argc > 1 ? atoi(argv[1]) : 1024;
    int n = argc > 2 ? atoi(argv[2]) : 1024;
    int k = argc > 3 ? atoi(argv[3]) : 1024;
    int executions = argc > 4 ? atoi(argv[4]) : 20;

    if (m % TILE || n % TILE || k % TILE) {
        printf("M, N and K must all be multiples of %d (got %d, %d, %d).\n", TILE, m, n, k);
        return 1;
    }

    printf("CUTLASS fused epilogue demo: C[%dx%d] = relu(scale(A[%dx%d]) * B[%dx%d] + bias), fp16\n",
           m, n, m, k, k, n);
    printf("  %d executions per mode, steady-state median reported (first execution excluded)\n\n",
           executions);

    std::vector<Element> hostA((size_t) m * k), hostB((size_t) k * n), hostBias(n);
    deterministic(hostA, 1, 16.0f);
    deterministic(hostB, 7, 16.0f);
    deterministic(hostBias, 3, 32.0f);
    std::vector<float> ref = reference(hostA, hostB, hostBias, m, n, k);

    Element *dA, *dB, *dBias, *dBiasMatrix, *dScaled, *dC;
    CUDA_CHECK(cudaMalloc(&dA, (size_t) m * k * sizeof(Element)));
    CUDA_CHECK(cudaMalloc(&dB, (size_t) k * n * sizeof(Element)));
    CUDA_CHECK(cudaMalloc(&dBias, (size_t) n * sizeof(Element)));
    CUDA_CHECK(cudaMalloc(&dBiasMatrix, (size_t) m * n * sizeof(Element)));
    CUDA_CHECK(cudaMalloc(&dScaled, (size_t) m * k * sizeof(Element)));
    CUDA_CHECK(cudaMalloc(&dC, (size_t) m * n * sizeof(Element)));
    CUDA_CHECK(cudaMemcpy(dA, hostA.data(), (size_t) m * k * sizeof(Element), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hostB.data(), (size_t) k * n * sizeof(Element), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBias, hostBias.data(), (size_t) n * sizeof(Element), cudaMemcpyHostToDevice));

    const int threads = 256;
    broadcastBias<<<((size_t) m * n + threads - 1) / threads, threads>>>(dBias, dBiasMatrix, n, m * n);
    CUDA_CHECK(cudaGetLastError());

    std::vector<Element> hostC((size_t) m * n);
    cutlass::gemm::GemmCoord problem(m, n, k);

    auto timeMode = [&](const char *label, bool fused) {
        printf("=== %s ===\n", label);
        std::vector<double> wallUs(executions);
        for (int e = 0; e < executions; e++) {
            auto start = std::chrono::steady_clock::now();

            scaleKernel<<<((size_t) m * k + threads - 1) / threads, threads>>>(dA, dScaled, m * k);
            CUDA_CHECK(cudaGetLastError());

            if (fused) {
                // beta = 1 with C = the broadcast bias matrix, and the ReLU folded
                // into the epilogue: one kernel does GEMM + bias + activation.
                GemmRelu gemm;
                typename GemmRelu::Arguments args(problem,
                    {dScaled, k}, {dB, n}, {dBiasMatrix, n}, {dC, n}, {1.0f, 1.0f});
                CUTLASS_CHECK(gemm(args));
            } else {
                GemmPlain gemm;
                typename GemmPlain::Arguments args(problem,
                    {dScaled, k}, {dB, n}, {dC, n}, {dC, n}, {1.0f, 0.0f});
                CUTLASS_CHECK(gemm(args));
                biasReluKernel<<<((size_t) m * n + threads - 1) / threads, threads>>>(dC, dBias, n, m * n);
                CUDA_CHECK(cudaGetLastError());
            }

            CUDA_CHECK(cudaMemcpy(hostC.data(), dC, (size_t) m * n * sizeof(Element),
                                  cudaMemcpyDeviceToHost));
            auto elapsed = std::chrono::steady_clock::now() - start;
            wallUs[e] = std::chrono::duration<double, std::micro>(elapsed).count();
            if (e == 0) {
                printf("  first execution (CUTLASS plan setup): %.0f us\n", wallUs[e]);
            }
        }
        std::vector<double> steady(wallUs.begin() + 1, wallUs.end());
        std::sort(steady.begin(), steady.end());
        double median = steady[steady.size() / 2];
        printf("  steady-state median wall-clock (n=%zu): %.0f us\n", steady.size(), median);
        return median;
    };

    double fusedMedian = timeMode("FUSED (CUTLASS LinearCombinationRelu epilogue - one kernel)", true);
    bool okFused = validate("fused", hostC, ref);
    printf("\n");
    double unfusedMedian = timeMode("UNFUSED (CUTLASS LinearCombination + separate bias/ReLU kernel)", false);
    bool okUnfused = validate("unfused", hostC, ref);

    printf("\n=== Summary (steady-state median us, this run/this GPU) ===\n");
    printf("fused   : %.0f\n", fusedMedian);
    printf("unfused : %.0f (%.2fx the fused time)\n", unfusedMedian, unfusedMedian / fusedMedian);
    printf("%s\n", (okFused && okUnfused) ? "Both modes produce the same, correct result"
                                          : "Result is INCORRECT");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dBias));
    CUDA_CHECK(cudaFree(dBiasMatrix));
    CUDA_CHECK(cudaFree(dScaled));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
