// Hand-written CUDA equivalent of TileFlashAttention.java: attention two ways, both in
// CUDA Tile C++.
//
//   1. materialised  scores = Q Kt -> row softmax -> out = P V. Three kernels and an
//                    [S_Q, S_KV] score matrix in device memory.
//   2. flash         one kernel, one pass over the KV sequence, online softmax. The score
//                    matrix never exists.
//
// The flash kernel below and TileFlashAttention.flash in the Java file are the same
// program: same tiles, same order, same ct:: operations. That is the claim this pair of
// files exists to let a reader check.
//
// Build (needs CUDA Toolkit 13.3 or newer):
//   nvcc --enable-tile -std=c++20 -arch=sm_89 -o tileflash TileFlashAttention.cu
//   ./tileflash 128 256 20

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

namespace ct = cuda::tiles;
using namespace ct::literals;

constexpr int HEAD_DIM = 64;
constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 32;
constexpr int MAX_KV = 256;
constexpr float NEGATIVE_LIMIT = -1.0e30f;

// --- flash attention: one kernel, no score matrix -------------------------------------
__tile_global__ void flash(const __half *q, const __half *k, const __half *v, float *out, int queryRows, int kvRows, int kvBlocks, float scale) {
    auto qView = ct::partition_view{ct::tensor_span{q, ct::extents{queryRows, HEAD_DIM}}, ct::shape<BLOCK_M, HEAD_DIM>{}};
    auto kView = ct::partition_view{ct::tensor_span{k, ct::extents{kvRows, HEAD_DIM}}, ct::shape<BLOCK_N, HEAD_DIM>{}};
    auto vView = ct::partition_view{ct::tensor_span{v, ct::extents{kvRows, HEAD_DIM}}, ct::shape<BLOCK_N, HEAD_DIM>{}};
    auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{queryRows, HEAD_DIM}}, ct::shape<BLOCK_M, HEAD_DIM>{}};

    int queryBlock = ct::bid().x;
    auto query = qView.load(queryBlock, 0);

    auto rowMax = ct::full<ct::tile<float, ct::shape<BLOCK_M, 1>>>(NEGATIVE_LIMIT);
    auto rowSum = ct::zeros<ct::tile<float, ct::shape<BLOCK_M, 1>>>();
    auto acc = ct::zeros<ct::tile<float, ct::shape<BLOCK_M, HEAD_DIM>>>();

    for (auto block : ct::irange(0, kvBlocks)) {
        auto keys = ct::transpose(kView.load(block, 0));
        auto scores = ct::mma(query, keys, ct::zeros<ct::tile<float, ct::shape<BLOCK_M, BLOCK_N>>>()) * scale;

        auto newMax = ct::max(rowMax, ct::reduce_max(scores, 1_ic));
        auto probabilities = ct::exp(scores - newMax);
        auto correction = ct::exp(rowMax - newMax);

        rowSum = rowSum * correction + ct::sum(probabilities, 1_ic);
        acc = acc * correction;
        acc = ct::mma(ct::element_cast<__half>(probabilities), vView.load(block, 0), acc);
        rowMax = newMax;
    }

    outView.store(acc / rowSum, queryBlock, 0);
}

// --- the materialised path ------------------------------------------------------------
__tile_global__ void scores(const __half *q, const __half *k, float *scoreMatrix, int queryRows, int kvRows, float scale) {
    auto qView = ct::partition_view{ct::tensor_span{q, ct::extents{queryRows, HEAD_DIM}}, ct::shape<BLOCK_M, HEAD_DIM>{}};
    auto kView = ct::partition_view{ct::tensor_span{k, ct::extents{kvRows, HEAD_DIM}}, ct::shape<BLOCK_N, HEAD_DIM>{}};
    auto scoreView = ct::partition_view{ct::tensor_span{scoreMatrix, ct::extents{queryRows, kvRows}}, ct::shape<BLOCK_M, BLOCK_N>{}};

    int rowBlock = ct::bid().x;
    int columnBlock = ct::bid().y;
    auto keys = ct::transpose(kView.load(columnBlock, 0));
    auto product = ct::mma(qView.load(rowBlock, 0), keys, ct::zeros<ct::tile<float, ct::shape<BLOCK_M, BLOCK_N>>>());
    scoreView.store(product * scale, rowBlock, columnBlock);
}

__tile_global__ void softmaxRows(const float *scoreMatrix, __half *probabilities, int queryRows) {
    auto scoreView = ct::partition_view{ct::tensor_span{scoreMatrix, ct::extents{queryRows, MAX_KV}}, ct::shape<1, MAX_KV>{}};
    auto outView = ct::partition_view{ct::tensor_span{probabilities, ct::extents{queryRows, MAX_KV}}, ct::shape<1, MAX_KV>{}};

    int row = ct::bid().x;
    auto values = scoreView.load(row, 0);
    auto shifted = ct::exp(values - ct::reduce_max(values, 1_ic));
    outView.store(ct::element_cast<__half>(shifted / ct::sum(shifted, 1_ic)), row, 0);
}

__tile_global__ void weightedSum(const __half *probabilities, const __half *v, float *out, int queryRows, int kvRows, int kvBlocks) {
    auto pView = ct::partition_view{ct::tensor_span{probabilities, ct::extents{queryRows, kvRows}}, ct::shape<BLOCK_M, BLOCK_N>{}};
    auto vView = ct::partition_view{ct::tensor_span{v, ct::extents{kvRows, HEAD_DIM}}, ct::shape<BLOCK_N, HEAD_DIM>{}};
    auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{queryRows, HEAD_DIM}}, ct::shape<BLOCK_M, HEAD_DIM>{}};

    int rowBlock = ct::bid().x;
    auto acc = ct::zeros<ct::tile<float, ct::shape<BLOCK_M, HEAD_DIM>>>();
    for (auto block : ct::irange(0, kvBlocks)) {
        acc = ct::mma(pView.load(rowBlock, block), vView.load(block, 0), acc);
    }
    outView.store(acc, rowBlock, 0);
}

// --------------------------------------------------------------------------------------

int main(int argc, char **argv) {
    int queryRows = argc > 1 ? atoi(argv[1]) : 128;
    int kvRows = argc > 2 ? atoi(argv[2]) : MAX_KV;
    int executions = argc > 3 ? atoi(argv[3]) : 20;
    if (queryRows % BLOCK_M != 0 || kvRows % BLOCK_N != 0) {
        printf("queryRows must be a multiple of %d and kvRows of %d\n", BLOCK_M, BLOCK_N);
        return 1;
    }
    const float scale = 1.0f / sqrtf((float) HEAD_DIM);
    printf("Q = [%d, %d], K = V = [%d, %d], %d executions\n\n", queryRows, HEAD_DIM, kvRows, HEAD_DIM, executions);

    std::vector<__half> q(queryRows * HEAD_DIM), k(kvRows * HEAD_DIM), v(kvRows * HEAD_DIM);
    srand(101);
    for (auto &value : q) value = __float2half((float) rand() / RAND_MAX - 0.5f);
    for (auto &value : k) value = __float2half((float) rand() / RAND_MAX - 0.5f);
    for (auto &value : v) value = __float2half((float) rand() / RAND_MAX - 0.5f);

    // Two-pass reference on the CPU.
    std::vector<double> reference(queryRows * HEAD_DIM, 0.0);
    for (int row = 0; row < queryRows; row++) {
        std::vector<double> rowScores(kvRows);
        double maximum = -INFINITY;
        for (int key = 0; key < kvRows; key++) {
            double dot = 0.0;
            for (int d = 0; d < HEAD_DIM; d++)
                dot += __half2float(q[row * HEAD_DIM + d]) * __half2float(k[key * HEAD_DIM + d]);
            rowScores[key] = dot * scale;
            maximum = fmax(maximum, rowScores[key]);
        }
        double denominator = 0.0;
        for (int key = 0; key < kvRows; key++) {
            rowScores[key] = exp(rowScores[key] - maximum);
            denominator += rowScores[key];
        }
        for (int key = 0; key < kvRows; key++) {
            double weight = rowScores[key] / denominator;
            for (int d = 0; d < HEAD_DIM; d++)
                reference[row * HEAD_DIM + d] += weight * __half2float(v[key * HEAD_DIM + d]);
        }
    }

    __half *dQ, *dK, *dV, *dProbabilities;
    float *dOut, *dScores;
    cudaMalloc(&dQ, q.size() * sizeof(__half));
    cudaMalloc(&dK, k.size() * sizeof(__half));
    cudaMalloc(&dV, v.size() * sizeof(__half));
    cudaMalloc(&dOut, queryRows * HEAD_DIM * sizeof(float));
    cudaMalloc(&dScores, queryRows * kvRows * sizeof(float));
    cudaMalloc(&dProbabilities, queryRows * kvRows * sizeof(__half));
    cudaMemcpy(dQ, q.data(), q.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, k.data(), k.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, v.data(), v.size() * sizeof(__half), cudaMemcpyHostToDevice);

    std::vector<float> out(queryRows * HEAD_DIM);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto verify = [&](const char *label, float ms) {
        cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double worst = 0.0;
        for (size_t i = 0; i < out.size(); i++) worst = fmax(worst, fabs(out[i] - reference[i]));
        printf("%-46s %14.3f %14.4f   %s\n", label, ms, worst, worst <= 0.01 ? "correct" : "WRONG");
    };

    printf("%-46s %14s %14s   %s\n", "path", "ms/execution", "max error", "result");

    // --- materialised ---
    float materialisedMs = NAN;
    if (kvRows == MAX_KV) {
        auto enqueueMaterialised = [&]() {
            scores<<<dim3(queryRows / BLOCK_M, kvRows / BLOCK_N), 1>>>(dQ, dK, dScores, queryRows, kvRows, scale);
            softmaxRows<<<queryRows, 1>>>(dScores, dProbabilities, queryRows);
            weightedSum<<<queryRows / BLOCK_M, 1>>>(dProbabilities, dV, dOut, queryRows, kvRows, kvRows / BLOCK_N);
        };
        enqueueMaterialised();
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < executions; i++) enqueueMaterialised();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&materialisedMs, start, stop);
        materialisedMs /= executions;
        verify("1. materialised (3 kernels + score matrix)", materialisedMs);
    } else {
        printf("1. materialised: skipped, its softmax kernel is compiled for kvRows = %d\n", MAX_KV);
    }

    // --- flash ---
    flash<<<queryRows / BLOCK_M, 1>>>(dQ, dK, dV, dOut, queryRows, kvRows, kvRows / BLOCK_N, scale);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < executions; i++)
        flash<<<queryRows / BLOCK_M, 1>>>(dQ, dK, dV, dOut, queryRows, kvRows, kvRows / BLOCK_N, scale);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float flashMs = 0.0f;
    cudaEventElapsedTime(&flashMs, start, stop);
    flashMs /= executions;
    verify("2. flash (1 kernel, online softmax)", flashMs);

    if (!isnan(materialisedMs)) {
        printf("\nfusion: %.3f ms -> %.3f ms per execution, %.2fx faster\n", materialisedMs, flashMs, materialisedMs / flashMs);
        printf("Kernel time only (CUDA events), so this isolates the fusion from host dispatch.\n");
    }

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dOut);
    cudaFree(dScores);
    cudaFree(dProbabilities);
    return 0;
}
