// CUDA equivalent of IntegratedShowcase.java.
//
// Everything at once: N independent chains of (your kernel -> cuBLAS sgemv ->
// your kernel), run four ways -- baseline, multi-stream, CUDA graph, and both
// combined -- plus the single-tile Tensor Core kernel from demo 08 as a bonus.
//
// In Java the four modes differ by one method call each on the same plan
// (nothing, withIntraPlanConcurrency(), withCUDAGraph(), or both). Here each
// mode is a different piece of host code: a stream pool to build and destroy,
// a capture/instantiate/launch cycle, pinned host buffers so the captured
// copies stay valid, and cuBLAS's stream binding to keep in sync with all of it.
//
// Build & run:
//   nvcc -arch=sm_89 -lcublas -o integrated_showcase IntegratedShowcase.cu
//   ./integrated_showcase [units] [m] [n] [executions] [baseline|concurrent|graph|combined|all]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st_ = (call);                                            \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int) st_,            \
                    __FILE__, __LINE__);                                        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static const float SCALE = 2.0f;
static const float BIAS = 1.0f;
static const int STREAM_POOL = 4;

__global__ void scaleKernel(float *matrix, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        matrix[i] = matrix[i] * SCALE;
    }
}

__global__ void biasKernel(float *output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = output[i] + BIAS;
    }
}

// ---- demo 08's Tensor Core kernel, reused as the bonus stage ----------------
static const int MM = 16, NN = 8, KK = 16;

__global__ void gemmMMASingleTile(const __half *A, const __half *B, float *C) {
    int lane = threadIdx.x;
    int groupID = lane >> 2;
    int tig = lane & 3;

    __half a[8];
    a[0] = A[(groupID) * KK + (2 * tig)];
    a[1] = A[(groupID) * KK + (2 * tig + 1)];
    a[2] = A[(groupID + 8) * KK + (2 * tig)];
    a[3] = A[(groupID + 8) * KK + (2 * tig + 1)];
    a[4] = A[(groupID) * KK + (2 * tig + 8)];
    a[5] = A[(groupID) * KK + (2 * tig + 9)];
    a[6] = A[(groupID + 8) * KK + (2 * tig + 8)];
    a[7] = A[(groupID + 8) * KK + (2 * tig + 9)];

    __half b[4];
    b[0] = B[(2 * tig) * NN + groupID];
    b[1] = B[(2 * tig + 1) * NN + groupID];
    b[2] = B[(2 * tig + 8) * NN + groupID];
    b[3] = B[(2 * tig + 9) * NN + groupID];

    const uint32_t *A32 = reinterpret_cast<const uint32_t *>(a);
    const uint32_t *B32 = reinterpret_cast<const uint32_t *>(b);
    float d[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(A32[0]), "r"(A32[1]), "r"(A32[2]), "r"(A32[3]),
          "r"(B32[0]), "r"(B32[1]));

    C[(groupID) * NN + (2 * tig)] = d[0];
    C[(groupID) * NN + (2 * tig + 1)] = d[1];
    C[(groupID + 8) * NN + (2 * tig)] = d[2];
    C[(groupID + 8) * NN + (2 * tig + 1)] = d[3];
}

static void fillMatrix(float *matrix, int m, int n, int unit, int exec) {
    for (int i = 0; i < m * n; i++) {
        matrix[i] = (float) ((i + unit + exec) % 7) + 1.0f;
    }
}

static void fillVector(float *vector, int n, int unit, int exec) {
    for (int j = 0; j < n; j++) {
        vector[j] = (float) ((j + unit + exec) % 5) + 1.0f;
    }
}

static std::vector<float> expectedChain(int m, int n, int unit, int exec) {
    std::vector<float> mat(m * n), vec(n), out(m);
    for (int i = 0; i < m * n; i++) {
        mat[i] = ((float) ((i + unit + exec) % 7) + 1.0f) * SCALE;
    }
    for (int j = 0; j < n; j++) {
        vec[j] = (float) ((j + unit + exec) % 5) + 1.0f;
    }
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += mat[i * n + j] * vec[j];
        }
        out[i] = sum + BIAS;
    }
    return out;
}

static double runMode(const char *mode, int units, int m, int n, int executions) {
    bool useGraph = (strcmp(mode, "graph") == 0) || (strcmp(mode, "combined") == 0);
    bool concurrent = (strcmp(mode, "concurrent") == 0) || (strcmp(mode, "combined") == 0);
    printf("\n=== %s ===\n", mode);

    const size_t matBytes = (size_t) m * n * sizeof(float);
    const size_t vecBytes = (size_t) n * sizeof(float);
    const size_t outBytes = (size_t) m * sizeof(float);

    std::vector<float *> hMat(units), hVec(units), hOut(units);
    std::vector<float *> dMat(units), dVec(units), dOut(units);
    for (int u = 0; u < units; u++) {
        // Pinned: required for the copies to be capturable into a CUDA graph.
        CUDA_CHECK(cudaMallocHost(&hMat[u], matBytes));
        CUDA_CHECK(cudaMallocHost(&hVec[u], vecBytes));
        CUDA_CHECK(cudaMallocHost(&hOut[u], outBytes));
        CUDA_CHECK(cudaMalloc(&dMat[u], matBytes));
        CUDA_CHECK(cudaMalloc(&dVec[u], vecBytes));
        CUDA_CHECK(cudaMalloc(&dOut[u], outBytes));
    }

    const int nStreams = concurrent ? STREAM_POOL : 1;
    std::vector<cudaStream_t> streams(nStreams);
    for (int s = 0; s < nStreams; s++) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f, beta = 0.0f;
    const int threads = 256;

    auto issue = [&]() {
        for (int u = 0; u < units; u++) {
            cudaStream_t s = streams[u % nStreams];
            // cuBLAS calls go to whichever stream the handle is bound to, so the
            // binding has to be updated per unit to match the stream pool.
            CUBLAS_CHECK(cublasSetStream(handle, s));
            CUDA_CHECK(cudaMemcpyAsync(dMat[u], hMat[u], matBytes, cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(dVec[u], hVec[u], vecBytes, cudaMemcpyHostToDevice, s));
            scaleKernel<<<(m * n + threads - 1) / threads, threads, 0, s>>>(dMat[u], m * n);
            CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, n, m, &alpha,
                                     dMat[u], m, dVec[u], 1, &beta, dOut[u], 1));
            biasKernel<<<(m + threads - 1) / threads, threads, 0, s>>>(dOut[u], m);
            CUDA_CHECK(cudaMemcpyAsync(hOut[u], dOut[u], outBytes, cudaMemcpyDeviceToHost, s));
        }
    };

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    if (useGraph) {
        for (int u = 0; u < units; u++) {
            fillMatrix(hMat[u], m, n, u, 0);
            fillVector(hVec[u], n, u, 0);
        }
        // Capture on the first stream; work issued to the pool is captured as a
        // forked graph as long as the streams are joined before EndCapture.
        CUDA_CHECK(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));
        if (concurrent) {
            // Fork: make every pool stream depend on stream 0's capture point.
            cudaEvent_t fork;
            CUDA_CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(fork, streams[0]));
            for (int s = 1; s < nStreams; s++) {
                CUDA_CHECK(cudaStreamWaitEvent(streams[s], fork, 0));
            }
            issue();
            // Join: stream 0 waits for every other pool stream.
            for (int s = 1; s < nStreams; s++) {
                cudaEvent_t join;
                CUDA_CHECK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(join, streams[s]));
                CUDA_CHECK(cudaStreamWaitEvent(streams[0], join, 0));
                CUDA_CHECK(cudaEventDestroy(join));
            }
            CUDA_CHECK(cudaEventDestroy(fork));
        } else {
            issue();
        }
        CUDA_CHECK(cudaStreamEndCapture(streams[0], &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    }

    std::vector<double> wallUs(executions);
    bool allCorrect = true;
    for (int e = 0; e < executions; e++) {
        for (int u = 0; u < units; u++) {
            fillMatrix(hMat[u], m, n, u, useGraph ? 0 : e);
            fillVector(hVec[u], n, u, useGraph ? 0 : e);
        }

        auto start = std::chrono::steady_clock::now();
        if (useGraph) {
            CUDA_CHECK(cudaGraphLaunch(graphExec, streams[0]));
            CUDA_CHECK(cudaStreamSynchronize(streams[0]));
        } else {
            issue();
            for (int s = 0; s < nStreams; s++) {
                CUDA_CHECK(cudaStreamSynchronize(streams[s]));
            }
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        wallUs[e] = std::chrono::duration<double, std::micro>(elapsed).count();

        for (int u = 0; u < units && allCorrect; u++) {
            std::vector<float> expected = expectedChain(m, n, u, useGraph ? 0 : e);
            for (int i = 0; i < m; i++) {
                if (fabsf(expected[i] - hOut[u][i]) > 0.01f) {
                    allCorrect = false;
                    break;
                }
            }
        }
        if (e < 3 || e == executions - 1) {
            printf("execution %d: %s, wall=%.0f us\n", e, allCorrect ? "correct" : "WRONG", wallUs[e]);
        }
    }

    std::vector<double> steady(wallUs.begin() + 1, wallUs.end());
    std::sort(steady.begin(), steady.end());
    double median = steady[steady.size() / 2];
    printf("%s first execution: %.0f us\n", mode, wallUs[0]);
    printf("%s steady-state median wall-clock (n=%zu): %.0f us\n", mode, steady.size(), median);
    printf("%s: %s\n", mode, allCorrect ? "All executions correct" : "Some executions WRONG");

    if (useGraph) {
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    for (int s = 0; s < nStreams; s++) {
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
    for (int u = 0; u < units; u++) {
        CUDA_CHECK(cudaFree(dMat[u]));
        CUDA_CHECK(cudaFree(dVec[u]));
        CUDA_CHECK(cudaFree(dOut[u]));
        CUDA_CHECK(cudaFreeHost(hMat[u]));
        CUDA_CHECK(cudaFreeHost(hVec[u]));
        CUDA_CHECK(cudaFreeHost(hOut[u]));
    }
    return median;
}

static void runTensorCoreBonus() {
    printf("\n=== BONUS: Tensor Core mma.sync single-tile GEMM (demo 08 kernel, reused) ===\n");
    __half hA[MM * KK], hB[KK * NN];
    float hC[MM * NN], ref[MM * NN];
    for (int i = 0; i < MM * KK; i++) {
        hA[i] = __float2half((float) ((i * 31 + 1) % 17 - 8) / 8.0f);
    }
    for (int i = 0; i < KK * NN; i++) {
        hB[i] = __float2half((float) ((i * 31 + 7) % 17 - 8) / 8.0f);
    }
    for (int i = 0; i < MM; i++) {
        for (int j = 0; j < NN; j++) {
            float sum = 0.0f;
            for (int k = 0; k < KK; k++) {
                sum += __half2float(hA[i * KK + k]) * __half2float(hB[k * NN + j]);
            }
            ref[i * NN + j] = sum;
        }
    }

    __half *dA, *dB;
    float *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(hC)));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    gemmMMASingleTile<<<1, 32>>>(dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost));

    float maxAbs = 0.0f;
    int bad = 0;
    for (int i = 0; i < MM * NN; i++) {
        float diff = fabsf(hC[i] - ref[i]);
        maxAbs = fmaxf(maxAbs, diff);
        if (diff > 1e-2f) {
            bad++;
        }
    }
    printf("Tensor Core tile validation: %s (max abs err %.5f, %d/%d cells out of tol)\n",
           bad == 0 ? "PASSED" : "FAILED", maxAbs, bad, MM * NN);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

int main(int argc, char **argv) {
    int units = argc > 1 ? atoi(argv[1]) : 6;
    int m = argc > 2 ? atoi(argv[2]) : 8;
    int n = argc > 3 ? atoi(argv[3]) : 8;
    int executions = argc > 4 ? atoi(argv[4]) : 20;
    const char *mode = argc > 5 ? argv[5] : "all";

    printf("Integrated showcase: %d chains of (scale -> cuBLAS sgemv -> bias), %dx%d, %d executions, mode=%s\n",
           units, m, n, executions, mode);

    bool all = strcmp(mode, "all") == 0;
    double base = -1, conc = -1, gr = -1, comb = -1;
    if (all || strcmp(mode, "baseline") == 0)   base = runMode("baseline", units, m, n, executions);
    if (all || strcmp(mode, "concurrent") == 0) conc = runMode("concurrent", units, m, n, executions);
    if (all || strcmp(mode, "graph") == 0)      gr   = runMode("graph", units, m, n, executions);
    if (all || strcmp(mode, "combined") == 0)   comb = runMode("combined", units, m, n, executions);

    if (base > 0) {
        printf("\n=== Summary (steady-state median us, this run/this GPU) ===\n");
        printf("baseline   : %.1f\n", base);
        if (conc > 0) printf("concurrent : %.1f (%.2fx vs baseline)\n", conc, base / conc);
        if (gr > 0)   printf("graph      : %.1f (%.2fx vs baseline)\n", gr, base / gr);
        if (comb > 0) printf("combined   : %.1f (%.2fx vs baseline)\n", comb, base / comb);
    }

    if (all) {
        runTensorCoreBonus();
    }
    return 0;
}
