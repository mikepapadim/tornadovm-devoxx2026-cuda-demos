// Hand-written CUDA counterpart to demos/17-matmul-ladder.
//
// The same rungs as the Java version, written directly in CUDA C++, so the
// generated code can be compared against the handwritten equivalent at each
// step of the ladder rather than only at the top:
//
//   1. naive          one thread per output, no reuse
//   2. tiled          shared-memory tile, TILE x TILE
//   3. registerTiled  shared memory + a TM x TN register micro-tile per thread
//   4. cuBLAS sgemm   the vendor kernel
//   5. cuBLAS TF32    same call, tensor cores
//
// CUTLASS is deliberately absent here: it needs a CUTLASS checkout, and
// scripts/run-all-cuda.sh builds this file with the plain CUDA toolkit. The
// Java side covers that rung, and demo 12 covers CUTLASS on the CUDA side.
//
// Build & run:
//   nvcc -arch=sm_89 -O3 -lcublas -o matmul_ladder MatMulLadder.cu && ./matmul_ladder 2048 10
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cublas_v2.h>
#define CK(c) do{cudaError_t e=(c); if(e){printf("cuda err %s\n",cudaGetErrorString(e));exit(1);} }while(0)
#define CB(c) do{cublasStatus_t s=(c); if(s){printf("cublas err %d\n",(int)s);exit(1);} }while(0)

static const int TILE = 16;
static const int BK = 16, TM = 4, TN = 4, THREADS = 16;
static const int BM = THREADS * TM;   // 64
static const int BN = THREADS * TN;   // 64

// ---- rung 1
__global__ void naive(const float *a, const float *b, float *c, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) sum += a[i * n + k] * b[k * n + j];
        c[i * n + j] = sum;
    }
}

// ---- rung 2
__global__ void tiled(const float *a, const float *b, float *c, int n) {
    __shared__ float tileA[TILE * TILE];
    __shared__ float tileB[TILE * TILE];
    int lx = threadIdx.x, ly = threadIdx.y;
    int row = blockIdx.y * TILE + ly;
    int col = blockIdx.x * TILE + lx;
    float sum = 0.0f;
    for (int kTile = 0; kTile < n; kTile += TILE) {
        tileA[ly * TILE + lx] = a[row * n + kTile + lx];
        tileB[ly * TILE + lx] = b[(kTile + ly) * n + col];
        __syncthreads();
        for (int k = 0; k < TILE; k++) sum += tileA[ly * TILE + k] * tileB[k * TILE + lx];
        __syncthreads();
    }
    c[row * n + col] = sum;
}

// ---- rung 3
__global__ void registerTiled(const float *a, const float *b, float *c, int n) {
    __shared__ float tileA[BM * BK];
    __shared__ float tileB[BK * BN];
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * THREADS + tx;
    int blockRow = blockIdx.y * BM, blockCol = blockIdx.x * BN;

    float acc[TM * TN];
    for (int i = 0; i < TM * TN; i++) acc[i] = 0.0f;
    float regA[TM], regB[TN];

    for (int kTile = 0; kTile < n; kTile += BK) {
        for (int load = 0; load < (BM * BK) / (THREADS * THREADS); load++) {
            int idx = tid + load * THREADS * THREADS;
            tileA[(idx / BK) * BK + (idx % BK)] = a[(blockRow + idx / BK) * n + kTile + (idx % BK)];
        }
        for (int load = 0; load < (BK * BN) / (THREADS * THREADS); load++) {
            int idx = tid + load * THREADS * THREADS;
            tileB[(idx / BN) * BN + (idx % BN)] = b[(kTile + idx / BN) * n + blockCol + (idx % BN)];
        }
        __syncthreads();
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++) regA[i] = tileA[(ty * TM + i) * BK + k];
            for (int j = 0; j < TN; j++) regB[j] = tileB[k * BN + tx * TN + j];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++) acc[i * TN + j] += regA[i] * regB[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            c[(blockRow + ty * TM + i) * n + blockCol + tx * TN + j] = acc[i * TN + j];
}

static double medianOf(std::vector<double> &v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

template <typename F>
static double timeRung(F launch, int reps) {
    for (int i = 0; i < 3; i++) launch();          // warm-up
    CK(cudaDeviceSynchronize());
    std::vector<double> t(reps);
    for (int r = 0; r < reps; r++) {
        auto s = std::chrono::steady_clock::now();
        launch();
        CK(cudaDeviceSynchronize());
        t[r] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - s).count();
    }
    return medianOf(t);
}

static bool validate(const char *label, const std::vector<float> &got, const std::vector<float> &ref, int n, float tol) {
    float maxErr = 0.0f; int bad = 0;
    for (int i = 0; i < n * n; i++) {
        float e = fabsf(got[i] - ref[i]);
        if (e > maxErr) maxErr = e;
        if (e > tol * fmaxf(1.0f, fabsf(ref[i]))) bad++;
    }
    printf("    %-32s validation %s (max abs err %.5f, %d/%d out of tol)\n",
           label, bad == 0 ? "PASSED" : "FAILED", maxErr, bad, n * n);
    return bad == 0;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 2048;
    int reps = argc > 2 ? atoi(argv[2]) : 10;
    if (n % BM != 0) { printf("size must be a multiple of %d; got %d\n", BM, n); return 1; }

    std::vector<float> hA(n * n), hB(n * n), hC(n * n), ref(n * n);
    for (int i = 0; i < n * n; i++) {
        hA[i] = ((i * 7 + 3) % 17) * 0.0625f - 0.5f;
        hB[i] = ((i * 5 + 11) % 13) * 0.0769f - 0.5f;
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++) s += hA[i * n + k] * hB[k * n + j];
            ref[i * n + j] = s;
        }

    float *dA, *dB, *dC;
    CK(cudaMalloc(&dA, (size_t)n * n * sizeof(float)));
    CK(cudaMalloc(&dB, (size_t)n * n * sizeof(float)));
    CK(cudaMalloc(&dC, (size_t)n * n * sizeof(float)));
    CK(cudaMemcpy(dA, hA.data(), (size_t)n * n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), (size_t)n * n * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle; CB(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    double gflop = 2.0 * n * n * n / 1e9;
    bool ok = true;
    const char *labels[5] = { "1. naive", "2. tiled", "3. register-tiled", "4. cuBLAS sgemm", "5. cuBLAS sgemm TF32" };
    double times[5];

    printf("Matrix-multiply ladder (hand-written CUDA): C = A * B, FP32, %dx%d\n", n, n);
    printf("  %d reps per rung, median, inputs uploaded once\n\n", reps);

    dim3 blockN(16, 16), gridN((n + 15) / 16, (n + 15) / 16);
    times[0] = timeRung([&]{ naive<<<gridN, blockN>>>(dA, dB, dC, n); }, reps);
    CK(cudaMemcpy(hC.data(), dC, (size_t)n * n * sizeof(float), cudaMemcpyDeviceToHost));
    ok &= validate(labels[0], hC, ref, n, 1e-3f);

    times[1] = timeRung([&]{ tiled<<<gridN, blockN>>>(dA, dB, dC, n); }, reps);
    CK(cudaMemcpy(hC.data(), dC, (size_t)n * n * sizeof(float), cudaMemcpyDeviceToHost));
    ok &= validate(labels[1], hC, ref, n, 1e-3f);

    dim3 blockR(THREADS, THREADS), gridR(n / BN, n / BM);
    times[2] = timeRung([&]{ registerTiled<<<gridR, blockR>>>(dA, dB, dC, n); }, reps);
    CK(cudaMemcpy(hC.data(), dC, (size_t)n * n * sizeof(float), cudaMemcpyDeviceToHost));
    ok &= validate(labels[2], hC, ref, n, 1e-3f);

    // Column-major cuBLAS: row-major C = A*B is C_cm = B_cm * A_cm, so the operands swap.
    CB(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    times[3] = timeRung([&]{ cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                         &alpha, dB, n, dA, n, &beta, dC, n); }, reps);
    CK(cudaMemcpy(hC.data(), dC, (size_t)n * n * sizeof(float), cudaMemcpyDeviceToHost));
    ok &= validate(labels[3], hC, ref, n, 1e-3f);

    CB(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    times[4] = timeRung([&]{ cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                         &alpha, dB, n, dA, n, &beta, dC, n); }, reps);
    CK(cudaMemcpy(hC.data(), dC, (size_t)n * n * sizeof(float), cudaMemcpyDeviceToHost));
    ok &= validate(labels[4], hC, ref, n, 2e-2f);
    CB(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    printf("\n=== Summary (median us, this run / this GPU) ===\n");
    printf("%-24s %10s %12s %10s\n", "rung", "median us", "GFLOP/s", "vs naive");
    for (int i = 0; i < 5; i++)
        printf("%-24s %10.0f %12.1f %9.1fx\n", labels[i], times[i], gflop / (times[i] / 1e6), times[0] / times[i]);

    printf("\n%s\n", ok ? "All five rungs produced the same, correct result" : "At least one rung FAILED");
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ok ? 0 : 1;
}
