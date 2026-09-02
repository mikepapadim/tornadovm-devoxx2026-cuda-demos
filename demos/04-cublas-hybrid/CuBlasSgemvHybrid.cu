// CUDA equivalent of CuBlasSgemvHybrid.java.
//
// One TaskGraph with three stages in Java becomes: create a cuBLAS handle,
// allocate and copy every buffer yourself, launch your own kernels around the
// library call, and keep the intermediate results on the device between them.
//
// cuBLAS is column-major; the matrix is built and validated row-major, so the
// call passes CUBLAS_OP_T -- exactly the same argument the Java demo passes.
//
// Build & run:
//   nvcc -arch=sm_89 -lcublas -o cublas_sgemv CuBlasSgemvHybrid.cu
//   ./cublas_sgemv [m] [n] [iterations]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>

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

static void fillMatrix(float *matrix, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        matrix[i] = (float) (i % 7) + 1.0f;
    }
}

static void fillVector(float *vector, int n) {
    for (int j = 0; j < n; j++) {
        vector[j] = (float) (j % 5) + 1.0f;
    }
}

int main(int argc, char **argv) {
    const int m = argc > 1 ? atoi(argv[1]) : 8;
    const int n = argc > 2 ? atoi(argv[2]) : 8;
    const int iterations = argc > 3 ? atoi(argv[3]) : 5;

    const size_t matrixBytes = (size_t) m * n * sizeof(float);
    const size_t vectorBytes = (size_t) n * sizeof(float);
    const size_t outputBytes = (size_t) m * sizeof(float);

    float *hostMatrix = (float *) malloc(matrixBytes);
    float *hostVector = (float *) malloc(vectorBytes);
    float *hostOutput = (float *) malloc(outputBytes);

    float *devMatrix, *devVector, *devOutput;
    CUDA_CHECK(cudaMalloc(&devMatrix, matrixBytes));
    CUDA_CHECK(cudaMalloc(&devVector, vectorBytes));
    CUDA_CHECK(cudaMalloc(&devOutput, outputBytes));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int lda = m;
    const int threads = 256;

    bool allCorrect = true;
    for (int it = 0; it < iterations; it++) {
        fillMatrix(hostMatrix, m, n);
        fillVector(hostVector, n);
        CUDA_CHECK(cudaMemcpy(devMatrix, hostMatrix, matrixBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devVector, hostVector, vectorBytes, cudaMemcpyHostToDevice));

        // Stage 1: your own kernel.
        scaleKernel<<<(m * n + threads - 1) / threads, threads>>>(devMatrix, m * n);
        CUDA_CHECK(cudaGetLastError());

        // Stage 2: the vendor library, on the buffer stage 1 just wrote.
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, n, m, &alpha,
                                 devMatrix, lda, devVector, 1, &beta, devOutput, 1));

        // Stage 3: your own kernel again, on the library's output.
        biasKernel<<<(m + threads - 1) / threads, threads>>>(devOutput, m);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(hostOutput, devOutput, outputBytes, cudaMemcpyDeviceToHost));

        // Sequential CPU reference for the identical three-step pipeline.
        bool correct = true;
        float expected0 = 0.0f;
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += (((float) ((i * n + j) % 7) + 1.0f) * SCALE) * ((float) (j % 5) + 1.0f);
            }
            float expected = sum + BIAS;
            if (i == 0) {
                expected0 = expected;
            }
            if (fabsf(expected - hostOutput[i]) > 0.01f) {
                correct = false;
            }
        }
        allCorrect &= correct;
        printf("iteration %d: %s output[0]=%.1f expected[0]=%.1f\n",
               it, correct ? "correct" : "WRONG", hostOutput[0], expected0);
    }
    printf("%s\n", allCorrect ? "All iterations correct" : "SOME ITERATIONS WRONG");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(devMatrix));
    CUDA_CHECK(cudaFree(devVector));
    CUDA_CHECK(cudaFree(devOutput));
    free(hostMatrix);
    free(hostVector);
    free(hostOutput);
    return 0;
}
