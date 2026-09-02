// CUDA equivalent of VectorAddKernel.java.
//
// The Java version shows the generated CUDA with `tornado --printKernel`;
// this is the same kernel written by hand, so you can put the two side by side.
//
// Build & run:
//   nvcc -arch=sm_89 -o vector_add VectorAddKernel.cu && ./vector_add [size]

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

__global__ void vectorAdd(const float *a, const float *b, float *c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv) {
    const int size = argc > 1 ? atoi(argv[1]) : 1024;
    const size_t bytes = (size_t) size * sizeof(float);

    float *hostA = (float *) malloc(bytes);
    float *hostB = (float *) malloc(bytes);
    float *hostC = (float *) malloc(bytes);
    for (int i = 0; i < size; i++) {
        hostA[i] = (float) i;
        hostB[i] = (float) (2 * i);
    }

    float *devA, *devB, *devC;
    CUDA_CHECK(cudaMalloc(&devA, bytes));
    CUDA_CHECK(cudaMalloc(&devB, bytes));
    CUDA_CHECK(cudaMalloc(&devC, bytes));

    CUDA_CHECK(cudaMemcpy(devA, hostA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devB, hostB, bytes, cudaMemcpyHostToDevice));

    // TornadoVM reports a profiler total time; time the kernel with events here.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    CUDA_CHECK(cudaEventRecord(start));
    vectorAdd<<<blocks, threads>>>(devA, devB, devC, size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(hostC, devC, bytes, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < size; i++) {
        if (hostC[i] != hostA[i] + hostB[i]) {
            correct = false;
            break;
        }
    }
    printf("%s\n", correct ? "Result is correct" : "Result is WRONG");
    printf("Kernel time: %.0f ns\n", ms * 1.0e6f);

    CUDA_CHECK(cudaFree(devA));
    CUDA_CHECK(cudaFree(devB));
    CUDA_CHECK(cudaFree(devC));
    free(hostA);
    free(hostB);
    free(hostC);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
