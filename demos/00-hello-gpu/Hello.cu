// CUDA equivalent of Hello.java.
//
// Same program, written the way you would without TornadoVM: an explicit
// __global__ kernel, manual device allocation, manual copies in both
// directions, an explicit launch configuration, and manual frees.
//
// Build & run:
//   nvcc -arch=sm_89 -o hello Hello.cu && ./hello

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

// The @Parallel loop in Hello.java becomes this kernel: one thread per element,
// with the bounds check TornadoVM inserts for you.
__global__ void addOne(const int *in, int *out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = in[i] + 1;
    }
}

static void printArray(const char *label, const int *values, int size) {
    printf("%s[", label);
    for (int i = 0; i < size; i++) {
        printf("%d%s", values[i], i + 1 < size ? ", " : "");
    }
    printf("]\n");
}

int main() {
    const int size = 8;
    const size_t bytes = size * sizeof(int);

    int hostIn[size];
    int hostOut[size];
    for (int i = 0; i < size; i++) {
        hostIn[i] = i;
    }

    int *devIn = nullptr;
    int *devOut = nullptr;
    CUDA_CHECK(cudaMalloc(&devIn, bytes));
    CUDA_CHECK(cudaMalloc(&devOut, bytes));

    CUDA_CHECK(cudaMemcpy(devIn, hostIn, bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    addOne<<<blocks, threads>>>(devIn, devOut, size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(hostOut, devOut, bytes, cudaMemcpyDeviceToHost));

    printArray("in:  ", hostIn, size);
    printArray("out: ", hostOut, size);

    CUDA_CHECK(cudaFree(devIn));
    CUDA_CHECK(cudaFree(devOut));
    return 0;
}
