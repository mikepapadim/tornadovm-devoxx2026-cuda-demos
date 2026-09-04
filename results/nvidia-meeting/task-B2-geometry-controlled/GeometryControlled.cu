// Hand-written CUDA side of task B2. Same kernel, same problem size; block size
// is a command-line argument so the geometry can be matched to the TornadoVM run.
//   nvcc -arch=sm_89 -O3 -o geom GeometryControlled.cu && ./geom <n> <block> <exec>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define CK(c) do{cudaError_t e=(c); if(e){printf("err %s\n",cudaGetErrorString(e));exit(1);} }while(0)

__global__ void elementwise(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 0.25f + 0.1f;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : (1 << 24);
    int block = argc > 2 ? atoi(argv[2]) : 256;
    int executions = argc > 3 ? atoi(argv[3]) : 20;

    float *hIn = (float *) malloc((size_t) n * sizeof(float));
    float *hOut = (float *) malloc((size_t) n * sizeof(float));
    for (int i = 0; i < n; i++) hIn[i] = i * 0.001f;

    float *a, *b;
    CK(cudaMalloc(&a, (size_t) n * sizeof(float)));
    CK(cudaMalloc(&b, (size_t) n * sizeof(float)));
    CK(cudaMemcpy(a, hIn, (size_t) n * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (n + block - 1) / block;
    for (int e = 0; e < executions; e++) {
        elementwise<<<blocks, block>>>(a, b, n);
    }
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hOut, b, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));

    int wrong = 0;
    for (int i = 0; i < n; i += 4096) {
        float expected = hIn[i] * 0.25f + 0.1f;
        if (fabsf(hOut[i] - expected) > 1e-5f) wrong++;
    }
    printf("n=%d block=%d executions=%d sampled-mismatches=%d %s\n",
           n, block, executions, wrong, wrong == 0 ? "PASSED" : "FAILED");
    return 0;
}
