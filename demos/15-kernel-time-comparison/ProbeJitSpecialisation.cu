// Probe for demo 15's second finding.
//
// TornadoVM JIT-compiles the kernel with the actual `degree` value already
// known, and Graal fully unrolls the FMA chain -- visible in `--printKernel` as
// a straight-line sequence of fma() calls with no loop. nvcc compiles ahead of
// time and must emit a real loop with a counter and a branch.
//
// This probe gives nvcc the same information (degree as a compile-time
// constant) to check whether that alone accounts for the difference.
//
// Build & run:
//   nvcc -arch=sm_89 -o probe_specialisation ProbeJitSpecialisation.cu && ./probe_specialisation
#include <cstdio>
#include <vector>
#include <algorithm>
#include <chrono>
#define CK(c) do{cudaError_t e=(c); if(e){printf("err %s\n",cudaGetErrorString(e));exit(1);} }while(0)

__global__ void polyRuntime(const float *in, float *out, int degree, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i], acc = 1.0f;
        for (int d = 0; d < degree; d++) acc = acc * v + 0.5f;
        out[i] = acc;
    }
}
template <int DEGREE>
__global__ void polyConst(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i], acc = 1.0f;
        #pragma unroll
        for (int d = 0; d < DEGREE; d++) acc = acc * v + 0.5f;
        out[i] = acc;
    }
}

int main() {
    const int n = 1 << 22, BLOCK = 256, reps = 20, degree = 256;
    float *a, *b;
    CK(cudaMalloc(&a, (size_t)n * sizeof(float)));
    CK(cudaMalloc(&b, (size_t)n * sizeof(float)));
    int blocks = (n + BLOCK - 1) / BLOCK;

    auto bench = [&](const char *name, bool constant) {
        for (int r = 0; r < 3; r++)
            constant ? polyConst<256><<<blocks,BLOCK>>>(a,b,n) : polyRuntime<<<blocks,BLOCK>>>(a,b,degree,n);
        CK(cudaDeviceSynchronize());
        std::vector<double> t(reps);
        for (int r = 0; r < reps; r++) {
            auto s = std::chrono::steady_clock::now();
            constant ? polyConst<256><<<blocks,BLOCK>>>(a,b,n) : polyRuntime<<<blocks,BLOCK>>>(a,b,degree,n);
            CK(cudaDeviceSynchronize());
            t[r] = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-s).count();
        }
        std::sort(t.begin(), t.end());
        printf("%-34s median %.1f us\n", name, t[t.size()/2]);
    };
    bench("degree as runtime argument", false);
    bench("degree as compile-time constant", true);
    return 0;
}
