// Probe for demo 15's first finding.
//
// TornadoVM's FloatArray places its payload 16 bytes (4 floats) after the
// allocation base -- visible in the generated CUDA as `l_5 = l_4 + 4L`. A warp
// reads 32 x 4 = 128 bytes, so that offset makes every warp-wide access straddle
// a 128-byte boundary and cost two memory transactions instead of one.
//
// This probe runs the identical kernel at offset 0 and at offset 4 floats, so
// the effect can be attributed rather than guessed at.
//
// Build & run:
//   nvcc -arch=sm_89 -o probe_alignment ProbeHeaderAlignment.cu && ./probe_alignment
#include <cstdio>
#include <vector>
#include <algorithm>
#include <chrono>
#define CK(c) do{cudaError_t e=(c); if(e){printf("err %s\n",cudaGetErrorString(e));exit(1);} }while(0)

__global__ void elementwise(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 0.25f + 0.1f;
}
__global__ void stencil(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float l = i > 0 ? in[i-1] : in[0];
        float r = i < n-1 ? in[i+1] : in[n-1];
        out[i] = (l + in[i] + r) * (1.0f/3.0f);
    }
}

int main() {
    const int n = 1 << 22, BLOCK = 256, reps = 30;
    float *a, *b;
    CK(cudaMalloc(&a, (size_t)(n + 8) * sizeof(float)));
    CK(cudaMalloc(&b, (size_t)(n + 8) * sizeof(float)));
    int blocks = (n + BLOCK - 1) / BLOCK;

    for (int off = 0; off <= 4; off += 4) {
        for (int k = 0; k < 2; k++) {
            const char *name = k == 0 ? "elementwise" : "stencil";
            // warm-up
            for (int r = 0; r < 3; r++)
                k == 0 ? elementwise<<<blocks,BLOCK>>>(a+off, b+off, n)
                       : stencil<<<blocks,BLOCK>>>(a+off, b+off, n);
            CK(cudaDeviceSynchronize());
            std::vector<double> t(reps);
            for (int r = 0; r < reps; r++) {
                auto s = std::chrono::steady_clock::now();
                k == 0 ? elementwise<<<blocks,BLOCK>>>(a+off, b+off, n)
                       : stencil<<<blocks,BLOCK>>>(a+off, b+off, n);
                CK(cudaDeviceSynchronize());
                t[r] = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-s).count();
            }
            std::sort(t.begin(), t.end());
            printf("%-12s offset=%d floats : median %.1f us\n", name, off, t[t.size()/2]);
        }
    }
    return 0;
}
