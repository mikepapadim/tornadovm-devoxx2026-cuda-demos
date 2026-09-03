// Alignment isolation for TornadoVM#1065 (task B of the NVIDIA evidence pack).
//
// One kernel, one problem size, one launch geometry. The ONLY variable is the
// byte offset of the base pointer. This separates a buffer-layout effect from a
// code-generation effect: if generated arithmetic mattered, changing an offset
// on the *same* compiled kernel could not reproduce the difference.
//
// A warp issues 32 x 4 = 128 contiguous bytes per request. The L1<->L2 path is
// addressed in 32-byte sectors, so the request costs
//   4 sectors  when the base is 32-byte aligned,
//   5 sectors  when it is not (the access straddles a fifth sector).
// TornadoVM's FloatArray places its payload 16 bytes after the allocation base
// (offset 4 floats), which is the misaligned case.
//
// Offsets swept, in floats: 0, 4, 8, 16, 32.
//   0  ->   0 B   aligned (reference)
//   4  ->  16 B   TornadoVM's current layout
//   8  ->  32 B   one sector
//   16 ->  64 B   two sectors
//   32 -> 128 B   full 128-byte alignment
//
// Build:
//   nvcc -arch=sm_89 -O3 -o alignment_sweep AlignmentSweep.cu
// Counters (one launch per offset, no timing loop):
//   ncu --csv --metrics <...> ./alignment_sweep --counters
// Timing (30 reps, median):
//   ./alignment_sweep
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#define CK(c) do{cudaError_t e=(c); if(e){printf("err %s\n",cudaGetErrorString(e));exit(1);} }while(0)

__global__ void elementwise(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 0.25f + 0.1f;
}

int main(int argc, char **argv) {
    const bool counters = argc > 1 && strcmp(argv[1], "--counters") == 0;
    const int n = 1 << 22, BLOCK = 256, reps = 30;
    const int offsets[] = { 0, 4, 8, 16, 32 };

    float *a, *b;
    CK(cudaMalloc(&a, (size_t)(n + 64) * sizeof(float)));
    CK(cudaMalloc(&b, (size_t)(n + 64) * sizeof(float)));
    int blocks = (n + BLOCK - 1) / BLOCK;

    for (int oi = 0; oi < 5; oi++) {
        int off = offsets[oi];
        if (counters) {
            elementwise<<<blocks, BLOCK>>>(a + off, b + off, n);
            CK(cudaDeviceSynchronize());
            printf("offset=%2d floats (%3d B) : launched\n", off, off * 4);
            continue;
        }
        for (int r = 0; r < 3; r++) elementwise<<<blocks, BLOCK>>>(a + off, b + off, n);
        CK(cudaDeviceSynchronize());
        std::vector<double> t(reps);
        for (int r = 0; r < reps; r++) {
            auto s = std::chrono::steady_clock::now();
            elementwise<<<blocks, BLOCK>>>(a + off, b + off, n);
            CK(cudaDeviceSynchronize());
            t[r] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - s).count();
        }
        std::sort(t.begin(), t.end());
        printf("offset=%2d floats (%3d B) : median %.1f us\n", off, off * 4, t[t.size() / 2]);
    }
    return 0;
}
