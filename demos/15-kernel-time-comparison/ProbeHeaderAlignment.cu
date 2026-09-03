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
//
// `./probe_alignment --counters` launches each kernel exactly once per offset
// and skips the timing loop, so Nsight Compute can attribute the effect to the
// memory-transaction counters directly instead of inferring it from wall time:
//   /opt/nvidia/nsight-compute/2024.3.2/ncu \
//     --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
//   l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
//   l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
//   l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
//   dram__bytes_read.sum,dram__bytes_write.sum \
//     ./probe_alignment --counters
#include <cstdio>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
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

int main(int argc, char **argv) {
    const bool counters = argc > 1 && std::string(argv[1]) == "--counters";
    const int n = 1 << 22, BLOCK = 256, reps = 30;
    float *a, *b;
    CK(cudaMalloc(&a, (size_t)(n + 8) * sizeof(float)));
    CK(cudaMalloc(&b, (size_t)(n + 8) * sizeof(float)));
    int blocks = (n + BLOCK - 1) / BLOCK;

    for (int off = 0; off <= 4; off += 4) {
        for (int k = 0; k < 2; k++) {
            const char *name = k == 0 ? "elementwise" : "stencil";
            if (counters) {
                // One launch per configuration: ncu profiles it, nothing else.
                k == 0 ? elementwise<<<blocks,BLOCK>>>(a+off, b+off, n)
                       : stencil<<<blocks,BLOCK>>>(a+off, b+off, n);
                CK(cudaDeviceSynchronize());
                printf("%-12s offset=%d floats : launched\n", name, off);
                continue;
            }
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
