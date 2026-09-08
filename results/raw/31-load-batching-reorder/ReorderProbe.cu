// ReorderProbe.cu -- can the interleaved staging schedule be reproduced in hand-written CUDA?
//
// TornadoVM's register-tiled kernel emitted its tile staging as an alternating chain --
// global load, shared store, global load, shared store -- so one global load was in flight
// at a time. This probe asks whether that schedule is reachable from CUDA C at all.
//
// It compiles ONE kernel (the register-tiled sgemm from demo 17's rung 3, same geometry,
// same arithmetic, same shared layout) three ways, differing only in staging order:
//
//   A  interleaved + compiler barrier   load, store, load, store, with an empty asm memory
//                                       clobber between pairs, to try to pin the order
//   B  interleaved, no barrier          the same source order, left to nvcc
//   C  batched                          all loads into registers, then all shared stores
//
// RESULT (RTX 4090, CUDA 12.6, sm_89): all three produce BYTE-IDENTICAL SASS -- 448
// instructions, and in every case the schedule is
//
//     LDG x8  then  STS x8
//
// so the timings are identical too and the answer is no: the interleaved schedule cannot be
// written in CUDA C. ptxas chooses the batched one regardless of source order, and the empty
// asm barrier constrains only the NVVM frontend, not ptxas.
//
// That negative result is the useful one, because of WHY ptxas is free to reorder here.
// Hand-written CUDA loads through `const float *` parameters, which ptxas lowers to LDG --
// a load proven to target global memory, and therefore proven not to alias the STS beside
// it. TornadoVM's generated kernel casts an integer address to a plain pointer, which
// lowers to a GENERIC `LD`. A generic load may target shared memory, so ptxas must keep it
// ordered against every shared store. See sass-schedules.txt:
//
//     tornadovm, batching off    L S L S L S L S L S L S L S L S     (generic LD)
//     tornadovm, batching on     L L L L L L L L S S S S S S S S     (generic LD)
//     hand-written CUDA          L L L L L L L L S S S S S S S S     (LDG)
//
// TornadoVM knows the address space at LIR level even though its emitted C does not encode
// it, so the batching has to happen in the code generator. That is what
// CUDAGlobalLoadBatching does, and it recovers the hand-written schedule exactly.
//
// Build & run:
//   nvcc -arch=sm_89 -O3 -std=c++17 -o reorder_probe ReorderProbe.cu && ./reorder_probe 1024 100
//   cuobjdump -sass <cubin>          to check the three variants really are identical
//
// Prints one CSV row per variant to stdout.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CK(c) do{cudaError_t e=(c); if(e){printf("cuda err %s\n",cudaGetErrorString(e));exit(1);} }while(0)

static const int BK = 16, TM = 4, TN = 4, THREADS = 16;
static const int BM = THREADS * TM;   // 64
static const int BN = THREADS * TN;   // 64
static const int A_LOADS = (BM * BK) / (THREADS * THREADS);   // 4
static const int B_LOADS = (BK * BN) / (THREADS * THREADS);   // 4

// An empty asm with a memory clobber: no instruction is emitted, but ptxas may not move a
// load across it. It is the standard way to pin a schedule without changing the arithmetic.
#define SCHED_BARRIER() asm volatile("" ::: "memory")

enum Variant { INTERLEAVED_PINNED, INTERLEAVED, BATCHED };

template <Variant V>
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
        if constexpr (V == BATCHED) {
            // All global loads first, then all shared stores.
            float sa[A_LOADS], sb[B_LOADS];
#pragma unroll
            for (int load = 0; load < A_LOADS; load++) {
                int idx = tid + load * THREADS * THREADS;
                sa[load] = a[(blockRow + idx / BK) * n + kTile + (idx % BK)];
            }
#pragma unroll
            for (int load = 0; load < B_LOADS; load++) {
                int idx = tid + load * THREADS * THREADS;
                sb[load] = b[(kTile + idx / BN) * n + blockCol + (idx % BN)];
            }
#pragma unroll
            for (int load = 0; load < A_LOADS; load++) {
                int idx = tid + load * THREADS * THREADS;
                tileA[(idx / BK) * BK + (idx % BK)] = sa[load];
            }
#pragma unroll
            for (int load = 0; load < B_LOADS; load++) {
                int idx = tid + load * THREADS * THREADS;
                tileB[(idx / BN) * BN + (idx % BN)] = sb[load];
            }
        } else {
            // Load, store, load, store ... exactly as TornadoVM emitted it.
#pragma unroll
            for (int load = 0; load < A_LOADS; load++) {
                int idx = tid + load * THREADS * THREADS;
                tileA[(idx / BK) * BK + (idx % BK)] = a[(blockRow + idx / BK) * n + kTile + (idx % BK)];
                if constexpr (V == INTERLEAVED_PINNED) SCHED_BARRIER();
            }
#pragma unroll
            for (int load = 0; load < B_LOADS; load++) {
                int idx = tid + load * THREADS * THREADS;
                tileB[(idx / BN) * BN + (idx % BN)] = b[(kTile + idx / BN) * n + blockCol + (idx % BN)];
                if constexpr (V == INTERLEAVED_PINNED) SCHED_BARRIER();
            }
        }
        __syncthreads();
#pragma unroll
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

template <Variant V>
static double timeVariant(const float *dA, const float *dB, float *dC, int n, int reps) {
    dim3 block(THREADS, THREADS), grid(n / BN, n / BM);
    cudaEvent_t evStart, evStop;
    CK(cudaEventCreate(&evStart));
    CK(cudaEventCreate(&evStop));
    registerTiled<V><<<grid, block>>>(dA, dB, dC, n);   // warm up / JIT the cubin
    CK(cudaDeviceSynchronize());

    std::vector<double> ms;
    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(evStart));
        registerTiled<V><<<grid, block>>>(dA, dB, dC, n);
        CK(cudaEventRecord(evStop));
        CK(cudaEventSynchronize(evStop));
        float t;
        CK(cudaEventElapsedTime(&t, evStart, evStop));
        ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    CK(cudaEventDestroy(evStart));
    CK(cudaEventDestroy(evStop));
    return ms[ms.size() / 2] * 1000.0;   // median, microseconds
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 2048;
    int reps = argc > 2 ? atoi(argv[2]) : 20;
    if (n % BN || n % BM) { printf("n must be a multiple of %d\n", BN); return 1; }

    size_t bytes = (size_t)n * n * sizeof(float);
    std::vector<float> hA((size_t)n * n), hB((size_t)n * n), hC((size_t)n * n), ref((size_t)n * n);
    for (size_t i = 0; i < hA.size(); i++) { hA[i] = (float)((i * 7) % 13) * 0.1f; hB[i] = (float)((i * 11) % 17) * 0.1f; }

    float *dA, *dB, *dC;
    CK(cudaMalloc(&dA, bytes)); CK(cudaMalloc(&dB, bytes)); CK(cudaMalloc(&dC, bytes));
    CK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    const char *names[3] = { "interleaved_pinned", "interleaved_nvcc", "batched" };
    double us[3];
    us[0] = timeVariant<INTERLEAVED_PINNED>(dA, dB, dC, n, reps);
    CK(cudaMemcpy(ref.data(), dC, bytes, cudaMemcpyDeviceToHost));
    us[1] = timeVariant<INTERLEAVED>(dA, dB, dC, n, reps);
    CK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < ref.size(); i++) if (fabsf(ref[i] - hC[i]) > 1e-3f) { printf("MISMATCH interleaved_nvcc at %zu\n", i); return 1; }
    us[2] = timeVariant<BATCHED>(dA, dB, dC, n, reps);
    CK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < ref.size(); i++) if (fabsf(ref[i] - hC[i]) > 1e-3f) { printf("MISMATCH batched at %zu\n", i); return 1; }

    printf("variant,n,reps,median_us,gflops,vs_interleaved_pinned\n");
    for (int i = 0; i < 3; i++) {
        double gf = 2.0 * n * n * n / (us[i] * 1e3);
        printf("%s,%d,%d,%.2f,%.1f,%.3f\n", names[i], n, reps, us[i], gf, us[0] / us[i]);
    }
    CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC));
    return 0;
}
