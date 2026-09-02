// CUDA equivalent of TensorCoreMMA.java.
//
// One warp computes one M16N8K16 fp16 GEMM tile with exactly one
// mma.sync.aligned instruction, next to a scalar reference kernel that uses
// none. The Java version reaches this through KernelContext.mmaLoadA/mmaLoadB/
// mma/mmaStore; here you write the inline PTX and, more importantly, the
// register-to-matrix-element mapping that those helpers encapsulate.
//
// That mapping is the whole reason the Java API exists: each lane holds 8
// halves of A, 4 of B and 4 floats of C, at positions fixed by the PTX ISA.
// Get it wrong and the result is silently incorrect.
//
// Build & run:
//   nvcc -arch=sm_89 -o tensor_core_mma TensorCoreMMA.cu && ./tensor_core_mma
//   cuobjdump -sass tensor_core_mma | grep -c HMMA      # count the tensor-core instructions

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static const int M = 16, N = 8, K = 16;
static const int WARP_SIZE = 32;

// Scalar (no Tensor Core) reference kernel: one thread per output element.
__global__ void gemmScalarFp16(const __half *A, const __half *B, float *C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = sum;
    }
}

// Single-warp, single-tile Tensor Core kernel: exactly one mma.sync.aligned.
__global__ void gemmMMASingleTile(const __half *A, const __half *B, float *C) {
    int lane = threadIdx.x;                 // 0..31, one warp
    int groupID = lane >> 2;                // 0..7
    int threadInGroup = lane & 3;           // 0..3

    // --- A fragment: 8 halves per lane, packed into 4 x b32 ---------------
    // a[0..1] -> row groupID,     cols 2*tig + {0,1}
    // a[2..3] -> row groupID + 8, cols 2*tig + {0,1}
    // a[4..5] -> row groupID,     cols 2*tig + 8 + {0,1}
    // a[6..7] -> row groupID + 8, cols 2*tig + 8 + {0,1}
    __half a[8];
    a[0] = A[(groupID) * K + (2 * threadInGroup)];
    a[1] = A[(groupID) * K + (2 * threadInGroup + 1)];
    a[2] = A[(groupID + 8) * K + (2 * threadInGroup)];
    a[3] = A[(groupID + 8) * K + (2 * threadInGroup + 1)];
    a[4] = A[(groupID) * K + (2 * threadInGroup + 8)];
    a[5] = A[(groupID) * K + (2 * threadInGroup + 9)];
    a[6] = A[(groupID + 8) * K + (2 * threadInGroup + 8)];
    a[7] = A[(groupID + 8) * K + (2 * threadInGroup + 9)];

    // --- B fragment: 4 halves per lane, packed into 2 x b32 ---------------
    // b[0..1] -> rows 2*tig + {0,1},     col groupID
    // b[2..3] -> rows 2*tig + 8 + {0,1}, col groupID
    __half b[4];
    b[0] = B[(2 * threadInGroup) * N + groupID];
    b[1] = B[(2 * threadInGroup + 1) * N + groupID];
    b[2] = B[(2 * threadInGroup + 8) * N + groupID];
    b[3] = B[(2 * threadInGroup + 9) * N + groupID];

    const uint32_t *A32 = reinterpret_cast<const uint32_t *>(a);
    const uint32_t *B32 = reinterpret_cast<const uint32_t *>(b);
    float d[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    // The one Tensor Core instruction.
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(A32[0]), "r"(A32[1]), "r"(A32[2]), "r"(A32[3]),
          "r"(B32[0]), "r"(B32[1]));

    // --- C fragment: 4 floats per lane ------------------------------------
    // d[0..1] -> row groupID,     cols 2*tig + {0,1}
    // d[2..3] -> row groupID + 8, cols 2*tig + {0,1}
    C[(groupID) * N + (2 * threadInGroup)] = d[0];
    C[(groupID) * N + (2 * threadInGroup + 1)] = d[1];
    C[(groupID + 8) * N + (2 * threadInGroup)] = d[2];
    C[(groupID + 8) * N + (2 * threadInGroup + 1)] = d[3];
}

static void deterministicFp16(__half *arr, int size, int seed) {
    for (int i = 0; i < size; i++) {
        float v = (float) ((i * 31 + seed) % 17 - 8) / 8.0f;
        arr[i] = __float2half(v);
    }
}

static void gemmReference(const __half *A, const __half *B, float *ref) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            ref[i * N + j] = sum;
        }
    }
}

static bool validate(const char *name, const float *got, const float *ref) {
    float maxAbs = 0.0f;
    int bad = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(got[i] - ref[i]);
        maxAbs = fmaxf(maxAbs, diff);
        if (diff > 1e-2f) {
            bad++;
        }
    }
    bool ok = bad == 0;
    printf("  [%s] validation %s (max abs err %.5f, %d/%d cells out of tol)\n",
           name, ok ? "PASSED" : "FAILED", maxAbs, bad, M * N);
    return ok;
}

int main() {
    printf("Single-tile Tensor Core (MMA) demo: C[16x8] = A[16x16] * B[16x8], fp16 -> f32\n");

    __half hostA[M * K], hostB[K * N];
    float hostScalar[M * N], hostMMA[M * N], ref[M * N];
    deterministicFp16(hostA, M * K, 1);
    deterministicFp16(hostB, K * N, 7);
    gemmReference(hostA, hostB, ref);

    __half *devA, *devB;
    float *devScalar, *devMMA;
    CUDA_CHECK(cudaMalloc(&devA, sizeof(hostA)));
    CUDA_CHECK(cudaMalloc(&devB, sizeof(hostB)));
    CUDA_CHECK(cudaMalloc(&devScalar, sizeof(hostScalar)));
    CUDA_CHECK(cudaMalloc(&devMMA, sizeof(hostMMA)));
    CUDA_CHECK(cudaMemcpy(devA, hostA, sizeof(hostA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devB, hostB, sizeof(hostB), cudaMemcpyHostToDevice));

    gemmScalarFp16<<<dim3(1, 1), dim3(M, N)>>>(devA, devB, devScalar);
    CUDA_CHECK(cudaGetLastError());
    gemmMMASingleTile<<<1, WARP_SIZE>>>(devA, devB, devMMA);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(hostScalar, devScalar, sizeof(hostScalar), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostMMA, devMMA, sizeof(hostMMA), cudaMemcpyDeviceToHost));

    bool okScalar = validate("scalar (no MMA)", hostScalar, ref);
    bool okMMA = validate("mma.sync (Tensor Core)", hostMMA, ref);
    printf("%s\n", (okScalar && okMMA) ? "Result is correct" : "Result is INCORRECT");

    CUDA_CHECK(cudaFree(devA));
    CUDA_CHECK(cudaFree(devB));
    CUDA_CHECK(cudaFree(devScalar));
    CUDA_CHECK(cudaFree(devMMA));
    return 0;
}
