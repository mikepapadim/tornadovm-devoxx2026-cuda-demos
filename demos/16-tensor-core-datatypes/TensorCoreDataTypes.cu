// Hand-written CUDA equivalent of demos/16-tensor-core-datatypes.
//
// The same four tensor-core operand types the Java version drives through
// KernelContext, written directly as inline PTX. Same problem shape
// (C[16x16] = A[16x32] * B[32x16], one warp), same test values, same CPU
// reference, so the two can be compared line for line.
//
// What the CUDA programmer has to get right here, and TornadoVM does for you:
// the per-lane fragment register mapping. Each `mma.sync` shape distributes A,
// B and C across the warp's 32 lanes in a specific pattern, and getting it
// wrong produces plausible-looking wrong numbers rather than an error. The
// m16n8k32 8-bit layout below is the m16n8k16 16-bit layout scaled by two.
//
// Build & run:
//   nvcc -arch=sm_89 -o tc_datatypes TensorCoreDataTypes.cu && ./tc_datatypes
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CUDA_CHECK(c) do { cudaError_t e = (c); if (e) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

static const int M = 16;
static const int N = 16;
static const int K = 32;
static const int WARP_SIZE = 32;

// --------------------------------------------------------------------- BF16

__global__ void gemmBF16(const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C) {
    int lane = threadIdx.x;
    int groupID = lane >> 2;        // 0..7
    int tig = lane & 3;             // 0..3

    // Each mma produces only 8 columns (the "n8" in m16n8k16), so N = 16 needs
    // two panels, each with its own accumulator.
    for (int panel = 0; panel < 2; panel++) {
    float d[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (int kBase = 0; kBase < K; kBase += 16) {
        __nv_bfloat16 a[8];
        a[0] = A[(groupID) * K + kBase + 2 * tig];
        a[1] = A[(groupID) * K + kBase + 2 * tig + 1];
        a[2] = A[(groupID + 8) * K + kBase + 2 * tig];
        a[3] = A[(groupID + 8) * K + kBase + 2 * tig + 1];
        a[4] = A[(groupID) * K + kBase + 2 * tig + 8];
        a[5] = A[(groupID) * K + kBase + 2 * tig + 9];
        a[6] = A[(groupID + 8) * K + kBase + 2 * tig + 8];
        a[7] = A[(groupID + 8) * K + kBase + 2 * tig + 9];

        __nv_bfloat16 b[4];
        b[0] = B[(kBase + 2 * tig) * N + 8 * panel + groupID];
        b[1] = B[(kBase + 2 * tig + 1) * N + 8 * panel + groupID];
        b[2] = B[(kBase + 2 * tig + 8) * N + 8 * panel + groupID];
        b[3] = B[(kBase + 2 * tig + 9) * N + 8 * panel + groupID];

        const uint32_t *A32 = reinterpret_cast<const uint32_t *>(a);
        const uint32_t *B32 = reinterpret_cast<const uint32_t *>(b);

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(A32[0]), "r"(A32[1]), "r"(A32[2]), "r"(A32[3]),
              "r"(B32[0]), "r"(B32[1]));
    }

    C[(groupID) * N + 8 * panel + 2 * tig] = d[0];
    C[(groupID) * N + 8 * panel + 2 * tig + 1] = d[1];
    C[(groupID + 8) * N + 8 * panel + 2 * tig] = d[2];
    C[(groupID + 8) * N + 8 * panel + 2 * tig + 1] = d[3];
    }
}

// --------------------------------------------------------------------- int8

__global__ void gemmInt8(const int8_t *A, const int8_t *B, int *C) {
    int lane = threadIdx.x;
    int groupID = lane >> 2;
    int tig = lane & 3;

    // m16n8k32, 8-bit operands: A is 16 bytes per lane, B is 8.
    int8_t a[16];
    for (int i = 0; i < 4; i++) {
        a[i] = A[(groupID) * K + 4 * tig + i];
        a[4 + i] = A[(groupID + 8) * K + 4 * tig + i];
        a[8 + i] = A[(groupID) * K + 4 * tig + 16 + i];
        a[12 + i] = A[(groupID + 8) * K + 4 * tig + 16 + i];
    }
    const uint32_t *A32 = reinterpret_cast<const uint32_t *>(a);

    // n8 again: two panels of 8 columns each.
    for (int panel = 0; panel < 2; panel++) {
        int8_t b[8];
        for (int i = 0; i < 4; i++) {
            b[i] = B[(4 * tig + i) * N + 8 * panel + groupID];
            b[4 + i] = B[(4 * tig + 16 + i) * N + 8 * panel + groupID];
        }
        const uint32_t *B32 = reinterpret_cast<const uint32_t *>(b);
        int d[4] = { 0, 0, 0, 0 };

        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(A32[0]), "r"(A32[1]), "r"(A32[2]), "r"(A32[3]),
              "r"(B32[0]), "r"(B32[1]));

        C[(groupID) * N + 8 * panel + 2 * tig] = d[0];
        C[(groupID) * N + 8 * panel + 2 * tig + 1] = d[1];
        C[(groupID + 8) * N + 8 * panel + 2 * tig] = d[2];
        C[(groupID + 8) * N + 8 * panel + 2 * tig + 1] = d[3];
    }
}

// ---------------------------------------------------------------------- FP8

// e4m3 and e5m2 differ only in the instruction: the byte staging is identical.
#define FP8_KERNEL(NAME, PTXTYPE)                                              \
__global__ void NAME(const uint8_t *A, const uint8_t *B, float *C) {            \
    int lane = threadIdx.x;                                                    \
    int groupID = lane >> 2;                                                   \
    int tig = lane & 3;                                                        \
    uint8_t a[16];                                                             \
    for (int i = 0; i < 4; i++) {                                              \
        a[i] = A[(groupID) * K + 4 * tig + i];                                 \
        a[4 + i] = A[(groupID + 8) * K + 4 * tig + i];                         \
        a[8 + i] = A[(groupID) * K + 4 * tig + 16 + i];                        \
        a[12 + i] = A[(groupID + 8) * K + 4 * tig + 16 + i];                   \
    }                                                                          \
    const uint32_t *A32 = reinterpret_cast<const uint32_t *>(a);               \
    for (int panel = 0; panel < 2; panel++) {                                  \
        uint8_t b[8];                                                          \
        for (int i = 0; i < 4; i++) {                                          \
            b[i] = B[(4 * tig + i) * N + 8 * panel + groupID];                 \
            b[4 + i] = B[(4 * tig + 16 + i) * N + 8 * panel + groupID];        \
        }                                                                      \
        const uint32_t *B32 = reinterpret_cast<const uint32_t *>(b);           \
        float d[4] = { 0.0f, 0.0f, 0.0f, 0.0f };                               \
        asm volatile(                                                          \
            "mma.sync.aligned.m16n8k32.row.col.f32." PTXTYPE "." PTXTYPE ".f32 " \
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n" \
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                   \
            : "r"(A32[0]), "r"(A32[1]), "r"(A32[2]), "r"(A32[3]),              \
              "r"(B32[0]), "r"(B32[1]));                                       \
        C[(groupID) * N + 8 * panel + 2 * tig] = d[0];                         \
        C[(groupID) * N + 8 * panel + 2 * tig + 1] = d[1];                     \
        C[(groupID + 8) * N + 8 * panel + 2 * tig] = d[2];                     \
        C[(groupID + 8) * N + 8 * panel + 2 * tig + 1] = d[3];                 \
    }                                                                          \
}

FP8_KERNEL(gemmFP8E4M3, "e4m3")
FP8_KERNEL(gemmFP8E5M2, "e5m2")

// -------------------------------------------------------------------- host

// Same generators as the Java version: small multiples of 0.5 in [-2, 2],
// exactly representable in bf16, e4m3 and e5m2 alike.
static float value(int i) { return ((i * 7 + 3) % 9) * 0.5f - 2.0f; }
static int intValue(int i) { return ((i * 7 + 3) % 9) - 4; }

static bool report(const char *label, const float *ref, const float *got) {
    float maxErr = 0.0f;
    int bad = 0;
    for (int i = 0; i < M * N; i++) {
        float e = fabsf(ref[i] - got[i]);
        if (e > maxErr) maxErr = e;
        if (e > 1e-3f) bad++;
    }
    printf("  %s  %s (max abs err %.5f, %d/%d cells out of tol)\n",
           label, bad == 0 ? "PASSED" : "FAILED", maxErr, bad, M * N);
    return bad == 0;
}

int main() {
    printf("Tensor-core operand types in CUDA: C[%dx%d] = A[%dx%d] * B[%dx%d], one warp\n", M, N, M, K, K, N);
    printf("  each result validated against a CPU reference over the same stored values\n\n");

    float ref[M * N], got[M * N];
    bool allOk = true;

    // ---- BF16
    {
        __nv_bfloat16 hA[M * K], hB[K * N];
        for (int i = 0; i < M * K; i++) hA[i] = __float2bfloat16(value(i));
        for (int i = 0; i < K * N; i++) hB[i] = __float2bfloat16(value(i + 5));
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++)
                    acc += __bfloat162float(hA[i * K + k]) * __bfloat162float(hB[k * N + j]);
                ref[i * N + j] = acc;
            }
        __nv_bfloat16 *dA, *dB; float *dC;
        CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
        CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
        CUDA_CHECK(cudaMalloc(&dC, sizeof(ref)));
        CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
        gemmBF16<<<1, WARP_SIZE>>>(dA, dB, dC);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(got, dC, sizeof(got), cudaMemcpyDeviceToHost));
        allOk &= report("BF16     m16n8k16  f32.bf16.bf16.f32", ref, got);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    // ---- int8
    {
        int8_t hA[M * K], hB[K * N];
        for (int i = 0; i < M * K; i++) hA[i] = (int8_t) intValue(i);
        for (int i = 0; i < K * N; i++) hB[i] = (int8_t) intValue(i + 5);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                int acc = 0;
                for (int k = 0; k < K; k++) acc += hA[i * K + k] * hB[k * N + j];
                ref[i * N + j] = (float) acc;
            }
        int8_t *dA, *dB; int *dC; int gotI[M * N];
        CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
        CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
        CUDA_CHECK(cudaMalloc(&dC, sizeof(gotI)));
        CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
        gemmInt8<<<1, WARP_SIZE>>>(dA, dB, dC);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(gotI, dC, sizeof(gotI), cudaMemcpyDeviceToHost));
        for (int i = 0; i < M * N; i++) got[i] = (float) gotI[i];
        allOk &= report("int8     m16n8k32  s32.s8.s8.s32    ", ref, got);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    // ---- FP8 e4m3 and e5m2
    for (int pass = 0; pass < 2; pass++) {
        bool e4m3 = (pass == 0);
        uint8_t hA[M * K], hB[K * N];
        float aF[M * K], bF[K * N];
        for (int i = 0; i < M * K; i++) {
            float v = value(i);
            if (e4m3) { __nv_fp8_e4m3 q(v); hA[i] = q.__x; aF[i] = (float) q; }
            else      { __nv_fp8_e5m2 q(v); hA[i] = q.__x; aF[i] = (float) q; }
        }
        for (int i = 0; i < K * N; i++) {
            float v = value(i + 5);
            if (e4m3) { __nv_fp8_e4m3 q(v); hB[i] = q.__x; bF[i] = (float) q; }
            else      { __nv_fp8_e5m2 q(v); hB[i] = q.__x; bF[i] = (float) q; }
        }
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) acc += aF[i * K + k] * bF[k * N + j];
                ref[i * N + j] = acc;
            }
        uint8_t *dA, *dB; float *dC;
        CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
        CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
        CUDA_CHECK(cudaMalloc(&dC, sizeof(ref)));
        CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
        if (e4m3) gemmFP8E4M3<<<1, WARP_SIZE>>>(dA, dB, dC);
        else      gemmFP8E5M2<<<1, WARP_SIZE>>>(dA, dB, dC);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(got, dC, sizeof(got), cudaMemcpyDeviceToHost));
        allOk &= report(e4m3 ? "FP8 e4m3 m16n8k32  f32.e4m3.e4m3.f32"
                             : "FP8 e5m2 m16n8k32  f32.e5m2.e5m2.f32", ref, got);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    printf("\n%s\n", allOk ? "All four operand types produced correct results"
                           : "At least one operand type FAILED - see above");
    return allOk ? 0 : 1;
}
