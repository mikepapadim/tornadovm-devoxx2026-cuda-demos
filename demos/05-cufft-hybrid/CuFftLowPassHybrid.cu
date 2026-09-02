// CUDA equivalent of CuFftLowPassHybrid.java.
//
// A GPU-resident low-pass filter: cuFFT forward (R2C) -> your own zeroing
// kernel -> cuFFT inverse (C2R) -> your own scaling kernel. In Java that is one
// TaskGraph with four stages; here you manage two cuFFT plans, the device
// buffers shared between library and hand-written kernels, and the fact that
// cuFFT's inverse is unnormalised.
//
// Build & run:
//   nvcc -arch=sm_89 -lcufft -o cufft_lowpass CuFftLowPassHybrid.cu
//   ./cufft_lowpass [n] [cutoff] [iterations]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cufft.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUFFT_CHECK(call)                                                       \
    do {                                                                        \
        cufftResult r_ = (call);                                                \
        if (r_ != CUFFT_SUCCESS) {                                              \
            fprintf(stderr, "cuFFT error %d at %s:%d\n", (int) r_,              \
                    __FILE__, __LINE__);                                        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static const int LOW_TONE_A = 3;
static const int LOW_TONE_B = 7;
static const int HIGH_TONE = 200;

// Stage 2: zero every bin at or above the cutoff. Operates on the same device
// buffer cuFFT just wrote, with no host round trip.
__global__ void lowPass(cufftComplex *spectrum, int cutoff, int bins) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < bins && k >= cutoff) {
        spectrum[k].x = 0.0f;
        spectrum[k].y = 0.0f;
    }
}

// Stage 4: cuFFT's C2R inverse is unnormalised, so divide by n.
__global__ void scaleBy(float *array, float factor, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        array[i] = array[i] * factor;
    }
}

static void fillSignal(float *signal, int n) {
    for (int t = 0; t < n; t++) {
        float lowTones = (float) (sin((2.0 * M_PI * LOW_TONE_A * t) / n)
                                  + 0.5 * cos((2.0 * M_PI * LOW_TONE_B * t) / n));
        float highTone = (float) (0.8 * sin((2.0 * M_PI * HIGH_TONE * t) / n));
        signal[t] = lowTones + highTone;
    }
}

static float analyticLowFrequencyExpected(int t, int n) {
    return (float) (sin((2.0 * M_PI * LOW_TONE_A * t) / n)
                    + 0.5 * cos((2.0 * M_PI * LOW_TONE_B * t) / n));
}

int main(int argc, char **argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 4096;
    const int cutoff = argc > 2 ? atoi(argv[2]) : 16;
    const int iterations = argc > 3 ? atoi(argv[3]) : 5;
    const int bins = n / 2 + 1;

    printf("Low-pass filter via hybrid cuFFT pipeline: n=%d, cutoff bin=%d, iterations=%d\n",
           n, cutoff, iterations);

    float *hostSignal = (float *) malloc((size_t) n * sizeof(float));
    float *hostFiltered = (float *) malloc((size_t) n * sizeof(float));
    fillSignal(hostSignal, n);

    float *devSignal, *devFiltered;
    cufftComplex *devSpectrum;
    CUDA_CHECK(cudaMalloc(&devSignal, (size_t) n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devFiltered, (size_t) n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devSpectrum, (size_t) bins * sizeof(cufftComplex)));

    // Two plans: cuFFT needs a separate plan per direction/type.
    cufftHandle planForward, planInverse;
    CUFFT_CHECK(cufftPlan1d(&planForward, n, CUFFT_R2C, 1));
    CUFFT_CHECK(cufftPlan1d(&planInverse, n, CUFFT_C2R, 1));

    const int threads = 256;
    bool allCorrect = true;
    for (int it = 0; it < iterations; it++) {
        CUDA_CHECK(cudaMemcpy(devSignal, hostSignal, (size_t) n * sizeof(float),
                              cudaMemcpyHostToDevice));

        CUFFT_CHECK(cufftExecR2C(planForward, devSignal, devSpectrum));
        lowPass<<<(bins + threads - 1) / threads, threads>>>(devSpectrum, cutoff, bins);
        CUDA_CHECK(cudaGetLastError());
        CUFFT_CHECK(cufftExecC2R(planInverse, devSpectrum, devFiltered));
        scaleBy<<<(n + threads - 1) / threads, threads>>>(devFiltered, 1.0f / n, n);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(hostFiltered, devFiltered, (size_t) n * sizeof(float),
                              cudaMemcpyDeviceToHost));

        float maxError = 0.0f;
        bool correct = true;
        for (int t = 0; t < n; t++) {
            float error = fabsf(analyticLowFrequencyExpected(t, n) - hostFiltered[t]);
            maxError = fmaxf(maxError, error);
            if (error > 1e-2f) {
                correct = false;
            }
        }
        allCorrect &= correct;
        printf("iteration %d: %s maxError=%g filtered[0]=%.8f\n",
               it, correct ? "correct" : "WRONG", maxError, hostFiltered[0]);
    }
    printf("%s\n", allCorrect ? "All iterations correct" : "SOME ITERATIONS WRONG");

    CUFFT_CHECK(cufftDestroy(planForward));
    CUFFT_CHECK(cufftDestroy(planInverse));
    CUDA_CHECK(cudaFree(devSignal));
    CUDA_CHECK(cudaFree(devFiltered));
    CUDA_CHECK(cudaFree(devSpectrum));
    free(hostSignal);
    free(hostFiltered);
    return 0;
}
