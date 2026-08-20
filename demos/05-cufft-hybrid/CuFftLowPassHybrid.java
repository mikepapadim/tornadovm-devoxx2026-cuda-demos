/*
 * Devoxx 2026 CUDA demo — Java kernel + cuFFT in one execution graph.
 *
 * One TaskGraph, four stages, all on the same device buffers, no host
 * round trip between them:
 *   1. cuFFT library task ("forward")  — real-to-complex FFT (R2C)
 *   2. JIT-compiled @Parallel task ("lowPass") — zero bins >= cutoff
 *   3. cuFFT library task ("inverse")  — complex-to-real inverse FFT (C2R)
 *   4. JIT-compiled @Parallel task ("normalize", method name scaleBy —
 *      "normalize" itself is a reserved OpenCL token rejected by the
 *      sketcher) — cuFFT is unnormalized, so divide by n
 *
 * Input signal is a deterministic sum of two low-frequency tones (kept by
 * the filter) and one high-frequency tone (removed). The filtered output is
 * validated every iteration against the exact analytic low-frequency signal
 * (not a numeric DFT reference), since the tone frequencies and cutoff are
 * chosen so the filtered result is known in closed form.
 */
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.cufft.CuFft;

public class CuFftLowPassHybrid {

    private static final int LOW_TONE_A = 3;
    private static final int LOW_TONE_B = 7;
    private static final int HIGH_TONE = 200;

    private static void lowPass(FloatArray spectrum, int cutoff, int bins) {
        for (@Parallel int k = 0; k < bins; k++) {
            if (k >= cutoff) {
                spectrum.set(2 * k, 0.0f);
                spectrum.set(2 * k + 1, 0.0f);
            }
        }
    }

    /** Named scaleBy, not "normalize" — TornadoSketcher rejects that name as a reserved OpenCL token. */
    private static void scaleBy(FloatArray array, float factor) {
        for (@Parallel int i = 0; i < array.getSize(); i++) {
            array.set(i, array.get(i) * factor);
        }
    }

    /** Deterministic, presenter-friendly signal: two low tones (kept) + one high tone (removed). */
    private static void fillSignal(FloatArray signal, int n) {
        for (int t = 0; t < n; t++) {
            float lowTones = (float) (Math.sin((2 * Math.PI * LOW_TONE_A * t) / n) + 0.5 * Math.cos((2 * Math.PI * LOW_TONE_B * t) / n));
            float highTone = (float) (0.8 * Math.sin((2 * Math.PI * HIGH_TONE * t) / n));
            signal.set(t, lowTones + highTone);
        }
    }

    private static float analyticLowFrequencyExpected(int t, int n) {
        return (float) (Math.sin((2 * Math.PI * LOW_TONE_A * t) / n) + 0.5 * Math.cos((2 * Math.PI * LOW_TONE_B * t) / n));
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 4096;
        int cutoff = args.length > 1 ? Integer.parseInt(args[1]) : 16;
        int iterations = args.length > 2 ? Integer.parseInt(args[2]) : 5;
        int bins = n / 2 + 1;

        System.out.println("Low-pass filter via hybrid cuFFT pipeline: n=" + n + ", cutoff bin=" + cutoff + ", iterations=" + iterations);

        FloatArray signal = new FloatArray(n);
        FloatArray spectrum = new FloatArray(2 * bins);
        FloatArray filtered = new FloatArray(n);
        fillSignal(signal, n);

        TaskGraph taskGraph = new TaskGraph("cufftHybrid") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, signal) //
                .libraryTask("forward", CuFft::cufftForwardR2C, signal, spectrum, n, 1) //
                .task("lowPass", CuFftLowPassHybrid::lowPass, spectrum, cutoff, bins) //
                .libraryTask("inverse", CuFft::cufftInverseC2R, spectrum, filtered, n, 1) //
                .task("normalize", CuFftLowPassHybrid::scaleBy, filtered, 1.0f / n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, filtered);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        boolean allCorrect = true;
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutableTaskGraph)) {
            for (int it = 0; it < iterations; it++) {
                TornadoExecutionResult result = plan.execute();

                float maxError = 0.0f;
                boolean correct = true;
                for (int t = 0; t < n; t++) {
                    float expected = analyticLowFrequencyExpected(t, n);
                    float error = Math.abs(expected - filtered.get(t));
                    maxError = Math.max(maxError, error);
                    if (error > 1e-2f) {
                        correct = false;
                    }
                }
                allCorrect &= correct;
                System.out.println("iteration " + it + ": " + (correct ? "correct" : "WRONG") //
                        + " maxError=" + maxError + " filtered[0]=" + filtered.get(0));
                System.out.println("  total task-graph time: " + result.getProfilerResult().getTotalTime() + " ns");
            }
        }
        System.out.println(allCorrect ? "All iterations correct" : "SOME ITERATIONS WRONG");
    }
}
