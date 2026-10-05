import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.cufft.CuFft;

/**
 * The Hybrid API, live: one TaskGraph mixes NVIDIA cuFFT library tasks with Java kernels JIT-compiled by TornadoVM,
 * on shared device buffers:
 *
 *   signal -> cuFFT forward (NVIDIA) -> lowPass (Java @Parallel) -> cuFFT inverse (NVIDIA) -> normalize (Java) -> out
 *
 * 1. sweep: the low-pass cutoff sweeps down, one execution per frame (~10 frames/s). Every frame is checked against
 *    the exact answer (the signal is a sum of tones, so a low-pass keeps exactly the tones below the cutoff), and
 *    TornadoVM's profiler reports each task's GPU time and the bytes copied.
 * 2. graph: the same graph, executed back to back, plain and with withCUDAGraph().
 *
 * Protocol (stdout, read by dashboard.py): IN, FRAME, PROFILE, OUT, SPEC, GRAPH, END lines, then the verdict.
 */
public class HybridLive {

    static final int N = 8192, BINS = N / 2 + 1;
    /** frequency (cycles per window) and amplitude: two "voice" tones, then the "noise" the filter removes */
    static final int[] FREQ = { 3, 7, 40, 90, 150, 230, 330, 470, 700, 1000 };
    static final float[] AMP = { 1.0f, 0.5f, 0.35f, 0.3f, 0.25f, 0.25f, 0.2f, 0.2f, 0.15f, 0.15f };
    static final int POINTS = 440, SPEC_BINS = 1200;

    // ---- the two Java kernels: plain Java, JIT-compiled to CUDA by TornadoVM ----

    /** Zeroes every frequency bin at or above the cutoff (interleaved complex spectrum from cuFFT R2C). */
    static void lowPass(FloatArray spectrum, IntArray cutoff, int bins) {
        for (@Parallel int k = 0; k < bins; k++) {
            if (k >= cutoff.get(0)) {
                spectrum.set(2 * k, 0f);
                spectrum.set(2 * k + 1, 0f);
            }
        }
    }

    /** cuFFT's inverse is unnormalised: divide by n. (Named scaleBy: "normalize" is a reserved OpenCL token.) */
    static void scaleBy(FloatArray signal, float factor) {
        for (@Parallel int i = 0; i < signal.getSize(); i++) {
            signal.set(i, signal.get(i) * factor);
        }
    }

    static float tones(int t, int below) {
        double v = 0;
        for (int i = 0; i < FREQ.length; i++) {
            if (FREQ[i] < below) {
                v += AMP[i] * Math.sin(2 * Math.PI * FREQ[i] * t / N + i);
            }
        }
        return (float) v;
    }

    public static void main(String[] args) throws Exception {
        int frames = args.length > 0 ? Integer.parseInt(args[0]) : 110;
        boolean wait = args.length > 1 && args[1].equals("wait");
        FloatArray signal = new FloatArray(N), spectrum = new FloatArray(2 * BINS), filtered = new FloatArray(N);
        IntArray cutoff = IntArray.fromElements(BINS);
        for (int t = 0; t < N; t++) {
            signal.set(t, tones(t, Integer.MAX_VALUE));
        }
        System.out.println("IN " + sample(signal));

        TaskGraph tg = new TaskGraph("hybrid")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, signal, cutoff)
                .libraryTask("forward", CuFft::cufftForwardR2C, signal, spectrum, N, 1)   // NVIDIA cuFFT
                .task("lowPass", HybridLive::lowPass, spectrum, cutoff, BINS)               // Java, JIT-compiled
                .libraryTask("inverse", CuFft::cufftInverseC2R, spectrum, filtered, N, 1) // NVIDIA cuFFT
                .task("normalize", HybridLive::scaleBy, filtered, 1.0f / N)                // Java, JIT-compiled
                .transferToHost(DataTransferMode.EVERY_EXECUTION, filtered, spectrum);     // spectrum: for the display

        boolean allCorrect = true;
        // 1. the sweep
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            plan.withProfiler(ProfilerMode.SILENT);
            for (int f = 0; f < frames; f++) {
                int cut = (int) Math.round(1200 * Math.pow(5.0 / 1200, f / (double) (frames - 1)));
                cutoff.set(0, cut);
                long t = System.nanoTime();
                TornadoExecutionResult result = plan.execute();
                double us = (System.nanoTime() - t) / 1e3;
                float maxError = 0;
                for (int i = 0; i < N; i++) {
                    maxError = Math.max(maxError, Math.abs(filtered.get(i) - tones(i, cut)));
                }
                allCorrect &= maxError < 1e-2f;
                System.out.printf("FRAME %d %d %.1f %s %.2e%n", f, cut, us, maxError < 1e-2f ? "correct" : "WRONG", maxError);
                String log = result.getProfilerResult().getProfileLog();
                System.out.println("PROFILE " + log.substring(log.lastIndexOf("{\n    \"hybrid\"") < 0 ? 0 : log.lastIndexOf("{\n    \"hybrid\"")).replace("\n", " "));
                System.out.println("OUT " + sample(filtered));
                StringBuilder spec = new StringBuilder("SPEC");
                for (int k = 0; k < SPEC_BINS; k++) {
                    float re = spectrum.get(2 * k), im = spectrum.get(2 * k + 1);
                    spec.append(String.format(" %.3f", Math.sqrt(re * re + im * im) / (N / 2.0)));
                }
                System.out.println(spec);
                System.out.flush();
                if (f == 0 && wait) { // live.py starts the sweep after the pipeline is on screen
                    new java.io.BufferedReader(new java.io.InputStreamReader(System.in)).readLine();
                }
                long rest = 90_000_000L - (System.nanoTime() - t);
                if (rest > 0) {
                    Thread.sleep(rest / 1_000_000);
                }
            }
        }

        // 2. the same graph back to back: plain execute() vs one captured CUDA graph per execution
        cutoff.set(0, 20);
        double plain = medianMicros(tg, false), graph = medianMicros(tg, true);
        float maxError = 0;
        for (int i = 0; i < N; i++) {
            maxError = Math.max(maxError, Math.abs(filtered.get(i) - tones(i, 20)));
        }
        System.out.printf("GRAPH %.1f %.1f %s%n", plain, graph, maxError < 1e-2f ? "correct" : "WRONG");
        System.out.println("END");
        allCorrect &= maxError < 1e-2f;
        System.out.println(allCorrect ? "HybridLive: PASSED -- every execution matches the exact answer, plain and as a CUDA graph"
                : "HybridLive: FAILED -- an execution differs from the exact answer");
    }

    static double medianMicros(TaskGraph tg, boolean cudaGraph) throws Exception {
        double[] us = new double[300];
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            if (cudaGraph) {
                plan.withCUDAGraph();
            }
            for (int i = 0; i < 20; i++) {
                plan.execute(); // warm-up (and the graph capture)
            }
            for (int i = 0; i < us.length; i++) {
                long t = System.nanoTime();
                plan.execute();
                us[i] = (System.nanoTime() - t) / 1e3;
            }
        }
        java.util.Arrays.sort(us);
        return us[us.length / 2];
    }

    static String sample(FloatArray a) {
        StringBuilder s = new StringBuilder();
        for (int p = 0; p < POINTS; p++) {
            s.append(String.format("%.3f ", a.get(p * N / POINTS)));
        }
        return s.toString().trim();
    }
}
