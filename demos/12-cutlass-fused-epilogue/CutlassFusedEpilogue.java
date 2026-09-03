/*
 * TornadoVM CUDA demo — CUTLASS fused epilogues next to Java JIT kernels.
 *
 * CUTLASS's selling point over a plain GEMM library is the *epilogue*: the
 * bias add and the activation are folded into the GEMM kernel itself, so the
 * MxN result never leaves registers between the matrix multiply and the
 * activation. The alternative — GEMM, then a separate elementwise pass — has
 * to write the result to global memory and read it straight back.
 *
 * This demo runs both shapes on the same inputs, in the same JVM:
 *
 *   fused   : JIT "scale" -> CUTLASS gemmBiasRelu                (1 GPU kernel)
 *   unfused : JIT "scale" -> CUTLASS hgemm -> JIT "biasRelu"     (2 GPU kernels)
 *
 * Both are one TaskGraph mixing Java-JIT tasks with a CUTLASS library task on
 * shared device buffers, and both are validated against the same sequential
 * Java reference.
 *
 * Usage:
 *   tornado --classpath . CutlassFusedEpilogue [M N K executions]
 */
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.cutlass.Cutlass;

public class CutlassFusedEpilogue {

    /** CUTLASS tiles the problem; every dimension must be a multiple of this. */
    private static final int TILE = 32;

    private static final float SCALE = 0.5f;

    /**
     * JIT task: pre-scale the activations before the GEMM.
     *
     * Writes to a separate output buffer rather than scaling in place, so the
     * inputs can be uploaded once (FIRST_EXECUTION) and every execution still
     * computes exactly the same thing. Scaling A in place would force an
     * EVERY_EXECUTION re-upload and make the steady-state timings below measure
     * PCIe bandwidth instead of GPU work.
     */
    private static void scale(HalfFloatArray a, HalfFloatArray scaled) {
        for (@Parallel int i = 0; i < a.getSize(); i++) {
            scaled.set(i, new HalfFloat(a.get(i).getFloat32() * SCALE));
        }
    }

    /**
     * JIT task: the epilogue CUTLASS would otherwise fuse — bias add + ReLU as a
     * separate pass over the MxN result. Only used by the "unfused" graph.
     */
    private static void biasRelu(HalfFloatArray c, HalfFloatArray bias, int n) {
        for (@Parallel int i = 0; i < c.getSize(); i++) {
            float v = c.get(i).getFloat32() + bias.get(i % n).getFloat32();
            // Math.max, not `v > 0 ? v : 0`: a ternary here leaves a merge point that
            // defeats escape analysis on the following `new HalfFloat(...)`, and the
            // CUDA backend then fails to lower it ("Node implementing Lowerable not
            // handled: NewInstance"). See this demo's README.
            c.set(i, new HalfFloat(Math.max(v, 0.0f)));
        }
    }

    /** Deterministic, small, exactly-fp16-representable values: no RNG, no rounding noise. */
    private static HalfFloatArray deterministic(int size, int seed, float span) {
        HalfFloatArray arr = new HalfFloatArray(size);
        for (int i = 0; i < size; i++) {
            arr.set(i, new HalfFloat(((i * 31 + seed) % 17 - 8) / span));
        }
        return arr;
    }

    /** Sequential Java reference: relu(scale(A) * B + bias), row-major throughout. */
    private static float[] reference(HalfFloatArray a, HalfFloatArray b, HalfFloatArray bias, int m, int n, int k) {
        float[] out = new float[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int p = 0; p < k; p++) {
                    sum += (a.get(i * k + p).getFloat32() * SCALE) * b.get(p * n + j).getFloat32();
                }
                sum += bias.get(j).getFloat32();
                out[i * n + j] = sum > 0.0f ? sum : 0.0f;
            }
        }
        return out;
    }

    private static boolean validate(String label, HalfFloatArray got, float[] ref) {
        float maxAbs = 0.0f;
        int bad = 0;
        for (int i = 0; i < ref.length; i++) {
            float diff = Math.abs(got.get(i).getFloat32() - ref[i]);
            maxAbs = Math.max(maxAbs, diff);
            // fp16 accumulate over K terms; tolerance scales with the magnitudes involved.
            if (diff > 0.05f) {
                bad++;
            }
        }
        boolean ok = bad == 0;
        System.out.printf("  [%-7s] validation %s (max abs err %.5f, %d/%d cells out of tol)%n",
                label, ok ? "PASSED" : "FAILED", maxAbs, bad, ref.length);
        return ok;
    }

    private static boolean isCUDABackend() {
        int backendIndex = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice().getBackendIndex();
        return TornadoRuntimeProvider.getTornadoRuntime().getBackendType(backendIndex) == TornadoVMBackendType.CUDA;
    }

    private static long medianOf(long[] samples) {
        long[] copy = samples.clone();
        java.util.Arrays.sort(copy);
        return copy[copy.length / 2];
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (!isCUDABackend()) {
            System.out.println("CUTLASS library tasks are only available on the CUDA backend.");
            return;
        }

        int m = args.length > 0 ? Integer.parseInt(args[0]) : 1024;
        int n = args.length > 1 ? Integer.parseInt(args[1]) : 1024;
        int k = args.length > 2 ? Integer.parseInt(args[2]) : 1024;
        int executions = args.length > 3 ? Integer.parseInt(args[3]) : 20;

        if (m % TILE != 0 || n % TILE != 0 || k % TILE != 0) {
            System.out.printf("M, N and K must all be multiples of %d (got %d, %d, %d).%n", TILE, m, n, k);
            return;
        }

        System.out.printf("CUTLASS fused epilogue demo: C[%dx%d] = relu(scale(A[%dx%d]) * B[%dx%d] + bias), fp16%n",
                m, n, m, k, k, n);
        System.out.printf("  %d executions per mode, steady-state median reported (first execution excluded)%n%n",
                executions);

        HalfFloatArray bias = deterministic(n, 3, 32.0f);
        float[] ref;
        {
            HalfFloatArray a0 = deterministic(m * k, 1, 16.0f);
            HalfFloatArray b0 = deterministic(k * n, 7, 16.0f);
            ref = reference(a0, b0, bias, m, n, k);
        }

        HalfFloatArray a = deterministic(m * k, 1, 16.0f);
        HalfFloatArray b = deterministic(k * n, 7, 16.0f);

        // ---- fused: one CUTLASS kernel does GEMM + bias + ReLU ----
        HalfFloatArray aScaledFused = new HalfFloatArray(m * k);
        HalfFloatArray cFused = new HalfFloatArray(m * n);
        TaskGraph fusedGraph = new TaskGraph("fused") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, bias) //
                .task("scale", CutlassFusedEpilogue::scale, a, aScaledFused) //
                .libraryTask("gemmBiasRelu", Cutlass::cutlassGemmBiasRelu, //
                        m, n, k, aScaledFused, b, bias, cFused) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cFused);

        // ---- unfused: CUTLASS GEMM, then a separate JIT elementwise pass ----
        HalfFloatArray aScaledUnfused = new HalfFloatArray(m * k);
        HalfFloatArray cUnfused = new HalfFloatArray(m * n);
        TaskGraph unfusedGraph = new TaskGraph("unfused") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, bias) //
                .task("scale", CutlassFusedEpilogue::scale, a, aScaledUnfused) //
                .libraryTask("hgemm", Cutlass::cutlassHgemm, //
                        m, n, k, 1.0f, aScaledUnfused, b, 0.0f, cUnfused) //
                .task("biasRelu", CutlassFusedEpilogue::biasRelu, cUnfused, bias, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cUnfused);

        long fusedMedian;
        long unfusedMedian;
        boolean okFused;
        boolean okUnfused;

        ImmutableTaskGraph fusedSnapshot = fusedGraph.snapshot();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(fusedSnapshot)) {
            System.out.println("=== FUSED (CUTLASS gemmBiasRelu — one kernel) ===");
            long[] samples = new long[executions - 1];
            for (int e = 0; e < executions; e++) {
                long start = System.nanoTime();
                plan.execute();
                long elapsed = (System.nanoTime() - start) / 1000;
                if (e == 0) {
                    System.out.printf("  first execution (JIT compile + CUTLASS plan setup): %d us%n", elapsed);
                } else {
                    samples[e - 1] = elapsed;
                }
            }
            fusedMedian = medianOf(samples);
            System.out.printf("  steady-state median wall-clock (n=%d): %d us%n", samples.length, fusedMedian);
            okFused = validate("fused", cFused, ref);
        }

        System.out.println();
        ImmutableTaskGraph unfusedSnapshot = unfusedGraph.snapshot();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(unfusedSnapshot)) {
            System.out.println("=== UNFUSED (CUTLASS hgemm + JIT bias/ReLU — two kernels) ===");
            long[] samples = new long[executions - 1];
            for (int e = 0; e < executions; e++) {
                long start = System.nanoTime();
                plan.execute();
                long elapsed = (System.nanoTime() - start) / 1000;
                if (e == 0) {
                    System.out.printf("  first execution (JIT compile + CUTLASS plan setup): %d us%n", elapsed);
                } else {
                    samples[e - 1] = elapsed;
                }
            }
            unfusedMedian = medianOf(samples);
            System.out.printf("  steady-state median wall-clock (n=%d): %d us%n", samples.length, unfusedMedian);
            okUnfused = validate("unfused", cUnfused, ref);
        }

        System.out.println();
        System.out.println("=== Summary (steady-state median us, this run/this GPU) ===");
        System.out.printf("fused   : %d%n", fusedMedian);
        System.out.printf("unfused : %d (%.2fx the fused time)%n", unfusedMedian, (double) unfusedMedian / fusedMedian);
        System.out.println(okFused && okUnfused
                ? "Both modes produce the same, correct result"
                : "Result is INCORRECT");
    }
}
