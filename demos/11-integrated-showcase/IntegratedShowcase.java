/*
 * Devoxx 2026 CUDA demo — integrated showcase, combining every Hybrid-API
 * mechanism verified individually in demos 04/05/06/07/08 into one program:
 *
 *   - JIT-compiled @Parallel Java kernels running on the CUDA backend (all demos)
 *   - a vendor-library call (cuBLAS sgemv) inside the same TaskGraph as the
 *     Java kernels, sharing device buffers with no host round-trip (demo 04)
 *   - N independent JIT+cuBLAS+JIT chains in ONE TaskGraph, run with
 *     TornadoExecutionPlan#withIntraPlanConcurrency() so DAG-independent
 *     units are routed to separate CUDA streams and can co-reside on the
 *     GPU (demo 06's mechanism, applied to a cuBLAS-bearing graph instead
 *     of pure-JIT units)
 *   - the same N-chain graph captured once and replayed every later
 *     execution with TornadoExecutionPlan#withCUDAGraph() (demo 07's
 *     mechanism); this combination (cuBLAS library task + JIT tasks inside
 *     one CUDA-graph capture) is itself an upstream-verified pattern, see
 *     vendor/tornadovm/tornado-cublas/.../tests/TestCuBlasSgemvWithTasksCudaGraph.java
 *   - an EXPERIMENTAL "combined" mode stacking withCUDAGraph() AND
 *     withIntraPlanConcurrency() on the same plan — not a pattern found
 *     anywhere in the pinned upstream tree (grepped tornado-unittests/ and
 *     tornado-cublas/tests/ before writing this demo); this program probes
 *     it live and reports Observed (works, with numbers) or Blocked (exact
 *     exception), it does not assume either outcome
 *   - a Tensor Core mma.sync tile computation (demo 08's kernel, reused
 *     verbatim) as a final "capability inventory" stage — kept as its own
 *     TaskGraph/TornadoExecutionPlan rather than forced into the sgemv
 *     pipeline above, since its WorkerGrid (one warp) and data types
 *     (fp16 in / f32 out) are unrelated to the sgemv chain's shapes
 *
 * Every mode validates its output against a closed-form CPU reference
 * before any wall-clock number is printed, same rigor as demos 04/06/07/08.
 *
 * Usage:
 *   tornado --classpath . IntegratedShowcase [units] [m] [n] [executions] [mode]
 *     mode: baseline | concurrent | graph | combined | all   (default: all)
 */
import java.util.Arrays;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.enums.CuBlasOperation;

public class IntegratedShowcase {

    private static final float SCALE = 2.0f;
    private static final float BIAS = 1.0f;
    private static final float DELTA = 0.05f;

    // ---- Stage 1..3: JIT scale -> cuBLAS sgemv -> JIT bias, one unit ----

    private static void scale(FloatArray matrix) {
        for (@Parallel int i = 0; i < matrix.getSize(); i++) {
            matrix.set(i, matrix.get(i) * SCALE);
        }
    }

    private static void bias(FloatArray output) {
        for (@Parallel int i = 0; i < output.getSize(); i++) {
            output.set(i, output.get(i) + BIAS);
        }
    }

    private static void fillMatrix(FloatArray matrix, int m, int n, int unit, int exec) {
        for (int i = 0; i < m * n; i++) {
            matrix.set(i, ((i + unit + exec) % 7) + 1.0f);
        }
    }

    private static void fillVector(FloatArray vector, int n, int unit, int exec) {
        for (int j = 0; j < n; j++) {
            vector.set(j, ((j + unit + exec) % 5) + 1.0f);
        }
    }

    private static float[] expectedChain(int m, int n, int unit, int exec) {
        float[] mat = new float[m * n];
        float[] vec = new float[n];
        for (int i = 0; i < m * n; i++) {
            mat[i] = (((i + unit + exec) % 7) + 1.0f) * SCALE;
        }
        for (int j = 0; j < n; j++) {
            vec[j] = ((j + unit + exec) % 5) + 1.0f;
        }
        float[] out = new float[m];
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += mat[i * n + j] * vec[j];
            }
            out[i] = sum + BIAS;
        }
        return out;
    }

    private static TaskGraph buildChainGraph(String name, int units, int m, int n, FloatArray[] matrix,
            FloatArray[] vector, FloatArray[] output) {
        TaskGraph tg = new TaskGraph(name);
        final float alpha = 1.0f;
        final float beta = 0.0f;
        final int incx = 1;
        final int incy = 1;
        final int lda = m;
        for (int u = 0; u < units; u++) {
            tg.transferToDevice(DataTransferMode.EVERY_EXECUTION, matrix[u], vector[u]) //
                    .task("scale" + u, IntegratedShowcase::scale, matrix[u]) //
                    .libraryTask("sgemv" + u, CuBlas::cublasSgemv, //
                            CuBlasOperation.CUBLAS_OP_T.operation(), //
                            m, n, //
                            alpha, matrix[u], lda, vector[u], incx, beta, output[u], incy) //
                    .task("bias" + u, IntegratedShowcase::bias, output[u]) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, output[u]);
        }
        return tg;
    }

    /**
     * Runs one mode of the N-chain pipeline for `executions` iterations, validating
     * every iteration against the closed-form CPU reference. Returns the steady-state
     * median wall-clock in microseconds (excludes the first, capture/JIT execution),
     * or -1 if the mode raised an exception (caller decides whether that means Blocked).
     */
    private static double runChainMode(String label, boolean useGraph, boolean useConcurrency, int units, int m,
            int n, int executions) {
        System.out.println();
        System.out.println("=== " + label + " ===");

        FloatArray[] matrix = new FloatArray[units];
        FloatArray[] vector = new FloatArray[units];
        FloatArray[] output = new FloatArray[units];
        for (int u = 0; u < units; u++) {
            matrix[u] = new FloatArray(m * n);
            vector[u] = new FloatArray(n);
            output[u] = new FloatArray(m);
        }

        TaskGraph tg = buildChainGraph("chain_" + label.replaceAll("[^A-Za-z0-9]", ""), units, m, n, matrix, vector,
                output);
        ImmutableTaskGraph itg = tg.snapshot();

        long[] wallNanos = new long[executions];
        boolean allCorrect = true;
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(itg)) {
            if (useGraph) {
                plan.withCUDAGraph();
            }
            if (useConcurrency) {
                plan.withIntraPlanConcurrency();
            }
            for (int e = 0; e < executions; e++) {
                for (int u = 0; u < units; u++) {
                    fillMatrix(matrix[u], m, n, u, e);
                    fillVector(vector[u], n, u, e);
                }

                long start = System.nanoTime();
                plan.execute();
                long elapsed = System.nanoTime() - start;
                wallNanos[e] = elapsed;

                boolean correct = true;
                for (int u = 0; u < units; u++) {
                    float[] expected = expectedChain(m, n, u, e);
                    for (int i = 0; i < m; i++) {
                        if (Math.abs(output[u].get(i) - expected[i]) > DELTA) {
                            correct = false;
                            break;
                        }
                    }
                    if (!correct) {
                        break;
                    }
                }
                allCorrect &= correct;
                System.out.println("execution " + e + ": " + (correct ? "correct" : "WRONG") + ", wall="
                        + (elapsed / 1000) + " us");
            }
        } catch (TornadoExecutionPlanException e) {
            System.out.println(label + ": BLOCKED — " + e.getClass().getSimpleName() + ": " + e.getMessage());
            return -1;
        } catch (RuntimeException e) {
            System.out.println(label + ": BLOCKED — " + e.getClass().getSimpleName() + ": " + e.getMessage());
            return -1;
        }

        long[] sorted = Arrays.copyOf(wallNanos, wallNanos.length);
        Arrays.sort(sorted);
        long firstExecUs = wallNanos[0] / 1000;
        long[] steadyState = executions > 1 ? Arrays.copyOfRange(sorted, 1, sorted.length) : sorted;
        long steadyMedianUs = steadyState[steadyState.length / 2] / 1000;
        System.out.println(label + " first execution (JIT/capture): " + firstExecUs + " us");
        System.out.println(label + " steady-state median wall-clock (n=" + steadyState.length + "): "
                + steadyMedianUs + " us");
        System.out.println(label + ": " + (allCorrect ? "All executions correct" : "SOME EXECUTIONS WRONG"));
        return allCorrect ? steadyMedianUs : -1;
    }

    // ---- Bonus stage: Tensor Core mma.sync single-tile GEMM (demo 08's kernel) ----

    private static final int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    private static final int WARP_SIZE = 32;

    private static void gemmMMASingleTile(KernelContext ctx, HalfFloatArray A, HalfFloatArray B, FloatArray C) {
        int tid = ctx.localIdx;
        int[] aTile = ctx.allocateIntLocalArray(MMA_M * MMA_K / 2);
        int[] bTile = ctx.allocateIntLocalArray(MMA_K * MMA_N / 2);
        for (int idx = tid; idx < MMA_M * MMA_K / 2; idx += WARP_SIZE) {
            int row = idx / (MMA_K / 2);
            int kPair = idx % (MMA_K / 2);
            int kBase = kPair * 2;
            int g = row * MMA_K + kBase;
            int lo = A.get(g).getHalfFloatValue() & 0xFFFF;
            int hi = A.get(g + 1).getHalfFloatValue() & 0xFFFF;
            aTile[row * (MMA_K / 2) + kPair] = lo | (hi << 16);
        }
        for (int idx = tid; idx < MMA_K * MMA_N / 2; idx += WARP_SIZE) {
            int kRow = idx / (MMA_N / 2);
            int jPair = idx % (MMA_N / 2);
            int jBase = jPair * 2;
            int g = kRow * MMA_N + jBase;
            int lo = B.get(g).getHalfFloatValue() & 0xFFFF;
            int hi = B.get(g + 1).getHalfFloatValue() & 0xFFFF;
            bTile[idx] = lo | (hi << 16);
        }
        ctx.localBarrier();
        HalfFloat[] fragA = ctx.mmaLoadA(aTile, MMA_K);
        HalfFloat[] fragB = ctx.mmaLoadB(bTile, MMA_K);
        float[] fragC = ctx.mmaFragment(0.0f);
        fragC = ctx.mma(fragA, fragB, fragC, MMAShape.M16N8K16);
        ctx.mmaStore(fragC, C, 0, 0, MMA_N);
    }

    private static HalfFloatArray deterministicFp16(int size, int seed) {
        HalfFloatArray arr = new HalfFloatArray(size);
        for (int i = 0; i < size; i++) {
            float v = ((i * 31 + seed) % 17 - 8) / 8.0f;
            arr.set(i, new HalfFloat(v));
        }
        return arr;
    }

    private static void runTensorCoreBonus() throws TornadoExecutionPlanException {
        System.out.println();
        System.out.println("=== BONUS: Tensor Core mma.sync single-tile GEMM (demo 08 kernel, reused) ===");
        HalfFloatArray A = deterministicFp16(MMA_M * MMA_K, 1);
        HalfFloatArray B = deterministicFp16(MMA_K * MMA_N, 7);
        FloatArray cMMA = new FloatArray(MMA_M * MMA_N);

        float[] ref = new float[MMA_M * MMA_N];
        for (int i = 0; i < MMA_M; i++) {
            for (int j = 0; j < MMA_N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < MMA_K; k++) {
                    sum += A.get(i * MMA_K + k).getFloat32() * B.get(k * MMA_N + j).getFloat32();
                }
                ref[i * MMA_N + j] = sum;
            }
        }

        WorkerGrid1D wgMMA = new WorkerGrid1D(WARP_SIZE);
        wgMMA.setLocalWork(WARP_SIZE, 1, 1);
        GridScheduler gsMMA = new GridScheduler("mma.gemm", wgMMA);
        TaskGraph tgMMA = new TaskGraph("mma") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, A, B) //
                .task("gemm", IntegratedShowcase::gemmMMASingleTile, new KernelContext(), A, B, cMMA) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cMMA);

        try (TornadoExecutionPlan planMMA = new TornadoExecutionPlan(tgMMA.snapshot())) {
            planMMA.withGridScheduler(gsMMA).execute();
        }

        float maxAbs = 0.0f;
        int bad = 0;
        for (int i = 0; i < MMA_M * MMA_N; i++) {
            float diff = Math.abs(cMMA.get(i) - ref[i]);
            maxAbs = Math.max(maxAbs, diff);
            if (diff > 1e-2f) {
                bad++;
            }
        }
        System.out.printf("Tensor Core tile validation: %s (max abs err %.5f, %d/%d cells out of tol)%n",
                bad == 0 ? "PASSED" : "FAILED", maxAbs, bad, MMA_M * MMA_N);
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int units = args.length > 0 ? Integer.parseInt(args[0]) : 6;
        int m = args.length > 1 ? Integer.parseInt(args[1]) : 8;
        int n = args.length > 2 ? Integer.parseInt(args[2]) : 8;
        int executions = args.length > 3 ? Integer.parseInt(args[3]) : 20;
        String mode = args.length > 4 ? args[4] : "all";

        System.out.println("Integrated CUDA showcase: units=" + units + " m=" + m + " n=" + n + " executions="
                + executions + " mode=" + mode);
        System.out.println("Pipeline per unit: JIT scale -> cuBLAS sgemv -> JIT bias (demo 04's shape x " + units
                + " independent chains, one TaskGraph)");

        Double baselineUs = null, concurrentUs = null, graphUs = null, combinedUs = null;
        if (mode.equals("all") || mode.equals("baseline")) {
            double r = runChainMode("BASELINE (single stream, no graph)", false, false, units, m, n, executions);
            baselineUs = r >= 0 ? r : null;
        }
        if (mode.equals("all") || mode.equals("concurrent")) {
            double r = runChainMode("CONCURRENT (withIntraPlanConcurrency)", false, true, units, m, n, executions);
            concurrentUs = r >= 0 ? r : null;
        }
        if (mode.equals("all") || mode.equals("graph")) {
            double r = runChainMode("GRAPH (withCUDAGraph)", true, false, units, m, n, executions);
            graphUs = r >= 0 ? r : null;
        }
        if (mode.equals("all") || mode.equals("combined")) {
            double r = runChainMode("COMBINED-EXPERIMENTAL (withCUDAGraph + withIntraPlanConcurrency)", true, true,
                    units, m, n, executions);
            combinedUs = r >= 0 ? r : null;
        }

        System.out.println();
        System.out.println("=== Summary (steady-state median us, this run/this GPU) ===");
        System.out.println("baseline   : " + (baselineUs == null ? "n/a or blocked" : baselineUs));
        System.out.println("concurrent : " + (concurrentUs == null ? "n/a or blocked" : concurrentUs)
                + (baselineUs != null && concurrentUs != null ? String.format(" (%.2fx vs baseline)", baselineUs / concurrentUs) : ""));
        System.out.println("graph      : " + (graphUs == null ? "n/a or blocked" : graphUs)
                + (baselineUs != null && graphUs != null ? String.format(" (%.2fx vs baseline)", baselineUs / graphUs) : ""));
        System.out.println("combined   : " + (combinedUs == null ? "n/a or blocked" : combinedUs)
                + (baselineUs != null && combinedUs != null ? String.format(" (%.2fx vs baseline)", baselineUs / combinedUs) : ""));

        if (mode.equals("all") || mode.equals("mma")) {
            runTensorCoreBonus();
        }
    }
}
