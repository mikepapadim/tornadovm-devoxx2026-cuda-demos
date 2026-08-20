/*
 * Devoxx 2026 CUDA demo — CUDA streams and async overlap from Java.
 *
 * TornadoExecutionPlan#withIntraPlanConcurrency() routes DAG-independent operations
 * within one execution plan to separate H2D / COMPUTE-pool / D2H role streams instead
 * of the single default stream, ordering them with device-side events. This is CUDA-only
 * runtime behaviour (uk.ac.manchester.tornado.api.TornadoExecutionPlan javadoc: "currently
 * realised on the CUDA backend"), the same family as withCUDAGraph() (demo 02) and
 * withStagedTransfers().
 *
 * This demo builds N independent H2D -> compute -> D2H pipelines in one TaskGraph (same
 * shape as the upstream correctness test
 * tornado-unittests/.../streams/TestCUDAStreams#testManyIndependentUnitsMultiStream, and the
 * wall-clock-focused tornado-unittests/.../streams/TestStreamsPerformance#testSmallKernelConcurrency)
 * and runs it two ways from the same JVM:
 *
 *   sequential mode  - no withIntraPlanConcurrency(): every unit serialised on one stream.
 *   concurrent mode  - withIntraPlanConcurrency(): units round-robin across CUDA streams and
 *                       can genuinely co-reside on the GPU.
 *
 * Each unit's kernel is deliberately small-grid + heavy-inner-loop (does not saturate the
 * SMs on its own), matching TestStreamsPerformance's documented condition for kernels to
 * actually co-reside across streams rather than merely being issued on different streams.
 * Every unit's output is validated against a closed-form CPU reference after every run, in
 * both modes. Median wall-clock time per mode is reported (printed, not asserted): overlap
 * speedup is workload/GPU/driver-dependent, so it is evidence, not a pass/fail gate.
 */
import java.util.Arrays;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class CudaStreamsOverlap {

    private static final float ALPHA = 0.5f;
    private static final float DELTA = 1e-2f;

    /** Small-grid, heavy-inner-loop kernel: a single unit does not saturate the SMs on its own. */
    private static void computeSmall(FloatArray x, FloatArray y, FloatArray result, float alpha, int innerIterations) {
        for (@Parallel int i = 0; i < result.getSize(); i++) {
            float xi = x.get(i);
            float yi = y.get(i);
            float val = alpha * xi + yi;
            for (int j = 0; j < innerIterations; j++) {
                val = val * alpha + yi;
            }
            result.set(i, val);
        }
    }

    private static float expectedSmall(float x, float y, float alpha, int innerIterations) {
        float val = alpha * x + y;
        for (int j = 0; j < innerIterations; j++) {
            val = val * alpha + y;
        }
        return val;
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int units = args.length > 0 ? Integer.parseInt(args[0]) : 8;
        int unitSize = args.length > 1 ? Integer.parseInt(args[1]) : 32 * 1024;
        int innerIterations = args.length > 2 ? Integer.parseInt(args[2]) : 1 << 16;
        int executions = args.length > 3 ? Integer.parseInt(args[3]) : 8;
        String mode = args.length > 4 ? args[4] : "both"; // sequential | concurrent | both

        System.out.println("units=" + units + " unitSize=" + unitSize + " innerIterations=" + innerIterations
                + " executions=" + executions + " mode=" + mode);

        if (!mode.equals("concurrent")) {
            runMode(false, units, unitSize, innerIterations, executions);
        }
        if (!mode.equals("sequential")) {
            runMode(true, units, unitSize, innerIterations, executions);
        }
    }

    private static void runMode(boolean concurrent, int units, int unitSize, int innerIterations, int executions)
            throws TornadoExecutionPlanException {
        String label = concurrent ? "CONCURRENT (withIntraPlanConcurrency)" : "SEQUENTIAL (single stream)";
        System.out.println();
        System.out.println("=== " + label + " ===");

        FloatArray[] x = new FloatArray[units];
        FloatArray[] y = new FloatArray[units];
        FloatArray[] r = new FloatArray[units];

        TaskGraph tg = new TaskGraph((concurrent ? "streamsConcurrent" : "streamsSequential"));
        for (int u = 0; u < units; u++) {
            x[u] = new FloatArray(unitSize);
            y[u] = new FloatArray(unitSize);
            r[u] = new FloatArray(unitSize);
            x[u].init(u + 1.0f);
            y[u].init(u + 2.0f);
            final int idx = u;
            tg.transferToDevice(DataTransferMode.EVERY_EXECUTION, x[u], y[u])
                    .task("unit" + u, CudaStreamsOverlap::computeSmall, x[u], y[u], r[u], ALPHA, innerIterations)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, r[u]);
        }

        ImmutableTaskGraph itg = tg.snapshot();
        long[] wallNanos = new long[executions];
        boolean allCorrect = true;
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(itg)) {
            if (concurrent) {
                plan.withIntraPlanConcurrency();
            }
            for (int e = 0; e < executions; e++) {
                long start = System.nanoTime();
                plan.execute();
                long elapsed = System.nanoTime() - start;
                wallNanos[e] = elapsed;

                boolean correct = true;
                for (int u = 0; u < units; u++) {
                    float expected = expectedSmall(u + 1.0f, u + 2.0f, ALPHA, innerIterations);
                    for (int i = 0; i < unitSize; i++) {
                        if (Math.abs(r[u].get(i) - expected) > DELTA) {
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
        }

        long[] sorted = Arrays.copyOf(wallNanos, wallNanos.length);
        Arrays.sort(sorted);
        long median = sorted[sorted.length / 2];
        System.out.println(label + " median wall-clock (all " + executions + " executions incl. first/JIT): "
                + (median / 1000) + " us");
        if (executions > 1) {
            long[] steadyState = Arrays.copyOfRange(sorted, 1, sorted.length);
            long steadyMedian = steadyState[steadyState.length / 2];
            System.out.println(label + " median wall-clock (excl. first execution): " + (steadyMedian / 1000) + " us");
        }
        System.out.println(label + ": " + (allCorrect ? "All executions correct" : "Some executions WRONG"));
    }
}
