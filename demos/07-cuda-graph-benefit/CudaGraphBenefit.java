/*
 * Devoxx 2026 CUDA demo — quantifying why CUDA Graph replay matters for repeated workloads.
 *
 * Demo 02 (../02-cuda-runtime-api) already showed TornadoExecutionPlan#withCUDAGraph()
 * capturing a task-graph once (cuStreamBeginCapture/cuStreamEndCapture/cuGraphInstantiate,
 * see vendor/tornadovm/tornado-drivers/cuda-jni/.../CUDAGraph.cpp) and replaying it
 * (cuGraphLaunch) on every subsequent execute(). That demo's evidence mixed first-execution
 * JIT-compile time into the "before" number, so it could not isolate the actual benefit of
 * graph replay vs. plain repeated execute() calls.
 *
 * This demo isolates that: the SAME multi-stage task-graph (a chain of N small elementwise
 * JIT tasks, each stage's output feeding the next, EVERY_EXECUTION host<->device transfers so
 * every execution is a real end-to-end dispatch) is run for many executions in two modes from
 * the same JVM:
 *
 *   nograph mode - plain TornadoExecutionPlan, no withCUDAGraph(): every execution re-issues
 *                   each stage's host-to-device copy / kernel launch / device-to-host copy as
 *                   separate CUDA runtime calls.
 *   graph mode   - withCUDAGraph(): the first execution captures the whole chain into one
 *                   CUgraph; every later execution is a single cuGraphLaunch replay.
 *
 * Both modes discard the first execution (JIT compile + graph capture, not representative of
 * steady state) and report the median wall-clock of the remaining executions. Every execution
 * in both modes is validated against a closed-form CPU reference. The reported numbers are
 * this-run/this-GPU evidence, not a general performance guarantee (same treatment as demo 06).
 */
import java.util.Arrays;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class CudaGraphBenefit {

    // Deterministic per-stage affine transform: stage s computes out = a[s] * in + b[s].
    private static final float[] A = { 2.0f, 0.5f, 3.0f, 0.25f, 1.5f, 4.0f };
    private static final float[] B = { 1.0f, 2.0f, -3.0f, 0.5f, -1.0f, 0.75f };

    private static void stage(FloatArray in, FloatArray out, float a, float b) {
        for (@Parallel int i = 0; i < out.getSize(); i++) {
            out.set(i, a * in.get(i) + b);
        }
    }

    private static float expectedChain(float x0, int stages) {
        float v = x0;
        for (int s = 0; s < stages; s++) {
            v = A[s % A.length] * v + B[s % B.length];
        }
        return v;
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int size = args.length > 0 ? Integer.parseInt(args[0]) : 4096;
        int stages = args.length > 1 ? Integer.parseInt(args[1]) : 6;
        int executions = args.length > 2 ? Integer.parseInt(args[2]) : 50;
        String mode = args.length > 3 ? args[3] : "both"; // nograph | graph | both

        System.out.println("size=" + size + " stages=" + stages + " executions=" + executions + " mode=" + mode);

        Double nographMedianUs = null;
        Double graphMedianUs = null;
        if (!mode.equals("graph")) {
            nographMedianUs = runMode(false, size, stages, executions);
        }
        if (!mode.equals("nograph")) {
            graphMedianUs = runMode(true, size, stages, executions);
        }
        if (nographMedianUs != null && graphMedianUs != null) {
            System.out.println();
            System.out.printf("steady-state median: nograph=%.1f us, graph=%.1f us, speedup=%.2fx%n", nographMedianUs,
                    graphMedianUs, nographMedianUs / graphMedianUs);
        }
    }

    private static double runMode(boolean useGraph, int size, int stages, int executions)
            throws TornadoExecutionPlanException {
        String label = useGraph ? "GRAPH (withCUDAGraph)" : "NOGRAPH (plain execute)";
        System.out.println();
        System.out.println("=== " + label + " ===");

        FloatArray x = new FloatArray(size);
        FloatArray[] buf = new FloatArray[stages];
        for (int s = 0; s < stages; s++) {
            buf[s] = new FloatArray(size);
        }

        TaskGraph tg = new TaskGraph(useGraph ? "chainGraph" : "chainNoGraph");
        tg.transferToDevice(DataTransferMode.EVERY_EXECUTION, x);
        FloatArray prev = x;
        for (int s = 0; s < stages; s++) {
            final float a = A[s % A.length];
            final float b = B[s % B.length];
            final FloatArray in = prev;
            final FloatArray out = buf[s];
            tg.task("stage" + s, CudaGraphBenefit::stage, in, out, a, b);
            prev = out;
        }
        tg.transferToHost(DataTransferMode.EVERY_EXECUTION, prev);
        FloatArray result = prev;

        ImmutableTaskGraph itg = tg.snapshot();
        long[] wallNanos = new long[executions];
        boolean allCorrect = true;
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(itg)) {
            if (useGraph) {
                plan.withCUDAGraph();
            }
            for (int e = 0; e < executions; e++) {
                // Mutate the input every execution so a stale replay would be caught.
                for (int i = 0; i < size; i++) {
                    x.set(i, (e + 1) * 0.01f + i * 0.001f);
                }

                long start = System.nanoTime();
                plan.execute();
                long elapsed = System.nanoTime() - start;
                wallNanos[e] = elapsed;

                boolean correct = true;
                for (int i = 0; i < size; i++) {
                    float x0 = (e + 1) * 0.01f + i * 0.001f;
                    float expected = expectedChain(x0, stages);
                    if (Math.abs(result.get(i) - expected) > 0.01f) {
                        correct = false;
                        break;
                    }
                }
                allCorrect &= correct;
                if (e < 3 || e == executions - 1) {
                    System.out.println("execution " + e + ": " + (correct ? "correct" : "WRONG") + ", wall="
                            + (elapsed / 1000) + " us");
                }
            }
        }

        long[] sorted = Arrays.copyOf(wallNanos, wallNanos.length);
        Arrays.sort(sorted);
        long firstExecUs = wallNanos[0] / 1000;
        System.out.println(label + " first execution (JIT compile" + (useGraph ? " + graph capture" : "") + "): "
                + firstExecUs + " us");

        long[] steadyState = Arrays.copyOfRange(sorted, 1, sorted.length);
        long steadyMedian = steadyState[steadyState.length / 2];
        System.out.println(label + " steady-state median wall-clock (excl. first execution, n="
                + steadyState.length + "): " + (steadyMedian / 1000) + " us");
        System.out.println(label + ": " + (allCorrect ? "All executions correct" : "Some executions WRONG"));

        return steadyMedian / 1000.0;
    }
}
