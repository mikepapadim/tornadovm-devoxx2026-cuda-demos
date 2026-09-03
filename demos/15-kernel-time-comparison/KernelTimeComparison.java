/*
 * TornadoVM CUDA demo — TornadoVM vs hand-written CUDA, kernel time only.
 *
 * Every other timed demo in this repo reports wall-clock, which on TornadoVM is
 * dominated by host-side dispatch. This one exists to answer the narrower and
 * more interesting question: once a kernel is actually running on the GPU, how
 * does TornadoVM's generated code compare with CUDA written by hand?
 *
 * Three kernels with deliberately different bottlenecks, chained:
 *
 *   1. elementwise  memory-bound   1 read + 1 write per element
 *   2. polynomial   compute-bound  a dependent chain of `degree` FMAs, 1 read + 1 write
 *   3. stencil      memory-bound   3 reads + 1 write per element (neighbour access)
 *
 * Both implementations use the SAME launch configuration (256-thread blocks) and
 * the same kernel names, so an Nsight Systems kernel summary from each can be
 * compared row by row. Measure with `nsys`, not with the wall-clock this program
 * prints -- see the README.
 *
 * Usage:
 *   tornado --classpath . KernelTimeComparison [n] [degree] [executions]
 */
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class KernelTimeComparison {

    /** Matches the CUDA version's block size exactly, so neither side is advantaged. */
    private static final int BLOCK = 256;

    private static final float SCALE = 0.25f;
    private static final float OFFSET = 0.1f;
    private static final float COEF = 0.5f;

    /** Memory-bound: one read, one write, no reuse. */
    public static void elementwise(KernelContext ctx, FloatArray in, FloatArray out) {
        int i = ctx.globalIdx;
        if (i < out.getSize()) {
            out.set(i, in.get(i) * SCALE + OFFSET);
        }
    }

    /**
     * Compute-bound: a dependent chain of `degree` fused multiply-adds on one
     * value. Inputs are kept in [0.1, 0.35] so the recurrence acc = acc*v + COEF
     * converges to COEF/(1-v) -- the result is numerically stable and identical
     * on CPU and GPU regardless of how the compiler contracts mul+add into fma.
     */
    public static void polynomial(KernelContext ctx, FloatArray in, FloatArray out, int degree) {
        int i = ctx.globalIdx;
        if (i < out.getSize()) {
            float v = in.get(i);
            float acc = 1.0f;
            for (int d = 0; d < degree; d++) {
                acc = acc * v + COEF;
            }
            out.set(i, acc);
        }
    }

    /** Memory-bound with neighbour access: three reads, one write. */
    public static void stencil(KernelContext ctx, FloatArray in, FloatArray out, int n) {
        int i = ctx.globalIdx;
        if (i < n) {
            float left = i > 0 ? in.get(i - 1) : in.get(0);
            float right = i < n - 1 ? in.get(i + 1) : in.get(n - 1);
            out.set(i, (left + in.get(i) + right) * (1.0f / 3.0f));
        }
    }

    private static float[] reference(FloatArray input, int n, int degree) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = input.get(i) * SCALE + OFFSET;
        }
        float[] b = new float[n];
        for (int i = 0; i < n; i++) {
            float v = a[i];
            float acc = 1.0f;
            for (int d = 0; d < degree; d++) {
                acc = acc * v + COEF;
            }
            b[i] = acc;
        }
        float[] c = new float[n];
        for (int i = 0; i < n; i++) {
            float left = i > 0 ? b[i - 1] : b[0];
            float right = i < n - 1 ? b[i + 1] : b[n - 1];
            c[i] = (left + b[i] + right) * (1.0f / 3.0f);
        }
        return c;
    }

    private static boolean isCUDABackend() {
        int backendIndex = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice().getBackendIndex();
        return TornadoRuntimeProvider.getTornadoRuntime().getBackendType(backendIndex) == TornadoVMBackendType.CUDA;
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (!isCUDABackend()) {
            System.out.println("This comparison demo targets the CUDA backend.");
            return;
        }

        int n = args.length > 0 ? Integer.parseInt(args[0]) : (1 << 22);
        int degree = args.length > 1 ? Integer.parseInt(args[1]) : 256;
        int executions = args.length > 2 ? Integer.parseInt(args[2]) : 20;

        System.out.printf("Kernel-time comparison: n=%d, polynomial degree=%d, %d executions%n",
                n, degree, executions);
        System.out.printf("  block size %d (identical to the CUDA version)%n", BLOCK);
        System.out.println("  kernels: elementwise (memory-bound) -> polynomial (compute-bound) -> stencil (memory-bound)");
        System.out.println("  NOTE: the wall-clock below includes host dispatch and transfers.");
        System.out.println("        Compare kernel time with nsys -- see this demo's README.");
        System.out.println();

        FloatArray input = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            input.set(i, (float) (i % 1024) / 1024.0f);
        }
        FloatArray tmpA = new FloatArray(n);
        FloatArray tmpB = new FloatArray(n);
        FloatArray output = new FloatArray(n);

        float[] ref = reference(input, n, degree);

        WorkerGrid1D worker = new WorkerGrid1D(n);
        worker.setLocalWork(BLOCK, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("pipeline.elementwise", worker);
        scheduler.addWorkerGrid("pipeline.polynomial", worker);
        scheduler.addWorkerGrid("pipeline.stencil", worker);

        TaskGraph graph = new TaskGraph("pipeline") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input) //
                .task("elementwise", KernelTimeComparison::elementwise, new KernelContext(), input, tmpA) //
                .task("polynomial", KernelTimeComparison::polynomial, new KernelContext(), tmpA, tmpB, degree) //
                .task("stencil", KernelTimeComparison::stencil, new KernelContext(), tmpB, output, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph snapshot = graph.snapshot();
        long[] wallUs = new long[executions];
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            plan.withGridScheduler(scheduler);
            for (int e = 0; e < executions; e++) {
                long start = System.nanoTime();
                plan.execute();
                wallUs[e] = (System.nanoTime() - start) / 1000;
            }
        }

        long[] steady = new long[executions - 1];
        System.arraycopy(wallUs, 1, steady, 0, executions - 1);
        java.util.Arrays.sort(steady);
        System.out.printf("first execution (JIT compile): %d us%n", wallUs[0]);
        System.out.printf("steady-state median wall-clock (n=%d): %d us%n", steady.length, steady[steady.length / 2]);

        float maxAbs = 0.0f;
        int bad = 0;
        for (int i = 0; i < n; i++) {
            float diff = Math.abs(output.get(i) - ref[i]);
            maxAbs = Math.max(maxAbs, diff);
            if (diff > 1e-4f) {
                bad++;
            }
        }
        System.out.printf("validation %s (max abs err %.7f, %d/%d elements out of tol)%n",
                bad == 0 ? "PASSED" : "FAILED", maxAbs, bad, n);
        System.out.println(bad == 0 ? "Result is correct" : "Result is INCORRECT");
    }
}
