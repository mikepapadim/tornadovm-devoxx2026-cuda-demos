import java.util.Arrays;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * FP16 C = A * B on tensor cores, one warp per 16x16 output tile, with the m16n8k16 fragments
 * loaded straight from global memory (KernelContext.mmaLoadA/mmaLoadB with a HalfFloatArray,
 * TornadoVM PR #1195): no shared memory, no packing, no barrier.
 *
 * <p>The same kernel body runs with two launch geometries, because with no shared memory the
 * only reuse is through the cache, and the launch decides which warps share it:</p>
 * <ol>
 * <li>1D: 256-thread work-groups, the 8 warps of a group along one row of output tiles;</li>
 * <li>2D: 128x4 work-groups, the 16 warps of a group covering a 4x4 block of output tiles.</li>
 * </ol>
 *
 * <pre>
 * tornado --classpath . MmaGlobalLoads 2048 10
 * </pre>
 */
public class MmaGlobalLoads {

    private static final int WARP = 32;

    private static final int TILE = 16;

    /** Fragment loads from global memory; the warp's output tile comes from a 1D launch. */
    public static void mmaGlobal1D(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int warpId = ctx.globalIdx / WARP;
        int tilesPerRow = n / TILE;
        int tileRow = (warpId / tilesPerRow) * TILE;
        int tileCol = (warpId % tilesPerRow) * TILE;
        float[] accLeft = ctx.mmaFragment(0.0f);
        float[] accRight = ctx.mmaFragment(0.0f);
        for (int k = 0; k < n; k += TILE) {
            HalfFloat[] fragA = ctx.mmaLoadA(a, tileRow, k, n);
            HalfFloat[] fragBLeft = ctx.mmaLoadB(b, k, tileCol, n);
            HalfFloat[] fragBRight = ctx.mmaLoadB(b, k, tileCol + 8, n);
            accLeft = ctx.mma(fragA, fragBLeft, accLeft, MMAShape.M16N8K16);
            accRight = ctx.mma(fragA, fragBRight, accRight, MMAShape.M16N8K16);
        }
        ctx.mmaStore(accLeft, c, tileRow, tileCol, n);
        ctx.mmaStore(accRight, c, tileRow, tileCol + 8, n);
    }

    /** The same body; x of the 2D launch counts warps down the rows of C, y the 16-column tiles. */
    public static void mmaGlobal2D(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int tileRow = (ctx.globalIdx / WARP) * TILE;
        int tileCol = ctx.globalIdy * TILE;
        float[] accLeft = ctx.mmaFragment(0.0f);
        float[] accRight = ctx.mmaFragment(0.0f);
        for (int k = 0; k < n; k += TILE) {
            HalfFloat[] fragA = ctx.mmaLoadA(a, tileRow, k, n);
            HalfFloat[] fragBLeft = ctx.mmaLoadB(b, k, tileCol, n);
            HalfFloat[] fragBRight = ctx.mmaLoadB(b, k, tileCol + 8, n);
            accLeft = ctx.mma(fragA, fragBLeft, accLeft, MMAShape.M16N8K16);
            accRight = ctx.mma(fragA, fragBRight, accRight, MMAShape.M16N8K16);
        }
        ctx.mmaStore(accLeft, c, tileRow, tileCol, n);
        ctx.mmaStore(accRight, c, tileRow, tileCol + 8, n);
    }

    private static long median(long[] samples) {
        long[] copy = samples.clone();
        Arrays.sort(copy);
        return copy[copy.length / 2];
    }

    private static long time(TaskGraph graph, String taskName, WorkerGrid grid, int executions) {
        long[] samples = new long[executions - 1];
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler(taskName, grid));
            for (int e = 0; e < executions; e++) {
                long start = System.nanoTime();
                plan.execute();
                if (e > 0) {
                    samples[e - 1] = (System.nanoTime() - start) / 1000;
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return median(samples);
    }

    /** Full-matrix check against a CPU reference over the same FP16-rounded inputs. */
    private static boolean validate(String label, float[] expected, FloatArray actual, float tolerance) {
        int bad = 0;
        float worst = 0.0f;
        for (int i = 0; i < expected.length; i++) {
            float err = Math.abs(expected[i] - actual.get(i));
            worst = Math.max(worst, err);
            if (err > tolerance * Math.max(1.0f, Math.abs(expected[i]))) {
                bad++;
            }
        }
        System.out.printf("   %-30s %s (max abs err %.2e, %d/%d out of tol)%n", label, bad == 0 ? "PASSED" : "FAILED", worst, bad, expected.length);
        return bad == 0;
    }

    public static void main(String[] args) {
        if (TornadoRuntimeProvider.getTornadoRuntime().getBackendType(0) != TornadoVMBackendType.CUDA) {
            System.out.println("This demo requires the CUDA backend.");
            return;
        }
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 2048;
        int executions = args.length > 1 ? Integer.parseInt(args[1]) : 10;
        if (n % 64 != 0 || executions < 2) {
            System.out.printf("n must be a multiple of 64 and executions >= 2; got %d, %d%n", n, executions);
            return;
        }

        HalfFloatArray a = new HalfFloatArray(n * n);
        HalfFloatArray b = new HalfFloatArray(n * n);
        float[] af = new float[n * n];
        float[] bf = new float[n * n];
        java.util.Random random = new java.util.Random(42);
        for (int i = 0; i < n * n; i++) {
            a.set(i, new HalfFloat(random.nextFloat() - 0.5f));
            b.set(i, new HalfFloat(random.nextFloat() - 0.5f));
            af[i] = a.get(i).getFloat32();
            bf[i] = b.get(i).getFloat32();
        }
        float[] expected = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                float av = af[i * n + k];
                for (int j = 0; j < n; j++) {
                    expected[i * n + j] += av * bf[k * n + j];
                }
            }
        }
        // FP16 inputs are exact in FP32; only the FP32 summation order differs from the reference.
        float tol = 1e-4f * (float) Math.sqrt(n);

        FloatArray c1 = new FloatArray(n * n);
        FloatArray c2 = new FloatArray(n * n);
        System.out.printf("FP16 C = A * B on tensor cores, %dx%d, FP32 accumulate, fragments loaded from global memory%n", n, n);
        System.out.printf("  %d executions per launch, steady-state median wall clock (first excluded); kernel time: nsys (README.md)%n%n", executions);

        System.out.println("1. 1D launch, 256-thread groups (1x8 tiles)");
        WorkerGrid1D w1 = new WorkerGrid1D((n / TILE) * (n / TILE) * WARP);
        w1.setLocalWork(8 * WARP, 1, 1);
        long t1 = time(new TaskGraph("g1").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("t", MmaGlobalLoads::mmaGlobal1D, new KernelContext(), a, b, c1, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c1), "g1.t", w1, executions);
        boolean ok = validate("1D launch", expected, c1, tol);

        System.out.println("2. 2D launch, 128x4 groups (4x4 tiles)");
        WorkerGrid2D w2 = new WorkerGrid2D((n / TILE) * WARP, n / TILE);
        w2.setLocalWork(4 * WARP, 4, 1);
        long t2 = time(new TaskGraph("g2").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("t", MmaGlobalLoads::mmaGlobal2D, new KernelContext(), a, b, c2, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c2), "g2.t", w2, executions);
        ok &= validate("2D launch", expected, c2, tol);

        double gflop = 2.0 * n * n * n / 1e9;
        System.out.printf("%n=== Summary (steady-state median wall clock, this run / this GPU) ===%n");
        System.out.printf("%-40s %10s %10s%n", "launch", "median us", "GFLOP/s");
        System.out.printf("%-40s %10d %10.0f%n", "1. 1D, 256-thread groups (1x8 tiles)", t1, gflop / (t1 / 1e6));
        System.out.printf("%-40s %10d %10.0f%n", "2. 2D, 128x4 groups (4x4 tiles)", t2, gflop / (t2 / 1e6));
        System.out.println();
        System.out.println(ok ? "Both launches produced the same, correct result" : "At least one launch FAILED");
    }
}
