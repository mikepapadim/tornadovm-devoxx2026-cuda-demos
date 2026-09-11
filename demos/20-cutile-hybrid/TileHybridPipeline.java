/*
 * TornadoVM CUDA demo — a JIT kernel, a CUDA Tile kernel and a cuBLAS call in ONE graph.
 *
 * One TaskGraph, four stages, all on the same device buffers, no host round-trip:
 *   1. KernelContext JIT task ("scale")  — thread-level Java kernel, SIMT path
 *   2. TileContext task ("gemm")         — CUDA Tile kernel, tensor cores via ct::mma
 *   3. cuBLAS library task ("project")   — y = A * x, vendor library on the same stream
 *   4. @Parallel JIT task ("biasRelu")   — thread-level Java kernel again
 *
 * The whole pipeline is then captured into a single CUDA graph with withCUDAGraph() and
 * replayed, which is the claim this demo exists to check: a tile kernel is an ordinary module
 * launch, so it captures and replays like any other task.
 *
 * Every execution is validated against a sequential Java reference over the same
 * FP16-rounded inputs.
 */
import java.util.Random;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.tile.DType;
import uk.ac.manchester.tornado.api.tile.PartitionView;
import uk.ac.manchester.tornado.api.tile.Tile;
import uk.ac.manchester.tornado.api.tile.TileContext;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.enums.CuBlasOperation;

public class TileHybridPipeline {

    private static final int TILE = 32;

    private static final float BIAS = 0.5f;

    /** Stage 1: a thread-level kernel. One thread per element, the SIMT model. */
    public static void scale(KernelContext ctx, FloatArray inOut, float factor) {
        int i = ctx.globalIdx;
        inOut.set(i, inOut.get(i) * factor);
    }

    /**
     * Stage 2: a tile-level kernel. {@code C = A * B} in FP16 with FP32 accumulation, where
     * the only thing said about the hardware is {@code mma}.
     */
    public static void gemm(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int m, int n, int k) {
        PartitionView aView = tc.partition(tc.view(a, m, k), TILE, TILE);
        PartitionView bView = tc.partition(tc.view(b, k, n), TILE, TILE);
        PartitionView cView = tc.partition(tc.view(c, m, n), TILE, TILE);

        int rowBlock = tc.bidX();
        int columnBlock = tc.bidY();
        Tile acc = tc.zeros(DType.F32, TILE, TILE);
        for (int step = 0; step < k / TILE; step++) {
            acc = tc.mma(aView.load(rowBlock, step), bView.load(step, columnBlock), acc);
        }
        cView.store(acc, rowBlock, columnBlock);
    }

    /** Stage 4: a thread-level kernel again, this time from a plain {@code @Parallel} loop. */
    public static void biasRelu(FloatArray inOut) {
        for (@Parallel int i = 0; i < inOut.getSize(); i++) {
            float value = inOut.get(i) + BIAS;
            inOut.set(i, value > 0.0f ? value : 0.0f);
        }
    }

    public static void main(String[] args) {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 256;
        int executions = args.length > 1 ? Integer.parseInt(args[1]) : 20;
        String mode = args.length > 2 ? args[2] : "both";
        if (n % TILE != 0) {
            System.out.println("n must be a multiple of " + TILE);
            return;
        }

        System.out.printf("n = %d, %d executions, mode = %s%n", n, executions, mode);
        System.out.println("pipeline: KernelContext scale -> CUDA Tile gemm -> cuBLAS sgemv -> @Parallel biasRelu");
        System.out.println();

        double withoutGraph = mode.equals("nograph") || mode.equals("both") ? run(n, executions, false) : Double.NaN;
        double withGraph = mode.equals("graph") || mode.equals("both") ? run(n, executions, true) : Double.NaN;

        if (!Double.isNaN(withoutGraph) && !Double.isNaN(withGraph) && withGraph > 0.0) {
            System.out.println();
            System.out.printf("CUDA graph replay: %.3f ms -> %.3f ms per execution, %.2fx faster%n", withoutGraph, withGraph, withoutGraph / withGraph);
            System.out.printf("%.3f ms of host dispatch saved per execution of this 4-stage graph%n", withoutGraph - withGraph);
            System.out.println("The kernels are identical in both modes; what changes is how they are submitted,");
            System.out.println("and the tile task captures and replays like every other task in the graph.");
        }
    }

    /** @return milliseconds per execution, or NaN when the run failed or was wrong */
    private static double run(int n, int executions, boolean cudaGraph) {
        final float factor = 1.5f;

        HalfFloatArray a = new HalfFloatArray(n * n);
        HalfFloatArray b = new HalfFloatArray(n * n);
        FloatArray x = new FloatArray(n);
        FloatArray hidden = new FloatArray(n * n);
        FloatArray projected = new FloatArray(n);

        Random random = new Random(11);
        for (int i = 0; i < n * n; i++) {
            a.set(i, new HalfFloat(random.nextFloat() - 0.5f));
            b.set(i, new HalfFloat(random.nextFloat() - 0.5f));
        }
        float[] xScaled = new float[n];
        for (int i = 0; i < n; i++) {
            x.set(i, random.nextFloat() - 0.5f);
            xScaled[i] = x.get(i) * factor;      // what stage 1 leaves on the device
        }

        WorkerGrid1D scaleWorker = new WorkerGrid1D(n);
        scaleWorker.setLocalWork(64, 1, 1);
        WorkerGrid2D gemmWorker = new WorkerGrid2D(n / TILE, n / TILE);   // tile blocks
        GridScheduler grid = new GridScheduler();
        grid.addWorkerGrid("hybrid.scale", scaleWorker);
        grid.addWorkerGrid("hybrid.gemm", gemmWorker);

        TaskGraph graph = new TaskGraph("hybrid") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b, x) //
                .task("scale", TileHybridPipeline::scale, new KernelContext(), x, factor) //
                .task("gemm", TileHybridPipeline::gemm, new TileContext(), a, b, hidden, n, n, n) //
                .libraryTask("project", CuBlas::cublasSgemv, //
                        CuBlasOperation.CUBLAS_OP_T.operation(), n, n, 1.0f, hidden, n, x, 1, 0.0f, projected, 1) //
                .task("biasRelu", TileHybridPipeline::biasRelu, projected) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, projected);

        String label = cudaGraph ? "withCUDAGraph()" : "no CUDA graph  ";
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(grid);
            if (cudaGraph) {
                plan.withCUDAGraph();
            }
            plan.execute();                                   // warm up: JIT and capture
            long start = System.nanoTime();
            for (int i = 0; i < executions; i++) {
                plan.execute();
            }
            long elapsed = System.nanoTime() - start;

            boolean correct = verify(a, b, xScaled, projected, n);
            double perExecution = elapsed / 1e6 / executions;
            System.out.printf("%s  %8.3f ms/execution   %s%n", label, perExecution, correct ? "correct" : "WRONG");
            return correct ? perExecution : Double.NaN;
        } catch (Exception e) {
            System.out.printf("%s  FAILED: %s%n", label, e);
            return Double.NaN;
        }
    }

    /** The same four stages, sequentially, on the CPU. */
    private static boolean verify(HalfFloatArray a, HalfFloatArray b, float[] xScaled, FloatArray actual, int n) {
        double worst = 0.0;
        double magnitude = 0.0;
        for (int row = 0; row < n; row++) {
            double sum = 0.0;
            for (int column = 0; column < n; column++) {
                double hidden = 0.0;
                for (int inner = 0; inner < n; inner++) {
                    hidden += a.get(row * n + inner).getFloat32() * b.get(inner * n + column).getFloat32();
                }
                sum += hidden * xScaled[column];
            }
            double expected = Math.max(0.0, sum + BIAS);
            magnitude = Math.max(magnitude, Math.abs(expected));
            worst = Math.max(worst, Math.abs(expected - actual.get(row)));
        }
        // FP16 operands accumulated in FP32 over n terms, then summed again over n terms:
        // the tolerance is relative to the magnitude the pipeline actually produces.
        double tolerance = 0.01 * magnitude + 0.01;
        System.out.printf("   max error %.4f of max |value| %.4f (tolerance %.4f)%n", worst, magnitude, tolerance);
        return worst <= tolerance;
    }
}
