import java.util.Random;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.tile.DType;
import uk.ac.manchester.tornado.api.tile.PartitionView;
import uk.ac.manchester.tornado.api.tile.Tile;
import uk.ac.manchester.tornado.api.tile.TileContext;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * The same FP16 GEMM written three ways, in increasing order of how much the hardware is
 * hidden from the author:
 *
 * <ol>
 * <li>{@code @Parallel} — one thread per output element, no tiles at all.</li>
 * <li>{@code KernelContext} tiled — shared-memory tiles the author allocates, indexes and
 *     barriers by hand. This is the thread-level model: every index is a thread's business.</li>
 * <li>{@code TileContext} — a tile is the unit of work. The kernel says
 *     {@code acc = tc.mma(a, b, acc)} and nothing about threads, warps, fragments, shared
 *     memory or layout. It compiles through NVIDIA CUDA Tile, and {@code tileiras} picks the
 *     tensor-core instruction.</li>
 * </ol>
 *
 * All three compute the same {@code C = A * B} and are validated against a CPU reference
 * computed over the same FP16-rounded inputs.
 *
 * <p>
 * The point of the demo is what {@code --printKernel} shows: rung 3 emits
 * {@code extern "C" __tile_global__} with {@code ct::partition_view} / {@code ct::mma} calls
 * and no inline PTX, and its cubin still disassembles to {@code HMMA.16816.F32}.
 * </p>
 *
 * Usage:
 *   tornado --classpath . TileMatMul [n] [executions]
 *   tornado --printKernel --classpath . TileMatMul 128 1
 */
public class TileMatMul {

    /** Tile shape. A CUDA Tile shape is part of the kernel's type, so it must be a constant. */
    private static final int TILE_M = 32;

    private static final int TILE_N = 32;

    private static final int TILE_K = 32;

    /** Thread-block tile for the KernelContext rung; the same 32x32 geometry. */
    private static final int BLOCK = 32;

    // ---------------------------------------------------------------------------------------
    // Rung 1: one thread per output element.
    // ---------------------------------------------------------------------------------------

    public static void naive(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n, int k) {
        int row = ctx.globalIdx;
        int column = ctx.globalIdy;
        float sum = 0.0f;
        for (int inner = 0; inner < k; inner++) {
            sum += a.get(row * k + inner).getFloat32() * b.get(inner * n + column).getFloat32();
        }
        c.set(row * n + column, sum);
    }

    // ---------------------------------------------------------------------------------------
    // Rung 2: shared-memory tiles, by hand.
    // ---------------------------------------------------------------------------------------

    public static void tiledByHand(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n, int k) {
        int localRow = ctx.localIdx;
        int localColumn = ctx.localIdy;
        int row = ctx.groupIdx * BLOCK + localRow;
        int column = ctx.groupIdy * BLOCK + localColumn;

        float[] aTile = ctx.allocateFloatLocalArray(BLOCK * BLOCK);
        float[] bTile = ctx.allocateFloatLocalArray(BLOCK * BLOCK);

        float sum = 0.0f;
        for (int step = 0; step < k / BLOCK; step++) {
            aTile[localRow * BLOCK + localColumn] = a.get(row * k + step * BLOCK + localColumn).getFloat32();
            bTile[localRow * BLOCK + localColumn] = b.get((step * BLOCK + localRow) * n + column).getFloat32();
            ctx.localBarrier();
            for (int inner = 0; inner < BLOCK; inner++) {
                sum += aTile[localRow * BLOCK + inner] * bTile[inner * BLOCK + localColumn];
            }
            ctx.localBarrier();
        }
        c.set(row * n + column, sum);
    }

    // ---------------------------------------------------------------------------------------
    // Rung 3: tiles as the unit of work. This is the whole kernel.
    // ---------------------------------------------------------------------------------------

    public static void tiles(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int m, int n, int k) {
        PartitionView aView = tc.partition(tc.view(a, m, k), TILE_M, TILE_K);
        PartitionView bView = tc.partition(tc.view(b, k, n), TILE_K, TILE_N);
        PartitionView cView = tc.partition(tc.view(c, m, n), TILE_M, TILE_N);

        int rowBlock = tc.bidX();
        int columnBlock = tc.bidY();

        Tile acc = tc.zeros(DType.F32, TILE_M, TILE_N);
        for (int step = 0; step < k / TILE_K; step++) {
            acc = tc.mma(aView.load(rowBlock, step), bView.load(step, columnBlock), acc);
        }
        cView.store(acc, rowBlock, columnBlock);
    }

    // ---------------------------------------------------------------------------------------

    public static void main(String[] args) {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 256;
        int executions = args.length > 1 ? Integer.parseInt(args[1]) : 10;
        if (n % TILE_M != 0 || n % BLOCK != 0) {
            System.out.println("n must be a multiple of " + Math.max(TILE_M, BLOCK));
            return;
        }
        System.out.printf("n = %d, %d executions per rung%n%n", n, executions);

        HalfFloatArray a = new HalfFloatArray(n * n);
        HalfFloatArray b = new HalfFloatArray(n * n);
        Random random = new Random(7);
        for (int i = 0; i < n * n; i++) {
            a.set(i, new HalfFloat(random.nextFloat() - 0.5f));
            b.set(i, new HalfFloat(random.nextFloat() - 0.5f));
        }

        // Reference over the FP16-rounded inputs the GPU actually sees, so the comparison
        // measures the kernels and not the quantisation.
        float[] reference = new float[n * n];
        for (int row = 0; row < n; row++) {
            for (int column = 0; column < n; column++) {
                float sum = 0.0f;
                for (int inner = 0; inner < n; inner++) {
                    sum += a.get(row * n + inner).getFloat32() * b.get(inner * n + column).getFloat32();
                }
                reference[row * n + column] = sum;
            }
        }

        FloatArray naiveOut = new FloatArray(n * n);
        FloatArray handOut = new FloatArray(n * n);
        FloatArray tileOut = new FloatArray(n * n);

        long naiveTime = runKernelContext("naive", false, a, b, naiveOut, n, executions, 1);
        long handTime = runKernelContext("hand", true, a, b, handOut, n, executions, BLOCK);
        long tileTime = runTiles(a, b, tileOut, n, executions);

        System.out.println();
        System.out.printf("%-42s %14s %12s %11s   %s%n", "rung", "ms/execution", "vs naive", "max error", "result");
        report("1. @Parallel-style naive (KernelContext)", naiveOut, reference, n, naiveTime, executions, naiveTime);
        report("2. KernelContext tiled (shared memory)", handOut, reference, n, handTime, executions, naiveTime);
        report("3. TileContext (CUDA Tile, ct::mma)", tileOut, reference, n, tileTime, executions, naiveTime);

        System.out.println();
        System.out.println("Wall clock above includes host dispatch, which at these sizes is most of it:");
        System.out.println("all three rungs submit the same number of launches, so the column compares");
        System.out.println("dispatch as much as kernels. For kernel time use nsys (see README.md), and for");
        System.out.println("what actually differs use --printKernel: rung 3 has no thread indices at all.");
    }

    /**
     * The kernel is selected with a flag rather than passed in as a functional-interface
     * value: a method reference wrapped in another lambda has no bytecode the sketcher can
     * read, and it fails with "Cannot read the array length because this.code is null". A
     * task method reference has to reach {@code .task(...)} directly.
     */
    private static long runKernelContext(String name, boolean tiled, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n, int executions, int localSize) {
        WorkerGrid2D worker = new WorkerGrid2D(n, n);
        worker.setLocalWork(localSize, localSize, 1);
        GridScheduler grid = new GridScheduler(name + ".k", worker);
        TaskGraph graph = new TaskGraph(name) //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b);
        if (tiled) {
            graph.task("k", TileMatMul::tiledByHand, new KernelContext(), a, b, c, n, n);
        } else {
            graph.task("k", TileMatMul::naive, new KernelContext(), a, b, c, n, n);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(grid).execute();     // warm up: JIT plus first transfer
            long start = System.nanoTime();
            for (int i = 0; i < executions; i++) {
                plan.execute();
            }
            return System.nanoTime() - start;
        } catch (Exception e) {
            System.out.println("[" + name + "] FAILED: " + e.getMessage());
            return -1;
        }
    }

    private static long runTiles(HalfFloatArray a, HalfFloatArray b, FloatArray c, int n, int executions) {
        // The worker grid counts TILE BLOCKS, not threads: one point per output tile. The
        // runtime pins the thread block to 1x1x1 for a tile task, which is what CUDA Tile
        // requires -- the tile compiler decides how many threads actually run.
        WorkerGrid2D worker = new WorkerGrid2D(n / TILE_M, n / TILE_N);
        GridScheduler grid = new GridScheduler("tile.k", worker);
        TaskGraph graph = new TaskGraph("tile") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("k", TileMatMul::tiles, new TileContext(), a, b, c, n, n, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(grid).execute();
            long start = System.nanoTime();
            for (int i = 0; i < executions; i++) {
                plan.execute();
            }
            return System.nanoTime() - start;
        } catch (Exception e) {
            System.out.println("[tile] FAILED: " + e.getMessage());
            return -1;
        }
    }

    private static void report(String label, FloatArray out, float[] reference, int n, long elapsed, int executions, long baseline) {
        if (elapsed < 0) {
            System.out.printf("%-42s %14s%n", label, "did not run");
            return;
        }
        double worst = 0.0;
        for (int i = 0; i < n * n; i++) {
            worst = Math.max(worst, Math.abs(out.get(i) - reference[i]));
        }
        // FP16 operands with FP32 accumulation over n terms: the tolerance grows with n.
        double tolerance = 0.02 * Math.sqrt(n);
        String speedup = baseline > 0 ? String.format("%.2fx", (double) baseline / elapsed) : "-";
        System.out.printf("%-42s %14.3f %12s %11.4f   %s%n", label, elapsed / 1e6 / executions, speedup, worst, worst <= tolerance ? "correct" : "WRONG");
    }
}
