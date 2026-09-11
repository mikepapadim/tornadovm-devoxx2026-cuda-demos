/*
 * TornadoVM CUDA demo — flash attention written in Java, compiled through CUDA Tile.
 *
 * Two ways to compute the same attention output:
 *
 *   1. materialised: scores = Q Kt (tile GEMM) -> softmax over rows (tile kernel)
 *                    -> out = P V (tile GEMM). Three kernels and an [S_Q, S_KV]
 *                    score matrix in device memory.
 *   2. flash:        one kernel, one pass over the KV sequence, online softmax.
 *                    The score matrix never exists.
 *
 * Both are validated against a sequential CPU reference over the same FP16-rounded inputs.
 * The comparison is the point: the fused kernel is the reason a tile-level API is worth
 * having, because rewriting it with threads, warps and shared-memory staging is the part
 * nobody wants to do twice.
 */
import java.util.Random;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.tile.DType;
import uk.ac.manchester.tornado.api.tile.PartitionView;
import uk.ac.manchester.tornado.api.tile.Tile;
import uk.ac.manchester.tornado.api.tile.TileContext;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public class TileFlashAttention {

    /** Head dimension, and the full width of every tile, so it is a compile-time constant. */
    private static final int HEAD_DIM = 64;

    private static final int BLOCK_M = 32;

    private static final int BLOCK_N = 32;

    /**
     * Stands in for negative infinity as the initial row maximum. With a true -inf the first
     * rescale is exp(-inf - m) = 0 multiplying a zero accumulator, which is a NaN on some
     * paths; NVIDIA's own TileGym kernels use a large negative constant for the same reason.
     */
    private static final float NEGATIVE_LIMIT = -1.0e30f;

    // ---------------------------------------------------------------------------------------
    // Flash attention: one kernel, one pass, no score matrix.
    // ---------------------------------------------------------------------------------------

    public static void flash(TileContext tc, HalfFloatArray q, HalfFloatArray k, HalfFloatArray v, FloatArray out, int queryRows, int kvRows, int kvBlocks, float scale) {
        PartitionView qView = tc.partition(tc.view(q, queryRows, HEAD_DIM), BLOCK_M, HEAD_DIM);
        PartitionView kView = tc.partition(tc.view(k, kvRows, HEAD_DIM), BLOCK_N, HEAD_DIM);
        PartitionView vView = tc.partition(tc.view(v, kvRows, HEAD_DIM), BLOCK_N, HEAD_DIM);
        PartitionView outView = tc.partition(tc.view(out, queryRows, HEAD_DIM), BLOCK_M, HEAD_DIM);

        int queryBlock = tc.bidX();
        Tile query = qView.load(queryBlock, 0);

        // The three running tiles of the online softmax: row maximum, row denominator and
        // the unnormalised output.
        Tile rowMax = tc.full(DType.F32, NEGATIVE_LIMIT, BLOCK_M, 1);
        Tile rowSum = tc.zeros(DType.F32, BLOCK_M, 1);
        Tile acc = tc.zeros(DType.F32, BLOCK_M, HEAD_DIM);

        for (int block = 0; block < kvBlocks; block++) {
            Tile keys = tc.transpose(kView.load(block, 0));
            Tile scores = tc.scale(tc.mma(query, keys, tc.zeros(DType.F32, BLOCK_M, BLOCK_N)), scale);

            Tile newMax = tc.maximum(rowMax, tc.max(scores, 1));
            Tile probabilities = tc.exp(tc.sub(scores, newMax));
            // When this block raises a row's maximum, everything accumulated so far for that
            // row is rescaled -- that is what makes one pass equal to a two-pass softmax.
            Tile correction = tc.exp(tc.sub(rowMax, newMax));

            rowSum = tc.add(tc.mul(rowSum, correction), tc.sum(probabilities, 1));
            acc = tc.mul(acc, correction);
            acc = tc.mma(tc.cast(probabilities, DType.F16), vView.load(block, 0), acc);
            rowMax = newMax;
        }

        outView.store(tc.div(acc, rowSum), queryBlock, 0);
    }

    // ---------------------------------------------------------------------------------------
    // The materialised path: three kernels and a score matrix in device memory.
    // ---------------------------------------------------------------------------------------

    /** scores = Q Kt * scale, a tiled GEMM with one operand transposed. */
    public static void scores(TileContext tc, HalfFloatArray q, HalfFloatArray k, FloatArray scores, int queryRows, int kvRows, float scale) {
        PartitionView qView = tc.partition(tc.view(q, queryRows, HEAD_DIM), BLOCK_M, HEAD_DIM);
        PartitionView kView = tc.partition(tc.view(k, kvRows, HEAD_DIM), BLOCK_N, HEAD_DIM);
        PartitionView scoreView = tc.partition(tc.view(scores, queryRows, kvRows), BLOCK_M, BLOCK_N);

        int rowBlock = tc.bidX();
        int columnBlock = tc.bidY();
        Tile keys = tc.transpose(kView.load(columnBlock, 0));
        Tile product = tc.mma(qView.load(rowBlock, 0), keys, tc.zeros(DType.F32, BLOCK_M, BLOCK_N));
        scoreView.store(tc.scale(product, scale), rowBlock, columnBlock);
    }

    /**
     * Row softmax over the whole score matrix, one row per tile block, storing FP16 so the
     * second GEMM can consume it without a host round-trip. The row is a single
     * {@code [1, kvRows]} tile, which is why this kernel needs the KV length to be a constant
     * and the fused one does not.
     */
    public static void softmaxRows(TileContext tc, FloatArray scores, HalfFloatArray probabilities, int queryRows) {
        PartitionView scoreView = tc.partition(tc.view(scores, queryRows, MAX_KV), 1, MAX_KV);
        PartitionView outView = tc.partition(tc.view(probabilities, queryRows, MAX_KV), 1, MAX_KV);

        int row = tc.bidX();
        Tile values = scoreView.load(row, 0);
        Tile shifted = tc.exp(tc.sub(values, tc.max(values, 1)));
        outView.store(tc.cast(tc.div(shifted, tc.sum(shifted, 1)), DType.F16), row, 0);
    }

    /** out = P V, a second tiled GEMM. */
    public static void weightedSum(TileContext tc, HalfFloatArray probabilities, HalfFloatArray v, FloatArray out, int queryRows, int kvRows, int kvBlocks) {
        PartitionView pView = tc.partition(tc.view(probabilities, queryRows, kvRows), BLOCK_M, BLOCK_N);
        PartitionView vView = tc.partition(tc.view(v, kvRows, HEAD_DIM), BLOCK_N, HEAD_DIM);
        PartitionView outView = tc.partition(tc.view(out, queryRows, HEAD_DIM), BLOCK_M, HEAD_DIM);

        int rowBlock = tc.bidX();
        Tile acc = tc.zeros(DType.F32, BLOCK_M, HEAD_DIM);
        for (int block = 0; block < kvBlocks; block++) {
            acc = tc.mma(pView.load(rowBlock, block), vView.load(block, 0), acc);
        }
        outView.store(acc, rowBlock, 0);
    }

    /** The KV length the materialised softmax kernel is compiled for. */
    private static final int MAX_KV = 256;

    // ---------------------------------------------------------------------------------------

    public static void main(String[] args) {
        int queryRows = args.length > 0 ? Integer.parseInt(args[0]) : 128;
        int kvRows = args.length > 1 ? Integer.parseInt(args[1]) : MAX_KV;
        int executions = args.length > 2 ? Integer.parseInt(args[2]) : 20;

        if (queryRows % BLOCK_M != 0 || kvRows % BLOCK_N != 0) {
            System.out.printf("queryRows must be a multiple of %d and kvRows of %d%n", BLOCK_M, BLOCK_N);
            return;
        }
        System.out.printf("Q = [%d, %d], K = V = [%d, %d], %d executions%n%n", queryRows, HEAD_DIM, kvRows, HEAD_DIM, executions);

        final float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));
        HalfFloatArray q = randomHalf(queryRows * HEAD_DIM, 101);
        HalfFloatArray k = randomHalf(kvRows * HEAD_DIM, 103);
        HalfFloatArray v = randomHalf(kvRows * HEAD_DIM, 107);

        double[] reference = reference(q, k, v, queryRows, kvRows, scale);

        FloatArray flashOut = new FloatArray(queryRows * HEAD_DIM);
        double flashTime = runFlash(q, k, v, flashOut, queryRows, kvRows, scale, executions);

        double materialisedTime = Double.NaN;
        FloatArray materialisedOut = new FloatArray(queryRows * HEAD_DIM);
        if (kvRows == MAX_KV) {
            materialisedTime = runMaterialised(q, k, v, materialisedOut, queryRows, kvRows, scale, executions);
        } else {
            System.out.printf("materialised path skipped: its softmax kernel is compiled for kvRows = %d%n%n", MAX_KV);
        }

        System.out.printf("%-46s %14s %14s   %s%n", "path", "ms/execution", "max error", "result");
        report("1. materialised (3 kernels + score matrix)", materialisedOut, reference, queryRows, materialisedTime);
        report("2. flash (1 kernel, online softmax)", flashOut, reference, queryRows, flashTime);

        if (!Double.isNaN(materialisedTime) && !Double.isNaN(flashTime)) {
            System.out.println();
            System.out.printf("fusion: %.3f ms -> %.3f ms per execution, %.2fx faster%n", materialisedTime, flashTime, materialisedTime / flashTime);
            System.out.printf("and %d fewer bytes of device traffic: the [%d, %d] score matrix is never%n", queryRows * kvRows * 6, queryRows, kvRows);
            System.out.println("written, read back, or allocated at all. Both paths run entirely on the GPU.");
        }
        System.out.println();
        System.out.println("Both paths are CUDA Tile kernels; neither contains a thread index.");
        System.out.println("See README.md for the generated source and the HMMA count in the cubin.");
    }

    private static double runFlash(HalfFloatArray q, HalfFloatArray k, HalfFloatArray v, FloatArray out, int queryRows, int kvRows, float scale, int executions) {
        WorkerGrid1D worker = new WorkerGrid1D(queryRows / BLOCK_M);
        GridScheduler grid = new GridScheduler("flash.k", worker);
        TaskGraph graph = new TaskGraph("flash") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, q, k, v) //
                .task("k", TileFlashAttention::flash, new TileContext(), q, k, v, out, queryRows, kvRows, kvRows / BLOCK_N, scale) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        return timed(graph, grid, executions, "flash");
    }

    private static double runMaterialised(HalfFloatArray q, HalfFloatArray k, HalfFloatArray v, FloatArray out, int queryRows, int kvRows, float scale, int executions) {
        FloatArray scoreMatrix = new FloatArray(queryRows * kvRows);
        HalfFloatArray probabilities = new HalfFloatArray(queryRows * kvRows);

        // Three tile kernels in ONE graph on shared device buffers: the intermediates never
        // leave the GPU, so the comparison against the fused kernel is a fair one. What the
        // fused kernel saves is the score matrix and two extra launches, not a host copy.
        WorkerGrid2D scoreWorker = new WorkerGrid2D(queryRows / BLOCK_M, kvRows / BLOCK_N);
        WorkerGrid1D softmaxWorker = new WorkerGrid1D(queryRows);
        WorkerGrid1D sumWorker = new WorkerGrid1D(queryRows / BLOCK_M);
        GridScheduler grid = new GridScheduler();
        grid.addWorkerGrid("materialised.scores", scoreWorker);
        grid.addWorkerGrid("materialised.softmax", softmaxWorker);
        grid.addWorkerGrid("materialised.sum", sumWorker);

        TaskGraph graph = new TaskGraph("materialised") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, q, k, v) //
                .task("scores", TileFlashAttention::scores, new TileContext(), q, k, scoreMatrix, queryRows, kvRows, scale) //
                .task("softmax", TileFlashAttention::softmaxRows, new TileContext(), scoreMatrix, probabilities, queryRows) //
                .task("sum", TileFlashAttention::weightedSum, new TileContext(), probabilities, v, out, queryRows, kvRows, kvRows / BLOCK_N) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        return timed(graph, grid, executions, "materialised");
    }

    private static double timed(TaskGraph graph, GridScheduler grid, int executions, String label) {
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(grid).execute();     // warm up: JIT, nvcc, first transfer
            long start = System.nanoTime();
            for (int i = 0; i < executions; i++) {
                plan.execute();
            }
            return (System.nanoTime() - start) / 1e6 / executions;
        } catch (Exception e) {
            System.out.println("[" + label + "] FAILED: " + e);
            return Double.NaN;
        }
    }

    private static HalfFloatArray randomHalf(int elements, long seed) {
        HalfFloatArray array = new HalfFloatArray(elements);
        Random random = new Random(seed);
        for (int i = 0; i < elements; i++) {
            array.set(i, new HalfFloat(random.nextFloat() - 0.5f));
        }
        return array;
    }

    /** Two-pass softmax attention on the CPU, over the FP16 values the GPU sees. */
    private static double[] reference(HalfFloatArray q, HalfFloatArray k, HalfFloatArray v, int queryRows, int kvRows, float scale) {
        double[] result = new double[queryRows * HEAD_DIM];
        for (int row = 0; row < queryRows; row++) {
            double[] scores = new double[kvRows];
            double maximum = Double.NEGATIVE_INFINITY;
            for (int key = 0; key < kvRows; key++) {
                double dot = 0.0;
                for (int d = 0; d < HEAD_DIM; d++) {
                    dot += q.get(row * HEAD_DIM + d).getFloat32() * k.get(key * HEAD_DIM + d).getFloat32();
                }
                scores[key] = dot * scale;
                maximum = Math.max(maximum, scores[key]);
            }
            double denominator = 0.0;
            for (int key = 0; key < kvRows; key++) {
                scores[key] = Math.exp(scores[key] - maximum);
                denominator += scores[key];
            }
            for (int key = 0; key < kvRows; key++) {
                double weight = scores[key] / denominator;
                for (int d = 0; d < HEAD_DIM; d++) {
                    result[row * HEAD_DIM + d] += weight * v.get(key * HEAD_DIM + d).getFloat32();
                }
            }
        }
        return result;
    }

    private static void report(String label, FloatArray out, double[] reference, int queryRows, double milliseconds) {
        if (Double.isNaN(milliseconds)) {
            System.out.printf("%-46s %14s%n", label, "not run");
            return;
        }
        double worst = 0.0;
        for (int i = 0; i < queryRows * HEAD_DIM; i++) {
            worst = Math.max(worst, Math.abs(out.get(i) - reference[i]));
        }
        System.out.printf("%-46s %14.3f %14.4f   %s%n", label, milliseconds, worst, worst <= 0.01 ? "correct" : "WRONG");
    }
}
