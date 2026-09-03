/*
 * TornadoVM CUDA demo — three hand-tuned CUDA optimisations in one Java kernel:
 *
 *   1. ASYNCHRONOUS COPY  (cp.async)  ctx.asyncCopyToLocal / asyncCopyCommit /
 *                                     asyncCopyWaitGroup — stage a tile from
 *                                     global into shared memory without going
 *                                     through registers, then wait on it.
 *   2. SHARED MEMORY                  ctx.allocateIntLocalArray /
 *                                     allocateFloatLocalArray + localBarrier.
 *   3. WARP SHUFFLE      (shfl.sync)  ctx.simdShuffleDown — reduce 32 lanes with
 *                                     register-to-register moves, no shared
 *                                     memory and no barrier inside the warp.
 *
 * The workload is a per-row sum over int8 data — the shape a quantized
 * inference kernel actually has. Each thread block owns one row: it cp.async's
 * the row into shared memory as packed int32 words, unpacks four int8 lanes per
 * word, reduces within each warp by shuffling, then combines the per-warp
 * partials through a small shared-memory array.
 *
 * A plain @Parallel row-sum computes the same result as the baseline, and both
 * are validated against a sequential Java reference.
 *
 * Usage:
 *   tornado --classpath . WarpAsyncSharedReduce [rows rowLen executions]
 *   tornado --printKernel --classpath . WarpAsyncSharedReduce  (see cp.async/shfl in the CUDA source)
 */
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class WarpAsyncSharedReduce {

    private static final int WARP_SIZE = 32;
    /** One block per row. 4 warps: enough to need a cross-warp combine step. */
    private static final int BLOCK = 128;
    private static final int WARPS_PER_BLOCK = BLOCK / WARP_SIZE;

    /**
     * Baseline: one GPU thread per row, reading int8 values straight from global
     * memory — no staging, no shuffle, no shared memory. Uses KernelContext for
     * indexing so the only difference from the optimised kernel is the three
     * techniques themselves, not the programming model.
     */
    private static void rowSumNaive(KernelContext ctx, ByteArray data, FloatArray out, int rowLen) {
        int row = ctx.globalIdx;
        int sum = 0;
        for (int j = 0; j < rowLen; j++) {
            sum += data.get(row * rowLen + j);
        }
        out.set(row, sum);
    }

    /**
     * Optimised: cp.async staging into shared memory + warp-shuffle reduction.
     *
     * rowLen must be a multiple of 4 (int8 values are staged as packed int32 words).
     */
    private static void rowSumOptimised(KernelContext ctx, ByteArray data, FloatArray out, int rowLen) {
        int row = ctx.groupIdx;
        int tid = ctx.localIdx;
        int words = rowLen / 4; // one int32 word holds four int8 values

        // --- 1. shared memory + asynchronous copy -------------------------------
        // Each asyncCopyToLocal call moves exactly 4 bytes (one int32 slot) from
        // global memory. The source offset is in ByteArray *elements* (= bytes).
        // The copies are issued by the whole block, then committed and awaited as
        // one group, so the DMA engine runs them while the threads keep going.
        int[] tile = ctx.allocateIntLocalArray(BLOCK);
        float[] warpPartials = ctx.allocateFloatLocalArray(WARPS_PER_BLOCK);

        int total = 0;
        for (int base = 0; base < words; base += BLOCK) {
            int word = base + tid;
            if (word < words) {
                ctx.asyncCopyToLocal(tile, tid, data, row * rowLen + word * 4);
            }
            ctx.asyncCopyCommit();
            ctx.asyncCopyWaitGroup(0); // wait for this tile before reading it
            ctx.localBarrier();

            if (word < words) {
                // Unpack four sign-extended int8 lanes from the staged word.
                int packed = tile[tid];
                total += (packed << 24) >> 24;
                total += (packed << 16) >> 24;
                total += (packed << 8) >> 24;
                total += packed >> 24;
            }
            ctx.localBarrier(); // tile is reused by the next iteration
        }

        // --- 2. warp shuffle ----------------------------------------------------
        // Reduce the 32 lanes of each warp entirely in registers: no shared memory,
        // no barrier. simdShuffleDown(v, d) returns lane (laneId + d)'s value.
        float value = total;
        for (int delta = WARP_SIZE / 2; delta > 0; delta /= 2) {
            value += ctx.simdShuffleDown(value, delta);
        }

        // --- 3. cross-warp combine through shared memory ------------------------
        int lane = tid % WARP_SIZE;
        int warp = tid / WARP_SIZE;
        if (lane == 0) {
            warpPartials[warp] = value; // lane 0 holds the whole warp's sum
        }
        ctx.localBarrier();

        if (tid == 0) {
            float blockSum = 0.0f;
            for (int i = 0; i < WARPS_PER_BLOCK; i++) {
                blockSum += warpPartials[i];
            }
            out.set(row, blockSum);
        }
    }

    /** Deterministic int8 fill: no RNG, exactly representable, same values every run. */
    private static ByteArray deterministic(int size, int seed) {
        ByteArray arr = new ByteArray(size);
        for (int i = 0; i < size; i++) {
            arr.set(i, (byte) ((i * 31 + seed) % 17 - 8));
        }
        return arr;
    }

    private static float[] reference(ByteArray data, int rows, int rowLen) {
        float[] ref = new float[rows];
        for (int row = 0; row < rows; row++) {
            int sum = 0;
            for (int j = 0; j < rowLen; j++) {
                sum += data.get(row * rowLen + j);
            }
            ref[row] = sum;
        }
        return ref;
    }

    private static boolean validate(String label, FloatArray got, float[] ref) {
        float maxAbs = 0.0f;
        int bad = 0;
        for (int i = 0; i < ref.length; i++) {
            float diff = Math.abs(got.get(i) - ref[i]);
            maxAbs = Math.max(maxAbs, diff);
            if (diff > 1e-3f) {
                bad++;
            }
        }
        boolean ok = bad == 0;
        System.out.printf("  [%-9s] validation %s (max abs err %.5f, %d/%d rows out of tol)%n",
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

    public static void main(String[] args) {
        if (!isCUDABackend()) {
            System.out.println("cp.async and warp shuffle are only supported on the CUDA backend.");
            return;
        }

        int rows = args.length > 0 ? Integer.parseInt(args[0]) : 4096;
        int rowLen = args.length > 1 ? Integer.parseInt(args[1]) : 1024;
        int executions = args.length > 2 ? Integer.parseInt(args[2]) : 20;

        if (rowLen % 4 != 0) {
            System.out.printf("rowLen must be a multiple of 4 (got %d).%n", rowLen);
            return;
        }

        System.out.printf("Warp-shuffle + cp.async + shared-memory row reduction: %d rows x %d int8 values%n",
                rows, rowLen);
        System.out.printf("  block = %d threads (%d warps), one block per row%n", BLOCK, WARPS_PER_BLOCK);
        System.out.printf("  %d executions per kernel, steady-state median reported (first execution excluded)%n%n",
                executions, executions);

        ByteArray data = deterministic(rows * rowLen, 1);
        FloatArray outNaive = new FloatArray(rows);
        FloatArray outOptimised = new FloatArray(rows);
        float[] ref = reference(data, rows, rowLen);

        WorkerGrid1D naiveWorker = new WorkerGrid1D(rows);
        GridScheduler naiveScheduler = new GridScheduler("naive.rowSum", naiveWorker);
        TaskGraph naiveGraph = new TaskGraph("naive") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, data) //
                .task("rowSum", WarpAsyncSharedReduce::rowSumNaive, new KernelContext(), data, outNaive, rowLen) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outNaive);

        WorkerGrid1D worker = new WorkerGrid1D(rows * BLOCK);
        worker.setLocalWork(BLOCK, 1, 1);
        GridScheduler scheduler = new GridScheduler("opt.rowSum", worker);
        TaskGraph optimisedGraph = new TaskGraph("opt") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, data) //
                .task("rowSum", WarpAsyncSharedReduce::rowSumOptimised, new KernelContext(), data, outOptimised, rowLen) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outOptimised);

        long naiveMedian = time("NAIVE (one thread per row, global memory only)",
                new TornadoExecutionPlan(naiveGraph.snapshot()).withGridScheduler(naiveScheduler), executions);
        System.out.println();
        long optimisedMedian = time("OPTIMISED (cp.async + shared + shuffle)",
                new TornadoExecutionPlan(optimisedGraph.snapshot()).withGridScheduler(scheduler), executions);

        System.out.println();
        boolean okNaive = validate("naive", outNaive, ref);
        boolean okOptimised = validate("optimised", outOptimised, ref);

        System.out.println();
        System.out.println("=== Summary (steady-state median us, this run/this GPU) ===");
        System.out.printf("naive     : %d%n", naiveMedian);
        System.out.printf("optimised : %d (%.2fx vs naive)%n",
                optimisedMedian, (double) naiveMedian / optimisedMedian);
        System.out.println(okNaive && okOptimised
                ? "Both kernels produce the same, correct result"
                : "Result is INCORRECT");
    }

    private static long time(String label, TornadoExecutionPlan plan, int executions) {
        System.out.println("=== " + label + " ===");
        long[] samples = new long[executions - 1];
        for (int e = 0; e < executions; e++) {
            long start = System.nanoTime();
            plan.execute();
            long elapsed = (System.nanoTime() - start) / 1000;
            if (e == 0) {
                System.out.printf("  first execution (JIT compile): %d us%n", elapsed);
            } else {
                samples[e - 1] = elapsed;
            }
        }
        long median = medianOf(samples);
        System.out.printf("  steady-state median wall-clock (n=%d): %d us%n", samples.length, median);
        return median;
    }
}
