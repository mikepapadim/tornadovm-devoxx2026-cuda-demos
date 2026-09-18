import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoDeviceTileNotSupported;
import uk.ac.manchester.tornado.api.tile.DType;
import uk.ac.manchester.tornado.api.tile.PartitionView;
import uk.ac.manchester.tornado.api.tile.Tile;
import uk.ac.manchester.tornado.api.tile.TileContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * 24 - CUDA Tile histogram: every tile block accumulates into every bin, atomically.
 *
 * A histogram is the textbook case where GPU work collides: many inputs, few output slots,
 * and every slot contended. In the tile model the contention is the same but the unit is a
 * tile - each tile block reduces its own chunk to a per-bin count and folds that count into
 * the shared bin with {@code PartitionView.atomicAdd}.
 *
 * This is the only demo here that uses the tile API's atomics. Demos 19-23 use none of the
 * ten atomic operations. Without one, the last block to write would overwrite the others and
 * the total would come out as a single chunk's worth instead of the whole input.
 *
 * Bins are chosen by VALUE, so the selection is data-dependent. CUDA Tile has no
 * gather/scatter, so the bin is not an index computed per element - it is a predicate over
 * the whole tile:
 *
 *     hit   = (values >= bin) AND (values < bin + 1)   a PRED tile
 *     count = sum(select(hit, 1, 0))                   a [1,1] tile
 *     bins.atomicAdd(count, 0, bin)                    folded into the shared bin
 *
 * Usage:
 *   tornado --jvm="-Dtornado.recover.bailout=False" --classpath . CuTileHistogram [n] [bins] [iterations]
 *   tornado --printKernel --jvm="-Dtornado.recover.bailout=False" --classpath . CuTileHistogram
 *
 * Always run with -Dtornado.recover.bailout=False. Every TileContext method also has a
 * plain-Java implementation, so with bailout left on (the default) a tile kernel that fails
 * to compile falls back to the host, computes the right answer, and reports "correct" while
 * the GPU did nothing.
 */
public class CuTileHistogram {

    // Compile-time constant, power of two. Never a command-line argument: a tile shape is
    // part of the kernel's type, and a shape passed as a parameter is rejected at sketch time.
    private static final int TILE = 256;

    /**
     * counts[b] = number of values v with b <= v < b+1.
     *
     * One tile block per chunk of TILE values. Each block walks every bin, counts its own
     * chunk's contribution with a predicate, and folds that [1,1] count into the shared bin.
     * Every block targets every bin, so the adds collide by construction.
     */
    public static void histogram(TileContext tc, FloatArray values, FloatArray counts, //
            int n, int bins) {
        PartitionView vv = tc.partition(tc.view(values, 1, n), 1, TILE);
        // The bin row is partitioned 1x1, so a bin index IS a block index.
        PartitionView bv = tc.partition(tc.view(counts, 1, bins), 1, 1);

        Tile ones = tc.full(DType.F32, 1.0, 1, TILE);
        Tile zeros = tc.zeros(DType.F32, 1, TILE);

        Tile chunk = vv.load(0, tc.bidX());
        for (int bin = 0; bin < bins; bin++) {
            Tile atOrAbove = tc.greaterOrEqual(chunk, bin);
            Tile below = tc.lessThan(chunk, bin + 1);
            Tile hit = tc.logicalAnd(atOrAbove, below);
            Tile count = tc.sum(tc.select(hit, ones, zeros), 1);
            bv.atomicAdd(count, 0, bin);
        }
    }

    public static void main(String[] args) {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 1048576;
        int bins = args.length > 1 ? Integer.parseInt(args[1]) : 256;
        int iterations = args.length > 2 ? Integer.parseInt(args[2]) : 20;

        if (n % TILE != 0) {
            System.out.printf("CuTileHistogram: n must be a multiple of %d (got %d)%n", TILE, n);
            return;
        }

        FloatArray values = new FloatArray(n);
        FloatArray counts = new FloatArray(bins);

        // Deterministic and integral: every value is a whole number in [0, bins), so each
        // lands in exactly one bin and every count is an exact integer in fp32. That makes
        // the comparison below == rather than a tolerance, however the adds interleave.
        for (int i = 0; i < n; i++) {
            values.set(i, (i * 31 + 7) % bins);
        }

        TaskGraph graph = new TaskGraph("tile") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, values, counts) //
                .task("histogram", CuTileHistogram::histogram, new TileContext(), values, counts, n, bins) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, counts);

        // One tile block per chunk of TILE values. The grid counts tile blocks, not threads,
        // and the block is pinned to 1x1x1 by CUDATileScheduler -- so no setLocalWork here.
        WorkerGrid1D worker = new WorkerGrid1D(n / TILE);
        GridScheduler grid = new GridScheduler("tile.histogram", worker);

        System.out.printf("CuTileHistogram: %d values into %d bins, tile width %d, %d tile blocks%n",
                n, bins, TILE, n / TILE);
        System.out.printf("CuTileHistogram: every one of the %d blocks folds into every one of the %d bins -- %d contended adds%n",
                n / TILE, bins, (long) (n / TILE) * bins);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(grid);
            // An accumulator has to start at zero, and the host array is what gets uploaded
            // each execution -- so it must be cleared here, not just relied on to be fresh.
            // Without this the bins keep summing across iterations.
            // Probe once; never retry a failed tile launch in a loop.
            for (int b = 0; b < bins; b++) {
                counts.set(b, 0.0f);
            }
            plan.execute();
            for (int i = 1; i < iterations; i++) {
                for (int b = 0; b < bins; b++) {
                    counts.set(b, 0.0f);
                }
                plan.execute();
            }
        } catch (TornadoDeviceTileNotSupported e) {
            System.out.println("[UNSUPPORTED] CuTileHistogram: " + e.getMessage());
            return;
        } catch (Exception e) {
            System.out.println("CuTileHistogram: FAILED -- " + e);
            return;
        }

        long[] expected = new long[bins];
        for (int i = 0; i < n; i++) {
            expected[(int) values.get(i)]++;
        }
        long mismatches = 0;
        long total = 0;
        for (int b = 0; b < bins; b++) {
            total += (long) counts.get(b);
            if (counts.get(b) != expected[b]) {
                mismatches++;
            }
        }

        System.out.printf("CuTileHistogram: bins wrong %d/%d, counted %d of %d values%n",
                mismatches, bins, total, n);
        System.out.println(mismatches == 0 && total == n
                ? "CuTileHistogram: result is correct (every bin exact, all values counted)"
                : "CuTileHistogram: WRONG");
    }
}
