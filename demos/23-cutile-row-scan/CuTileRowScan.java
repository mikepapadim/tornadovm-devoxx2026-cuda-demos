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
 * 23 - CUDA Tile row scan: a per-row prefix sum over a deliberately ragged row.
 *
 * A prefix sum is the GPU primitive that normally costs a lecture -- Blelloch, up-sweep,
 * down-sweep, bank conflicts. Here it is one call: {@code tc.prefixSum(tile, 1)}.
 *
 * The row is 1000 wide against a 128-wide tile, which is where the API's two rules meet:
 * tile SHAPES are compile-time constants, but tensor EXTENTS are runtime. The last tile of
 * every row hangs 24 lanes off the end of the data, so {@code loadMasked} zero-pads it and
 * {@code storeMasked} writes back only the lanes that exist. A [1,1] carry tile rides the
 * loop to join the tiles into one running total.
 *
 * Materially different from demo 17: a scan and a loop-carried reduction, not a matmul,
 * and no tensor cores at all.
 *
 * Usage:
 *   tornado --jvm="-Dtornado.recover.bailout=False" --classpath . CuTileRowScan [rows] [cols] [iterations]
 *   tornado --printKernel --jvm="-Dtornado.recover.bailout=False" --classpath . CuTileRowScan
 *
 * Always run with -Dtornado.recover.bailout=False. Every TileContext method also has a
 * plain-Java implementation, so with bailout left on (the default) a tile kernel that fails
 * to compile falls back to the host, computes the right answer, and reports "correct"
 * while the GPU did nothing.
 */
public class CuTileRowScan {

    // Compile-time constant, power of two. Never a command-line argument.
    private static final int TILE = 128;

    /**
     * out[r][c] = sum of in[r][0..c], and totals[r] = sum of the whole row.
     *
     * One tile block per row. Reductions and scans are rank-2 only, so the tile stays
     * [1,TILE] and the axis is 1.
     */
    public static void rowScan(TileContext tc, FloatArray in, FloatArray out, FloatArray totals, //
            int rows, int cols, int colBlocks) {
        PartitionView iv = tc.partition(tc.view(in, rows, cols), 1, TILE);
        PartitionView ov = tc.partition(tc.view(out, rows, cols), 1, TILE);
        PartitionView tv = tc.partition(tc.view(totals, rows, 1), 1, 1);

        int row = tc.bidX();
        Tile carry = tc.zeros(DType.F32, 1, 1);
        for (int block = 0; block < colBlocks; block++) {
            // Zero-pads the ragged tail, so the padded lanes contribute nothing to either
            // the scan or the running total.
            Tile values = iv.loadMasked(row, block);
            // [1,TILE] + [1,1] broadcasts the carry across the tile implicitly.
            ov.storeMasked(tc.add(tc.prefixSum(values, 1), carry), row, block);
            carry = tc.add(carry, tc.sum(values, 1));
        }
        // A [1,1] reduction cannot be stored back into the [1,TILE] view it came from,
        // so the totals get their own [rows,1] view partitioned 1x1.
        tv.store(carry, row, 0);
    }

    public static void main(String[] args) {
        int rows = args.length > 0 ? Integer.parseInt(args[0]) : 4096;
        int cols = args.length > 1 ? Integer.parseInt(args[1]) : 1000;
        int iterations = args.length > 2 ? Integer.parseInt(args[2]) : 20;
        int colBlocks = (cols + TILE - 1) / TILE;

        FloatArray in = new FloatArray(rows * cols);
        FloatArray out = new FloatArray(rows * cols);
        FloatArray totals = new FloatArray(rows);

        // Small integers, so every prefix is an integer well inside fp32's exact range and
        // the comparison below can be exact rather than tolerance-based.
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                in.set(r * cols + c, ((r + c) % 7) - 3);
            }
        }

        TaskGraph graph = new TaskGraph("tile") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, in) //
                .task("scan", CuTileRowScan::rowScan, new TileContext(), in, out, totals, rows, cols, colBlocks) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out, totals);

        // One tile block per row. The grid counts tile blocks, not threads.
        WorkerGrid1D worker = new WorkerGrid1D(rows);
        GridScheduler grid = new GridScheduler("tile.scan", worker);

        System.out.printf("CuTileRowScan: %d rows x %d cols, tile width %d, %d tiles per row%n",
                rows, cols, TILE, colBlocks);
        System.out.printf("CuTileRowScan: last tile of each row has %d padded lane(s) handled by loadMasked/storeMasked%n",
                colBlocks * TILE - cols);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(grid);
            // Probe once; never retry a failed tile launch in a loop.
            plan.execute();
            for (int i = 1; i < iterations; i++) {
                plan.execute();
            }
        } catch (TornadoDeviceTileNotSupported e) {
            System.out.println("[UNSUPPORTED] CuTileRowScan: " + e.getMessage());
            return;
        } catch (Exception e) {
            System.out.println("CuTileRowScan: FAILED -- " + e);
            return;
        }

        long scanMismatches = 0;
        long totalMismatches = 0;
        for (int r = 0; r < rows; r++) {
            float running = 0.0f;
            for (int c = 0; c < cols; c++) {
                running += in.get(r * cols + c);
                if (out.get(r * cols + c) != running) {
                    scanMismatches++;
                }
            }
            if (totals.get(r) != running) {
                totalMismatches++;
            }
        }

        System.out.printf("CuTileRowScan: scan mismatches %d/%d, row-total mismatches %d/%d%n",
                scanMismatches, (long) rows * cols, totalMismatches, rows);
        System.out.println(scanMismatches == 0 && totalMismatches == 0
                ? "CuTileRowScan: result is correct (bit-exact prefix sums and row totals)"
                : "CuTileRowScan: WRONG");
    }
}
