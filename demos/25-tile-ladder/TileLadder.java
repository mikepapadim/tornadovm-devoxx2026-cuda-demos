/*
 * TornadoVM CUDA demo — the TileContext ladder, against a fully optimised KernelContext
 * GEMM and native CUDA Tile.
 *
 * The same FP16 C = A * B (FP32 accumulate, FP32 out) written three ways from Java:
 *
 *   1. KernelContext, simple      one warp per 16x16 output tile (demo 22's rung 3)
 *   2. KernelContext, optimised   128x128 block tile, 8 warps x 64x32 warp tiles,
 *                                 16 tensor-core fragments per warp, cp.async
 *                                 double-buffered shared-memory staging
 *   3-6. TileContext              the same ten-line tile kernel at four tile shapes
 *   7. TileContext + hint         rung 6 compiled with the CUDA Tile launch hint
 *                                 occupancy=2 -- the other knob a tile kernel has
 *   8. cuBLAS GemmEx FP16->FP32   the vendor ceiling
 *
 * Rung 2 is the most a KernelContext kernel can do here: warp arithmetic, fragment
 * offsets and a hand-built pipeline. Rungs 3-7 never name a thread. The tile shape alone
 * is worth ~5.5x in kernel time, and with the launch hint the ten-line tile kernel
 * edges past rung 2. See README.md for the measured numbers.
 *
 * The hand-written counterpart, TileLadder.cu, runs the same tile kernels in native CUDA
 * Tile C++. Written the idiomatic way they are up to 4.5x SLOWER than the Java rungs:
 * TornadoVM compiles after `n` is known, so its generated tile code carries constant
 * extents, an alignment promise and a fully unrolled k-loop. Given the same three facts,
 * native CUDA Tile matches TileContext exactly. See README.md.
 *
 * Requires an SDK with the CUDA Tile API (TornadoVM 7.0.0+), CUDA Toolkit 13.3+ for the
 * tile path, and a GPU with tensor cores.
 *
 * Usage:
 *   tornado --classpath . TileLadder [n] [executions]      # n: multiple of 128, default 2048
 */
import java.util.Arrays;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoDeviceTileNotSupported;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.tile.DType;
import uk.ac.manchester.tornado.api.tile.PartitionView;
import uk.ac.manchester.tornado.api.tile.Tile;
import uk.ac.manchester.tornado.api.tile.TileContext;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.enums.CuBlasOperation;

public class TileLadder {

    private static final int WARP = 32;

    // ================================================================ rung 1

    /**
     * One warp per 16x16 output tile, two m16n8k16 calls per k-step, operands packed from
     * global memory by the warp itself. Tensor cores, but no reuse across warps and one
     * load in flight at a time. This is demo 22's rung 3, unchanged.
     */
    public static void kcSimple(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int warpId = ctx.groupIdx;
        int lane = ctx.localIdx;
        int tilesPerRow = n / 16;
        int tileRow = (warpId / tilesPerRow) * 16;
        int tileCol = (warpId % tilesPerRow) * 16;

        int[] aTile = ctx.allocateIntLocalArray(128);
        int[] bTile0 = ctx.allocateIntLocalArray(64);
        int[] bTile1 = ctx.allocateIntLocalArray(64);
        float[] accLeft = ctx.mmaFragment(0.0f);
        float[] accRight = ctx.mmaFragment(0.0f);

        for (int kBase = 0; kBase < n; kBase += 16) {
            for (int idx = lane; idx < 128; idx += WARP) {
                int elem = idx * 2;
                int r = elem / 16;
                int kk = elem % 16;
                int g = (tileRow + r) * n + kBase + kk;
                int lo = a.get(g).getHalfFloatValue() & 0xFFFF;
                int hi = a.get(g + 1).getHalfFloatValue() & 0xFFFF;
                aTile[r * 8 + kk / 2] = lo | (hi << 16);
            }
            for (int idx = lane; idx < 64; idx += WARP) {
                int kRow = idx / 4;
                int jPair = idx % 4;
                int gl = (kBase + kRow) * n + tileCol + jPair * 2;
                bTile0[kRow * 4 + jPair] = (b.get(gl).getHalfFloatValue() & 0xFFFF) | ((b.get(gl + 1).getHalfFloatValue() & 0xFFFF) << 16);
                int gr = gl + 8;
                bTile1[kRow * 4 + jPair] = (b.get(gr).getHalfFloatValue() & 0xFFFF) | ((b.get(gr + 1).getHalfFloatValue() & 0xFFFF) << 16);
            }
            ctx.localBarrier();
            HalfFloat[] fragA = ctx.mmaLoadA(aTile, 16);
            HalfFloat[] fragB0 = ctx.mmaLoadB(bTile0, 16);
            accLeft = ctx.mma(fragA, fragB0, accLeft, MMAShape.M16N8K16);
            HalfFloat[] fragB1 = ctx.mmaLoadB(bTile1, 16);
            accRight = ctx.mma(fragA, fragB1, accRight, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        ctx.mmaStore(accLeft, c, tileRow, tileCol, n);
        ctx.mmaStore(accRight, c, tileRow, tileCol + 8, n);
    }

    // ================================================================ rung 2

    // Block tile 128x128, k-step 16. 8 warps as 2 rows x 4 columns, so each warp owns a
    // 64x32 region: 4 A fragments (m16) x 4 B fragments (n8) = 16 mma per k-step.
    // Chosen by measurement over 11 tilings -- see README.md "How rung 2 was tuned".
    private static final int OPT_BM = 128;
    private static final int OPT_BN = 128;
    private static final int OPT_THREADS = 256;
    // Shared memory per pipeline stage, in ints (two fp16 per int):
    //   A: 128 rows x 16 k   = 1024 ints, rows of 32 bytes -- the layout mmaLoadA expects.
    //   B: 16 k x 128 cols   = 1024 ints, as 16 panels of 16x8 (256 bytes each) -- the
    //      layout mmaLoadB expects; panel p holds columns 8p..8p+7.
    private static final int OPT_STAGE_INTS = 1024;

    /**
     * The fastest KernelContext GEMM found for this GPU. What it does that rung 1 does not:
     * <ul>
     * <li>a 128x128 output block shared by 8 warps, so every staged operand byte feeds
     * many tensor-core instructions instead of one;</li>
     * <li>16 accumulator fragments per warp, held in registers for the whole k-loop;</li>
     * <li>operands staged with {@code cp.async} straight into shared memory, coalesced
     * across the block, and double-buffered: the next k-tile is in flight while the
     * current one is being multiplied;</li>
     * <li>fragments read out of the shared tiles by byte offset ({@code mmaLoadA/B} with a
     * third argument), which TornadoVM lowers to {@code ldmatrix}.</li>
     * </ul>
     * The fragments are separate variables because a fragment is a value, not an array
     * element: there is no array-of-fragments in the KernelContext API.
     */
    public static void kcOptimised(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int tid = ctx.localIdx;
        int blocksPerRow = n / OPT_BN;
        int blockRow = (ctx.groupIdx / blocksPerRow) * OPT_BM;
        int blockCol = (ctx.groupIdx % blocksPerRow) * OPT_BN;
        int warp = tid / WARP;
        int warpRow = warp / 4;          // 0..1: which 64-row band
        int warpCol = warp % 4;          // 0..3: which 32-column band

        // Two stages each, back to back: stage s starts at int s * OPT_STAGE_INTS.
        int[] aSmem = ctx.allocateIntLocalArray(2 * OPT_STAGE_INTS);
        int[] bSmem = ctx.allocateIntLocalArray(2 * OPT_STAGE_INTS);

        float[] acc00 = ctx.mmaFragment(0.0f);
        float[] acc01 = ctx.mmaFragment(0.0f);
        float[] acc02 = ctx.mmaFragment(0.0f);
        float[] acc03 = ctx.mmaFragment(0.0f);
        float[] acc10 = ctx.mmaFragment(0.0f);
        float[] acc11 = ctx.mmaFragment(0.0f);
        float[] acc12 = ctx.mmaFragment(0.0f);
        float[] acc13 = ctx.mmaFragment(0.0f);
        float[] acc20 = ctx.mmaFragment(0.0f);
        float[] acc21 = ctx.mmaFragment(0.0f);
        float[] acc22 = ctx.mmaFragment(0.0f);
        float[] acc23 = ctx.mmaFragment(0.0f);
        float[] acc30 = ctx.mmaFragment(0.0f);
        float[] acc31 = ctx.mmaFragment(0.0f);
        float[] acc32 = ctx.mmaFragment(0.0f);
        float[] acc33 = ctx.mmaFragment(0.0f);

        int numK = n / 16;

        // Prologue: k-tile 0 into stage 0. 256 threads x 4 copies = 1024 ints each for A and B.
        // A: consecutive threads take consecutive 4-byte pairs along a row of A -- coalesced.
        // B: consecutive threads take consecutive pairs along a row of B, scattered into panels.
        for (int q = 0; q < 4; q++) {
            int idx = tid + q * OPT_THREADS;
            ctx.asyncCopyToLocal(aSmem, idx, a, (blockRow + idx / 8) * n + (idx % 8) * 2);
            int kRow = idx / 64;
            int rem = idx % 64;
            ctx.asyncCopyToLocal(bSmem, (rem / 4) * 64 + kRow * 4 + (rem % 4), b, kRow * n + blockCol + rem * 2);
        }
        ctx.asyncCopyCommit();

        for (int kt = 0; kt < numK; kt++) {
            int cur = kt % 2;
            if (kt + 1 < numK) {
                // Issue k-tile kt+1 into the other stage, then wait only for k-tile kt.
                int next = (1 - cur) * OPT_STAGE_INTS;
                int kNext = (kt + 1) * 16;
                for (int q = 0; q < 4; q++) {
                    int idx = tid + q * OPT_THREADS;
                    ctx.asyncCopyToLocal(aSmem, next + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                    int kRow = idx / 64;
                    int rem = idx % 64;
                    ctx.asyncCopyToLocal(bSmem, next + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(1);
            } else {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();

            // Byte offsets into the current stage: A rows are 32 bytes, B panels 256 bytes.
            int aBase = cur * OPT_STAGE_INTS * 4 + warpRow * 64 * 32;
            int bBase = cur * OPT_STAGE_INTS * 4 + warpCol * 4 * 256;
            HalfFloat[] fa0 = ctx.mmaLoadA(aSmem, 16, aBase);
            HalfFloat[] fa1 = ctx.mmaLoadA(aSmem, 16, aBase + 16 * 32);
            HalfFloat[] fa2 = ctx.mmaLoadA(aSmem, 16, aBase + 32 * 32);
            HalfFloat[] fa3 = ctx.mmaLoadA(aSmem, 16, aBase + 48 * 32);
            HalfFloat[] fb0 = ctx.mmaLoadB(bSmem, 16, bBase);
            HalfFloat[] fb1 = ctx.mmaLoadB(bSmem, 16, bBase + 256);
            HalfFloat[] fb2 = ctx.mmaLoadB(bSmem, 16, bBase + 512);
            HalfFloat[] fb3 = ctx.mmaLoadB(bSmem, 16, bBase + 768);

            acc00 = ctx.mma(fa0, fb0, acc00, MMAShape.M16N8K16);
            acc01 = ctx.mma(fa0, fb1, acc01, MMAShape.M16N8K16);
            acc02 = ctx.mma(fa0, fb2, acc02, MMAShape.M16N8K16);
            acc03 = ctx.mma(fa0, fb3, acc03, MMAShape.M16N8K16);
            acc10 = ctx.mma(fa1, fb0, acc10, MMAShape.M16N8K16);
            acc11 = ctx.mma(fa1, fb1, acc11, MMAShape.M16N8K16);
            acc12 = ctx.mma(fa1, fb2, acc12, MMAShape.M16N8K16);
            acc13 = ctx.mma(fa1, fb3, acc13, MMAShape.M16N8K16);
            acc20 = ctx.mma(fa2, fb0, acc20, MMAShape.M16N8K16);
            acc21 = ctx.mma(fa2, fb1, acc21, MMAShape.M16N8K16);
            acc22 = ctx.mma(fa2, fb2, acc22, MMAShape.M16N8K16);
            acc23 = ctx.mma(fa2, fb3, acc23, MMAShape.M16N8K16);
            acc30 = ctx.mma(fa3, fb0, acc30, MMAShape.M16N8K16);
            acc31 = ctx.mma(fa3, fb1, acc31, MMAShape.M16N8K16);
            acc32 = ctx.mma(fa3, fb2, acc32, MMAShape.M16N8K16);
            acc33 = ctx.mma(fa3, fb3, acc33, MMAShape.M16N8K16);

            ctx.localBarrier();   // nobody refills this stage until every warp is done with it
        }

        int row0 = blockRow + warpRow * 64;
        int col0 = blockCol + warpCol * 32;
        ctx.mmaStore(acc00, c, row0, col0, n);
        ctx.mmaStore(acc01, c, row0, col0 + 8, n);
        ctx.mmaStore(acc02, c, row0, col0 + 16, n);
        ctx.mmaStore(acc03, c, row0, col0 + 24, n);
        ctx.mmaStore(acc10, c, row0 + 16, col0, n);
        ctx.mmaStore(acc11, c, row0 + 16, col0 + 8, n);
        ctx.mmaStore(acc12, c, row0 + 16, col0 + 16, n);
        ctx.mmaStore(acc13, c, row0 + 16, col0 + 24, n);
        ctx.mmaStore(acc20, c, row0 + 32, col0, n);
        ctx.mmaStore(acc21, c, row0 + 32, col0 + 8, n);
        ctx.mmaStore(acc22, c, row0 + 32, col0 + 16, n);
        ctx.mmaStore(acc23, c, row0 + 32, col0 + 24, n);
        ctx.mmaStore(acc30, c, row0 + 48, col0, n);
        ctx.mmaStore(acc31, c, row0 + 48, col0 + 8, n);
        ctx.mmaStore(acc32, c, row0 + 48, col0 + 16, n);
        ctx.mmaStore(acc33, c, row0 + 48, col0 + 24, n);
    }

    // ================================================================ rungs 3-6

    // A tile shape is part of a tile kernel's type: it must be a compile-time constant, so
    // each shape is its own method. The body is identical in all four.

    private static final int T32_M = 32, T32_N = 32, T32_K = 32;

    public static void tile32x32x32(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T32_M, T32_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T32_K, T32_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T32_M, T32_N);
        int rowBlock = tc.bidX();
        int columnBlock = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T32_M, T32_N);
        for (int k = 0; k < n / T32_K; k++) {
            acc = tc.mma(av.load(rowBlock, k), bv.load(k, columnBlock), acc);
        }
        cv.store(acc, rowBlock, columnBlock);
    }

    private static final int T64_M = 64, T64_N = 64, T64_K = 64;

    public static void tile64x64x64(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T64_M, T64_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T64_K, T64_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T64_M, T64_N);
        int rowBlock = tc.bidX();
        int columnBlock = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T64_M, T64_N);
        for (int k = 0; k < n / T64_K; k++) {
            acc = tc.mma(av.load(rowBlock, k), bv.load(k, columnBlock), acc);
        }
        cv.store(acc, rowBlock, columnBlock);
    }

    private static final int T128K32_M = 128, T128K32_N = 128, T128K32_K = 32;

    public static void tile128x128x32(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T128K32_M, T128K32_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T128K32_K, T128K32_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T128K32_M, T128K32_N);
        int rowBlock = tc.bidX();
        int columnBlock = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T128K32_M, T128K32_N);
        for (int k = 0; k < n / T128K32_K; k++) {
            acc = tc.mma(av.load(rowBlock, k), bv.load(k, columnBlock), acc);
        }
        cv.store(acc, rowBlock, columnBlock);
    }

    private static final int T128K64_M = 128, T128K64_N = 128, T128K64_K = 64;

    public static void tile128x128x64(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T128K64_M, T128K64_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T128K64_K, T128K64_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T128K64_M, T128K64_N);
        int rowBlock = tc.bidX();
        int columnBlock = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T128K64_M, T128K64_N);
        for (int k = 0; k < n / T128K64_K; k++) {
            acc = tc.mma(av.load(rowBlock, k), bv.load(k, columnBlock), acc);
        }
        cv.store(acc, rowBlock, columnBlock);
    }

    /**
     * Rung 6's kernel, byte for byte, under its own name. Rung 7 compiles it with a launch
     * hint; a separate method guarantees it cannot reuse rung 6's cached cubin.
     */
    public static void tile128x128x64Hinted(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T128K64_M, T128K64_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T128K64_K, T128K64_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T128K64_M, T128K64_N);
        int rowBlock = tc.bidX();
        int columnBlock = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T128K64_M, T128K64_N);
        for (int k = 0; k < n / T128K64_K; k++) {
            acc = tc.mma(av.load(rowBlock, k), bv.load(k, columnBlock), acc);
        }
        cv.store(acc, rowBlock, columnBlock);
    }

    /**
     * CUDA Tile launch hint for rung 7. TornadoVM reads -Dtornado.cuda.tile.hints each time
     * it compiles a tile kernel and puts it on the kernel as
     * [[ using cutile : hint(0, occupancy=2) ]]. The keys are not validated: a misspelt key
     * compiles and does nothing, so the README shows the attribute in --printKernel output.
     */
    private static final String HINTS_PROPERTY = "tornado.cuda.tile.hints";
    private static final String OCCUPANCY_HINT = "occupancy=2";

    // ================================================================ host

    private static long median(long[] samples) {
        long[] copy = samples.clone();
        Arrays.sort(copy);
        return copy[copy.length / 2];
    }

    /** Steady-state median wall clock in microseconds, first (JIT) execution excluded. */
    private static long time(TaskGraph graph, GridScheduler scheduler, int executions) {
        long[] samples = new long[executions - 1];
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            if (scheduler != null) {
                plan.withGridScheduler(scheduler);
            }
            for (int e = 0; e < executions; e++) {
                long start = System.nanoTime();
                plan.execute();
                if (e > 0) {
                    samples[e - 1] = (System.nanoTime() - start) / 1000;
                }
            }
        } catch (TornadoDeviceTileNotSupported e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return median(samples);
    }

    private static boolean validate(String label, float[] expected, FloatArray actual, int n) {
        // FP16 inputs are exact in FP32; only the FP32 summation order differs between rungs.
        float tolerance = 0.05f * n / 256.0f;
        int bad = 0;
        float worst = 0.0f;
        for (int i = 0; i < expected.length; i++) {
            float err = Math.abs(expected[i] - actual.get(i));
            worst = Math.max(worst, err);
            if (err > tolerance * Math.max(1.0f, Math.abs(expected[i]))) {
                bad++;
            }
        }
        System.out.printf("   %-34s %s (max abs err %.4f, %d/%d out of tol)%n", label, bad == 0 ? "PASSED" : "FAILED", worst, bad, expected.length);
        return bad == 0;
    }

    private static TaskGraph graph(String name, HalfFloatArray a, HalfFloatArray b, FloatArray c) {
        return new TaskGraph(name).transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b);
    }

    public static void main(String[] args) {
        if (TornadoRuntimeProvider.getTornadoRuntime().getBackendType(0) != TornadoVMBackendType.CUDA) {
            System.out.println("This demo requires the CUDA backend.");
            return;
        }
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 2048;
        int executions = args.length > 1 ? Integer.parseInt(args[1]) : 10;
        if (n % 128 != 0 || executions < 2) {
            System.out.printf("n must be a multiple of 128 and executions >= 2; got %d, %d%n", n, executions);
            return;
        }

        HalfFloatArray a = new HalfFloatArray(n * n);
        HalfFloatArray b = new HalfFloatArray(n * n);
        for (int i = 0; i < n * n; i++) {
            a.set(i, new HalfFloat(((i * 7 + 3) % 17) * 0.0625f - 0.5f));
            b.set(i, new HalfFloat(((i * 5 + 11) % 13) * 0.0769f - 0.5f));
        }

        // CPU reference over the same FP16-rounded values, so validation measures the kernel.
        float[] af = new float[n * n];
        float[] bf = new float[n * n];
        for (int i = 0; i < n * n; i++) {
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

        System.out.printf("TileContext ladder vs optimised KernelContext: FP16 C = A * B, %dx%d, FP32 accumulate%n", n, n);
        System.out.printf("  %d executions per rung, steady-state median wall clock (first excluded)%n", executions);
        System.out.printf("  Wall clock includes host dispatch and the copy-out. For kernel time use nsys (README.md).%n%n");

        String[] labels = { "1. KernelContext, simple", "2. KernelContext, optimised", //
                "3. TileContext 32x32x32", "4. TileContext 64x64x64", "5. TileContext 128x128x32", //
                "6. TileContext 128x128x64", "7. TileContext 128x128x64 +hint", "8. cuBLAS GemmEx FP16->FP32" };
        String[] kernels = { "kcSimple", "kcOptimised", "tile32x32x32", "tile64x64x64", "tile128x128x32", //
                "tile128x128x64", "tile128x128x64Hinted", "(cuBLAS-chosen)" };
        long[] times = new long[labels.length];
        boolean ok = true;
        FloatArray[] out = new FloatArray[labels.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = new FloatArray(n * n);
        }

        // ---- rung 1
        System.out.println(labels[0]);
        WorkerGrid1D wSimple = new WorkerGrid1D((n / 16) * (n / 16) * WARP);
        wSimple.setLocalWork(WARP, 1, 1);
        times[0] = time(graph("simple", a, b, out[0]) //
                .task("t", TileLadder::kcSimple, new KernelContext(), a, b, out[0], n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out[0]), new GridScheduler("simple.t", wSimple), executions);
        ok &= validate("KernelContext, simple", expected, out[0], n);

        // ---- rung 2
        System.out.println(labels[1]);
        WorkerGrid1D wOpt = new WorkerGrid1D((n / OPT_BM) * (n / OPT_BN) * OPT_THREADS);
        wOpt.setLocalWork(OPT_THREADS, 1, 1);
        times[1] = time(graph("opt", a, b, out[1]) //
                .task("t", TileLadder::kcOptimised, new KernelContext(), a, b, out[1], n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out[1]), new GridScheduler("opt.t", wOpt), executions);
        ok &= validate("KernelContext, optimised", expected, out[1], n);

        // ---- rungs 3-6: the worker grid counts TILE BLOCKS; the runtime pins the block to 1x1x1.
        // Probe the tile path once, on the first tile rung; never retry a failed tile launch.
        int[][] shapes = { { T32_M, T32_N }, { T64_M, T64_N }, { T128K32_M, T128K32_N }, { T128K64_M, T128K64_N } };
        try {
            for (int s = 0; s < 4; s++) {
                int rung = 2 + s;
                System.out.println(labels[rung]);
                TaskGraph g = graph("tile" + s, a, b, out[rung]);
                switch (s) {
                    case 0 -> g.task("t", TileLadder::tile32x32x32, new TileContext(), a, b, out[rung], n);
                    case 1 -> g.task("t", TileLadder::tile64x64x64, new TileContext(), a, b, out[rung], n);
                    case 2 -> g.task("t", TileLadder::tile128x128x32, new TileContext(), a, b, out[rung], n);
                    default -> g.task("t", TileLadder::tile128x128x64, new TileContext(), a, b, out[rung], n);
                }
                g.transferToHost(DataTransferMode.EVERY_EXECUTION, out[rung]);
                WorkerGrid2D w = new WorkerGrid2D(n / shapes[s][0], n / shapes[s][1]);
                times[rung] = time(g, new GridScheduler("tile" + s + ".t", w), executions);
                ok &= validate(labels[rung].substring(3), expected, out[rung], n);
            }

            // ---- rung 7: rung 6's kernel with a launch hint, set for this compile only.
            System.out.println(labels[6]);
            String previous = System.getProperty(HINTS_PROPERTY);
            System.setProperty(HINTS_PROPERTY, OCCUPANCY_HINT);
            try {
                times[6] = time(graph("hinted", a, b, out[6]) //
                        .task("t", TileLadder::tile128x128x64Hinted, new TileContext(), a, b, out[6], n) //
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out[6]), //
                        new GridScheduler("hinted.t", new WorkerGrid2D(n / T128K64_M, n / T128K64_N)), executions);
            } finally {
                if (previous == null) {
                    System.clearProperty(HINTS_PROPERTY);
                } else {
                    System.setProperty(HINTS_PROPERTY, previous);
                }
            }
            ok &= validate("TileContext 128x128x64 +hint", expected, out[6], n);
        } catch (TornadoDeviceTileNotSupported e) {
            System.out.println("[UNSUPPORTED] TileLadder: " + e.getMessage());
            return;
        }

        // ---- rung 8: cuBLAS is column-major, so C^T = B^T * A^T -- swap the operands.
        System.out.println(labels[7]);
        times[7] = time(graph("cublas", a, b, out[7]) //
                .libraryTask("t", CuBlas::cublasGemmExFP16FP32, CuBlasOperation.CUBLAS_OP_N.operation(), CuBlasOperation.CUBLAS_OP_N.operation(), //
                        n, n, n, 1.0f, b, n, a, n, 0.0f, out[7], n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out[7]), null, executions);
        ok &= validate("cuBLAS GemmEx FP16->FP32", expected, out[7], n);

        double gflop = 2.0 * n * n * n / 1e9;
        System.out.printf("%n=== Summary (steady-state median wall clock, this run / this GPU) ===%n");
        System.out.printf("%-34s %-22s %10s %10s%n", "rung", "nsys kernel name", "median us", "GFLOP/s");
        for (int i = 0; i < labels.length; i++) {
            System.out.printf("%-34s %-22s %10d %10.0f%n", labels[i], kernels[i], times[i], gflop / (times[i] / 1e6));
        }
        System.out.println();
        System.out.println(ok ? "All rungs produced the same, correct result" : "At least one rung FAILED");
    }
}
