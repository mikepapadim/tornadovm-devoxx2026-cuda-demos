import java.util.Arrays;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoDeviceTileNotSupported;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.tile.DType;
import uk.ac.manchester.tornado.api.tile.PartitionView;
import uk.ac.manchester.tornado.api.tile.Tile;
import uk.ac.manchester.tornado.api.tile.TileContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * FP32 C = A * B written seven ways: three KernelContext kernels that follow the classic
 * SIMT progression (one thread per output, shared-memory tiles, register tiles), and the
 * same TileContext kernel at four tile shapes. Every rung is validated against the full
 * CPU reference.
 *
 * <pre>
 * tornado --jvm="-Dtornado.recover.bailout=False" --classpath . MatMulFP32Variants 1024 10
 * </pre>
 */
public class MatMulFP32Variants {

    // ================================================================ KernelContext rungs

    /** 1. One thread per output element, launched as 16x16 work-groups. */
    public static void oneThreadPerOutput(KernelContext ctx, FloatArray a, FloatArray b, FloatArray c, int n) {
        int col = ctx.globalIdx;
        int row = ctx.globalIdy;
        float acc = 0.0f;
        for (int k = 0; k < n; k++) {
            acc += a.get(row * n + k) * b.get(k * n + col);
        }
        c.set(row * n + col, acc);
    }

    private static final int TILE = 16;

    /** 2. 16x16 shared-memory tiles: each staged value is read TILE times from shared memory. */
    public static void sharedTiled(KernelContext ctx, FloatArray a, FloatArray b, FloatArray c, int n) {
        float[] tileA = ctx.allocateFloatLocalArray(TILE * TILE);
        float[] tileB = ctx.allocateFloatLocalArray(TILE * TILE);
        int localIdx = ctx.localIdx;
        int localIdy = ctx.localIdy;
        int row = ctx.groupIdy * TILE + localIdy;
        int col = ctx.groupIdx * TILE + localIdx;
        float sum = 0.0f;
        for (int tile = 0; tile < n / TILE; tile++) {
            tileA[localIdy * TILE + localIdx] = a.get(row * n + tile * TILE + localIdx);
            tileB[localIdy * TILE + localIdx] = b.get((tile * TILE + localIdy) * n + col);
            ctx.localBarrier();
            for (int k = 0; k < TILE; k++) {
                sum += tileA[localIdy * TILE + k] * tileB[k * TILE + localIdx];
            }
            ctx.localBarrier();
        }
        c.set(row * n + col, sum);
    }

    private static final int BM = 64;
    private static final int BN = 64;
    private static final int BK = 16;
    private static final int TM = 4;
    private static final int TN = 4;

    /**
     * 3. Shared-memory tiles plus a register micro-tile: a 64x64 block per 16x16 work-group, each
     * thread accumulating a 4x4 patch of C in registers over k-steps of 16.
     */
    public static void registerTiled(KernelContext ctx, FloatArray a, FloatArray b, FloatArray c, int n) {
        int bx = ctx.groupIdx;
        int by = ctx.groupIdy;
        int numThreadsBlockTile = (BM * BN) / (TM * TN);
        int linearLocalId = ctx.localIdy * ctx.localGroupSizeX + ctx.localIdx;
        int threadCol = ctx.localIdx;
        int threadRow = ctx.localIdy;

        float[] tileA = ctx.allocateFloatLocalArray(BM * BK);
        float[] tileB = ctx.allocateFloatLocalArray(BK * BN);

        int aFrom = by * BM * n;
        int bFrom = bx * BN;
        int cFrom = (by * BM * n) + (bx * BN);

        int innerRowA = linearLocalId / BK;
        int innerColA = linearLocalId % BK;
        int strideA = numThreadsBlockTile / BK;
        int innerRowB = linearLocalId / BN;
        int innerColB = linearLocalId % BN;
        int strideB = numThreadsBlockTile / BN;

        float[] threadResults = new float[TM * TN];
        float[] regM = new float[TM];
        float[] regN = new float[TN];
        for (int i = 0; i < TM * TN; i++) {
            threadResults[i] = 0.0f;
        }

        for (int bkIdx = 0; bkIdx < n; bkIdx += BK) {
            for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
                tileA[(innerRowA + loadOffset) * BK + innerColA] = a.get(((innerRowA + loadOffset) * n + innerColA) + aFrom);
            }
            for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
                tileB[(innerRowB + loadOffset) * BN + innerColB] = b.get(((innerRowB + loadOffset) * n + innerColB) + bFrom);
            }
            ctx.localBarrier();
            aFrom += BK;
            bFrom += BK * n;
            for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
                for (int i = 0; i < TM; i++) {
                    regM[i] = tileA[(threadRow * TM + i) * BK + dotIdx];
                }
                for (int i = 0; i < TN; i++) {
                    regN[i] = tileB[dotIdx * BN + threadCol * TN + i];
                }
                for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
                    for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
                        threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                    }
                }
            }
            ctx.localBarrier();
        }
        for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
            for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
                c.set(((threadRow * TM + resIdxM) * n + threadCol * TN + resIdxN) + cFrom, threadResults[resIdxM * TN + resIdxN]);
            }
        }
    }

    // ================================================================ TileContext rungs
    // FP32 operands into tc.mma. A tile shape is part of the kernel's type, so each shape is its
    // own method; the bodies are identical.

    public static void tile32x32x32(TileContext tc, FloatArray a, FloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), 32, 32);
        PartitionView bv = tc.partition(tc.view(b, n, n), 32, 32);
        PartitionView cv = tc.partition(tc.view(c, n, n), 32, 32);
        Tile acc = tc.zeros(DType.F32, 32, 32);
        for (int k = 0; k < n / 32; k++) {
            acc = tc.mma(av.load(tc.bidX(), k), bv.load(k, tc.bidY()), acc);
        }
        cv.store(acc, tc.bidX(), tc.bidY());
    }

    public static void tile64x64x64(TileContext tc, FloatArray a, FloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), 64, 64);
        PartitionView bv = tc.partition(tc.view(b, n, n), 64, 64);
        PartitionView cv = tc.partition(tc.view(c, n, n), 64, 64);
        Tile acc = tc.zeros(DType.F32, 64, 64);
        for (int k = 0; k < n / 64; k++) {
            acc = tc.mma(av.load(tc.bidX(), k), bv.load(k, tc.bidY()), acc);
        }
        cv.store(acc, tc.bidX(), tc.bidY());
    }

    public static void tile128x128x32(TileContext tc, FloatArray a, FloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), 128, 32);
        PartitionView bv = tc.partition(tc.view(b, n, n), 32, 128);
        PartitionView cv = tc.partition(tc.view(c, n, n), 128, 128);
        Tile acc = tc.zeros(DType.F32, 128, 128);
        for (int k = 0; k < n / 32; k++) {
            acc = tc.mma(av.load(tc.bidX(), k), bv.load(k, tc.bidY()), acc);
        }
        cv.store(acc, tc.bidX(), tc.bidY());
    }

    public static void tile128x128x64(TileContext tc, FloatArray a, FloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), 128, 64);
        PartitionView bv = tc.partition(tc.view(b, n, n), 64, 128);
        PartitionView cv = tc.partition(tc.view(c, n, n), 128, 128);
        Tile acc = tc.zeros(DType.F32, 128, 128);
        for (int k = 0; k < n / 64; k++) {
            acc = tc.mma(av.load(tc.bidX(), k), bv.load(k, tc.bidY()), acc);
        }
        cv.store(acc, tc.bidX(), tc.bidY());
    }

    // ================================================================ host

    private static long median(long[] samples) {
        long[] copy = samples.clone();
        Arrays.sort(copy);
        return copy[copy.length / 2];
    }

    /** Median wall clock of {@code executions - 1} runs (the first, with JIT compilation, excluded), in microseconds. */
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
        } catch (TornadoDeviceTileNotSupported e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return median(samples);
    }

    /** Full-matrix check against a CPU reference over the same inputs. */
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
        System.out.printf("   %-34s %s (max abs err %.2e, %d/%d out of tol)%n", label, bad == 0 ? "PASSED" : "FAILED", worst, bad, expected.length);
        return bad == 0;
    }

    private static TaskGraph graph(String name, FloatArray a, FloatArray b) {
        return new TaskGraph(name).transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b);
    }

    public static void main(String[] args) {
        if (TornadoRuntimeProvider.getTornadoRuntime().getBackendType(0) != TornadoVMBackendType.CUDA) {
            System.out.println("This demo requires the CUDA backend.");
            return;
        }
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 1024;
        int executions = args.length > 1 ? Integer.parseInt(args[1]) : 10;
        if (n % 128 != 0 || executions < 2) {
            System.out.printf("n must be a multiple of 128 and executions >= 2; got %d, %d%n", n, executions);
            return;
        }

        FloatArray a = new FloatArray(n * n);
        FloatArray b = new FloatArray(n * n);
        java.util.Random random = new java.util.Random(42);
        for (int i = 0; i < n * n; i++) {
            a.set(i, random.nextFloat() - 0.5f);
            b.set(i, random.nextFloat() - 0.5f);
        }
        float[] expected = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                float av = a.get(i * n + k);
                for (int j = 0; j < n; j++) {
                    expected[i * n + j] += av * b.get(k * n + j);
                }
            }
        }

        String[] labels = { "1. one thread per output", "2. shared-memory tiled 16x16", "3. register-tiled 64x64x16, 4x4", //
                "4. TileContext 32x32x32", "5. TileContext 64x64x64", "6. TileContext 128x128x32", "7. TileContext 128x128x64" };
        long[] times = new long[labels.length];
        FloatArray[] out = new FloatArray[labels.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = new FloatArray(n * n);
        }
        boolean ok = true;
        // FP32 accumulation in a different order than the CPU reference; the tolerance grows with k.
        float tol = 1e-4f * (float) Math.sqrt(n);

        System.out.printf("FP32 C = A * B, %dx%d, %d executions per rung, steady-state median wall clock (first excluded)%n", n, n, executions);
        System.out.printf("  Wall clock includes host dispatch and the copy-out. For kernel time use nsys (README.md).%n%n");

        // ---- rung 1
        System.out.println(labels[0]);
        WorkerGrid2D w1 = new WorkerGrid2D(n, n);
        w1.setLocalWork(TILE, TILE, 1);
        times[0] = time(graph("one", a, b).task("t", MatMulFP32Variants::oneThreadPerOutput, new KernelContext(), a, b, out[0], n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out[0]), "one.t", w1, executions);
        ok &= validate(labels[0].substring(3), expected, out[0], tol);

        // ---- rung 2
        System.out.println(labels[1]);
        WorkerGrid2D w2 = new WorkerGrid2D(n, n);
        w2.setLocalWork(TILE, TILE, 1);
        times[1] = time(graph("shared", a, b).task("t", MatMulFP32Variants::sharedTiled, new KernelContext(), a, b, out[1], n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out[1]), "shared.t", w2, executions);
        ok &= validate(labels[1].substring(3), expected, out[1], tol);

        // ---- rung 3: 16x16 threads per 64x64 block
        System.out.println(labels[2]);
        WorkerGrid2D w3 = new WorkerGrid2D(n / TN, n / TM);
        w3.setLocalWork(BN / TN, BM / TM, 1);
        times[2] = time(graph("reg", a, b).task("t", MatMulFP32Variants::registerTiled, new KernelContext(), a, b, out[2], n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out[2]), "reg.t", w3, executions);
        ok &= validate(labels[2].substring(3), expected, out[2], tol);

        // ---- rungs 4-7: the worker grid counts tile blocks
        int[][] shapes = { { 32, 32 }, { 64, 64 }, { 128, 128 }, { 128, 128 } };
        try {
            for (int s = 0; s < 4; s++) {
                int rung = 3 + s;
                System.out.println(labels[rung]);
                TaskGraph g = graph("tile" + s, a, b);
                switch (s) {
                    case 0 -> g.task("t", MatMulFP32Variants::tile32x32x32, new TileContext(), a, b, out[rung], n);
                    case 1 -> g.task("t", MatMulFP32Variants::tile64x64x64, new TileContext(), a, b, out[rung], n);
                    case 2 -> g.task("t", MatMulFP32Variants::tile128x128x32, new TileContext(), a, b, out[rung], n);
                    default -> g.task("t", MatMulFP32Variants::tile128x128x64, new TileContext(), a, b, out[rung], n);
                }
                g.transferToHost(DataTransferMode.EVERY_EXECUTION, out[rung]);
                times[rung] = time(g, "tile" + s + ".t", new WorkerGrid2D(n / shapes[s][0], n / shapes[s][1]), executions);
                ok &= validate(labels[rung].substring(3), expected, out[rung], tol);
            }
        } catch (TornadoDeviceTileNotSupported e) {
            System.out.println("[UNSUPPORTED] MatMulFP32Variants: " + e.getMessage());
            return;
        }

        double gflop = 2.0 * n * n * n / 1e9;
        System.out.printf("%n=== Summary (steady-state median wall clock, this run / this GPU) ===%n");
        System.out.printf("%-34s %10s %10s%n", "rung", "median us", "GFLOP/s");
        for (int i = 0; i < labels.length; i++) {
            System.out.printf("%-34s %10d %10.0f%n", labels[i], times[i], gflop / (times[i] / 1e6));
        }
        System.out.println();
        System.out.println(ok ? "All seven rungs produced the same, correct result" : "At least one rung FAILED");
    }
}
