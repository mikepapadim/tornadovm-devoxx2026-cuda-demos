/*
 * TornadoVM CUDA demo — the matrix-multiply ladder, one rung at a time.
 *
 * The same FP32 C = A * B computed six ways, on the same buffers, in one JVM,
 * validated against the same CPU reference. The point is not that the library
 * wins -- it is *how much* of the distance a Java programmer can close by hand,
 * and where hand-writing stops paying:
 *
 *   1. naive            @Parallel, no GPU knowledge. One thread per output,
 *                       K global loads per element, no reuse.
 *   2. kcTiled          KernelContext + shared memory. The classic tile: each
 *                       block stages a TILE x TILE square of A and B, so every
 *                       loaded value is reused TILE times.
 *   3. kcRegisterTiled  KernelContext + shared memory + a per-thread register
 *                       micro-tile. Each thread computes TM x TN outputs, so
 *                       each shared-memory read feeds TM (or TN) FMAs instead
 *                       of one. This is the rung most people never write.
 *   4. cutlassSgemm     CUTLASS library task -- NVIDIA's template GEMM.
 *   5. cublasSgemm      cuBLAS library task -- the vendor's tuned FP32 kernel.
 *   6. cublasSgemmTF32  Same call, tensor cores via TF32. Same FP32 data, ~10
 *                       mantissa bits in the multiply, FP32 accumulate.
 *
 * Rungs 1-3 are Java that TornadoVM JIT-compiles to CUDA. Rungs 4-6 are native
 * library calls on the *same* device buffers, in the same task graph shape --
 * no host round trip anywhere.
 *
 * Usage:
 *   tornado --classpath . MatMulLadder [size] [executions]
 *   tornado --printKernel --classpath . MatMulLadder 256 2   # see the generated CUDA
 */
import java.util.Arrays;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.enums.CuBlasOperation;
import uk.ac.manchester.tornado.cutlass.Cutlass;

public class MatMulLadder {

    /** Shared-memory tile edge for rung 2. 16x16 threads, one output each. */
    private static final int TILE = 16;

    /** Rung 3: 16x16 threads, each computing a TM x TN micro-tile, so a 64x64 block tile. */
    private static final int BK = 16;
    private static final int TM = 4;
    private static final int TN = 4;
    private static final int THREADS = 16;
    private static final int BM = THREADS * TM; // 64
    private static final int BN = THREADS * TN; // 64

    // ---------------------------------------------------------------- rung 1

    /** One thread per output element. Every thread reads a full row and column from global memory. */
    public static void naive(FloatArray a, FloatArray b, FloatArray c, int n) {
        for (@Parallel int i = 0; i < n; i++) {
            for (@Parallel int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += a.get(i * n + k) * b.get(k * n + j);
                }
                c.set(i * n + j, sum);
            }
        }
    }

    // ---------------------------------------------------------------- rung 2

    /**
     * Shared-memory tiling. The block cooperatively stages a TILE x TILE square of A and of B,
     * then every thread reads them TILE times from shared memory instead of global. Global traffic
     * drops by a factor of TILE.
     */
    public static void kcTiled(KernelContext ctx, FloatArray a, FloatArray b, FloatArray c, int n) {
        int localCol = ctx.localIdx;
        int localRow = ctx.localIdy;
        int row = ctx.groupIdy * TILE + localRow;
        int col = ctx.groupIdx * TILE + localCol;

        float[] tileA = ctx.allocateFloatLocalArray(TILE * TILE);
        float[] tileB = ctx.allocateFloatLocalArray(TILE * TILE);

        float sum = 0.0f;
        for (int kTile = 0; kTile < n; kTile += TILE) {
            tileA[localRow * TILE + localCol] = a.get(row * n + kTile + localCol);
            tileB[localRow * TILE + localCol] = b.get((kTile + localRow) * n + col);
            ctx.localBarrier();

            for (int k = 0; k < TILE; k++) {
                sum += tileA[localRow * TILE + k] * tileB[k * TILE + localCol];
            }
            ctx.localBarrier();
        }
        c.set(row * n + col, sum);
    }

    // ---------------------------------------------------------------- rung 3

    /**
     * Shared memory plus a register micro-tile. Each thread owns a TM x TN patch of C held in
     * registers, so one shared-memory read of A feeds TN fused multiply-adds and one read of B
     * feeds TM. That raises arithmetic intensity by roughly TM*TN/(TM+TN) over rung 2 -- the step
     * that turns a memory-bound kernel into a compute-bound one.
     */
    public static void kcRegisterTiled(KernelContext ctx, FloatArray a, FloatArray b, FloatArray c, int n) {
        int tx = ctx.localIdx;
        int ty = ctx.localIdy;
        int tid = ty * THREADS + tx;
        int blockRow = ctx.groupIdy * BM;
        int blockCol = ctx.groupIdx * BN;

        float[] tileA = ctx.allocateFloatLocalArray(BM * BK); // 64 x 16
        float[] tileB = ctx.allocateFloatLocalArray(BK * BN); // 16 x 64

        float[] acc = new float[TM * TN];
        for (int i = 0; i < TM * TN; i++) {
            acc[i] = 0.0f;
        }
        float[] regA = new float[TM];
        float[] regB = new float[TN];

        for (int kTile = 0; kTile < n; kTile += BK) {
            // 256 threads stage 1024 elements of each tile: four apiece.
            for (int load = 0; load < (BM * BK) / (THREADS * THREADS); load++) {
                int idx = tid + load * THREADS * THREADS;
                int r = idx / BK;
                int cc = idx % BK;
                tileA[r * BK + cc] = a.get((blockRow + r) * n + kTile + cc);
            }
            for (int load = 0; load < (BK * BN) / (THREADS * THREADS); load++) {
                int idx = tid + load * THREADS * THREADS;
                int r = idx / BN;
                int cc = idx % BN;
                tileB[r * BN + cc] = b.get((kTile + r) * n + blockCol + cc);
            }
            ctx.localBarrier();

            for (int k = 0; k < BK; k++) {
                for (int i = 0; i < TM; i++) {
                    regA[i] = tileA[(ty * TM + i) * BK + k];
                }
                for (int j = 0; j < TN; j++) {
                    regB[j] = tileB[k * BN + tx * TN + j];
                }
                for (int i = 0; i < TM; i++) {
                    for (int j = 0; j < TN; j++) {
                        acc[i * TN + j] += regA[i] * regB[j];
                    }
                }
            }
            ctx.localBarrier();
        }

        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < TN; j++) {
                c.set((blockRow + ty * TM + i) * n + blockCol + tx * TN + j, acc[i * TN + j]);
            }
        }
    }

    // ----------------------------------------------------------------- host

    private static long medianOf(long[] samples) {
        long[] copy = samples.clone();
        Arrays.sort(copy);
        return copy[copy.length / 2];
    }

    /** Runs one rung, discarding the first execution (JIT compile / plan setup). */
    private static long timeRung(TaskGraph graph, GridScheduler scheduler, int executions) throws TornadoExecutionPlanException {
        ImmutableTaskGraph snapshot = graph.snapshot();
        long[] samples = new long[executions - 1];
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            for (int e = 0; e < executions; e++) {
                long start = System.nanoTime();
                if (scheduler != null) {
                    plan.withGridScheduler(scheduler).execute();
                } else {
                    plan.execute();
                }
                long elapsed = (System.nanoTime() - start) / 1000;
                if (e > 0) {
                    samples[e - 1] = elapsed;
                }
            }
        }
        return medianOf(samples);
    }

    private static boolean validate(String label, FloatArray actual, float[] expected, int n, float tol) {
        float maxErr = 0.0f;
        int bad = 0;
        for (int i = 0; i < n * n; i++) {
            float err = Math.abs(actual.get(i) - expected[i]);
            maxErr = Math.max(maxErr, err);
            if (err > tol * Math.max(1.0f, Math.abs(expected[i]))) {
                bad++;
            }
        }
        System.out.printf("    validation %s (max abs err %.5f, %d/%d cells out of tol)%n", //
                bad == 0 ? "PASSED" : "FAILED", maxErr, bad, n * n);
        return bad == 0;
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (TornadoRuntimeProvider.getTornadoRuntime().getBackendType(0) != TornadoVMBackendType.CUDA) {
            System.out.println("This demo requires the CUDA backend.");
            return;
        }
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 1024;
        int executions = args.length > 1 ? Integer.parseInt(args[1]) : 20;

        if (n % BM != 0) {
            System.out.printf("size must be a multiple of %d (the register-tiled block); got %d%n", BM, n);
            return;
        }

        FloatArray a = new FloatArray(n * n);
        FloatArray b = new FloatArray(n * n);
        for (int i = 0; i < n * n; i++) {
            a.set(i, ((i * 7 + 3) % 17) * 0.0625f - 0.5f);
            b.set(i, ((i * 5 + 11) % 13) * 0.0769f - 0.5f);
        }

        float[] expected = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += a.get(i * n + k) * b.get(k * n + j);
                }
                expected[i * n + j] = sum;
            }
        }

        double gflop = 2.0 * n * n * n / 1e9;
        System.out.printf("Matrix-multiply ladder: C = A * B, FP32, %dx%d%n", n, n);
        System.out.printf("  %d executions per rung, steady-state median wall-clock (first excluded)%n%n", executions);

        FloatArray cNaive = new FloatArray(n * n);
        FloatArray cTiled = new FloatArray(n * n);
        FloatArray cReg = new FloatArray(n * n);
        FloatArray cCutlass = new FloatArray(n * n);
        FloatArray cCublas = new FloatArray(n * n);
        FloatArray cTf32 = new FloatArray(n * n);

        boolean ok = true;
        long[] times = new long[6];
        String[] labels = { "1. naive @Parallel", "2. KernelContext tiled", "3. KernelContext register-tiled", //
                "4. CUTLASS sgemm", "5. cuBLAS sgemm", "6. cuBLAS sgemm TF32" };

        // ---- rung 1
        System.out.println(labels[0]);
        times[0] = timeRung(new TaskGraph("naive") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("mm", MatMulLadder::naive, a, b, cNaive, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cNaive), null, executions);
        ok &= validate(labels[0], cNaive, expected, n, 1e-3f);

        // ---- rung 2
        System.out.println(labels[1]);
        WorkerGrid2D tiledGrid = new WorkerGrid2D(n, n);
        tiledGrid.setLocalWork(TILE, TILE, 1);
        times[1] = timeRung(new TaskGraph("tiled") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("mm", MatMulLadder::kcTiled, new KernelContext(), a, b, cTiled, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cTiled), //
                new GridScheduler("tiled.mm", tiledGrid), executions);
        ok &= validate(labels[1], cTiled, expected, n, 1e-3f);

        // ---- rung 3
        System.out.println(labels[2]);
        WorkerGrid2D regGrid = new WorkerGrid2D(n / TN, n / TM);
        regGrid.setLocalWork(THREADS, THREADS, 1);
        times[2] = timeRung(new TaskGraph("reg") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("mm", MatMulLadder::kcRegisterTiled, new KernelContext(), a, b, cReg, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cReg), //
                new GridScheduler("reg.mm", regGrid), executions);
        ok &= validate(labels[2], cReg, expected, n, 1e-3f);

        // ---- rung 4: CUTLASS is row-major, so operands go in the natural order
        System.out.println(labels[3]);
        times[3] = timeRung(new TaskGraph("cutlass") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .libraryTask("mm", Cutlass::cutlassSgemm, n, n, n, 1.0f, a, b, 0.0f, cCutlass) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cCutlass), null, executions);
        ok &= validate(labels[3], cCutlass, expected, n, 1e-3f);

        // ---- rung 5: cuBLAS is column-major. Row-major C = A*B is column-major C = B*A,
        //              so the operands are swapped rather than the data transposed.
        System.out.println(labels[4]);
        int opN = CuBlasOperation.CUBLAS_OP_N.operation();
        times[4] = timeRung(new TaskGraph("cublas") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .libraryTask("mm", CuBlas::cublasSgemm, opN, opN, n, n, n, 1.0f, b, n, a, n, 0.0f, cCublas, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cCublas), null, executions);
        ok &= validate(labels[4], cCublas, expected, n, 1e-3f);

        // ---- rung 6: same call, tensor cores. TF32 keeps ~10 mantissa bits, so the
        //              tolerance is looser by design -- that is the trade being shown.
        System.out.println(labels[5]);
        times[5] = timeRung(new TaskGraph("tf32") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .libraryTask("mm", CuBlas::cublasSgemmTF32, opN, opN, n, n, n, 1.0f, b, n, a, n, 0.0f, cTf32, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cTf32), null, executions);
        ok &= validate(labels[5], cTf32, expected, n, 2e-2f);

        System.out.printf("%n=== Summary (steady-state median, this run / this GPU) ===%n");
        System.out.printf("%-34s %10s %12s %10s%n", "rung", "median us", "GFLOP/s", "vs naive");
        for (int i = 0; i < 6; i++) {
            System.out.printf("%-34s %10d %12.1f %9.1fx%n", labels[i], times[i], gflop / (times[i] / 1e6), //
                    (double) times[0] / times[i]);
        }
        System.out.println();
        System.out.println(ok ? "All six rungs produced the same, correct result" : "At least one rung FAILED");
    }
}
