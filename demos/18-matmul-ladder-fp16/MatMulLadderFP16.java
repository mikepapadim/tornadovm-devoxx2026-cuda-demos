/*
 * TornadoVM CUDA demo — the FP16 matrix-multiply ladder.
 *
 * Demo 17 climbs the same GEMM in FP32. This is the FP16 ladder, and it has one
 * rung FP32 cannot have: tensor cores reached directly from Java through
 * KernelContext's mma intrinsics.
 *
 *   1. naive @Parallel            one thread per output element
 *   2. KernelContext tiled        shared-memory tiles, FP32 accumulate
 *   3. KernelContext MMA          ctx.mma -> mma.sync.aligned.m16n8k16, from Java
 *   4. CUTLASS hgemm              library task, tensor cores
 *   5. cuBLAS GemmEx FP16         FP16 in, FP16 out, FP32 accumulate
 *   6. cuBLAS GemmEx FP16->FP32   FP16 in, FP32 out, the inference configuration
 *
 * Every rung computes the same C = A * B and is validated against a CPU
 * reference computed over the same FP16-rounded inputs, so the comparison
 * measures the kernel and not the quantisation.
 *
 * Usage:
 *   tornado --classpath . MatMulLadderFP16 [n] [executions]
 */
import java.util.Arrays;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.enums.CuBlasOperation;
import uk.ac.manchester.tornado.cutlass.Cutlass;

public class MatMulLadderFP16 {

    private static final int TILE = 16;

    /** Rung 3: one warp per 16x16 output tile, built from two m16n8k16 mma calls. */
    private static final int WARP = 32;
    private static final int MMA_M = 16;
    private static final int MMA_N = 16;
    private static final int MMA_K = 16;

    // ---------------------------------------------------------------- rung 1

    /** One thread per output element, accumulating in FP32 as every rung does. */
    public static void naive(HalfFloatArray a, HalfFloatArray b, HalfFloatArray c, int n) {
        for (@Parallel int i = 0; i < n; i++) {
            for (@Parallel int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += a.get(i * n + k).getFloat32() * b.get(k * n + j).getFloat32();
                }
                c.set(i * n + j, new HalfFloat(sum));
            }
        }
    }

    // ---------------------------------------------------------------- rung 2

    /** Shared-memory tiles. The same structure as demo 17 rung 2, in FP16. */
    public static void kcTiled(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, HalfFloatArray c, int n) {
        int tx = ctx.localIdx;
        int ty = ctx.localIdy;
        int row = ctx.groupIdy * TILE + ty;
        int col = ctx.groupIdx * TILE + tx;

        HalfFloat[] tileA = ctx.allocateHalfFloatLocalArray(TILE * TILE);
        HalfFloat[] tileB = ctx.allocateHalfFloatLocalArray(TILE * TILE);

        float sum = 0.0f;
        for (int t = 0; t < n / TILE; t++) {
            tileA[ty * TILE + tx] = a.get(row * n + t * TILE + tx);
            tileB[ty * TILE + tx] = b.get((t * TILE + ty) * n + col);
            ctx.localBarrier();
            for (int k = 0; k < TILE; k++) {
                sum += tileA[ty * TILE + k].getFloat32() * tileB[k * TILE + tx].getFloat32();
            }
            ctx.localBarrier();
        }
        c.set(row * n + col, new HalfFloat(sum));
    }

    // ---------------------------------------------------------------- rung 3

    /**
     * Tensor cores from Java. One warp owns a 16x16 output tile; because
     * {@code m16n8k16} produces 8 columns, the tile is two mma calls over two
     * 8-column panels, accumulating across the k dimension in FP32 fragments.
     *
     * <p>This is the rung the FP32 ladder cannot have: it emits a real
     * {@code mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32}, verifiable with
     * {@code tornado --printKernel}.</p>
     */
    public static void kcMma(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int warpId = ctx.groupIdx;
        int lane = ctx.localIdx;
        int tilesPerRow = n / MMA_N;
        int tileRow = (warpId / tilesPerRow) * MMA_M;
        int tileCol = (warpId % tilesPerRow) * MMA_N;

        int[] aTile = ctx.allocateIntLocalArray(MMA_M * MMA_K / 2);
        int[] bTile0 = ctx.allocateIntLocalArray(MMA_K * 8 / 2);
        int[] bTile1 = ctx.allocateIntLocalArray(MMA_K * 8 / 2);

        float[] accLeft = ctx.mmaFragment(0.0f);
        float[] accRight = ctx.mmaFragment(0.0f);

        for (int kBase = 0; kBase < n; kBase += MMA_K) {
            // Pack two fp16 values per int32 word, cooperatively across the warp.
            for (int idx = lane; idx < (MMA_M * MMA_K) / 2; idx += WARP) {
                int elem = idx * 2;
                int r = elem / MMA_K;
                int kk = elem % MMA_K;
                int g = (tileRow + r) * n + kBase + kk;
                int lo = a.get(g).getHalfFloatValue() & 0xFFFF;
                int hi = a.get(g + 1).getHalfFloatValue() & 0xFFFF;
                aTile[r * (MMA_K / 2) + kk / 2] = lo | (hi << 16);
            }
            for (int idx = lane; idx < 64; idx += WARP) {
                int kRow = idx / 4;
                int jPair = idx % 4;
                int jBase = jPair * 2;
                int gl = (kBase + kRow) * n + tileCol + jBase;
                bTile0[kRow * 4 + jPair] = (b.get(gl).getHalfFloatValue() & 0xFFFF) //
                        | ((b.get(gl + 1).getHalfFloatValue() & 0xFFFF) << 16);
                int gr = (kBase + kRow) * n + tileCol + 8 + jBase;
                bTile1[kRow * 4 + jPair] = (b.get(gr).getHalfFloatValue() & 0xFFFF) //
                        | ((b.get(gr + 1).getHalfFloatValue() & 0xFFFF) << 16);
            }
            ctx.localBarrier();

            HalfFloat[] fragA = ctx.mmaLoadA(aTile, MMA_K);
            HalfFloat[] fragB0 = ctx.mmaLoadB(bTile0, MMA_K);
            accLeft = ctx.mma(fragA, fragB0, accLeft, MMAShape.M16N8K16);
            HalfFloat[] fragB1 = ctx.mmaLoadB(bTile1, MMA_K);
            accRight = ctx.mma(fragA, fragB1, accRight, MMAShape.M16N8K16);

            ctx.localBarrier();
        }

        ctx.mmaStore(accLeft, c, tileRow, tileCol, n);
        ctx.mmaStore(accRight, c, tileRow, tileCol + 8, n);
    }

    // ----------------------------------------------------------------- host

    private static long medianOf(long[] samples) {
        long[] copy = samples.clone();
        Arrays.sort(copy);
        return copy[copy.length / 2];
    }

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

    /** FP16 has ~11 bits of mantissa; the tolerance scales with the k-length of the dot product. */
    private static boolean validate(String label, float[] expected, float[] actual, int n) {
        float tol = 0.05f * n / 256.0f;
        int bad = 0;
        float maxErr = 0.0f;
        for (int i = 0; i < expected.length; i++) {
            float err = Math.abs(expected[i] - actual[i]);
            maxErr = Math.max(maxErr, err);
            if (err > tol * Math.max(1.0f, Math.abs(expected[i]))) {
                bad++;
            }
        }
        System.out.printf("     %-32s %s (max abs err %.4f, %d/%d out of tol)%n", //
                label, bad == 0 ? "PASSED" : "FAILED", maxErr, bad, expected.length);
        return bad == 0;
    }

    private static float[] toFloats(HalfFloatArray h) {
        float[] out = new float[h.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = h.get(i).getFloat32();
        }
        return out;
    }

    private static float[] toFloats(FloatArray f) {
        float[] out = new float[f.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = f.get(i);
        }
        return out;
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (TornadoRuntimeProvider.getTornadoRuntime().getBackendType(0) != TornadoVMBackendType.CUDA) {
            System.out.println("This demo requires the CUDA backend.");
            return;
        }
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 1024;
        int executions = args.length > 1 ? Integer.parseInt(args[1]) : 20;
        if (n % 64 != 0) {
            System.out.printf("size must be a multiple of 64; got %d%n", n);
            return;
        }

        HalfFloatArray a = new HalfFloatArray(n * n);
        HalfFloatArray b = new HalfFloatArray(n * n);
        for (int i = 0; i < n * n; i++) {
            a.set(i, new HalfFloat(((i * 7 + 3) % 17) * 0.0625f - 0.5f));
            b.set(i, new HalfFloat(((i * 5 + 11) % 13) * 0.0769f - 0.5f));
        }

        // Reference over the SAME fp16-rounded values, so this measures the kernel not the rounding.
        float[] af = toFloats(a);
        float[] bf = toFloats(b);
        float[] expected = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += af[i * n + k] * bf[k * n + j];
                }
                expected[i * n + j] = sum;
            }
        }

        double gflop = 2.0 * n * n * n / 1e9;
        System.out.printf("FP16 matrix-multiply ladder: C = A * B, %dx%d%n", n, n);
        System.out.printf("  %d executions per rung, steady-state median wall-clock (first excluded)%n%n", executions);

        HalfFloatArray cNaive = new HalfFloatArray(n * n);
        HalfFloatArray cTiled = new HalfFloatArray(n * n);
        FloatArray cMma = new FloatArray(n * n);
        HalfFloatArray cCutlass = new HalfFloatArray(n * n);
        HalfFloatArray cGemmEx = new HalfFloatArray(n * n);
        FloatArray cGemmExF32 = new FloatArray(n * n);

        String[] labels = { "1. naive @Parallel", "2. KernelContext tiled", "3. KernelContext MMA (tensor core)", //
                "4. CUTLASS hgemm", "5. cuBLAS GemmEx FP16", "6. cuBLAS GemmEx FP16->FP32" };
        long[] times = new long[6];
        boolean ok = true;

        // rung 1
        System.out.println(labels[0]);
        WorkerGrid2D wNaive = new WorkerGrid2D(n, n);
        times[0] = timeRung(new TaskGraph("naive") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("t", MatMulLadderFP16::naive, a, b, cNaive, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cNaive), //
                new GridScheduler("naive.t", wNaive), executions);
        ok &= validate("naive", expected, toFloats(cNaive), n);

        // rung 2
        System.out.println(labels[1]);
        WorkerGrid2D wTiled = new WorkerGrid2D(n, n);
        wTiled.setLocalWork(TILE, TILE, 1);
        times[1] = timeRung(new TaskGraph("tiled") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("t", MatMulLadderFP16::kcTiled, new KernelContext(), a, b, cTiled, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cTiled), //
                new GridScheduler("tiled.t", wTiled), executions);
        ok &= validate("KernelContext tiled", expected, toFloats(cTiled), n);

        // rung 3 — one warp per 16x16 tile
        System.out.println(labels[2]);
        int warps = (n / MMA_M) * (n / MMA_N);
        WorkerGrid1D wMma = new WorkerGrid1D(warps * WARP);
        wMma.setLocalWork(WARP, 1, 1);
        times[2] = timeRung(new TaskGraph("mma") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("t", MatMulLadderFP16::kcMma, new KernelContext(), a, b, cMma, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cMma), //
                new GridScheduler("mma.t", wMma), executions);
        ok &= validate("KernelContext MMA", expected, toFloats(cMma), n);

        // rung 4 — CUTLASS, row-major
        System.out.println(labels[3]);
        times[3] = timeRung(new TaskGraph("cutlass") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .libraryTask("t", Cutlass::cutlassHgemm, n, n, n, 1.0f, a, b, 0.0f, cCutlass) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cCutlass), null, executions);
        ok &= validate("CUTLASS hgemm", expected, toFloats(cCutlass), n);

        // rungs 5 and 6 — cuBLAS is column-major, so swap the operands
        System.out.println(labels[4]);
        times[4] = timeRung(new TaskGraph("gemmex") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .libraryTask("t", CuBlas::cublasGemmExFP16, //
                        CuBlasOperation.CUBLAS_OP_N.operation(), CuBlasOperation.CUBLAS_OP_N.operation(), //
                        n, n, n, 1.0f, b, n, a, n, 0.0f, cGemmEx, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cGemmEx), null, executions);
        ok &= validate("cuBLAS GemmEx FP16", expected, toFloats(cGemmEx), n);

        System.out.println(labels[5]);
        times[5] = timeRung(new TaskGraph("gemmexf32") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .libraryTask("t", CuBlas::cublasGemmExFP16FP32, //
                        CuBlasOperation.CUBLAS_OP_N.operation(), CuBlasOperation.CUBLAS_OP_N.operation(), //
                        n, n, n, 1.0f, b, n, a, n, 0.0f, cGemmExF32, n) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cGemmExF32), null, executions);
        ok &= validate("cuBLAS GemmEx FP16->FP32", expected, toFloats(cGemmExF32), n);

        System.out.printf("%n=== Summary (steady-state median wall-clock, this run / this GPU) ===%n");
        System.out.printf("%-36s %10s %12s %10s%n", "rung", "median us", "GFLOP/s", "vs naive");
        for (int i = 0; i < times.length; i++) {
            System.out.printf("%-36s %10d %12.0f %9.1fx%n", labels[i], times[i], //
                    gflop / (times[i] / 1e6), (double) times[0] / times[i]);
        }
        System.out.printf("%nWall-clock includes host dispatch. For kernel time alone use nsys;%n");
        System.out.printf("scripts/compare-ladder.sh 18 collects both.%n");
        System.out.println(ok ? "All rungs produced correct results" : "At least one rung FAILED");
    }
}
