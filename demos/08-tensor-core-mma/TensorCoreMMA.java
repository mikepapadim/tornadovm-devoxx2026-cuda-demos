import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Smallest possible Tensor Core / MMA demo on TornadoVM's CUDA backend:
 * a single warp (32 threads) computes one M16N8K16 fp16 GEMM tile
 * (C[16x8] = A[16x16] * B[16x8]) with exactly one {@code mma.sync} instruction,
 * next to a scalar (no-MMA) reference kernel computing the same tile.
 *
 * Usage:
 *   tornado --classpath . TensorCoreMMA
 *   tornado --printKernel --classpath . TensorCoreMMA   (dump generated CUDA source)
 */
public class TensorCoreMMA {

    private static final int M = 16, N = 8, K = 16;
    private static final int WARP_SIZE = 32;

    /** Scalar (no Tensor Core) reference kernel: one thread per output element. */
    public static void gemmScalarFp16(KernelContext ctx, HalfFloatArray A, HalfFloatArray B, FloatArray C) {
        int row = ctx.globalIdx;
        int col = ctx.globalIdy;
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A.get(row * K + k).getFloat32() * B.get(k * N + col).getFloat32();
        }
        C.set(row * N + col, sum);
    }

    /** Single-warp, single-tile Tensor Core kernel: one mma.sync.aligned.m16n8k16 per launch. */
    public static void gemmMMASingleTile(KernelContext ctx, HalfFloatArray A, HalfFloatArray B, FloatArray C) {
        int tid = ctx.localIdx;

        // Packed fp16 tiles in shared memory: two half-floats per int (lo | hi << 16).
        int[] aTile = ctx.allocateIntLocalArray(M * K / 2); // 128 elements
        int[] bTile = ctx.allocateIntLocalArray(K * N / 2); // 64 elements

        // Cooperatively pack A [16,16] row-major into aTile.
        for (int idx = tid; idx < M * K / 2; idx += WARP_SIZE) {
            int row = idx / (K / 2);
            int kPair = idx % (K / 2);
            int kBase = kPair * 2;
            int g = row * K + kBase;
            int lo = A.get(g).getHalfFloatValue() & 0xFFFF;
            int hi = A.get(g + 1).getHalfFloatValue() & 0xFFFF;
            aTile[row * (K / 2) + kPair] = lo | (hi << 16);
        }
        // Cooperatively pack B [16,8] row-major into bTile.
        for (int idx = tid; idx < K * N / 2; idx += WARP_SIZE) {
            int kRow = idx / (N / 2);
            int jPair = idx % (N / 2);
            int jBase = jPair * 2;
            int g = kRow * N + jBase;
            int lo = B.get(g).getHalfFloatValue() & 0xFFFF;
            int hi = B.get(g + 1).getHalfFloatValue() & 0xFFFF;
            bTile[idx] = lo | (hi << 16);
        }
        ctx.localBarrier();

        HalfFloat[] fragA = ctx.mmaLoadA(aTile, K); // 8 half-floats/lane, offset 0 (single warp/tile)
        HalfFloat[] fragB = ctx.mmaLoadB(bTile, K); // 4 half-floats/lane, offset 0
        float[] fragC = ctx.mmaFragment(0.0f);
        fragC = ctx.mma(fragA, fragB, fragC, MMAShape.M16N8K16); // the one mma.sync.aligned.m16n8k16 instruction
        ctx.mmaStore(fragC, C, 0, 0, N);
    }

    private static HalfFloatArray deterministicFp16(int size, int seed) {
        HalfFloatArray arr = new HalfFloatArray(size);
        for (int i = 0; i < size; i++) {
            // Deterministic, bounded values -> exact fp16 representation, no rounding noise.
            float v = ((i * 31 + seed) % 17 - 8) / 8.0f;
            arr.set(i, new HalfFloat(v));
        }
        return arr;
    }

    private static float[] gemmReference(HalfFloatArray A, HalfFloatArray B) {
        float[] ref = new float[M * N];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A.get(i * K + k).getFloat32() * B.get(k * N + j).getFloat32();
                }
                ref[i * N + j] = sum;
            }
        }
        return ref;
    }

    private static boolean validate(String name, FloatArray got, float[] ref) {
        float maxAbs = 0.0f;
        int bad = 0;
        for (int i = 0; i < M * N; i++) {
            float diff = Math.abs(got.get(i) - ref[i]);
            maxAbs = Math.max(maxAbs, diff);
            if (diff > 1e-2f) {
                bad++;
            }
        }
        boolean ok = bad == 0;
        System.out.printf("  [%s] validation %s (max abs err %.5f, %d/%d cells out of tol)%n",
                name, ok ? "PASSED" : "FAILED", maxAbs, bad, M * N);
        return ok;
    }

    private static boolean isCUDABackend() {
        int driverIndex = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice().getBackendIndex();
        TornadoVMBackendType backend = TornadoRuntimeProvider.getTornadoRuntime().getBackendType(driverIndex);
        return backend == TornadoVMBackendType.CUDA;
    }

    public static void main(String[] args) {
        if (!isCUDABackend()) {
            System.out.println("MMA instructions are only supported for the CUDA backend.");
            return;
        }

        System.out.println("Single-tile Tensor Core (MMA) demo: C[16x8] = A[16x16] * B[16x8], fp16 -> f32");

        HalfFloatArray A = deterministicFp16(M * K, 1);
        HalfFloatArray B = deterministicFp16(K * N, 7);
        FloatArray cScalar = new FloatArray(M * N);
        FloatArray cMMA = new FloatArray(M * N);
        float[] ref = gemmReference(A, B);

        WorkerGrid2D wgScalar = new WorkerGrid2D(M, N);
        wgScalar.setLocalWork(M, N, 1);
        GridScheduler gsScalar = new GridScheduler("s_scalar.gemm", wgScalar);
        TaskGraph tgScalar = new TaskGraph("s_scalar")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, A, B)
                .task("gemm", TensorCoreMMA::gemmScalarFp16, new KernelContext(), A, B, cScalar)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cScalar);
        TornadoExecutionPlan planScalar = new TornadoExecutionPlan(tgScalar.snapshot());

        WorkerGrid1D wgMMA = new WorkerGrid1D(WARP_SIZE);
        wgMMA.setLocalWork(WARP_SIZE, 1, 1);
        GridScheduler gsMMA = new GridScheduler("s_mma.gemm", wgMMA);
        TaskGraph tgMMA = new TaskGraph("s_mma")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, A, B)
                .task("gemm", TensorCoreMMA::gemmMMASingleTile, new KernelContext(), A, B, cMMA)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cMMA);
        TornadoExecutionPlan planMMA = new TornadoExecutionPlan(tgMMA.snapshot());

        planScalar.withGridScheduler(gsScalar).execute();
        planMMA.withGridScheduler(gsMMA).execute();

        boolean okScalar = validate("scalar (no MMA)", cScalar, ref);
        boolean okMMA = validate("mma.sync (Tensor Core)", cMMA, ref);

        System.out.println(okScalar && okMMA ? "Result is correct" : "Result is INCORRECT");
    }
}
