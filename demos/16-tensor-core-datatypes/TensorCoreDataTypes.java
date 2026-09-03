/*
 * TornadoVM CUDA demo — every tensor-core operand type TornadoVM can emit.
 *
 * Demo 08 shows one fp16 `mma.sync`. That is the easy one. This demo runs the
 * SAME GEMM through the four other operand types the CUDA backend can emit, so
 * the claim "TornadoVM reaches the tensor cores from Java" can be checked for
 * each numeric format rather than for fp16 alone:
 *
 *   BF16       mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
 *   int8       mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
 *   FP8 e4m3   mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
 *   FP8 e5m2   mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32
 *
 * Every kernel computes C[16x16] = A[16x32] * B[32x16] with one warp, and every
 * result is validated against a CPU reference computed from the SAME stored
 * values -- each input is round-tripped through its storage format first, so
 * the comparison measures the kernel and not the quantisation.
 *
 * Test values are small multiples of 0.5 in [-2, 2] (small integers for int8).
 * Those are exactly representable in bf16, e4m3 and e5m2 alike, and their
 * products and sums stay inside f32's exact range, so any mismatch is a real
 * mismatch rather than accumulated rounding.
 *
 * Usage:
 *   tornado --classpath . TensorCoreDataTypes
 *   tornado --printKernel --classpath . TensorCoreDataTypes   # see the PTX per type
 */
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FP8Array;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.ShortArray;

public class TensorCoreDataTypes {

    /** One warp owns the whole 16x16 output tile. */
    private static final int WARP = 32;

    private static final int M = 16;
    private static final int N = 16;
    private static final int K = 32;

    /** m16n8k16 shapes (bf16, fp16) consume 16 of K per mma. */
    private static final int K16 = 16;
    /** m16n8k32 shapes (int8, fp8) consume 32 of K per mma. */
    private static final int K32 = 32;

    // ------------------------------------------------------------------ BF16

    /**
     * BF16 shares fp16's tile shape and fragment layout: `mmaLoadA`/`mmaLoadB`
     * and `mmaStore` are bit-type-agnostic, and only the compute call differs.
     * That is why this kernel is the fp16 kernel with `mmaBF16` substituted.
     */
    public static void gemmBF16(KernelContext ctx, ShortArray a, ShortArray b, FloatArray c) {
        int lane = ctx.localIdx;

        int[] aTile = ctx.allocateIntLocalArray(M * K16 / 2);
        int[] bTile0 = ctx.allocateIntLocalArray(K16 * N / 2);
        int[] bTile1 = ctx.allocateIntLocalArray(K16 * N / 2);

        float[] fragC0 = ctx.mmaFragment(0.0f);
        float[] fragC1 = ctx.mmaFragment(0.0f);

        for (int kBase = 0; kBase < K; kBase += K16) {
            // Pack two bf16 bit patterns per int32 word.
            for (int idx = lane; idx < (M * K16) / 2; idx += WARP) {
                int elem = idx * 2;
                int r = elem / K16;
                int kk = elem % K16;
                int g = r * K + kBase + kk;
                aTile[r * (K16 / 2) + kk / 2] = (a.get(g) & 0xFFFF) | ((a.get(g + 1) & 0xFFFF) << 16);
            }
            for (int idx = lane; idx < 64; idx += WARP) {
                int kRow = idx / 4;
                int jPair = idx % 4;
                int jBase = jPair * 2;

                int gl = (kBase + kRow) * N + jBase;
                bTile0[kRow * 4 + jPair] = (b.get(gl) & 0xFFFF) | ((b.get(gl + 1) & 0xFFFF) << 16);

                int gr = (kBase + kRow) * N + 8 + jBase;
                bTile1[kRow * 4 + jPair] = (b.get(gr) & 0xFFFF) | ((b.get(gr + 1) & 0xFFFF) << 16);
            }
            ctx.localBarrier();

            HalfFloat[] fragA = ctx.mmaLoadA(aTile, K16);
            HalfFloat[] fragB0 = ctx.mmaLoadB(bTile0, K16);
            fragC0 = ctx.mmaBF16(fragA, fragB0, fragC0, MMAShape.M16N8K16);
            HalfFloat[] fragB1 = ctx.mmaLoadB(bTile1, K16);
            fragC1 = ctx.mmaBF16(fragA, fragB1, fragC1, MMAShape.M16N8K16);

            ctx.localBarrier();
        }

        ctx.mmaStore(fragC0, c, 0, 0, N);
        ctx.mmaStore(fragC1, c, 0, 8, N);
    }

    // ------------------------------------------------------------------ int8

    /** int8 uses the wider m16n8k32 shape and an s32 accumulator. */
    public static void gemmInt8(KernelContext ctx, ByteArray a, ByteArray b, IntArray c) {
        int lane = ctx.localIdx;

        int[] aTile = ctx.allocateIntLocalArray(M * K32 / 4);
        int[] bTile0 = ctx.allocateIntLocalArray(K32 * 8 / 4);
        int[] bTile1 = ctx.allocateIntLocalArray(K32 * 8 / 4);

        int[] fragC0 = ctx.mmaFragmentInt(0);
        int[] fragC1 = ctx.mmaFragmentInt(0);

        // Pack four int8 lanes per int32 word.
        for (int idx = lane; idx < (M * K32) / 4; idx += WARP) {
            int elem = idx * 4;
            int r = elem / K32;
            int kk = elem % K32;
            int g = r * K + kk;
            aTile[r * (K32 / 4) + kk / 4] = pack4(a.get(g), a.get(g + 1), a.get(g + 2), a.get(g + 3));
        }
        for (int idx = lane; idx < 64; idx += WARP) {
            int kRow = idx / 4;
            int jPair = idx % 4;
            int jBase = jPair * 2;
            int kb = 2 * kRow;

            bTile0[kRow * 4 + jPair] = pack4(b.get(kb * N + jBase), b.get((kb + 1) * N + jBase), //
                    b.get(kb * N + jBase + 1), b.get((kb + 1) * N + jBase + 1));
            bTile1[kRow * 4 + jPair] = pack4(b.get(kb * N + 8 + jBase), b.get((kb + 1) * N + 8 + jBase), //
                    b.get(kb * N + 8 + jBase + 1), b.get((kb + 1) * N + 8 + jBase + 1));
        }
        ctx.localBarrier();

        byte[] fragA = ctx.mmaLoadAInt8(aTile, K32);
        byte[] fragB0 = ctx.mmaLoadBInt8(bTile0, K32);
        fragC0 = ctx.mmaInt8(fragA, fragB0, fragC0, MMAShape.M16N8K32);
        byte[] fragB1 = ctx.mmaLoadBInt8(bTile1, K32);
        fragC1 = ctx.mmaInt8(fragA, fragB1, fragC1, MMAShape.M16N8K32);

        ctx.mmaStoreInt(fragC0, c, 0, 0, N);
        ctx.mmaStoreInt(fragC1, c, 0, 8, N);
    }

    // ------------------------------------------------------------------- FP8

    /**
     * e4m3 and e5m2 differ only in the compute call. The bytes are staged
     * identically, which is the point worth seeing: the *format* is chosen by
     * the instruction, not by the data layout.
     */
    public static void gemmFP8E4M3(KernelContext ctx, FP8Array a, FP8Array b, FloatArray c) {
        int lane = ctx.localIdx;

        int[] aTile = ctx.allocateIntLocalArray(M * K32 / 4);
        int[] bTile0 = ctx.allocateIntLocalArray(K32 * 8 / 4);
        int[] bTile1 = ctx.allocateIntLocalArray(K32 * 8 / 4);

        float[] fragC0 = ctx.mmaFragment(0.0f);
        float[] fragC1 = ctx.mmaFragment(0.0f);

        stageFP8(ctx, a, b, aTile, bTile0, bTile1, lane);

        byte[] fragA = ctx.mmaLoadAFP8(aTile, K32);
        byte[] fragB0 = ctx.mmaLoadBFP8(bTile0, K32);
        fragC0 = ctx.mmaFP8E4M3(fragA, fragB0, fragC0, MMAShape.M16N8K32);
        byte[] fragB1 = ctx.mmaLoadBFP8(bTile1, K32);
        fragC1 = ctx.mmaFP8E4M3(fragA, fragB1, fragC1, MMAShape.M16N8K32);

        ctx.mmaStore(fragC0, c, 0, 0, N);
        ctx.mmaStore(fragC1, c, 0, 8, N);
    }

    public static void gemmFP8E5M2(KernelContext ctx, FP8Array a, FP8Array b, FloatArray c) {
        int lane = ctx.localIdx;

        int[] aTile = ctx.allocateIntLocalArray(M * K32 / 4);
        int[] bTile0 = ctx.allocateIntLocalArray(K32 * 8 / 4);
        int[] bTile1 = ctx.allocateIntLocalArray(K32 * 8 / 4);

        float[] fragC0 = ctx.mmaFragment(0.0f);
        float[] fragC1 = ctx.mmaFragment(0.0f);

        stageFP8(ctx, a, b, aTile, bTile0, bTile1, lane);

        byte[] fragA = ctx.mmaLoadAFP8(aTile, K32);
        byte[] fragB0 = ctx.mmaLoadBFP8(bTile0, K32);
        fragC0 = ctx.mmaFP8E5M2(fragA, fragB0, fragC0, MMAShape.M16N8K32);
        byte[] fragB1 = ctx.mmaLoadBFP8(bTile1, K32);
        fragC1 = ctx.mmaFP8E5M2(fragA, fragB1, fragC1, MMAShape.M16N8K32);

        ctx.mmaStore(fragC0, c, 0, 0, N);
        ctx.mmaStore(fragC1, c, 0, 8, N);
    }

    private static void stageFP8(KernelContext ctx, FP8Array a, FP8Array b, //
            int[] aTile, int[] bTile0, int[] bTile1, int lane) {
        for (int idx = lane; idx < (M * K32) / 4; idx += WARP) {
            int elem = idx * 4;
            int r = elem / K32;
            int kk = elem % K32;
            int g = r * K + kk;
            aTile[r * (K32 / 4) + kk / 4] = pack4(a.get(g), a.get(g + 1), a.get(g + 2), a.get(g + 3));
        }
        for (int idx = lane; idx < 64; idx += WARP) {
            int kRow = idx / 4;
            int jPair = idx % 4;
            int jBase = jPair * 2;
            int kb = 2 * kRow;

            bTile0[kRow * 4 + jPair] = pack4(b.get(kb * N + jBase), b.get((kb + 1) * N + jBase), //
                    b.get(kb * N + jBase + 1), b.get((kb + 1) * N + jBase + 1));
            bTile1[kRow * 4 + jPair] = pack4(b.get(kb * N + 8 + jBase), b.get((kb + 1) * N + 8 + jBase), //
                    b.get(kb * N + 8 + jBase + 1), b.get((kb + 1) * N + 8 + jBase + 1));
        }
        ctx.localBarrier();
    }

    private static int pack4(byte b0, byte b1, byte b2, byte b3) {
        return (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }

    // --------------------------------------------------------------- driver

    /** bf16 is the top 16 bits of the f32 bit pattern; these values are exact, so truncation is lossless. */
    private static short toBF16(float v) {
        return (short) (Float.floatToIntBits(v) >>> 16);
    }

    private static float fromBF16(short s) {
        return Float.intBitsToFloat(s << 16);
    }

    /** Small multiples of 0.5 in [-2, 2] — exact in bf16, e4m3 and e5m2 alike. */
    private static float value(int i) {
        return ((i * 7 + 3) % 9) * 0.5f - 2.0f;
    }

    private static int intValue(int i) {
        return ((i * 7 + 3) % 9) - 4;
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (TornadoRuntimeProvider.getTornadoRuntime().getBackendType(0) != TornadoVMBackendType.CUDA) {
            System.out.println("This demo requires the CUDA backend.");
            return;
        }

        System.out.printf("Tensor-core operand types from Java: C[%dx%d] = A[%dx%d] * B[%dx%d], one warp%n", //
                M, N, M, K, K, N);
        System.out.println("  each result validated against a CPU reference over the same stored values");
        System.out.println();

        boolean allOk = true;
        allOk &= runBF16();
        allOk &= runInt8();
        allOk &= runFP8(true);
        allOk &= runFP8(false);

        System.out.println();
        System.out.println(allOk //
                ? "All four operand types produced correct results" //
                : "At least one operand type FAILED — see above");
    }

    private static boolean runBF16() throws TornadoExecutionPlanException {
        ShortArray a = new ShortArray(M * K);
        ShortArray b = new ShortArray(K * N);
        for (int i = 0; i < M * K; i++) {
            a.set(i, toBF16(value(i)));
        }
        for (int i = 0; i < K * N; i++) {
            b.set(i, toBF16(value(i + 5)));
        }

        FloatArray c = new FloatArray(M * N);
        float[] ref = new float[M * N];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    acc += fromBF16(a.get(i * K + k)) * fromBF16(b.get(k * N + j));
                }
                ref[i * N + j] = acc;
            }
        }

        execute("mma_bf16", g -> g.task("gemm", TensorCoreDataTypes::gemmBF16, new KernelContext(), a, b, c), a, b, c);
        return report("BF16     m16n8k16  f32.bf16.bf16.f32", ref, i -> c.get(i));
    }

    private static boolean runInt8() throws TornadoExecutionPlanException {
        ByteArray a = new ByteArray(M * K);
        ByteArray b = new ByteArray(K * N);
        for (int i = 0; i < M * K; i++) {
            a.set(i, (byte) intValue(i));
        }
        for (int i = 0; i < K * N; i++) {
            b.set(i, (byte) intValue(i + 5));
        }

        IntArray c = new IntArray(M * N);
        float[] ref = new float[M * N];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int acc = 0;
                for (int k = 0; k < K; k++) {
                    acc += a.get(i * K + k) * b.get(k * N + j);
                }
                ref[i * N + j] = acc;
            }
        }

        execute("mma_int8", g -> g.task("gemm", TensorCoreDataTypes::gemmInt8, new KernelContext(), a, b, c), a, b, c);
        return report("int8     m16n8k32  s32.s8.s8.s32    ", ref, i -> (float) c.get(i));
    }

    private static boolean runFP8(boolean e4m3) throws TornadoExecutionPlanException {
        FP8Array a = new FP8Array(M * K);
        FP8Array b = new FP8Array(K * N);
        for (int i = 0; i < M * K; i++) {
            if (e4m3) {
                a.setE4M3(i, value(i));
            } else {
                a.setE5M2(i, value(i));
            }
        }
        for (int i = 0; i < K * N; i++) {
            if (e4m3) {
                b.setE4M3(i, value(i + 5));
            } else {
                b.setE5M2(i, value(i + 5));
            }
        }

        FloatArray c = new FloatArray(M * N);
        float[] ref = new float[M * N];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    float av = e4m3 ? a.getE4M3(i * K + k) : a.getE5M2(i * K + k);
                    float bv = e4m3 ? b.getE4M3(k * N + j) : b.getE5M2(k * N + j);
                    acc += av * bv;
                }
                ref[i * N + j] = acc;
            }
        }

        String name = e4m3 ? "mma_fp8_e4m3" : "mma_fp8_e5m2";
        execute(name, g -> e4m3 //
                ? g.task("gemm", TensorCoreDataTypes::gemmFP8E4M3, new KernelContext(), a, b, c) //
                : g.task("gemm", TensorCoreDataTypes::gemmFP8E5M2, new KernelContext(), a, b, c), a, b, c);

        String label = e4m3 //
                ? "FP8 e4m3 m16n8k32  f32.e4m3.e4m3.f32" //
                : "FP8 e5m2 m16n8k32  f32.e5m2.e5m2.f32";
        return report(label, ref, i -> c.get(i));
    }

    @FunctionalInterface
    private interface TaskAdder {
        TaskGraph add(TaskGraph graph);
    }

    @FunctionalInterface
    private interface Reader {
        float get(int index);
    }

    private static void execute(String graphName, TaskAdder adder, Object a, Object b, Object c) //
            throws TornadoExecutionPlanException {
        WorkerGrid1D worker = new WorkerGrid1D(WARP);
        worker.setLocalWork(WARP, 1, 1);
        GridScheduler scheduler = new GridScheduler(graphName + ".gemm", worker);

        TaskGraph graph = new TaskGraph(graphName) //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b);
        graph = adder.add(graph);
        graph = graph.transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        ImmutableTaskGraph itg = graph.snapshot();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(itg)) {
            plan.withGridScheduler(scheduler).execute();
        }
    }

    private static boolean report(String label, float[] ref, Reader actual) {
        float maxErr = 0.0f;
        int bad = 0;
        for (int i = 0; i < ref.length; i++) {
            float err = Math.abs(ref[i] - actual.get(i));
            maxErr = Math.max(maxErr, err);
            if (err > 1e-3f) {
                bad++;
            }
        }
        System.out.printf("  %s  %s (max abs err %.5f, %d/%d cells out of tol)%n", //
                label, bad == 0 ? "PASSED" : "FAILED", maxErr, bad, ref.length);
        return bad == 0;
    }
}
