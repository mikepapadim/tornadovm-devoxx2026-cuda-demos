import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.tile.*;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.*;

public class TileProbe {

    private static final int T32X32X32_M = 32, T32X32X32_N = 32, T32X32X32_K = 32;
    public static void t32x32x32(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T32X32X32_M, T32X32X32_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T32X32X32_K, T32X32X32_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T32X32X32_M, T32X32X32_N);
        int rb = tc.bidX();
        int cb = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T32X32X32_M, T32X32X32_N);
        for (int s = 0; s < n / T32X32X32_K; s++) {
            acc = tc.mma(av.load(rb, s), bv.load(s, cb), acc);
        }
        cv.store(acc, rb, cb);
    }
    private static final int T64X64X32_M = 64, T64X64X32_N = 64, T64X64X32_K = 32;
    public static void t64x64x32(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T64X64X32_M, T64X64X32_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T64X64X32_K, T64X64X32_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T64X64X32_M, T64X64X32_N);
        int rb = tc.bidX();
        int cb = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T64X64X32_M, T64X64X32_N);
        for (int s = 0; s < n / T64X64X32_K; s++) {
            acc = tc.mma(av.load(rb, s), bv.load(s, cb), acc);
        }
        cv.store(acc, rb, cb);
    }
    private static final int T64X64X64_M = 64, T64X64X64_N = 64, T64X64X64_K = 64;
    public static void t64x64x64(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T64X64X64_M, T64X64X64_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T64X64X64_K, T64X64X64_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T64X64X64_M, T64X64X64_N);
        int rb = tc.bidX();
        int cb = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T64X64X64_M, T64X64X64_N);
        for (int s = 0; s < n / T64X64X64_K; s++) {
            acc = tc.mma(av.load(rb, s), bv.load(s, cb), acc);
        }
        cv.store(acc, rb, cb);
    }
    private static final int T128X64X32_M = 128, T128X64X32_N = 64, T128X64X32_K = 32;
    public static void t128x64x32(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T128X64X32_M, T128X64X32_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T128X64X32_K, T128X64X32_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T128X64X32_M, T128X64X32_N);
        int rb = tc.bidX();
        int cb = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T128X64X32_M, T128X64X32_N);
        for (int s = 0; s < n / T128X64X32_K; s++) {
            acc = tc.mma(av.load(rb, s), bv.load(s, cb), acc);
        }
        cv.store(acc, rb, cb);
    }
    private static final int T128X128X32_M = 128, T128X128X32_N = 128, T128X128X32_K = 32;
    public static void t128x128x32(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T128X128X32_M, T128X128X32_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T128X128X32_K, T128X128X32_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T128X128X32_M, T128X128X32_N);
        int rb = tc.bidX();
        int cb = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T128X128X32_M, T128X128X32_N);
        for (int s = 0; s < n / T128X128X32_K; s++) {
            acc = tc.mma(av.load(rb, s), bv.load(s, cb), acc);
        }
        cv.store(acc, rb, cb);
    }
    private static final int T128X128X64_M = 128, T128X128X64_N = 128, T128X128X64_K = 64;
    public static void t128x128x64(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T128X128X64_M, T128X128X64_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T128X128X64_K, T128X128X64_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T128X128X64_M, T128X128X64_N);
        int rb = tc.bidX();
        int cb = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T128X128X64_M, T128X128X64_N);
        for (int s = 0; s < n / T128X128X64_K; s++) {
            acc = tc.mma(av.load(rb, s), bv.load(s, cb), acc);
        }
        cv.store(acc, rb, cb);
    }
    private static final int T256X128X32_M = 256, T256X128X32_N = 128, T256X128X32_K = 32;
    public static void t256x128x32(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T256X128X32_M, T256X128X32_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T256X128X32_K, T256X128X32_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T256X128X32_M, T256X128X32_N);
        int rb = tc.bidX();
        int cb = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T256X128X32_M, T256X128X32_N);
        for (int s = 0; s < n / T256X128X32_K; s++) {
            acc = tc.mma(av.load(rb, s), bv.load(s, cb), acc);
        }
        cv.store(acc, rb, cb);
    }

    private static final int T128X128X128_M = 128, T128X128X128_N = 128, T128X128X128_K = 128;
    public static void t128x128x128(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T128X128X128_M, T128X128X128_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T128X128X128_K, T128X128X128_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T128X128X128_M, T128X128X128_N);
        int rb = tc.bidX();
        int cb = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T128X128X128_M, T128X128X128_N);
        for (int s = 0; s < n / T128X128X128_K; s++) {
            acc = tc.mma(av.load(rb, s), bv.load(s, cb), acc);
        }
        cv.store(acc, rb, cb);
    }
    private static final int T64X128X64_M = 64, T64X128X64_N = 128, T64X128X64_K = 64;
    public static void t64x128x64(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T64X128X64_M, T64X128X64_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T64X128X64_K, T64X128X64_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T64X128X64_M, T64X128X64_N);
        int rb = tc.bidX();
        int cb = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T64X128X64_M, T64X128X64_N);
        for (int s = 0; s < n / T64X128X64_K; s++) {
            acc = tc.mma(av.load(rb, s), bv.load(s, cb), acc);
        }
        cv.store(acc, rb, cb);
    }
    private static final int T128X64X64_M = 128, T128X64X64_N = 64, T128X64X64_K = 64;
    public static void t128x64x64(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T128X64X64_M, T128X64X64_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T128X64X64_K, T128X64X64_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T128X64X64_M, T128X64X64_N);
        int rb = tc.bidX();
        int cb = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T128X64X64_M, T128X64X64_N);
        for (int s = 0; s < n / T128X64X64_K; s++) {
            acc = tc.mma(av.load(rb, s), bv.load(s, cb), acc);
        }
        cv.store(acc, rb, cb);
    }
    private static final int T256X128X64_M = 256, T256X128X64_N = 128, T256X128X64_K = 64;
    public static void t256x128x64(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T256X128X64_M, T256X128X64_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T256X128X64_K, T256X128X64_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T256X128X64_M, T256X128X64_N);
        int rb = tc.bidX();
        int cb = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T256X128X64_M, T256X128X64_N);
        for (int s = 0; s < n / T256X128X64_K; s++) {
            acc = tc.mma(av.load(rb, s), bv.load(s, cb), acc);
        }
        cv.store(acc, rb, cb);
    }
    private static final int T128X256X64_M = 128, T128X256X64_N = 256, T128X256X64_K = 64;
    public static void t128x256x64(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T128X256X64_M, T128X256X64_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T128X256X64_K, T128X256X64_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T128X256X64_M, T128X256X64_N);
        int rb = tc.bidX();
        int cb = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T128X256X64_M, T128X256X64_N);
        for (int s = 0; s < n / T128X256X64_K; s++) {
            acc = tc.mma(av.load(rb, s), bv.load(s, cb), acc);
        }
        cv.store(acc, rb, cb);
    }
    private static final int T64X64X128_M = 64, T64X64X128_N = 64, T64X64X128_K = 128;
    public static void t64x64x128(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T64X64X128_M, T64X64X128_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T64X64X128_K, T64X64X128_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T64X64X128_M, T64X64X128_N);
        int rb = tc.bidX();
        int cb = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T64X64X128_M, T64X64X128_N);
        for (int s = 0; s < n / T64X64X128_K; s++) {
            acc = tc.mma(av.load(rb, s), bv.load(s, cb), acc);
        }
        cv.store(acc, rb, cb);
    }
    private static final int T256X256X64_M = 256, T256X256X64_N = 256, T256X256X64_K = 64;
    public static void t256x256x64(TileContext tc, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        PartitionView av = tc.partition(tc.view(a, n, n), T256X256X64_M, T256X256X64_K);
        PartitionView bv = tc.partition(tc.view(b, n, n), T256X256X64_K, T256X256X64_N);
        PartitionView cv = tc.partition(tc.view(c, n, n), T256X256X64_M, T256X256X64_N);
        int rb = tc.bidX();
        int cb = tc.bidY();
        Tile acc = tc.zeros(DType.F32, T256X256X64_M, T256X256X64_N);
        for (int s = 0; s < n / T256X256X64_K; s++) {
            acc = tc.mma(av.load(rb, s), bv.load(s, cb), acc);
        }
        cv.store(acc, rb, cb);
    }

    public static void main(String[] args) throws Exception {
        String shape = args[0];
        int n = Integer.parseInt(args[1]);
        int reps = Integer.parseInt(args[2]);
        HalfFloatArray a = new HalfFloatArray(n * n), b = new HalfFloatArray(n * n);
        for (int i = 0; i < n * n; i++) {
            a.set(i, new HalfFloat(((i * 7 + 3) % 17) * 0.0625f - 0.5f));
            b.set(i, new HalfFloat(((i * 5 + 11) % 13) * 0.0769f - 0.5f));
        }
        FloatArray c = new FloatArray(n * n);
        TaskGraph g; WorkerGrid w;
        switch (shape) {
            case "t32x32x32": g = new TaskGraph("g").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                    .task("t", TileProbe::t32x32x32, new TileContext(), a, b, c, n).transferToHost(DataTransferMode.EVERY_EXECUTION, c);
                    w = new WorkerGrid2D(n / 32, n / 32); break;
            case "t64x64x32": g = new TaskGraph("g").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                    .task("t", TileProbe::t64x64x32, new TileContext(), a, b, c, n).transferToHost(DataTransferMode.EVERY_EXECUTION, c);
                    w = new WorkerGrid2D(n / 64, n / 64); break;
            case "t64x64x64": g = new TaskGraph("g").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                    .task("t", TileProbe::t64x64x64, new TileContext(), a, b, c, n).transferToHost(DataTransferMode.EVERY_EXECUTION, c);
                    w = new WorkerGrid2D(n / 64, n / 64); break;
            case "t128x64x32": g = new TaskGraph("g").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                    .task("t", TileProbe::t128x64x32, new TileContext(), a, b, c, n).transferToHost(DataTransferMode.EVERY_EXECUTION, c);
                    w = new WorkerGrid2D(n / 128, n / 64); break;
            case "t128x128x32": g = new TaskGraph("g").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                    .task("t", TileProbe::t128x128x32, new TileContext(), a, b, c, n).transferToHost(DataTransferMode.EVERY_EXECUTION, c);
                    w = new WorkerGrid2D(n / 128, n / 128); break;
            case "t128x128x64": g = new TaskGraph("g").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                    .task("t", TileProbe::t128x128x64, new TileContext(), a, b, c, n).transferToHost(DataTransferMode.EVERY_EXECUTION, c);
                    w = new WorkerGrid2D(n / 128, n / 128); break;
            case "t256x128x32": g = new TaskGraph("g").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                    .task("t", TileProbe::t256x128x32, new TileContext(), a, b, c, n).transferToHost(DataTransferMode.EVERY_EXECUTION, c);
                    w = new WorkerGrid2D(n / 256, n / 128); break;
            case "t128x128x128": g = new TaskGraph("g").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                    .task("t", TileProbe::t128x128x128, new TileContext(), a, b, c, n).transferToHost(DataTransferMode.EVERY_EXECUTION, c);
                    w = new WorkerGrid2D(n / 128, n / 128); break;
            case "t64x128x64": g = new TaskGraph("g").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                    .task("t", TileProbe::t64x128x64, new TileContext(), a, b, c, n).transferToHost(DataTransferMode.EVERY_EXECUTION, c);
                    w = new WorkerGrid2D(n / 64, n / 128); break;
            case "t128x64x64": g = new TaskGraph("g").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                    .task("t", TileProbe::t128x64x64, new TileContext(), a, b, c, n).transferToHost(DataTransferMode.EVERY_EXECUTION, c);
                    w = new WorkerGrid2D(n / 128, n / 64); break;
            case "t256x128x64": g = new TaskGraph("g").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                    .task("t", TileProbe::t256x128x64, new TileContext(), a, b, c, n).transferToHost(DataTransferMode.EVERY_EXECUTION, c);
                    w = new WorkerGrid2D(n / 256, n / 128); break;
            case "t128x256x64": g = new TaskGraph("g").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                    .task("t", TileProbe::t128x256x64, new TileContext(), a, b, c, n).transferToHost(DataTransferMode.EVERY_EXECUTION, c);
                    w = new WorkerGrid2D(n / 128, n / 256); break;
            case "t64x64x128": g = new TaskGraph("g").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                    .task("t", TileProbe::t64x64x128, new TileContext(), a, b, c, n).transferToHost(DataTransferMode.EVERY_EXECUTION, c);
                    w = new WorkerGrid2D(n / 64, n / 64); break;
            case "t256x256x64": g = new TaskGraph("g").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                    .task("t", TileProbe::t256x256x64, new TileContext(), a, b, c, n).transferToHost(DataTransferMode.EVERY_EXECUTION, c);
                    w = new WorkerGrid2D(n / 256, n / 256); break;
            default: throw new IllegalArgumentException(shape);
        }
        try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
            p.withGridScheduler(new GridScheduler("g.t", w));
            for (int r = 0; r < reps; r++) p.execute();
        }
        // spot-check 256 cells against a CPU dot product over the same fp16 inputs
        int bad = 0; float maxErr = 0f; java.util.Random rnd = new java.util.Random(1);
        for (int t = 0; t < 256; t++) {
            int i = rnd.nextInt(n), j = rnd.nextInt(n); float s = 0f;
            for (int k = 0; k < n; k++) s += a.get(i * n + k).getFloat32() * b.get(k * n + j).getFloat32();
            float e = Math.abs(s - c.get(i * n + j)); maxErr = Math.max(maxErr, e);
            if (e > 0.05f * n / 256f * Math.max(1f, Math.abs(s))) bad++;
        }
        System.out.printf("%s n=%d %s maxErr=%.4f bad=%d/256%n", shape, n, bad == 0 ? "PASSED" : "FAILED", maxErr, bad);
    }
}
