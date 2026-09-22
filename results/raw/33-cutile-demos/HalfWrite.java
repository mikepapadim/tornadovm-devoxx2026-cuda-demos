import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public class HalfWrite {
    public static void scale(KernelContext ctx, HalfFloatArray a, float f) {
        int i = ctx.globalIdx;
        a.set(i, new HalfFloat(a.get(i).getFloat32() * f));
    }
    public static void main(String[] args) throws Exception {
        int n = 8;
        HalfFloatArray a = new HalfFloatArray(n);
        for (int i = 0; i < n; i++) a.set(i, new HalfFloat(i + 1.0f));
        WorkerGrid1D w = new WorkerGrid1D(n);
        w.setLocalWork(8, 1, 1);
        GridScheduler g = new GridScheduler("p.k", w);
        TaskGraph tg = new TaskGraph("p")
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, a)
            .task("k", HalfWrite::scale, new KernelContext(), a, 2.0f)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, a);
        try (TornadoExecutionPlan p = new TornadoExecutionPlan(tg.snapshot())) {
            p.withGridScheduler(g).execute();
        }
        for (int i = 0; i < n; i++) System.out.print(a.get(i).getFloat32() + " ");
        System.out.println("  (expected 2 4 6 8 10 12 14 16)");
    }
}
