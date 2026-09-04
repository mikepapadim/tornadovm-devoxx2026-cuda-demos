import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.*;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

/** Prints every value TestProfiler#testProfilerEnabled asserts on, instead of just failing. */
public class ProbeProfiler {
    public static void add(IntArray a, IntArray b, IntArray c) {
        for (@Parallel int i = 0; i < c.getSize(); i++) c.set(i, a.get(i) + b.get(i));
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int n = 16;
        IntArray a = new IntArray(n), b = new IntArray(n), c = new IntArray(n);
        a.init(1); b.init(2);
        TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice().clean();
        TaskGraph tg = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                .task("t0", ProbeProfiler::add, a, b, c)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            TornadoExecutionResult r = plan.withProfiler(ProfilerMode.SILENT).execute();
            var p = r.getProfilerResult();
            System.out.println("---- values the test asserts on ----");
            System.out.printf("getTotalTime()                > 0  : %d%n", p.getTotalTime());
            System.out.printf("getTornadoCompilerTime()      > 0  : %d%n", p.getTornadoCompilerTime());
            System.out.printf("getCompileTime()              > 0  : %d%n", p.getCompileTime());
            System.out.printf("getDataTransfersTime()       >= 0  : %d%n", p.getDataTransfersTime());
            System.out.printf("getDeviceReadTime()          >= 0  : %d%n", p.getDeviceReadTime());
            System.out.printf("getDeviceWriteTime()         >= 0  : %d%n", p.getDeviceWriteTime());
            System.out.printf("getDataTransferDispatchTime() > 0  : %d%n", p.getDataTransferDispatchTime());
            System.out.printf("getKernelDispatchTime()       > 0  : %d%n", p.getKernelDispatchTime());
            System.out.printf("getDeviceReadTime()           > 0  : %d%n", p.getDeviceReadTime());
            System.out.printf("write+read == dataTransfers   : %d + %d = %d  vs %d%n",
                    p.getDeviceWriteTime(), p.getDeviceReadTime(),
                    p.getDeviceWriteTime() + p.getDeviceReadTime(), p.getDataTransfersTime());
        }
    }
}
