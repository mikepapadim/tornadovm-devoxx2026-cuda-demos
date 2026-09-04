/*
 * Task B2 — separate the alignment effect from the launch-geometry effect.
 *
 * Iteration 3 retracted an attribution because TornadoVM's JIT elementwise
 * kernels ran at block=1024 while the hand-written CUDA ran at block=256, so
 * the ratio between them confounded two things. This probe pins the block size
 * explicitly on the TornadoVM side with a GridScheduler and sweeps it, giving a
 * 2x2 over {implementation} x {block size}:
 *
 *   TornadoVM@256 vs CUDA@256   -> layout + codegen, geometry controlled
 *   TornadoVM@1024 vs @256      -> pure occupancy effect within one implementation
 *   CUDA@1024 vs @256           -> the same effect, hand-written baseline
 *
 * Kernel is demo 15's elementwise (out = in * 0.25 + 0.1), the simplest
 * bandwidth-bound shape, so the sector penalty is not masked by arithmetic.
 *
 * Usage:
 *   tornado --classpath . GeometryControlled <n> <block> <executions>
 */
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class GeometryControlled {

    public static void elementwise(KernelContext ctx, FloatArray in, FloatArray out) {
        int i = ctx.globalIdx;
        if (i < out.getSize()) {
            out.set(i, in.get(i) * 0.25f + 0.1f);
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : (1 << 24);
        int block = args.length > 1 ? Integer.parseInt(args[1]) : 256;
        int executions = args.length > 2 ? Integer.parseInt(args[2]) : 20;

        FloatArray in = new FloatArray(n);
        FloatArray out = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            in.set(i, i * 0.001f);
        }

        WorkerGrid1D worker = new WorkerGrid1D(n);
        worker.setLocalWork(block, 1, 1);
        GridScheduler scheduler = new GridScheduler("geom.ew", worker);

        TaskGraph graph = new TaskGraph("geom") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, in) //
                .task("ew", GeometryControlled::elementwise, new KernelContext(), in, out) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        ImmutableTaskGraph itg = graph.snapshot();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(itg)) {
            for (int e = 0; e < executions; e++) {
                plan.withGridScheduler(scheduler).execute();
            }
        }

        // Validate against the same arithmetic on the host.
        int wrong = 0;
        for (int i = 0; i < n; i += 4096) {
            float expected = in.get(i) * 0.25f + 0.1f;
            if (Math.abs(out.get(i) - expected) > 1e-5f) {
                wrong++;
            }
        }
        System.out.printf("n=%d block=%d executions=%d sampled-mismatches=%d %s%n", //
                n, block, executions, wrong, wrong == 0 ? "PASSED" : "FAILED");
    }
}
