/*
 * Minimal reproducer: per-execution 256-byte host<->device control transfers.
 *
 * One task graph, one trivial kernel, one input and one output. Executed N
 * times inside a single TornadoExecutionPlan. Nothing here asks for any
 * host<->device traffic per execution except the declared transfers.
 *
 * Build & run:
 *   javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d . DispatchOverhead.java
 *   nsys profile --trace=cuda -o dispatch \
 *     $JAVA_HOME/bin/java @$TORNADOVM_HOME/tornado-argfile -cp . DispatchOverhead <n> <execs>
 *   nsys stats --report cuda_api_sum --format csv dispatch.nsys-rep
 */
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class DispatchOverhead {

    public static void scale(FloatArray in, FloatArray out) {
        for (@Parallel int i = 0; i < out.getSize(); i++) {
            out.set(i, in.get(i) * 2.0f);
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 1024;
        int executions = args.length > 1 ? Integer.parseInt(args[1]) : 20;

        FloatArray in = new FloatArray(n);
        FloatArray out = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            in.set(i, i);
        }

        TaskGraph graph = new TaskGraph("s") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, in) //
                .task("t", DispatchOverhead::scale, in, out) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        ImmutableTaskGraph itg = graph.snapshot();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(itg)) {
            for (int i = 0; i < executions; i++) {
                plan.execute();
            }
        }

        System.out.printf("n=%d executions=%d out[0]=%.1f out[%d]=%.1f%n", //
                n, executions, out.get(0), n - 1, out.get(n - 1));
    }
}
