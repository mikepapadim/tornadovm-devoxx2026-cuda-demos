/*
 * Devoxx 2026 CUDA demo — first real CUDA kernel generated from Java.
 *
 * Element-wise vector addition. Run with `tornado --printKernel` to see the
 * actual generated CUDA source that TornadoVM compiles and launches on the
 * GPU for the @Parallel loop below.
 */
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class VectorAddKernel {

    private static void vectorAdd(FloatArray a, FloatArray b, FloatArray c) {
        for (@Parallel int i = 0; i < c.getSize(); i++) {
            c.set(i, a.get(i) + b.get(i));
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int size = args.length > 0 ? Integer.parseInt(args[0]) : 1024;

        FloatArray a = new FloatArray(size);
        FloatArray b = new FloatArray(size);
        FloatArray c = new FloatArray(size);
        for (int i = 0; i < size; i++) {
            a.set(i, i);
            b.set(i, 2 * i);
        }

        TaskGraph taskGraph = new TaskGraph("vadd") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                .task("add", VectorAddKernel::vectorAdd, a, b, c) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutableTaskGraph)) {
            TornadoExecutionResult result = plan.execute();

            boolean correct = true;
            for (int i = 0; i < size; i++) {
                if (c.get(i) != a.get(i) + b.get(i)) {
                    correct = false;
                    break;
                }
            }
            System.out.println(correct ? "Result is correct" : "Result is WRONG");
            System.out.println("Total time: " + result.getProfilerResult().getTotalTime() + " ns");
        }
    }
}
