/*
 * TornadoVM CUDA demo — the smallest possible TornadoVM GPU program.
 *
 * Adds 1 to every element of an int array, on the GPU, using the pinned
 * TornadoVM CUDA backend (see env/versions.env for the exact SHA/GPU).
 */
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.Arrays;

public class Hello {

    // @Parallel tells TornadoVM this loop body can run as one GPU thread
    // per iteration instead of a sequential CPU loop.
    private static void addOne(IntArray in, IntArray out) {
        for (@Parallel int i = 0; i < in.getSize(); i++) {
            out.set(i, in.get(i) + 1);
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int size = 8;
        IntArray in = new IntArray(size);
        IntArray out = new IntArray(size);
        for (int i = 0; i < size; i++) {
            in.set(i, i);
        }

        // A TaskGraph is TornadoVM's unit of GPU work: move data to the
        // device, run one or more tasks, move results back.
        TaskGraph taskGraph = new TaskGraph("hello") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, in) //
                .task("addOne", Hello::addOne, in, out) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutableTaskGraph)) {
            plan.execute();
        }

        System.out.println("in:  " + Arrays.toString(in.toHeapArray()));
        System.out.println("out: " + Arrays.toString(out.toHeapArray()));
    }
}
