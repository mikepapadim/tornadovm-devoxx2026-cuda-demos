/*
 * TornadoVM CUDA demo — CUDA runtime API access from Java, outside library tasks.
 *
 * TornadoExecutionPlan#withCUDAGraph() captures the whole task-graph (H2D copy,
 * kernel launch, D2H copy) as a single CUDA graph (cudaGraphCreate / cudaStreamBeginCapture
 * under the hood) the first time the plan executes, then replays that graph
 * (cudaGraphLaunch) on every subsequent execute() call instead of re-issuing each CUDA
 * runtime call individually. This is CUDA-only runtime behaviour: TornadoExecutionPlan's
 * javadoc records withCUDAGraph()/withIntraPlanConcurrency()/withStagedTransfers() as
 * "currently realised on the CUDA backend" (streams/graphs/pinned-memory staging), unlike
 * the vendor-library (cuBLAS/cuFFT/...) hybrid-API tasks covered elsewhere in this repo.
 *
 * The inputs are mutated before every replay and the result validated against a CPU
 * reference after every replay: a stale/incorrectly captured graph would keep returning
 * the first execution's output instead of observing the new input values.
 */
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class CudaGraphReplay {

    private static final float ALPHA = 0.5f;

    /** axpy: result = alpha * x + y */
    private static void axpy(FloatArray x, FloatArray y, FloatArray result, float alpha) {
        for (@Parallel int i = 0; i < result.getSize(); i++) {
            result.set(i, alpha * x.get(i) + y.get(i));
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int size = args.length > 0 ? Integer.parseInt(args[0]) : 8192;
        int replays = args.length > 1 ? Integer.parseInt(args[1]) : 8;

        FloatArray x = new FloatArray(size);
        FloatArray y = new FloatArray(size);
        FloatArray result = new FloatArray(size);

        TaskGraph taskGraph = new TaskGraph("cudaGraphDemo") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x, y) //
                .task("axpy", CudaGraphReplay::axpy, x, y, result, ALPHA) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, result);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutableTaskGraph)) {
            // CUDA runtime API, not a library task: captures the graph on first execute(),
            // replays it (cudaGraphLaunch) on every call after that.
            plan.withCUDAGraph();

            boolean allCorrect = true;
            for (int i = 0; i < replays; i++) {
                float xValue = 1.0f + i;
                float yValue = 2.0f + i;
                x.init(xValue);
                y.init(yValue);

                plan.execute();

                float expected = ALPHA * xValue + yValue;
                boolean correct = true;
                for (int j = 0; j < size; j++) {
                    if (Math.abs(result.get(j) - expected) > 1e-3f) {
                        correct = false;
                        break;
                    }
                }
                allCorrect &= correct;
                System.out.println("replay " + i + " (x=" + xValue + ", y=" + yValue + "): "
                        + (correct ? "correct" : "WRONG") + ", result[0]=" + result.get(0)
                        + " expected=" + expected);
            }
            System.out.println(allCorrect ? "All replays correct" : "Some replays WRONG");
        }
    }
}
