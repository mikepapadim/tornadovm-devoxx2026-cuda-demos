/*
 * Devoxx 2026 CUDA demo — Java kernel + cuBLAS in one execution graph.
 *
 * One TaskGraph, three stages, all on the same device buffers:
 *   1. JIT-compiled @Parallel task ("scale")  — Java kernel on the GPU
 *   2. cuBLAS library task ("sgemv")          — y = alpha * op(A) * x + beta * y
 *   3. JIT-compiled @Parallel task ("bias")   — Java kernel on the GPU
 *
 * cuBLAS assumes column-major storage; matrix is built and validated
 * row-major here, so CUBLAS_OP_T is passed (see CuBlas.java javadoc).
 *
 * Every execution is validated against a plain sequential Java reference
 * that performs the identical scale -> matvec -> bias pipeline on the CPU.
 */
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.enums.CuBlasOperation;

public class CuBlasSgemvHybrid {

    private static final float SCALE = 2.0f;
    private static final float BIAS = 1.0f;

    private static void scale(FloatArray matrix) {
        for (@Parallel int i = 0; i < matrix.getSize(); i++) {
            matrix.set(i, matrix.get(i) * SCALE);
        }
    }

    private static void bias(FloatArray output) {
        for (@Parallel int i = 0; i < output.getSize(); i++) {
            output.set(i, output.get(i) + BIAS);
        }
    }

    /** Deterministic, presenter-friendly fill: no RNG, same values every run. */
    private static void fillDeterministicMatrix(FloatArray matrix, int m, int n) {
        for (int i = 0; i < m * n; i++) {
            matrix.set(i, (i % 7) + 1.0f);
        }
    }

    private static void fillDeterministicVector(FloatArray vector, int n) {
        for (int j = 0; j < n; j++) {
            vector.set(j, (j % 5) + 1.0f);
        }
    }

    /** row-major matrix-vector product, computed purely on the CPU. */
    private static void matVecJava(FloatArray matrix, FloatArray vector, FloatArray output, int m, int n) {
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += matrix.get(i * n + j) * vector.get(j);
            }
            output.set(i, sum);
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int m = args.length > 0 ? Integer.parseInt(args[0]) : 8;
        int n = args.length > 1 ? Integer.parseInt(args[1]) : 8;
        int iterations = args.length > 2 ? Integer.parseInt(args[2]) : 5;

        FloatArray matrix = new FloatArray(m * n);
        FloatArray vector = new FloatArray(n);
        FloatArray output = new FloatArray(m);
        fillDeterministicMatrix(matrix, m, n);
        fillDeterministicVector(vector, n);

        final float alpha = 1.0f;
        final float beta = 0.0f;
        final int incx = 1;
        final int incy = 1;
        final int lda = m; // cuBLAS is column-major; row-major A is passed with CUBLAS_OP_T below.

        TaskGraph taskGraph = new TaskGraph("cublasHybrid") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, matrix, vector) //
                .task("scale", CuBlasSgemvHybrid::scale, matrix) //
                .libraryTask("sgemv", CuBlas::cublasSgemv, //
                        CuBlasOperation.CUBLAS_OP_T.operation(), //
                        m, n, //
                        alpha, matrix, lda, vector, incx, beta, output, incy) //
                .task("bias", CuBlasSgemvHybrid::bias, output) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        boolean allCorrect = true;
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutableTaskGraph)) {
            for (int it = 0; it < iterations; it++) {
                // Reset host matrix/vector before every execution: EVERY_EXECUTION
                // re-copies them, and the "scale" task mutates them in place on device.
                fillDeterministicMatrix(matrix, m, n);
                fillDeterministicVector(vector, n);

                TornadoExecutionResult result = plan.execute();

                FloatArray expected = new FloatArray(m);
                FloatArray seqMatrix = new FloatArray(m * n);
                FloatArray seqVector = new FloatArray(n);
                fillDeterministicMatrix(seqMatrix, m, n);
                fillDeterministicVector(seqVector, n);
                for (int i = 0; i < seqMatrix.getSize(); i++) {
                    seqMatrix.set(i, seqMatrix.get(i) * SCALE);
                }
                matVecJava(seqMatrix, seqVector, expected, m, n);
                for (int i = 0; i < expected.getSize(); i++) {
                    expected.set(i, expected.get(i) + BIAS);
                }

                boolean correct = true;
                for (int i = 0; i < m; i++) {
                    if (Math.abs(expected.get(i) - output.get(i)) > 0.01f) {
                        correct = false;
                        break;
                    }
                }
                allCorrect &= correct;
                System.out.println("iteration " + it + ": " + (correct ? "correct" : "WRONG") //
                        + " output[0]=" + output.get(0) + " expected[0]=" + expected.get(0));
                System.out.println("  total task-graph time: " + result.getProfilerResult().getTotalTime() + " ns");
            }
        }
        System.out.println(allCorrect ? "All iterations correct" : "SOME ITERATIONS WRONG");
    }
}
