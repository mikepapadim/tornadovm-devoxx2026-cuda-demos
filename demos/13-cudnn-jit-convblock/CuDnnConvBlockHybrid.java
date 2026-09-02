/*
 * Devoxx 2026 CUDA demo — a CNN block built from cuDNN library tasks and Java
 * JIT kernels interleaved in ONE TaskGraph.
 *
 * The classic conv block is conv -> bias -> activation. cuDNN owns the hard
 * part (the convolution, and the activation), but the bias add is trivial and
 * application-specific — exactly the kind of step you would rather write in
 * Java than look up in a vendor API. TornadoVM's hybrid API lets both live in
 * the same graph, on the same device buffers, with no round trip to the host:
 *
 *   1. JIT   "scale"    — Java @Parallel kernel, normalises the input
 *   2. cuDNN "conv2d"   — NCHW convolution, pad 1, stride 1
 *   3. JIT   "addBias"  — Java @Parallel kernel, per-output-channel bias
 *   4. cuDNN "relu"     — activation
 *
 * Every execution is validated against a sequential Java reference that runs
 * the identical four steps on the CPU.
 *
 * Usage:
 *   tornado --classpath . CuDnnConvBlockHybrid [N C H W K executions]
 */
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.cudnn.CuDnn;

public class CuDnnConvBlockHybrid {

    /** 3x3 filters, pad 1, stride 1 — output spatial dims equal input spatial dims. */
    private static final int R = 3, S = 3, PAD = 1, STRIDE = 1;

    private static final float SCALE = 0.25f;

    /** JIT task 1: normalise the input activations. */
    private static void scale(FloatArray in, FloatArray scaled) {
        for (@Parallel int i = 0; i < in.getSize(); i++) {
            scaled.set(i, in.get(i) * SCALE);
        }
    }

    /**
     * JIT task 2: add a per-output-channel bias to the NCHW convolution result.
     * hw = H*W, k = number of output channels; index i maps to channel (i / hw) % k.
     */
    private static void addBias(FloatArray conv, FloatArray bias, int hw, int k) {
        for (@Parallel int i = 0; i < conv.getSize(); i++) {
            conv.set(i, conv.get(i) + bias.get((i / hw) % k));
        }
    }

    private static FloatArray deterministic(int size, int seed, float span) {
        FloatArray arr = new FloatArray(size);
        for (int i = 0; i < size; i++) {
            arr.set(i, ((i * 31 + seed) % 17 - 8) / span);
        }
        return arr;
    }

    /** Sequential Java reference for the whole four-step block, NCHW layout. */
    private static float[] reference(FloatArray in, FloatArray filter, FloatArray bias,
            int n, int c, int h, int w, int k) {
        int hw = h * w;
        float[] out = new float[n * k * hw];
        for (int img = 0; img < n; img++) {
            for (int oc = 0; oc < k; oc++) {
                for (int oh = 0; oh < h; oh++) {
                    for (int ow = 0; ow < w; ow++) {
                        float sum = 0.0f;
                        for (int ic = 0; ic < c; ic++) {
                            for (int r = 0; r < R; r++) {
                                for (int s = 0; s < S; s++) {
                                    int ih = oh * STRIDE + r - PAD;
                                    int iw = ow * STRIDE + s - PAD;
                                    if (ih < 0 || ih >= h || iw < 0 || iw >= w) {
                                        continue; // zero padding
                                    }
                                    float x = in.get(((img * c + ic) * h + ih) * w + iw) * SCALE;
                                    float f = filter.get(((oc * c + ic) * R + r) * S + s);
                                    sum += x * f;
                                }
                            }
                        }
                        sum += bias.get(oc);
                        out[((img * k + oc) * h + oh) * w + ow] = Math.max(sum, 0.0f);
                    }
                }
            }
        }
        return out;
    }

    private static boolean validate(FloatArray got, float[] ref) {
        float maxAbs = 0.0f;
        int bad = 0;
        for (int i = 0; i < ref.length; i++) {
            float diff = Math.abs(got.get(i) - ref[i]);
            maxAbs = Math.max(maxAbs, diff);
            if (diff > 1e-3f) {
                bad++;
            }
        }
        boolean ok = bad == 0;
        System.out.printf("  validation %s (max abs err %.6f, %d/%d elements out of tol)%n",
                ok ? "PASSED" : "FAILED", maxAbs, bad, ref.length);
        return ok;
    }

    private static boolean isCUDABackend() {
        int backendIndex = TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice().getBackendIndex();
        return TornadoRuntimeProvider.getTornadoRuntime().getBackendType(backendIndex) == TornadoVMBackendType.CUDA;
    }

    private static long medianOf(long[] samples) {
        long[] copy = samples.clone();
        java.util.Arrays.sort(copy);
        return copy[copy.length / 2];
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        if (!isCUDABackend()) {
            System.out.println("cuDNN library tasks are only available on the CUDA backend.");
            return;
        }

        int n = args.length > 0 ? Integer.parseInt(args[0]) : 4;
        int c = args.length > 1 ? Integer.parseInt(args[1]) : 16;
        int h = args.length > 2 ? Integer.parseInt(args[2]) : 32;
        int w = args.length > 3 ? Integer.parseInt(args[3]) : 32;
        int k = args.length > 4 ? Integer.parseInt(args[4]) : 16;
        int executions = args.length > 5 ? Integer.parseInt(args[5]) : 10;

        int hw = h * w;
        int inSize = n * c * hw;
        int outSize = n * k * hw;
        int filterSize = k * c * R * S;

        System.out.printf("cuDNN + JIT conv block: NCHW %dx%dx%dx%d, %d filters %dx%d (pad %d, stride %d)%n",
                n, c, h, w, k, R, S, PAD, STRIDE);
        System.out.println("  graph: JIT scale -> cuDNN conv2d -> JIT addBias -> cuDNN relu");
        System.out.printf("  %d executions, steady-state median reported (first execution excluded)%n%n",
                executions);

        FloatArray input = deterministic(inSize, 1, 8.0f);
        FloatArray filter = deterministic(filterSize, 5, 16.0f);
        FloatArray bias = deterministic(k, 3, 32.0f);

        FloatArray scaled = new FloatArray(inSize);
        FloatArray conv = new FloatArray(outSize);
        FloatArray output = new FloatArray(outSize);

        float[] ref = reference(input, filter, bias, n, c, h, w, k);

        TaskGraph graph = new TaskGraph("convBlock") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, input, filter, bias) //
                .task("scale", CuDnnConvBlockHybrid::scale, input, scaled) //
                .libraryTask("conv2d", CuDnn::cudnnConv2d, //
                        scaled, filter, conv, n, c, h, w, k, R, S, PAD, STRIDE) //
                .task("addBias", CuDnnConvBlockHybrid::addBias, conv, bias, hw, k) //
                .libraryTask("relu", CuDnn::cudnnRelu, conv, output, outSize) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        ImmutableTaskGraph snapshot = graph.snapshot();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(snapshot)) {
            long[] samples = new long[executions - 1];
            for (int e = 0; e < executions; e++) {
                long start = System.nanoTime();
                plan.execute();
                long elapsed = (System.nanoTime() - start) / 1000;
                if (e == 0) {
                    System.out.printf("  first execution (JIT compile + cuDNN plan setup): %d us%n", elapsed);
                } else {
                    samples[e - 1] = elapsed;
                }
            }
            System.out.printf("  steady-state median wall-clock (n=%d): %d us%n", samples.length, medianOf(samples));
            System.out.println(validate(output, ref) ? "Result is correct" : "Result is INCORRECT");
        }
    }
}
