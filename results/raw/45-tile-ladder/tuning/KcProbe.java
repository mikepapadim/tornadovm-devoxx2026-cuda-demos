import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.*;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.enums.CuBlasOperation;

public class KcProbe {
    private static final int WARP = 32;

    // ---- demo 22 rung 3, verbatim: one warp per 16x16 output tile
    public static void kcSimple(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int warpId = ctx.groupIdx;
        int lane = ctx.localIdx;
        int tilesPerRow = n / 16;
        int tileRow = (warpId / tilesPerRow) * 16;
        int tileCol = (warpId % tilesPerRow) * 16;
        int[] aTile = ctx.allocateIntLocalArray(128);
        int[] bTile0 = ctx.allocateIntLocalArray(64);
        int[] bTile1 = ctx.allocateIntLocalArray(64);
        float[] accLeft = ctx.mmaFragment(0.0f);
        float[] accRight = ctx.mmaFragment(0.0f);
        for (int kBase = 0; kBase < n; kBase += 16) {
            for (int idx = lane; idx < 128; idx += WARP) {
                int elem = idx * 2;
                int r = elem / 16;
                int kk = elem % 16;
                int g = (tileRow + r) * n + kBase + kk;
                int lo = a.get(g).getHalfFloatValue() & 0xFFFF;
                int hi = a.get(g + 1).getHalfFloatValue() & 0xFFFF;
                aTile[r * 8 + kk / 2] = lo | (hi << 16);
            }
            for (int idx = lane; idx < 64; idx += WARP) {
                int kRow = idx / 4;
                int jPair = idx % 4;
                int jBase = jPair * 2;
                int gl = (kBase + kRow) * n + tileCol + jBase;
                bTile0[kRow * 4 + jPair] = (b.get(gl).getHalfFloatValue() & 0xFFFF) | ((b.get(gl + 1).getHalfFloatValue() & 0xFFFF) << 16);
                int gr = (kBase + kRow) * n + tileCol + 8 + jBase;
                bTile1[kRow * 4 + jPair] = (b.get(gr).getHalfFloatValue() & 0xFFFF) | ((b.get(gr + 1).getHalfFloatValue() & 0xFFFF) << 16);
            }
            ctx.localBarrier();
            HalfFloat[] fragA = ctx.mmaLoadA(aTile, 16);
            HalfFloat[] fragB0 = ctx.mmaLoadB(bTile0, 16);
            accLeft = ctx.mma(fragA, fragB0, accLeft, MMAShape.M16N8K16);
            HalfFloat[] fragB1 = ctx.mmaLoadB(bTile1, 16);
            accRight = ctx.mma(fragA, fragB1, accRight, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        ctx.mmaStore(accLeft, c, tileRow, tileCol, n);
        ctx.mmaStore(accRight, c, tileRow, tileCol + 8, n);
    }

    // ---- optimised: 64x64 block tile, 4 warps x 32x32 warp tiles, cp.async double buffering
    private static final int BM = 64, BN = 64, BK = 16, THREADS = 128;
    private static final int A_INTS = BM * BK / 2;      // 512 ints per A stage
    private static final int B_INTS = BK * BN / 2;      // 512 ints per B stage (8 panels of 16x8)

    public static void kcOpt(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int tid = ctx.localIdx;
        int blocksPerRow = n / BN;
        int blockRow = (ctx.groupIdx / blocksPerRow) * BM;
        int blockCol = (ctx.groupIdx % blocksPerRow) * BN;
        int warp = tid / WARP;
        int warpRow = warp / 2;   // 0..1 -> 32-row band
        int warpCol = warp % 2;   // 0..1 -> 32-column band

        int[] aSmem = ctx.allocateIntLocalArray(2 * A_INTS);
        int[] bSmem = ctx.allocateIntLocalArray(2 * B_INTS);

        float[] acc00 = ctx.mmaFragment(0.0f);
        float[] acc01 = ctx.mmaFragment(0.0f);
        float[] acc02 = ctx.mmaFragment(0.0f);
        float[] acc03 = ctx.mmaFragment(0.0f);
        float[] acc10 = ctx.mmaFragment(0.0f);
        float[] acc11 = ctx.mmaFragment(0.0f);
        float[] acc12 = ctx.mmaFragment(0.0f);
        float[] acc13 = ctx.mmaFragment(0.0f);

        int numK = n / BK;

        // stage 0 -> buffer 0
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * THREADS;
            int r = idx / 8;
            int k2 = idx % 8;
            ctx.asyncCopyToLocal(aSmem, idx, a, (blockRow + r) * n + k2 * 2);
            int kRow = idx / 32;
            int rem = idx % 32;
            ctx.asyncCopyToLocal(bSmem, (rem / 4) * 64 + kRow * 4 + (rem % 4), b, kRow * n + blockCol + rem * 2);
        }
        ctx.asyncCopyCommit();

        for (int kt = 0; kt < numK; kt++) {
            int cur = kt % 2;
            if (kt + 1 < numK) {
                int nxt = 1 - cur;
                int kBase = (kt + 1) * BK;
                for (int i = 0; i < 4; i++) {
                    int idx = tid + i * THREADS;
                    int r = idx / 8;
                    int k2 = idx % 8;
                    ctx.asyncCopyToLocal(aSmem, nxt * A_INTS + idx, a, (blockRow + r) * n + kBase + k2 * 2);
                    int kRow = idx / 32;
                    int rem = idx % 32;
                    ctx.asyncCopyToLocal(bSmem, nxt * B_INTS + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kBase + kRow) * n + blockCol + rem * 2);
                }
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(1);
            } else {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();

            int aBase = cur * A_INTS * 4 + warpRow * 32 * 32;   // bytes: buffer + 32-row band (32 B per row)
            int bBase = cur * B_INTS * 4 + warpCol * 4 * 256;   // bytes: buffer + 4 panels of 256 B
            HalfFloat[] fa0 = ctx.mmaLoadA(aSmem, BK, aBase);
            HalfFloat[] fa1 = ctx.mmaLoadA(aSmem, BK, aBase + 16 * 32);
            HalfFloat[] fb0 = ctx.mmaLoadB(bSmem, BK, bBase);
            HalfFloat[] fb1 = ctx.mmaLoadB(bSmem, BK, bBase + 256);
            HalfFloat[] fb2 = ctx.mmaLoadB(bSmem, BK, bBase + 512);
            HalfFloat[] fb3 = ctx.mmaLoadB(bSmem, BK, bBase + 768);
            acc00 = ctx.mma(fa0, fb0, acc00, MMAShape.M16N8K16);
            acc01 = ctx.mma(fa0, fb1, acc01, MMAShape.M16N8K16);
            acc02 = ctx.mma(fa0, fb2, acc02, MMAShape.M16N8K16);
            acc03 = ctx.mma(fa0, fb3, acc03, MMAShape.M16N8K16);
            acc10 = ctx.mma(fa1, fb0, acc10, MMAShape.M16N8K16);
            acc11 = ctx.mma(fa1, fb1, acc11, MMAShape.M16N8K16);
            acc12 = ctx.mma(fa1, fb2, acc12, MMAShape.M16N8K16);
            acc13 = ctx.mma(fa1, fb3, acc13, MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        int row0 = blockRow + warpRow * 32;
        int col0 = blockCol + warpCol * 32;
        ctx.mmaStore(acc00, c, row0, col0, n);
        ctx.mmaStore(acc01, c, row0, col0 + 8, n);
        ctx.mmaStore(acc02, c, row0, col0 + 16, n);
        ctx.mmaStore(acc03, c, row0, col0 + 24, n);
        ctx.mmaStore(acc10, c, row0 + 16, col0, n);
        ctx.mmaStore(acc11, c, row0 + 16, col0 + 8, n);
        ctx.mmaStore(acc12, c, row0 + 16, col0 + 16, n);
        ctx.mmaStore(acc13, c, row0 + 16, col0 + 24, n);
    }

    private static final int O64X64_2X2_BM = 64, O64X64_2X2_BN = 64, O64X64_2X2_T = 128;
    public static void o64x64_2x2(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int tid = ctx.localIdx;
        int blocksPerRow = n / 64;
        int blockRow = (ctx.groupIdx / blocksPerRow) * 64;
        int blockCol = (ctx.groupIdx % blocksPerRow) * 64;
        int warp = tid / 32;
        int warpRow = warp / 2;
        int warpCol = warp % 2;
        int[] aSmem = ctx.allocateIntLocalArray(1024);
        int[] bSmem = ctx.allocateIntLocalArray(1024);
        float[] acc0_0 = ctx.mmaFragment(0.0f);
        float[] acc0_1 = ctx.mmaFragment(0.0f);
        float[] acc0_2 = ctx.mmaFragment(0.0f);
        float[] acc0_3 = ctx.mmaFragment(0.0f);
        float[] acc1_0 = ctx.mmaFragment(0.0f);
        float[] acc1_1 = ctx.mmaFragment(0.0f);
        float[] acc1_2 = ctx.mmaFragment(0.0f);
        float[] acc1_3 = ctx.mmaFragment(0.0f);
        int numK = n / 16;
            { int idx = tid + 0;
              ctx.asyncCopyToLocal(aSmem, 0 * 512 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 32; int rem = idx % 32;
              ctx.asyncCopyToLocal(bSmem, 0 * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 128;
              ctx.asyncCopyToLocal(aSmem, 0 * 512 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 32; int rem = idx % 32;
              ctx.asyncCopyToLocal(bSmem, 0 * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 256;
              ctx.asyncCopyToLocal(aSmem, 0 * 512 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 32; int rem = idx % 32;
              ctx.asyncCopyToLocal(bSmem, 0 * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 384;
              ctx.asyncCopyToLocal(aSmem, 0 * 512 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 32; int rem = idx % 32;
              ctx.asyncCopyToLocal(bSmem, 0 * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
        ctx.asyncCopyCommit();
        for (int kt = 0; kt < numK; kt++) {
            int cur = kt % 2;
            if (kt + 1 < numK) {
                int nxt = 1 - cur;
                int kNext = (kt + 1) * 16;
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 512 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 32; int rem = idx % 32;
                  ctx.asyncCopyToLocal(bSmem, nxt * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 128;
                  ctx.asyncCopyToLocal(aSmem, nxt * 512 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 32; int rem = idx % 32;
                  ctx.asyncCopyToLocal(bSmem, nxt * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 512 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 32; int rem = idx % 32;
                  ctx.asyncCopyToLocal(bSmem, nxt * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 384;
                  ctx.asyncCopyToLocal(aSmem, nxt * 512 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 32; int rem = idx % 32;
                  ctx.asyncCopyToLocal(bSmem, nxt * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(1);
            } else {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();
            int aBase = cur * 2048 + warpRow * 1024;
            int bBase = cur * 2048 + warpCol * 1024;
            HalfFloat[] fa0 = ctx.mmaLoadA(aSmem, 16, aBase + 0);
            HalfFloat[] fa1 = ctx.mmaLoadA(aSmem, 16, aBase + 512);
            HalfFloat[] fb0 = ctx.mmaLoadB(bSmem, 16, bBase + 0);
            HalfFloat[] fb1 = ctx.mmaLoadB(bSmem, 16, bBase + 256);
            HalfFloat[] fb2 = ctx.mmaLoadB(bSmem, 16, bBase + 512);
            HalfFloat[] fb3 = ctx.mmaLoadB(bSmem, 16, bBase + 768);
            acc0_0 = ctx.mma(fa0, fb0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa0, fb1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa0, fb2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa0, fb3, acc0_3, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa1, fb0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa1, fb1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa1, fb2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa1, fb3, acc1_3, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        int row0 = blockRow + warpRow * 32;
        int col0 = blockCol + warpCol * 32;
        ctx.mmaStore(acc0_0, c, row0 + 0, col0 + 0, n);
        ctx.mmaStore(acc0_1, c, row0 + 0, col0 + 8, n);
        ctx.mmaStore(acc0_2, c, row0 + 0, col0 + 16, n);
        ctx.mmaStore(acc0_3, c, row0 + 0, col0 + 24, n);
        ctx.mmaStore(acc1_0, c, row0 + 16, col0 + 0, n);
        ctx.mmaStore(acc1_1, c, row0 + 16, col0 + 8, n);
        ctx.mmaStore(acc1_2, c, row0 + 16, col0 + 16, n);
        ctx.mmaStore(acc1_3, c, row0 + 16, col0 + 24, n);
    }

    private static final int O128X128_2X4_BM = 128, O128X128_2X4_BN = 128, O128X128_2X4_T = 256;
    public static void o128x128_2x4(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int tid = ctx.localIdx;
        int blocksPerRow = n / 128;
        int blockRow = (ctx.groupIdx / blocksPerRow) * 128;
        int blockCol = (ctx.groupIdx % blocksPerRow) * 128;
        int warp = tid / 32;
        int warpRow = warp / 4;
        int warpCol = warp % 4;
        int[] aSmem = ctx.allocateIntLocalArray(2048);
        int[] bSmem = ctx.allocateIntLocalArray(2048);
        float[] acc0_0 = ctx.mmaFragment(0.0f);
        float[] acc0_1 = ctx.mmaFragment(0.0f);
        float[] acc0_2 = ctx.mmaFragment(0.0f);
        float[] acc0_3 = ctx.mmaFragment(0.0f);
        float[] acc1_0 = ctx.mmaFragment(0.0f);
        float[] acc1_1 = ctx.mmaFragment(0.0f);
        float[] acc1_2 = ctx.mmaFragment(0.0f);
        float[] acc1_3 = ctx.mmaFragment(0.0f);
        float[] acc2_0 = ctx.mmaFragment(0.0f);
        float[] acc2_1 = ctx.mmaFragment(0.0f);
        float[] acc2_2 = ctx.mmaFragment(0.0f);
        float[] acc2_3 = ctx.mmaFragment(0.0f);
        float[] acc3_0 = ctx.mmaFragment(0.0f);
        float[] acc3_1 = ctx.mmaFragment(0.0f);
        float[] acc3_2 = ctx.mmaFragment(0.0f);
        float[] acc3_3 = ctx.mmaFragment(0.0f);
        int numK = n / 16;
            { int idx = tid + 0;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 256;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 512;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 768;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
        ctx.asyncCopyCommit();
        for (int kt = 0; kt < numK; kt++) {
            int cur = kt % 2;
            if (kt + 1 < numK) {
                int nxt = 1 - cur;
                int kNext = (kt + 1) * 16;
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 512;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 768;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(1);
            } else {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();
            int aBase = cur * 4096 + warpRow * 2048;
            int bBase = cur * 4096 + warpCol * 1024;
            HalfFloat[] fa0 = ctx.mmaLoadA(aSmem, 16, aBase + 0);
            HalfFloat[] fa1 = ctx.mmaLoadA(aSmem, 16, aBase + 512);
            HalfFloat[] fa2 = ctx.mmaLoadA(aSmem, 16, aBase + 1024);
            HalfFloat[] fa3 = ctx.mmaLoadA(aSmem, 16, aBase + 1536);
            HalfFloat[] fb0 = ctx.mmaLoadB(bSmem, 16, bBase + 0);
            HalfFloat[] fb1 = ctx.mmaLoadB(bSmem, 16, bBase + 256);
            HalfFloat[] fb2 = ctx.mmaLoadB(bSmem, 16, bBase + 512);
            HalfFloat[] fb3 = ctx.mmaLoadB(bSmem, 16, bBase + 768);
            acc0_0 = ctx.mma(fa0, fb0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa0, fb1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa0, fb2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa0, fb3, acc0_3, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa1, fb0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa1, fb1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa1, fb2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa1, fb3, acc1_3, MMAShape.M16N8K16);
            acc2_0 = ctx.mma(fa2, fb0, acc2_0, MMAShape.M16N8K16);
            acc2_1 = ctx.mma(fa2, fb1, acc2_1, MMAShape.M16N8K16);
            acc2_2 = ctx.mma(fa2, fb2, acc2_2, MMAShape.M16N8K16);
            acc2_3 = ctx.mma(fa2, fb3, acc2_3, MMAShape.M16N8K16);
            acc3_0 = ctx.mma(fa3, fb0, acc3_0, MMAShape.M16N8K16);
            acc3_1 = ctx.mma(fa3, fb1, acc3_1, MMAShape.M16N8K16);
            acc3_2 = ctx.mma(fa3, fb2, acc3_2, MMAShape.M16N8K16);
            acc3_3 = ctx.mma(fa3, fb3, acc3_3, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        int row0 = blockRow + warpRow * 64;
        int col0 = blockCol + warpCol * 32;
        ctx.mmaStore(acc0_0, c, row0 + 0, col0 + 0, n);
        ctx.mmaStore(acc0_1, c, row0 + 0, col0 + 8, n);
        ctx.mmaStore(acc0_2, c, row0 + 0, col0 + 16, n);
        ctx.mmaStore(acc0_3, c, row0 + 0, col0 + 24, n);
        ctx.mmaStore(acc1_0, c, row0 + 16, col0 + 0, n);
        ctx.mmaStore(acc1_1, c, row0 + 16, col0 + 8, n);
        ctx.mmaStore(acc1_2, c, row0 + 16, col0 + 16, n);
        ctx.mmaStore(acc1_3, c, row0 + 16, col0 + 24, n);
        ctx.mmaStore(acc2_0, c, row0 + 32, col0 + 0, n);
        ctx.mmaStore(acc2_1, c, row0 + 32, col0 + 8, n);
        ctx.mmaStore(acc2_2, c, row0 + 32, col0 + 16, n);
        ctx.mmaStore(acc2_3, c, row0 + 32, col0 + 24, n);
        ctx.mmaStore(acc3_0, c, row0 + 48, col0 + 0, n);
        ctx.mmaStore(acc3_1, c, row0 + 48, col0 + 8, n);
        ctx.mmaStore(acc3_2, c, row0 + 48, col0 + 16, n);
        ctx.mmaStore(acc3_3, c, row0 + 48, col0 + 24, n);
    }

    private static final int O128X64_2X2_BM = 128, O128X64_2X2_BN = 64, O128X64_2X2_T = 128;
    public static void o128x64_2x2(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int tid = ctx.localIdx;
        int blocksPerRow = n / 64;
        int blockRow = (ctx.groupIdx / blocksPerRow) * 128;
        int blockCol = (ctx.groupIdx % blocksPerRow) * 64;
        int warp = tid / 32;
        int warpRow = warp / 2;
        int warpCol = warp % 2;
        int[] aSmem = ctx.allocateIntLocalArray(2048);
        int[] bSmem = ctx.allocateIntLocalArray(1024);
        float[] acc0_0 = ctx.mmaFragment(0.0f);
        float[] acc0_1 = ctx.mmaFragment(0.0f);
        float[] acc0_2 = ctx.mmaFragment(0.0f);
        float[] acc0_3 = ctx.mmaFragment(0.0f);
        float[] acc1_0 = ctx.mmaFragment(0.0f);
        float[] acc1_1 = ctx.mmaFragment(0.0f);
        float[] acc1_2 = ctx.mmaFragment(0.0f);
        float[] acc1_3 = ctx.mmaFragment(0.0f);
        float[] acc2_0 = ctx.mmaFragment(0.0f);
        float[] acc2_1 = ctx.mmaFragment(0.0f);
        float[] acc2_2 = ctx.mmaFragment(0.0f);
        float[] acc2_3 = ctx.mmaFragment(0.0f);
        float[] acc3_0 = ctx.mmaFragment(0.0f);
        float[] acc3_1 = ctx.mmaFragment(0.0f);
        float[] acc3_2 = ctx.mmaFragment(0.0f);
        float[] acc3_3 = ctx.mmaFragment(0.0f);
        int numK = n / 16;
            { int idx = tid + 0;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 32; int rem = idx % 32;
              ctx.asyncCopyToLocal(bSmem, 0 * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 128;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 32; int rem = idx % 32;
              ctx.asyncCopyToLocal(bSmem, 0 * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 256;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 32; int rem = idx % 32;
              ctx.asyncCopyToLocal(bSmem, 0 * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 384;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 32; int rem = idx % 32;
              ctx.asyncCopyToLocal(bSmem, 0 * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 512;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
            }
            { int idx = tid + 640;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
            }
            { int idx = tid + 768;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
            }
            { int idx = tid + 896;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
            }
        ctx.asyncCopyCommit();
        for (int kt = 0; kt < numK; kt++) {
            int cur = kt % 2;
            if (kt + 1 < numK) {
                int nxt = 1 - cur;
                int kNext = (kt + 1) * 16;
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 32; int rem = idx % 32;
                  ctx.asyncCopyToLocal(bSmem, nxt * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 128;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 32; int rem = idx % 32;
                  ctx.asyncCopyToLocal(bSmem, nxt * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 32; int rem = idx % 32;
                  ctx.asyncCopyToLocal(bSmem, nxt * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 384;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 32; int rem = idx % 32;
                  ctx.asyncCopyToLocal(bSmem, nxt * 512 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 512;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                }
                { int idx = tid + 640;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                }
                { int idx = tid + 768;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                }
                { int idx = tid + 896;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                }
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(1);
            } else {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();
            int aBase = cur * 4096 + warpRow * 2048;
            int bBase = cur * 2048 + warpCol * 1024;
            HalfFloat[] fa0 = ctx.mmaLoadA(aSmem, 16, aBase + 0);
            HalfFloat[] fa1 = ctx.mmaLoadA(aSmem, 16, aBase + 512);
            HalfFloat[] fa2 = ctx.mmaLoadA(aSmem, 16, aBase + 1024);
            HalfFloat[] fa3 = ctx.mmaLoadA(aSmem, 16, aBase + 1536);
            HalfFloat[] fb0 = ctx.mmaLoadB(bSmem, 16, bBase + 0);
            HalfFloat[] fb1 = ctx.mmaLoadB(bSmem, 16, bBase + 256);
            HalfFloat[] fb2 = ctx.mmaLoadB(bSmem, 16, bBase + 512);
            HalfFloat[] fb3 = ctx.mmaLoadB(bSmem, 16, bBase + 768);
            acc0_0 = ctx.mma(fa0, fb0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa0, fb1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa0, fb2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa0, fb3, acc0_3, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa1, fb0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa1, fb1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa1, fb2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa1, fb3, acc1_3, MMAShape.M16N8K16);
            acc2_0 = ctx.mma(fa2, fb0, acc2_0, MMAShape.M16N8K16);
            acc2_1 = ctx.mma(fa2, fb1, acc2_1, MMAShape.M16N8K16);
            acc2_2 = ctx.mma(fa2, fb2, acc2_2, MMAShape.M16N8K16);
            acc2_3 = ctx.mma(fa2, fb3, acc2_3, MMAShape.M16N8K16);
            acc3_0 = ctx.mma(fa3, fb0, acc3_0, MMAShape.M16N8K16);
            acc3_1 = ctx.mma(fa3, fb1, acc3_1, MMAShape.M16N8K16);
            acc3_2 = ctx.mma(fa3, fb2, acc3_2, MMAShape.M16N8K16);
            acc3_3 = ctx.mma(fa3, fb3, acc3_3, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        int row0 = blockRow + warpRow * 64;
        int col0 = blockCol + warpCol * 32;
        ctx.mmaStore(acc0_0, c, row0 + 0, col0 + 0, n);
        ctx.mmaStore(acc0_1, c, row0 + 0, col0 + 8, n);
        ctx.mmaStore(acc0_2, c, row0 + 0, col0 + 16, n);
        ctx.mmaStore(acc0_3, c, row0 + 0, col0 + 24, n);
        ctx.mmaStore(acc1_0, c, row0 + 16, col0 + 0, n);
        ctx.mmaStore(acc1_1, c, row0 + 16, col0 + 8, n);
        ctx.mmaStore(acc1_2, c, row0 + 16, col0 + 16, n);
        ctx.mmaStore(acc1_3, c, row0 + 16, col0 + 24, n);
        ctx.mmaStore(acc2_0, c, row0 + 32, col0 + 0, n);
        ctx.mmaStore(acc2_1, c, row0 + 32, col0 + 8, n);
        ctx.mmaStore(acc2_2, c, row0 + 32, col0 + 16, n);
        ctx.mmaStore(acc2_3, c, row0 + 32, col0 + 24, n);
        ctx.mmaStore(acc3_0, c, row0 + 48, col0 + 0, n);
        ctx.mmaStore(acc3_1, c, row0 + 48, col0 + 8, n);
        ctx.mmaStore(acc3_2, c, row0 + 48, col0 + 16, n);
        ctx.mmaStore(acc3_3, c, row0 + 48, col0 + 24, n);
    }

    private static final int O128X128_2X2_BM = 128, O128X128_2X2_BN = 128, O128X128_2X2_T = 128;
    public static void o128x128_2x2(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int tid = ctx.localIdx;
        int blocksPerRow = n / 128;
        int blockRow = (ctx.groupIdx / blocksPerRow) * 128;
        int blockCol = (ctx.groupIdx % blocksPerRow) * 128;
        int warp = tid / 32;
        int warpRow = warp / 2;
        int warpCol = warp % 2;
        int[] aSmem = ctx.allocateIntLocalArray(2048);
        int[] bSmem = ctx.allocateIntLocalArray(2048);
        float[] acc0_0 = ctx.mmaFragment(0.0f);
        float[] acc0_1 = ctx.mmaFragment(0.0f);
        float[] acc0_2 = ctx.mmaFragment(0.0f);
        float[] acc0_3 = ctx.mmaFragment(0.0f);
        float[] acc0_4 = ctx.mmaFragment(0.0f);
        float[] acc0_5 = ctx.mmaFragment(0.0f);
        float[] acc0_6 = ctx.mmaFragment(0.0f);
        float[] acc0_7 = ctx.mmaFragment(0.0f);
        float[] acc1_0 = ctx.mmaFragment(0.0f);
        float[] acc1_1 = ctx.mmaFragment(0.0f);
        float[] acc1_2 = ctx.mmaFragment(0.0f);
        float[] acc1_3 = ctx.mmaFragment(0.0f);
        float[] acc1_4 = ctx.mmaFragment(0.0f);
        float[] acc1_5 = ctx.mmaFragment(0.0f);
        float[] acc1_6 = ctx.mmaFragment(0.0f);
        float[] acc1_7 = ctx.mmaFragment(0.0f);
        float[] acc2_0 = ctx.mmaFragment(0.0f);
        float[] acc2_1 = ctx.mmaFragment(0.0f);
        float[] acc2_2 = ctx.mmaFragment(0.0f);
        float[] acc2_3 = ctx.mmaFragment(0.0f);
        float[] acc2_4 = ctx.mmaFragment(0.0f);
        float[] acc2_5 = ctx.mmaFragment(0.0f);
        float[] acc2_6 = ctx.mmaFragment(0.0f);
        float[] acc2_7 = ctx.mmaFragment(0.0f);
        float[] acc3_0 = ctx.mmaFragment(0.0f);
        float[] acc3_1 = ctx.mmaFragment(0.0f);
        float[] acc3_2 = ctx.mmaFragment(0.0f);
        float[] acc3_3 = ctx.mmaFragment(0.0f);
        float[] acc3_4 = ctx.mmaFragment(0.0f);
        float[] acc3_5 = ctx.mmaFragment(0.0f);
        float[] acc3_6 = ctx.mmaFragment(0.0f);
        float[] acc3_7 = ctx.mmaFragment(0.0f);
        int numK = n / 16;
            { int idx = tid + 0;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 128;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 256;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 384;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 512;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 640;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 768;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 896;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
        ctx.asyncCopyCommit();
        for (int kt = 0; kt < numK; kt++) {
            int cur = kt % 2;
            if (kt + 1 < numK) {
                int nxt = 1 - cur;
                int kNext = (kt + 1) * 16;
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 128;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 384;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 512;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 640;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 768;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 896;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(1);
            } else {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();
            int aBase = cur * 4096 + warpRow * 2048;
            int bBase = cur * 4096 + warpCol * 2048;
            HalfFloat[] fa0 = ctx.mmaLoadA(aSmem, 16, aBase + 0);
            HalfFloat[] fa1 = ctx.mmaLoadA(aSmem, 16, aBase + 512);
            HalfFloat[] fa2 = ctx.mmaLoadA(aSmem, 16, aBase + 1024);
            HalfFloat[] fa3 = ctx.mmaLoadA(aSmem, 16, aBase + 1536);
            HalfFloat[] fb0 = ctx.mmaLoadB(bSmem, 16, bBase + 0);
            HalfFloat[] fb1 = ctx.mmaLoadB(bSmem, 16, bBase + 256);
            HalfFloat[] fb2 = ctx.mmaLoadB(bSmem, 16, bBase + 512);
            HalfFloat[] fb3 = ctx.mmaLoadB(bSmem, 16, bBase + 768);
            HalfFloat[] fb4 = ctx.mmaLoadB(bSmem, 16, bBase + 1024);
            HalfFloat[] fb5 = ctx.mmaLoadB(bSmem, 16, bBase + 1280);
            HalfFloat[] fb6 = ctx.mmaLoadB(bSmem, 16, bBase + 1536);
            HalfFloat[] fb7 = ctx.mmaLoadB(bSmem, 16, bBase + 1792);
            acc0_0 = ctx.mma(fa0, fb0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa0, fb1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa0, fb2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa0, fb3, acc0_3, MMAShape.M16N8K16);
            acc0_4 = ctx.mma(fa0, fb4, acc0_4, MMAShape.M16N8K16);
            acc0_5 = ctx.mma(fa0, fb5, acc0_5, MMAShape.M16N8K16);
            acc0_6 = ctx.mma(fa0, fb6, acc0_6, MMAShape.M16N8K16);
            acc0_7 = ctx.mma(fa0, fb7, acc0_7, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa1, fb0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa1, fb1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa1, fb2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa1, fb3, acc1_3, MMAShape.M16N8K16);
            acc1_4 = ctx.mma(fa1, fb4, acc1_4, MMAShape.M16N8K16);
            acc1_5 = ctx.mma(fa1, fb5, acc1_5, MMAShape.M16N8K16);
            acc1_6 = ctx.mma(fa1, fb6, acc1_6, MMAShape.M16N8K16);
            acc1_7 = ctx.mma(fa1, fb7, acc1_7, MMAShape.M16N8K16);
            acc2_0 = ctx.mma(fa2, fb0, acc2_0, MMAShape.M16N8K16);
            acc2_1 = ctx.mma(fa2, fb1, acc2_1, MMAShape.M16N8K16);
            acc2_2 = ctx.mma(fa2, fb2, acc2_2, MMAShape.M16N8K16);
            acc2_3 = ctx.mma(fa2, fb3, acc2_3, MMAShape.M16N8K16);
            acc2_4 = ctx.mma(fa2, fb4, acc2_4, MMAShape.M16N8K16);
            acc2_5 = ctx.mma(fa2, fb5, acc2_5, MMAShape.M16N8K16);
            acc2_6 = ctx.mma(fa2, fb6, acc2_6, MMAShape.M16N8K16);
            acc2_7 = ctx.mma(fa2, fb7, acc2_7, MMAShape.M16N8K16);
            acc3_0 = ctx.mma(fa3, fb0, acc3_0, MMAShape.M16N8K16);
            acc3_1 = ctx.mma(fa3, fb1, acc3_1, MMAShape.M16N8K16);
            acc3_2 = ctx.mma(fa3, fb2, acc3_2, MMAShape.M16N8K16);
            acc3_3 = ctx.mma(fa3, fb3, acc3_3, MMAShape.M16N8K16);
            acc3_4 = ctx.mma(fa3, fb4, acc3_4, MMAShape.M16N8K16);
            acc3_5 = ctx.mma(fa3, fb5, acc3_5, MMAShape.M16N8K16);
            acc3_6 = ctx.mma(fa3, fb6, acc3_6, MMAShape.M16N8K16);
            acc3_7 = ctx.mma(fa3, fb7, acc3_7, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        int row0 = blockRow + warpRow * 64;
        int col0 = blockCol + warpCol * 64;
        ctx.mmaStore(acc0_0, c, row0 + 0, col0 + 0, n);
        ctx.mmaStore(acc0_1, c, row0 + 0, col0 + 8, n);
        ctx.mmaStore(acc0_2, c, row0 + 0, col0 + 16, n);
        ctx.mmaStore(acc0_3, c, row0 + 0, col0 + 24, n);
        ctx.mmaStore(acc0_4, c, row0 + 0, col0 + 32, n);
        ctx.mmaStore(acc0_5, c, row0 + 0, col0 + 40, n);
        ctx.mmaStore(acc0_6, c, row0 + 0, col0 + 48, n);
        ctx.mmaStore(acc0_7, c, row0 + 0, col0 + 56, n);
        ctx.mmaStore(acc1_0, c, row0 + 16, col0 + 0, n);
        ctx.mmaStore(acc1_1, c, row0 + 16, col0 + 8, n);
        ctx.mmaStore(acc1_2, c, row0 + 16, col0 + 16, n);
        ctx.mmaStore(acc1_3, c, row0 + 16, col0 + 24, n);
        ctx.mmaStore(acc1_4, c, row0 + 16, col0 + 32, n);
        ctx.mmaStore(acc1_5, c, row0 + 16, col0 + 40, n);
        ctx.mmaStore(acc1_6, c, row0 + 16, col0 + 48, n);
        ctx.mmaStore(acc1_7, c, row0 + 16, col0 + 56, n);
        ctx.mmaStore(acc2_0, c, row0 + 32, col0 + 0, n);
        ctx.mmaStore(acc2_1, c, row0 + 32, col0 + 8, n);
        ctx.mmaStore(acc2_2, c, row0 + 32, col0 + 16, n);
        ctx.mmaStore(acc2_3, c, row0 + 32, col0 + 24, n);
        ctx.mmaStore(acc2_4, c, row0 + 32, col0 + 32, n);
        ctx.mmaStore(acc2_5, c, row0 + 32, col0 + 40, n);
        ctx.mmaStore(acc2_6, c, row0 + 32, col0 + 48, n);
        ctx.mmaStore(acc2_7, c, row0 + 32, col0 + 56, n);
        ctx.mmaStore(acc3_0, c, row0 + 48, col0 + 0, n);
        ctx.mmaStore(acc3_1, c, row0 + 48, col0 + 8, n);
        ctx.mmaStore(acc3_2, c, row0 + 48, col0 + 16, n);
        ctx.mmaStore(acc3_3, c, row0 + 48, col0 + 24, n);
        ctx.mmaStore(acc3_4, c, row0 + 48, col0 + 32, n);
        ctx.mmaStore(acc3_5, c, row0 + 48, col0 + 40, n);
        ctx.mmaStore(acc3_6, c, row0 + 48, col0 + 48, n);
        ctx.mmaStore(acc3_7, c, row0 + 48, col0 + 56, n);
    }

    private static final int O64X128_2X2_BM = 64, O64X128_2X2_BN = 128, O64X128_2X2_T = 128;
    public static void o64x128_2x2(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int tid = ctx.localIdx;
        int blocksPerRow = n / 128;
        int blockRow = (ctx.groupIdx / blocksPerRow) * 64;
        int blockCol = (ctx.groupIdx % blocksPerRow) * 128;
        int warp = tid / 32;
        int warpRow = warp / 2;
        int warpCol = warp % 2;
        int[] aSmem = ctx.allocateIntLocalArray(1024);
        int[] bSmem = ctx.allocateIntLocalArray(2048);
        float[] acc0_0 = ctx.mmaFragment(0.0f);
        float[] acc0_1 = ctx.mmaFragment(0.0f);
        float[] acc0_2 = ctx.mmaFragment(0.0f);
        float[] acc0_3 = ctx.mmaFragment(0.0f);
        float[] acc0_4 = ctx.mmaFragment(0.0f);
        float[] acc0_5 = ctx.mmaFragment(0.0f);
        float[] acc0_6 = ctx.mmaFragment(0.0f);
        float[] acc0_7 = ctx.mmaFragment(0.0f);
        float[] acc1_0 = ctx.mmaFragment(0.0f);
        float[] acc1_1 = ctx.mmaFragment(0.0f);
        float[] acc1_2 = ctx.mmaFragment(0.0f);
        float[] acc1_3 = ctx.mmaFragment(0.0f);
        float[] acc1_4 = ctx.mmaFragment(0.0f);
        float[] acc1_5 = ctx.mmaFragment(0.0f);
        float[] acc1_6 = ctx.mmaFragment(0.0f);
        float[] acc1_7 = ctx.mmaFragment(0.0f);
        int numK = n / 16;
            { int idx = tid + 0;
              ctx.asyncCopyToLocal(aSmem, 0 * 512 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 128;
              ctx.asyncCopyToLocal(aSmem, 0 * 512 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 256;
              ctx.asyncCopyToLocal(aSmem, 0 * 512 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 384;
              ctx.asyncCopyToLocal(aSmem, 0 * 512 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 512;
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 640;
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 768;
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 896;
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
        ctx.asyncCopyCommit();
        for (int kt = 0; kt < numK; kt++) {
            int cur = kt % 2;
            if (kt + 1 < numK) {
                int nxt = 1 - cur;
                int kNext = (kt + 1) * 16;
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 512 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 128;
                  ctx.asyncCopyToLocal(aSmem, nxt * 512 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 512 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 384;
                  ctx.asyncCopyToLocal(aSmem, nxt * 512 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 512;
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 640;
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 768;
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 896;
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(1);
            } else {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();
            int aBase = cur * 2048 + warpRow * 1024;
            int bBase = cur * 4096 + warpCol * 2048;
            HalfFloat[] fa0 = ctx.mmaLoadA(aSmem, 16, aBase + 0);
            HalfFloat[] fa1 = ctx.mmaLoadA(aSmem, 16, aBase + 512);
            HalfFloat[] fb0 = ctx.mmaLoadB(bSmem, 16, bBase + 0);
            HalfFloat[] fb1 = ctx.mmaLoadB(bSmem, 16, bBase + 256);
            HalfFloat[] fb2 = ctx.mmaLoadB(bSmem, 16, bBase + 512);
            HalfFloat[] fb3 = ctx.mmaLoadB(bSmem, 16, bBase + 768);
            HalfFloat[] fb4 = ctx.mmaLoadB(bSmem, 16, bBase + 1024);
            HalfFloat[] fb5 = ctx.mmaLoadB(bSmem, 16, bBase + 1280);
            HalfFloat[] fb6 = ctx.mmaLoadB(bSmem, 16, bBase + 1536);
            HalfFloat[] fb7 = ctx.mmaLoadB(bSmem, 16, bBase + 1792);
            acc0_0 = ctx.mma(fa0, fb0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa0, fb1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa0, fb2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa0, fb3, acc0_3, MMAShape.M16N8K16);
            acc0_4 = ctx.mma(fa0, fb4, acc0_4, MMAShape.M16N8K16);
            acc0_5 = ctx.mma(fa0, fb5, acc0_5, MMAShape.M16N8K16);
            acc0_6 = ctx.mma(fa0, fb6, acc0_6, MMAShape.M16N8K16);
            acc0_7 = ctx.mma(fa0, fb7, acc0_7, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa1, fb0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa1, fb1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa1, fb2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa1, fb3, acc1_3, MMAShape.M16N8K16);
            acc1_4 = ctx.mma(fa1, fb4, acc1_4, MMAShape.M16N8K16);
            acc1_5 = ctx.mma(fa1, fb5, acc1_5, MMAShape.M16N8K16);
            acc1_6 = ctx.mma(fa1, fb6, acc1_6, MMAShape.M16N8K16);
            acc1_7 = ctx.mma(fa1, fb7, acc1_7, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        int row0 = blockRow + warpRow * 32;
        int col0 = blockCol + warpCol * 64;
        ctx.mmaStore(acc0_0, c, row0 + 0, col0 + 0, n);
        ctx.mmaStore(acc0_1, c, row0 + 0, col0 + 8, n);
        ctx.mmaStore(acc0_2, c, row0 + 0, col0 + 16, n);
        ctx.mmaStore(acc0_3, c, row0 + 0, col0 + 24, n);
        ctx.mmaStore(acc0_4, c, row0 + 0, col0 + 32, n);
        ctx.mmaStore(acc0_5, c, row0 + 0, col0 + 40, n);
        ctx.mmaStore(acc0_6, c, row0 + 0, col0 + 48, n);
        ctx.mmaStore(acc0_7, c, row0 + 0, col0 + 56, n);
        ctx.mmaStore(acc1_0, c, row0 + 16, col0 + 0, n);
        ctx.mmaStore(acc1_1, c, row0 + 16, col0 + 8, n);
        ctx.mmaStore(acc1_2, c, row0 + 16, col0 + 16, n);
        ctx.mmaStore(acc1_3, c, row0 + 16, col0 + 24, n);
        ctx.mmaStore(acc1_4, c, row0 + 16, col0 + 32, n);
        ctx.mmaStore(acc1_5, c, row0 + 16, col0 + 40, n);
        ctx.mmaStore(acc1_6, c, row0 + 16, col0 + 48, n);
        ctx.mmaStore(acc1_7, c, row0 + 16, col0 + 56, n);
    }

    private static final int O128X128_4X2_BM = 128, O128X128_4X2_BN = 128, O128X128_4X2_T = 256;
    public static void o128x128_4x2(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int tid = ctx.localIdx;
        int blocksPerRow = n / 128;
        int blockRow = (ctx.groupIdx / blocksPerRow) * 128;
        int blockCol = (ctx.groupIdx % blocksPerRow) * 128;
        int warp = tid / 32;
        int warpRow = warp / 2;
        int warpCol = warp % 2;
        int[] aSmem = ctx.allocateIntLocalArray(2048);
        int[] bSmem = ctx.allocateIntLocalArray(2048);
        float[] acc0_0 = ctx.mmaFragment(0.0f);
        float[] acc0_1 = ctx.mmaFragment(0.0f);
        float[] acc0_2 = ctx.mmaFragment(0.0f);
        float[] acc0_3 = ctx.mmaFragment(0.0f);
        float[] acc0_4 = ctx.mmaFragment(0.0f);
        float[] acc0_5 = ctx.mmaFragment(0.0f);
        float[] acc0_6 = ctx.mmaFragment(0.0f);
        float[] acc0_7 = ctx.mmaFragment(0.0f);
        float[] acc1_0 = ctx.mmaFragment(0.0f);
        float[] acc1_1 = ctx.mmaFragment(0.0f);
        float[] acc1_2 = ctx.mmaFragment(0.0f);
        float[] acc1_3 = ctx.mmaFragment(0.0f);
        float[] acc1_4 = ctx.mmaFragment(0.0f);
        float[] acc1_5 = ctx.mmaFragment(0.0f);
        float[] acc1_6 = ctx.mmaFragment(0.0f);
        float[] acc1_7 = ctx.mmaFragment(0.0f);
        int numK = n / 16;
            { int idx = tid + 0;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 256;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 512;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
            { int idx = tid + 768;
              ctx.asyncCopyToLocal(aSmem, 0 * 1024 + idx, a, (blockRow + idx / 8) * n + 0 + (idx % 8) * 2);
              int kRow = idx / 64; int rem = idx % 64;
              ctx.asyncCopyToLocal(bSmem, 0 * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + kRow) * n + blockCol + rem * 2);
            }
        ctx.asyncCopyCommit();
        for (int kt = 0; kt < numK; kt++) {
            int cur = kt % 2;
            if (kt + 1 < numK) {
                int nxt = 1 - cur;
                int kNext = (kt + 1) * 16;
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 512;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 768;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + idx, a, (blockRow + idx / 8) * n + kNext + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + kRow) * n + blockCol + rem * 2);
                }
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(1);
            } else {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();
            int aBase = cur * 4096 + warpRow * 1024;
            int bBase = cur * 4096 + warpCol * 2048;
            HalfFloat[] fa0 = ctx.mmaLoadA(aSmem, 16, aBase + 0);
            HalfFloat[] fa1 = ctx.mmaLoadA(aSmem, 16, aBase + 512);
            HalfFloat[] fb0 = ctx.mmaLoadB(bSmem, 16, bBase + 0);
            HalfFloat[] fb1 = ctx.mmaLoadB(bSmem, 16, bBase + 256);
            HalfFloat[] fb2 = ctx.mmaLoadB(bSmem, 16, bBase + 512);
            HalfFloat[] fb3 = ctx.mmaLoadB(bSmem, 16, bBase + 768);
            HalfFloat[] fb4 = ctx.mmaLoadB(bSmem, 16, bBase + 1024);
            HalfFloat[] fb5 = ctx.mmaLoadB(bSmem, 16, bBase + 1280);
            HalfFloat[] fb6 = ctx.mmaLoadB(bSmem, 16, bBase + 1536);
            HalfFloat[] fb7 = ctx.mmaLoadB(bSmem, 16, bBase + 1792);
            acc0_0 = ctx.mma(fa0, fb0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa0, fb1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa0, fb2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa0, fb3, acc0_3, MMAShape.M16N8K16);
            acc0_4 = ctx.mma(fa0, fb4, acc0_4, MMAShape.M16N8K16);
            acc0_5 = ctx.mma(fa0, fb5, acc0_5, MMAShape.M16N8K16);
            acc0_6 = ctx.mma(fa0, fb6, acc0_6, MMAShape.M16N8K16);
            acc0_7 = ctx.mma(fa0, fb7, acc0_7, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa1, fb0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa1, fb1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa1, fb2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa1, fb3, acc1_3, MMAShape.M16N8K16);
            acc1_4 = ctx.mma(fa1, fb4, acc1_4, MMAShape.M16N8K16);
            acc1_5 = ctx.mma(fa1, fb5, acc1_5, MMAShape.M16N8K16);
            acc1_6 = ctx.mma(fa1, fb6, acc1_6, MMAShape.M16N8K16);
            acc1_7 = ctx.mma(fa1, fb7, acc1_7, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        int row0 = blockRow + warpRow * 32;
        int col0 = blockCol + warpCol * 64;
        ctx.mmaStore(acc0_0, c, row0 + 0, col0 + 0, n);
        ctx.mmaStore(acc0_1, c, row0 + 0, col0 + 8, n);
        ctx.mmaStore(acc0_2, c, row0 + 0, col0 + 16, n);
        ctx.mmaStore(acc0_3, c, row0 + 0, col0 + 24, n);
        ctx.mmaStore(acc0_4, c, row0 + 0, col0 + 32, n);
        ctx.mmaStore(acc0_5, c, row0 + 0, col0 + 40, n);
        ctx.mmaStore(acc0_6, c, row0 + 0, col0 + 48, n);
        ctx.mmaStore(acc0_7, c, row0 + 0, col0 + 56, n);
        ctx.mmaStore(acc1_0, c, row0 + 16, col0 + 0, n);
        ctx.mmaStore(acc1_1, c, row0 + 16, col0 + 8, n);
        ctx.mmaStore(acc1_2, c, row0 + 16, col0 + 16, n);
        ctx.mmaStore(acc1_3, c, row0 + 16, col0 + 24, n);
        ctx.mmaStore(acc1_4, c, row0 + 16, col0 + 32, n);
        ctx.mmaStore(acc1_5, c, row0 + 16, col0 + 40, n);
        ctx.mmaStore(acc1_6, c, row0 + 16, col0 + 48, n);
        ctx.mmaStore(acc1_7, c, row0 + 16, col0 + 56, n);
    }

    public static void p128x128_2x4_k32(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int tid = ctx.localIdx;
        int blocksPerRow = n / 128;
        int blockRow = (ctx.groupIdx / blocksPerRow) * 128;
        int blockCol = (ctx.groupIdx % blocksPerRow) * 128;
        int warp = tid / 32;
        int warpRow = warp / 4;
        int warpCol = warp % 4;
        int[] aSmem = ctx.allocateIntLocalArray(4096);
        int[] bSmem = ctx.allocateIntLocalArray(4096);
        float[] acc0_0 = ctx.mmaFragment(0.0f);
        float[] acc0_1 = ctx.mmaFragment(0.0f);
        float[] acc0_2 = ctx.mmaFragment(0.0f);
        float[] acc0_3 = ctx.mmaFragment(0.0f);
        float[] acc1_0 = ctx.mmaFragment(0.0f);
        float[] acc1_1 = ctx.mmaFragment(0.0f);
        float[] acc1_2 = ctx.mmaFragment(0.0f);
        float[] acc1_3 = ctx.mmaFragment(0.0f);
        float[] acc2_0 = ctx.mmaFragment(0.0f);
        float[] acc2_1 = ctx.mmaFragment(0.0f);
        float[] acc2_2 = ctx.mmaFragment(0.0f);
        float[] acc2_3 = ctx.mmaFragment(0.0f);
        float[] acc3_0 = ctx.mmaFragment(0.0f);
        float[] acc3_1 = ctx.mmaFragment(0.0f);
        float[] acc3_2 = ctx.mmaFragment(0.0f);
        float[] acc3_3 = ctx.mmaFragment(0.0f);
        int numK = n / 32;
        { int idx = tid + 0;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 256;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 512;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 768;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 0;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 256;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 512;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 768;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        ctx.asyncCopyCommit();
        for (int kt = 0; kt < numK; kt++) {
            int cur = kt % 2;
            if (kt + 1 < numK) {
                int nxt = 1 - cur;
                int kNext = (kt + 1) * 32;
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 512;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 768;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 512;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 768;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(1);
            } else {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();
            int aBase0 = cur * 8192 + 0 + warpRow * 2048;
            int bBase0 = cur * 8192 + 0 + warpCol * 1024;
            HalfFloat[] fa0_0 = ctx.mmaLoadA(aSmem, 16, aBase0 + 0);
            HalfFloat[] fa0_1 = ctx.mmaLoadA(aSmem, 16, aBase0 + 512);
            HalfFloat[] fa0_2 = ctx.mmaLoadA(aSmem, 16, aBase0 + 1024);
            HalfFloat[] fa0_3 = ctx.mmaLoadA(aSmem, 16, aBase0 + 1536);
            HalfFloat[] fb0_0 = ctx.mmaLoadB(bSmem, 16, bBase0 + 0);
            HalfFloat[] fb0_1 = ctx.mmaLoadB(bSmem, 16, bBase0 + 256);
            HalfFloat[] fb0_2 = ctx.mmaLoadB(bSmem, 16, bBase0 + 512);
            HalfFloat[] fb0_3 = ctx.mmaLoadB(bSmem, 16, bBase0 + 768);
            acc0_0 = ctx.mma(fa0_0, fb0_0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa0_0, fb0_1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa0_0, fb0_2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa0_0, fb0_3, acc0_3, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa0_1, fb0_0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa0_1, fb0_1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa0_1, fb0_2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa0_1, fb0_3, acc1_3, MMAShape.M16N8K16);
            acc2_0 = ctx.mma(fa0_2, fb0_0, acc2_0, MMAShape.M16N8K16);
            acc2_1 = ctx.mma(fa0_2, fb0_1, acc2_1, MMAShape.M16N8K16);
            acc2_2 = ctx.mma(fa0_2, fb0_2, acc2_2, MMAShape.M16N8K16);
            acc2_3 = ctx.mma(fa0_2, fb0_3, acc2_3, MMAShape.M16N8K16);
            acc3_0 = ctx.mma(fa0_3, fb0_0, acc3_0, MMAShape.M16N8K16);
            acc3_1 = ctx.mma(fa0_3, fb0_1, acc3_1, MMAShape.M16N8K16);
            acc3_2 = ctx.mma(fa0_3, fb0_2, acc3_2, MMAShape.M16N8K16);
            acc3_3 = ctx.mma(fa0_3, fb0_3, acc3_3, MMAShape.M16N8K16);
            int aBase1 = cur * 8192 + 4096 + warpRow * 2048;
            int bBase1 = cur * 8192 + 4096 + warpCol * 1024;
            HalfFloat[] fa1_0 = ctx.mmaLoadA(aSmem, 16, aBase1 + 0);
            HalfFloat[] fa1_1 = ctx.mmaLoadA(aSmem, 16, aBase1 + 512);
            HalfFloat[] fa1_2 = ctx.mmaLoadA(aSmem, 16, aBase1 + 1024);
            HalfFloat[] fa1_3 = ctx.mmaLoadA(aSmem, 16, aBase1 + 1536);
            HalfFloat[] fb1_0 = ctx.mmaLoadB(bSmem, 16, bBase1 + 0);
            HalfFloat[] fb1_1 = ctx.mmaLoadB(bSmem, 16, bBase1 + 256);
            HalfFloat[] fb1_2 = ctx.mmaLoadB(bSmem, 16, bBase1 + 512);
            HalfFloat[] fb1_3 = ctx.mmaLoadB(bSmem, 16, bBase1 + 768);
            acc0_0 = ctx.mma(fa1_0, fb1_0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa1_0, fb1_1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa1_0, fb1_2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa1_0, fb1_3, acc0_3, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa1_1, fb1_0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa1_1, fb1_1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa1_1, fb1_2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa1_1, fb1_3, acc1_3, MMAShape.M16N8K16);
            acc2_0 = ctx.mma(fa1_2, fb1_0, acc2_0, MMAShape.M16N8K16);
            acc2_1 = ctx.mma(fa1_2, fb1_1, acc2_1, MMAShape.M16N8K16);
            acc2_2 = ctx.mma(fa1_2, fb1_2, acc2_2, MMAShape.M16N8K16);
            acc2_3 = ctx.mma(fa1_2, fb1_3, acc2_3, MMAShape.M16N8K16);
            acc3_0 = ctx.mma(fa1_3, fb1_0, acc3_0, MMAShape.M16N8K16);
            acc3_1 = ctx.mma(fa1_3, fb1_1, acc3_1, MMAShape.M16N8K16);
            acc3_2 = ctx.mma(fa1_3, fb1_2, acc3_2, MMAShape.M16N8K16);
            acc3_3 = ctx.mma(fa1_3, fb1_3, acc3_3, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        int row0 = blockRow + warpRow * 64;
        int col0 = blockCol + warpCol * 32;
        ctx.mmaStore(acc0_0, c, row0 + 0, col0 + 0, n);
        ctx.mmaStore(acc0_1, c, row0 + 0, col0 + 8, n);
        ctx.mmaStore(acc0_2, c, row0 + 0, col0 + 16, n);
        ctx.mmaStore(acc0_3, c, row0 + 0, col0 + 24, n);
        ctx.mmaStore(acc1_0, c, row0 + 16, col0 + 0, n);
        ctx.mmaStore(acc1_1, c, row0 + 16, col0 + 8, n);
        ctx.mmaStore(acc1_2, c, row0 + 16, col0 + 16, n);
        ctx.mmaStore(acc1_3, c, row0 + 16, col0 + 24, n);
        ctx.mmaStore(acc2_0, c, row0 + 32, col0 + 0, n);
        ctx.mmaStore(acc2_1, c, row0 + 32, col0 + 8, n);
        ctx.mmaStore(acc2_2, c, row0 + 32, col0 + 16, n);
        ctx.mmaStore(acc2_3, c, row0 + 32, col0 + 24, n);
        ctx.mmaStore(acc3_0, c, row0 + 48, col0 + 0, n);
        ctx.mmaStore(acc3_1, c, row0 + 48, col0 + 8, n);
        ctx.mmaStore(acc3_2, c, row0 + 48, col0 + 16, n);
        ctx.mmaStore(acc3_3, c, row0 + 48, col0 + 24, n);
    }

    public static void p128x128_2x2_k32(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int tid = ctx.localIdx;
        int blocksPerRow = n / 128;
        int blockRow = (ctx.groupIdx / blocksPerRow) * 128;
        int blockCol = (ctx.groupIdx % blocksPerRow) * 128;
        int warp = tid / 32;
        int warpRow = warp / 2;
        int warpCol = warp % 2;
        int[] aSmem = ctx.allocateIntLocalArray(4096);
        int[] bSmem = ctx.allocateIntLocalArray(4096);
        float[] acc0_0 = ctx.mmaFragment(0.0f);
        float[] acc0_1 = ctx.mmaFragment(0.0f);
        float[] acc0_2 = ctx.mmaFragment(0.0f);
        float[] acc0_3 = ctx.mmaFragment(0.0f);
        float[] acc0_4 = ctx.mmaFragment(0.0f);
        float[] acc0_5 = ctx.mmaFragment(0.0f);
        float[] acc0_6 = ctx.mmaFragment(0.0f);
        float[] acc0_7 = ctx.mmaFragment(0.0f);
        float[] acc1_0 = ctx.mmaFragment(0.0f);
        float[] acc1_1 = ctx.mmaFragment(0.0f);
        float[] acc1_2 = ctx.mmaFragment(0.0f);
        float[] acc1_3 = ctx.mmaFragment(0.0f);
        float[] acc1_4 = ctx.mmaFragment(0.0f);
        float[] acc1_5 = ctx.mmaFragment(0.0f);
        float[] acc1_6 = ctx.mmaFragment(0.0f);
        float[] acc1_7 = ctx.mmaFragment(0.0f);
        float[] acc2_0 = ctx.mmaFragment(0.0f);
        float[] acc2_1 = ctx.mmaFragment(0.0f);
        float[] acc2_2 = ctx.mmaFragment(0.0f);
        float[] acc2_3 = ctx.mmaFragment(0.0f);
        float[] acc2_4 = ctx.mmaFragment(0.0f);
        float[] acc2_5 = ctx.mmaFragment(0.0f);
        float[] acc2_6 = ctx.mmaFragment(0.0f);
        float[] acc2_7 = ctx.mmaFragment(0.0f);
        float[] acc3_0 = ctx.mmaFragment(0.0f);
        float[] acc3_1 = ctx.mmaFragment(0.0f);
        float[] acc3_2 = ctx.mmaFragment(0.0f);
        float[] acc3_3 = ctx.mmaFragment(0.0f);
        float[] acc3_4 = ctx.mmaFragment(0.0f);
        float[] acc3_5 = ctx.mmaFragment(0.0f);
        float[] acc3_6 = ctx.mmaFragment(0.0f);
        float[] acc3_7 = ctx.mmaFragment(0.0f);
        int numK = n / 32;
        { int idx = tid + 0;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 128;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 256;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 384;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 512;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 640;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 768;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 896;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 0;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 128;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 256;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 384;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 512;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 640;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 768;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 896;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        ctx.asyncCopyCommit();
        for (int kt = 0; kt < numK; kt++) {
            int cur = kt % 2;
            if (kt + 1 < numK) {
                int nxt = 1 - cur;
                int kNext = (kt + 1) * 32;
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 128;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 384;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 512;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 640;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 768;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 896;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 128;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 384;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 512;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 640;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 768;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 896;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(1);
            } else {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();
            int aBase0 = cur * 8192 + 0 + warpRow * 2048;
            int bBase0 = cur * 8192 + 0 + warpCol * 2048;
            HalfFloat[] fa0_0 = ctx.mmaLoadA(aSmem, 16, aBase0 + 0);
            HalfFloat[] fa0_1 = ctx.mmaLoadA(aSmem, 16, aBase0 + 512);
            HalfFloat[] fa0_2 = ctx.mmaLoadA(aSmem, 16, aBase0 + 1024);
            HalfFloat[] fa0_3 = ctx.mmaLoadA(aSmem, 16, aBase0 + 1536);
            HalfFloat[] fb0_0 = ctx.mmaLoadB(bSmem, 16, bBase0 + 0);
            HalfFloat[] fb0_1 = ctx.mmaLoadB(bSmem, 16, bBase0 + 256);
            HalfFloat[] fb0_2 = ctx.mmaLoadB(bSmem, 16, bBase0 + 512);
            HalfFloat[] fb0_3 = ctx.mmaLoadB(bSmem, 16, bBase0 + 768);
            HalfFloat[] fb0_4 = ctx.mmaLoadB(bSmem, 16, bBase0 + 1024);
            HalfFloat[] fb0_5 = ctx.mmaLoadB(bSmem, 16, bBase0 + 1280);
            HalfFloat[] fb0_6 = ctx.mmaLoadB(bSmem, 16, bBase0 + 1536);
            HalfFloat[] fb0_7 = ctx.mmaLoadB(bSmem, 16, bBase0 + 1792);
            acc0_0 = ctx.mma(fa0_0, fb0_0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa0_0, fb0_1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa0_0, fb0_2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa0_0, fb0_3, acc0_3, MMAShape.M16N8K16);
            acc0_4 = ctx.mma(fa0_0, fb0_4, acc0_4, MMAShape.M16N8K16);
            acc0_5 = ctx.mma(fa0_0, fb0_5, acc0_5, MMAShape.M16N8K16);
            acc0_6 = ctx.mma(fa0_0, fb0_6, acc0_6, MMAShape.M16N8K16);
            acc0_7 = ctx.mma(fa0_0, fb0_7, acc0_7, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa0_1, fb0_0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa0_1, fb0_1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa0_1, fb0_2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa0_1, fb0_3, acc1_3, MMAShape.M16N8K16);
            acc1_4 = ctx.mma(fa0_1, fb0_4, acc1_4, MMAShape.M16N8K16);
            acc1_5 = ctx.mma(fa0_1, fb0_5, acc1_5, MMAShape.M16N8K16);
            acc1_6 = ctx.mma(fa0_1, fb0_6, acc1_6, MMAShape.M16N8K16);
            acc1_7 = ctx.mma(fa0_1, fb0_7, acc1_7, MMAShape.M16N8K16);
            acc2_0 = ctx.mma(fa0_2, fb0_0, acc2_0, MMAShape.M16N8K16);
            acc2_1 = ctx.mma(fa0_2, fb0_1, acc2_1, MMAShape.M16N8K16);
            acc2_2 = ctx.mma(fa0_2, fb0_2, acc2_2, MMAShape.M16N8K16);
            acc2_3 = ctx.mma(fa0_2, fb0_3, acc2_3, MMAShape.M16N8K16);
            acc2_4 = ctx.mma(fa0_2, fb0_4, acc2_4, MMAShape.M16N8K16);
            acc2_5 = ctx.mma(fa0_2, fb0_5, acc2_5, MMAShape.M16N8K16);
            acc2_6 = ctx.mma(fa0_2, fb0_6, acc2_6, MMAShape.M16N8K16);
            acc2_7 = ctx.mma(fa0_2, fb0_7, acc2_7, MMAShape.M16N8K16);
            acc3_0 = ctx.mma(fa0_3, fb0_0, acc3_0, MMAShape.M16N8K16);
            acc3_1 = ctx.mma(fa0_3, fb0_1, acc3_1, MMAShape.M16N8K16);
            acc3_2 = ctx.mma(fa0_3, fb0_2, acc3_2, MMAShape.M16N8K16);
            acc3_3 = ctx.mma(fa0_3, fb0_3, acc3_3, MMAShape.M16N8K16);
            acc3_4 = ctx.mma(fa0_3, fb0_4, acc3_4, MMAShape.M16N8K16);
            acc3_5 = ctx.mma(fa0_3, fb0_5, acc3_5, MMAShape.M16N8K16);
            acc3_6 = ctx.mma(fa0_3, fb0_6, acc3_6, MMAShape.M16N8K16);
            acc3_7 = ctx.mma(fa0_3, fb0_7, acc3_7, MMAShape.M16N8K16);
            int aBase1 = cur * 8192 + 4096 + warpRow * 2048;
            int bBase1 = cur * 8192 + 4096 + warpCol * 2048;
            HalfFloat[] fa1_0 = ctx.mmaLoadA(aSmem, 16, aBase1 + 0);
            HalfFloat[] fa1_1 = ctx.mmaLoadA(aSmem, 16, aBase1 + 512);
            HalfFloat[] fa1_2 = ctx.mmaLoadA(aSmem, 16, aBase1 + 1024);
            HalfFloat[] fa1_3 = ctx.mmaLoadA(aSmem, 16, aBase1 + 1536);
            HalfFloat[] fb1_0 = ctx.mmaLoadB(bSmem, 16, bBase1 + 0);
            HalfFloat[] fb1_1 = ctx.mmaLoadB(bSmem, 16, bBase1 + 256);
            HalfFloat[] fb1_2 = ctx.mmaLoadB(bSmem, 16, bBase1 + 512);
            HalfFloat[] fb1_3 = ctx.mmaLoadB(bSmem, 16, bBase1 + 768);
            HalfFloat[] fb1_4 = ctx.mmaLoadB(bSmem, 16, bBase1 + 1024);
            HalfFloat[] fb1_5 = ctx.mmaLoadB(bSmem, 16, bBase1 + 1280);
            HalfFloat[] fb1_6 = ctx.mmaLoadB(bSmem, 16, bBase1 + 1536);
            HalfFloat[] fb1_7 = ctx.mmaLoadB(bSmem, 16, bBase1 + 1792);
            acc0_0 = ctx.mma(fa1_0, fb1_0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa1_0, fb1_1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa1_0, fb1_2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa1_0, fb1_3, acc0_3, MMAShape.M16N8K16);
            acc0_4 = ctx.mma(fa1_0, fb1_4, acc0_4, MMAShape.M16N8K16);
            acc0_5 = ctx.mma(fa1_0, fb1_5, acc0_5, MMAShape.M16N8K16);
            acc0_6 = ctx.mma(fa1_0, fb1_6, acc0_6, MMAShape.M16N8K16);
            acc0_7 = ctx.mma(fa1_0, fb1_7, acc0_7, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa1_1, fb1_0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa1_1, fb1_1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa1_1, fb1_2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa1_1, fb1_3, acc1_3, MMAShape.M16N8K16);
            acc1_4 = ctx.mma(fa1_1, fb1_4, acc1_4, MMAShape.M16N8K16);
            acc1_5 = ctx.mma(fa1_1, fb1_5, acc1_5, MMAShape.M16N8K16);
            acc1_6 = ctx.mma(fa1_1, fb1_6, acc1_6, MMAShape.M16N8K16);
            acc1_7 = ctx.mma(fa1_1, fb1_7, acc1_7, MMAShape.M16N8K16);
            acc2_0 = ctx.mma(fa1_2, fb1_0, acc2_0, MMAShape.M16N8K16);
            acc2_1 = ctx.mma(fa1_2, fb1_1, acc2_1, MMAShape.M16N8K16);
            acc2_2 = ctx.mma(fa1_2, fb1_2, acc2_2, MMAShape.M16N8K16);
            acc2_3 = ctx.mma(fa1_2, fb1_3, acc2_3, MMAShape.M16N8K16);
            acc2_4 = ctx.mma(fa1_2, fb1_4, acc2_4, MMAShape.M16N8K16);
            acc2_5 = ctx.mma(fa1_2, fb1_5, acc2_5, MMAShape.M16N8K16);
            acc2_6 = ctx.mma(fa1_2, fb1_6, acc2_6, MMAShape.M16N8K16);
            acc2_7 = ctx.mma(fa1_2, fb1_7, acc2_7, MMAShape.M16N8K16);
            acc3_0 = ctx.mma(fa1_3, fb1_0, acc3_0, MMAShape.M16N8K16);
            acc3_1 = ctx.mma(fa1_3, fb1_1, acc3_1, MMAShape.M16N8K16);
            acc3_2 = ctx.mma(fa1_3, fb1_2, acc3_2, MMAShape.M16N8K16);
            acc3_3 = ctx.mma(fa1_3, fb1_3, acc3_3, MMAShape.M16N8K16);
            acc3_4 = ctx.mma(fa1_3, fb1_4, acc3_4, MMAShape.M16N8K16);
            acc3_5 = ctx.mma(fa1_3, fb1_5, acc3_5, MMAShape.M16N8K16);
            acc3_6 = ctx.mma(fa1_3, fb1_6, acc3_6, MMAShape.M16N8K16);
            acc3_7 = ctx.mma(fa1_3, fb1_7, acc3_7, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        int row0 = blockRow + warpRow * 64;
        int col0 = blockCol + warpCol * 64;
        ctx.mmaStore(acc0_0, c, row0 + 0, col0 + 0, n);
        ctx.mmaStore(acc0_1, c, row0 + 0, col0 + 8, n);
        ctx.mmaStore(acc0_2, c, row0 + 0, col0 + 16, n);
        ctx.mmaStore(acc0_3, c, row0 + 0, col0 + 24, n);
        ctx.mmaStore(acc0_4, c, row0 + 0, col0 + 32, n);
        ctx.mmaStore(acc0_5, c, row0 + 0, col0 + 40, n);
        ctx.mmaStore(acc0_6, c, row0 + 0, col0 + 48, n);
        ctx.mmaStore(acc0_7, c, row0 + 0, col0 + 56, n);
        ctx.mmaStore(acc1_0, c, row0 + 16, col0 + 0, n);
        ctx.mmaStore(acc1_1, c, row0 + 16, col0 + 8, n);
        ctx.mmaStore(acc1_2, c, row0 + 16, col0 + 16, n);
        ctx.mmaStore(acc1_3, c, row0 + 16, col0 + 24, n);
        ctx.mmaStore(acc1_4, c, row0 + 16, col0 + 32, n);
        ctx.mmaStore(acc1_5, c, row0 + 16, col0 + 40, n);
        ctx.mmaStore(acc1_6, c, row0 + 16, col0 + 48, n);
        ctx.mmaStore(acc1_7, c, row0 + 16, col0 + 56, n);
        ctx.mmaStore(acc2_0, c, row0 + 32, col0 + 0, n);
        ctx.mmaStore(acc2_1, c, row0 + 32, col0 + 8, n);
        ctx.mmaStore(acc2_2, c, row0 + 32, col0 + 16, n);
        ctx.mmaStore(acc2_3, c, row0 + 32, col0 + 24, n);
        ctx.mmaStore(acc2_4, c, row0 + 32, col0 + 32, n);
        ctx.mmaStore(acc2_5, c, row0 + 32, col0 + 40, n);
        ctx.mmaStore(acc2_6, c, row0 + 32, col0 + 48, n);
        ctx.mmaStore(acc2_7, c, row0 + 32, col0 + 56, n);
        ctx.mmaStore(acc3_0, c, row0 + 48, col0 + 0, n);
        ctx.mmaStore(acc3_1, c, row0 + 48, col0 + 8, n);
        ctx.mmaStore(acc3_2, c, row0 + 48, col0 + 16, n);
        ctx.mmaStore(acc3_3, c, row0 + 48, col0 + 24, n);
        ctx.mmaStore(acc3_4, c, row0 + 48, col0 + 32, n);
        ctx.mmaStore(acc3_5, c, row0 + 48, col0 + 40, n);
        ctx.mmaStore(acc3_6, c, row0 + 48, col0 + 48, n);
        ctx.mmaStore(acc3_7, c, row0 + 48, col0 + 56, n);
    }

    public static void p128x256_2x4_k16(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int tid = ctx.localIdx;
        int blocksPerRow = n / 256;
        int blockRow = (ctx.groupIdx / blocksPerRow) * 128;
        int blockCol = (ctx.groupIdx % blocksPerRow) * 256;
        int warp = tid / 32;
        int warpRow = warp / 4;
        int warpCol = warp % 4;
        int[] aSmem = ctx.allocateIntLocalArray(2048);
        int[] bSmem = ctx.allocateIntLocalArray(4096);
        float[] acc0_0 = ctx.mmaFragment(0.0f);
        float[] acc0_1 = ctx.mmaFragment(0.0f);
        float[] acc0_2 = ctx.mmaFragment(0.0f);
        float[] acc0_3 = ctx.mmaFragment(0.0f);
        float[] acc0_4 = ctx.mmaFragment(0.0f);
        float[] acc0_5 = ctx.mmaFragment(0.0f);
        float[] acc0_6 = ctx.mmaFragment(0.0f);
        float[] acc0_7 = ctx.mmaFragment(0.0f);
        float[] acc1_0 = ctx.mmaFragment(0.0f);
        float[] acc1_1 = ctx.mmaFragment(0.0f);
        float[] acc1_2 = ctx.mmaFragment(0.0f);
        float[] acc1_3 = ctx.mmaFragment(0.0f);
        float[] acc1_4 = ctx.mmaFragment(0.0f);
        float[] acc1_5 = ctx.mmaFragment(0.0f);
        float[] acc1_6 = ctx.mmaFragment(0.0f);
        float[] acc1_7 = ctx.mmaFragment(0.0f);
        float[] acc2_0 = ctx.mmaFragment(0.0f);
        float[] acc2_1 = ctx.mmaFragment(0.0f);
        float[] acc2_2 = ctx.mmaFragment(0.0f);
        float[] acc2_3 = ctx.mmaFragment(0.0f);
        float[] acc2_4 = ctx.mmaFragment(0.0f);
        float[] acc2_5 = ctx.mmaFragment(0.0f);
        float[] acc2_6 = ctx.mmaFragment(0.0f);
        float[] acc2_7 = ctx.mmaFragment(0.0f);
        float[] acc3_0 = ctx.mmaFragment(0.0f);
        float[] acc3_1 = ctx.mmaFragment(0.0f);
        float[] acc3_2 = ctx.mmaFragment(0.0f);
        float[] acc3_3 = ctx.mmaFragment(0.0f);
        float[] acc3_4 = ctx.mmaFragment(0.0f);
        float[] acc3_5 = ctx.mmaFragment(0.0f);
        float[] acc3_6 = ctx.mmaFragment(0.0f);
        float[] acc3_7 = ctx.mmaFragment(0.0f);
        int numK = n / 16;
        { int idx = tid + 0;
          ctx.asyncCopyToLocal(aSmem, 0 * 1024 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 128; int rem = idx % 128;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 256;
          ctx.asyncCopyToLocal(aSmem, 0 * 1024 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 128; int rem = idx % 128;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 512;
          ctx.asyncCopyToLocal(aSmem, 0 * 1024 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 128; int rem = idx % 128;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 768;
          ctx.asyncCopyToLocal(aSmem, 0 * 1024 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 128; int rem = idx % 128;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 1024;
          int kRow = idx / 128; int rem = idx % 128;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 1280;
          int kRow = idx / 128; int rem = idx % 128;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 1536;
          int kRow = idx / 128; int rem = idx % 128;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 1792;
          int kRow = idx / 128; int rem = idx % 128;
          ctx.asyncCopyToLocal(bSmem, 0 * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        ctx.asyncCopyCommit();
        for (int kt = 0; kt < numK; kt++) {
            int cur = kt % 2;
            if (kt + 1 < numK) {
                int nxt = 1 - cur;
                int kNext = (kt + 1) * 16;
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 128; int rem = idx % 128;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 128; int rem = idx % 128;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 512;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 128; int rem = idx % 128;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 768;
                  ctx.asyncCopyToLocal(aSmem, nxt * 1024 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 128; int rem = idx % 128;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 1024;
                  int kRow = idx / 128; int rem = idx % 128;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 1280;
                  int kRow = idx / 128; int rem = idx % 128;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 1536;
                  int kRow = idx / 128; int rem = idx % 128;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 1792;
                  int kRow = idx / 128; int rem = idx % 128;
                  ctx.asyncCopyToLocal(bSmem, nxt * 2048 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(1);
            } else {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();
            int aBase0 = cur * 4096 + 0 + warpRow * 2048;
            int bBase0 = cur * 8192 + 0 + warpCol * 2048;
            HalfFloat[] fa0_0 = ctx.mmaLoadA(aSmem, 16, aBase0 + 0);
            HalfFloat[] fa0_1 = ctx.mmaLoadA(aSmem, 16, aBase0 + 512);
            HalfFloat[] fa0_2 = ctx.mmaLoadA(aSmem, 16, aBase0 + 1024);
            HalfFloat[] fa0_3 = ctx.mmaLoadA(aSmem, 16, aBase0 + 1536);
            HalfFloat[] fb0_0 = ctx.mmaLoadB(bSmem, 16, bBase0 + 0);
            HalfFloat[] fb0_1 = ctx.mmaLoadB(bSmem, 16, bBase0 + 256);
            HalfFloat[] fb0_2 = ctx.mmaLoadB(bSmem, 16, bBase0 + 512);
            HalfFloat[] fb0_3 = ctx.mmaLoadB(bSmem, 16, bBase0 + 768);
            HalfFloat[] fb0_4 = ctx.mmaLoadB(bSmem, 16, bBase0 + 1024);
            HalfFloat[] fb0_5 = ctx.mmaLoadB(bSmem, 16, bBase0 + 1280);
            HalfFloat[] fb0_6 = ctx.mmaLoadB(bSmem, 16, bBase0 + 1536);
            HalfFloat[] fb0_7 = ctx.mmaLoadB(bSmem, 16, bBase0 + 1792);
            acc0_0 = ctx.mma(fa0_0, fb0_0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa0_0, fb0_1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa0_0, fb0_2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa0_0, fb0_3, acc0_3, MMAShape.M16N8K16);
            acc0_4 = ctx.mma(fa0_0, fb0_4, acc0_4, MMAShape.M16N8K16);
            acc0_5 = ctx.mma(fa0_0, fb0_5, acc0_5, MMAShape.M16N8K16);
            acc0_6 = ctx.mma(fa0_0, fb0_6, acc0_6, MMAShape.M16N8K16);
            acc0_7 = ctx.mma(fa0_0, fb0_7, acc0_7, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa0_1, fb0_0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa0_1, fb0_1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa0_1, fb0_2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa0_1, fb0_3, acc1_3, MMAShape.M16N8K16);
            acc1_4 = ctx.mma(fa0_1, fb0_4, acc1_4, MMAShape.M16N8K16);
            acc1_5 = ctx.mma(fa0_1, fb0_5, acc1_5, MMAShape.M16N8K16);
            acc1_6 = ctx.mma(fa0_1, fb0_6, acc1_6, MMAShape.M16N8K16);
            acc1_7 = ctx.mma(fa0_1, fb0_7, acc1_7, MMAShape.M16N8K16);
            acc2_0 = ctx.mma(fa0_2, fb0_0, acc2_0, MMAShape.M16N8K16);
            acc2_1 = ctx.mma(fa0_2, fb0_1, acc2_1, MMAShape.M16N8K16);
            acc2_2 = ctx.mma(fa0_2, fb0_2, acc2_2, MMAShape.M16N8K16);
            acc2_3 = ctx.mma(fa0_2, fb0_3, acc2_3, MMAShape.M16N8K16);
            acc2_4 = ctx.mma(fa0_2, fb0_4, acc2_4, MMAShape.M16N8K16);
            acc2_5 = ctx.mma(fa0_2, fb0_5, acc2_5, MMAShape.M16N8K16);
            acc2_6 = ctx.mma(fa0_2, fb0_6, acc2_6, MMAShape.M16N8K16);
            acc2_7 = ctx.mma(fa0_2, fb0_7, acc2_7, MMAShape.M16N8K16);
            acc3_0 = ctx.mma(fa0_3, fb0_0, acc3_0, MMAShape.M16N8K16);
            acc3_1 = ctx.mma(fa0_3, fb0_1, acc3_1, MMAShape.M16N8K16);
            acc3_2 = ctx.mma(fa0_3, fb0_2, acc3_2, MMAShape.M16N8K16);
            acc3_3 = ctx.mma(fa0_3, fb0_3, acc3_3, MMAShape.M16N8K16);
            acc3_4 = ctx.mma(fa0_3, fb0_4, acc3_4, MMAShape.M16N8K16);
            acc3_5 = ctx.mma(fa0_3, fb0_5, acc3_5, MMAShape.M16N8K16);
            acc3_6 = ctx.mma(fa0_3, fb0_6, acc3_6, MMAShape.M16N8K16);
            acc3_7 = ctx.mma(fa0_3, fb0_7, acc3_7, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        int row0 = blockRow + warpRow * 64;
        int col0 = blockCol + warpCol * 64;
        ctx.mmaStore(acc0_0, c, row0 + 0, col0 + 0, n);
        ctx.mmaStore(acc0_1, c, row0 + 0, col0 + 8, n);
        ctx.mmaStore(acc0_2, c, row0 + 0, col0 + 16, n);
        ctx.mmaStore(acc0_3, c, row0 + 0, col0 + 24, n);
        ctx.mmaStore(acc0_4, c, row0 + 0, col0 + 32, n);
        ctx.mmaStore(acc0_5, c, row0 + 0, col0 + 40, n);
        ctx.mmaStore(acc0_6, c, row0 + 0, col0 + 48, n);
        ctx.mmaStore(acc0_7, c, row0 + 0, col0 + 56, n);
        ctx.mmaStore(acc1_0, c, row0 + 16, col0 + 0, n);
        ctx.mmaStore(acc1_1, c, row0 + 16, col0 + 8, n);
        ctx.mmaStore(acc1_2, c, row0 + 16, col0 + 16, n);
        ctx.mmaStore(acc1_3, c, row0 + 16, col0 + 24, n);
        ctx.mmaStore(acc1_4, c, row0 + 16, col0 + 32, n);
        ctx.mmaStore(acc1_5, c, row0 + 16, col0 + 40, n);
        ctx.mmaStore(acc1_6, c, row0 + 16, col0 + 48, n);
        ctx.mmaStore(acc1_7, c, row0 + 16, col0 + 56, n);
        ctx.mmaStore(acc2_0, c, row0 + 32, col0 + 0, n);
        ctx.mmaStore(acc2_1, c, row0 + 32, col0 + 8, n);
        ctx.mmaStore(acc2_2, c, row0 + 32, col0 + 16, n);
        ctx.mmaStore(acc2_3, c, row0 + 32, col0 + 24, n);
        ctx.mmaStore(acc2_4, c, row0 + 32, col0 + 32, n);
        ctx.mmaStore(acc2_5, c, row0 + 32, col0 + 40, n);
        ctx.mmaStore(acc2_6, c, row0 + 32, col0 + 48, n);
        ctx.mmaStore(acc2_7, c, row0 + 32, col0 + 56, n);
        ctx.mmaStore(acc3_0, c, row0 + 48, col0 + 0, n);
        ctx.mmaStore(acc3_1, c, row0 + 48, col0 + 8, n);
        ctx.mmaStore(acc3_2, c, row0 + 48, col0 + 16, n);
        ctx.mmaStore(acc3_3, c, row0 + 48, col0 + 24, n);
        ctx.mmaStore(acc3_4, c, row0 + 48, col0 + 32, n);
        ctx.mmaStore(acc3_5, c, row0 + 48, col0 + 40, n);
        ctx.mmaStore(acc3_6, c, row0 + 48, col0 + 48, n);
        ctx.mmaStore(acc3_7, c, row0 + 48, col0 + 56, n);
    }

    public static void p256x128_4x2_k16(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int tid = ctx.localIdx;
        int blocksPerRow = n / 128;
        int blockRow = (ctx.groupIdx / blocksPerRow) * 256;
        int blockCol = (ctx.groupIdx % blocksPerRow) * 128;
        int warp = tid / 32;
        int warpRow = warp / 2;
        int warpCol = warp % 2;
        int[] aSmem = ctx.allocateIntLocalArray(4096);
        int[] bSmem = ctx.allocateIntLocalArray(2048);
        float[] acc0_0 = ctx.mmaFragment(0.0f);
        float[] acc0_1 = ctx.mmaFragment(0.0f);
        float[] acc0_2 = ctx.mmaFragment(0.0f);
        float[] acc0_3 = ctx.mmaFragment(0.0f);
        float[] acc0_4 = ctx.mmaFragment(0.0f);
        float[] acc0_5 = ctx.mmaFragment(0.0f);
        float[] acc0_6 = ctx.mmaFragment(0.0f);
        float[] acc0_7 = ctx.mmaFragment(0.0f);
        float[] acc1_0 = ctx.mmaFragment(0.0f);
        float[] acc1_1 = ctx.mmaFragment(0.0f);
        float[] acc1_2 = ctx.mmaFragment(0.0f);
        float[] acc1_3 = ctx.mmaFragment(0.0f);
        float[] acc1_4 = ctx.mmaFragment(0.0f);
        float[] acc1_5 = ctx.mmaFragment(0.0f);
        float[] acc1_6 = ctx.mmaFragment(0.0f);
        float[] acc1_7 = ctx.mmaFragment(0.0f);
        float[] acc2_0 = ctx.mmaFragment(0.0f);
        float[] acc2_1 = ctx.mmaFragment(0.0f);
        float[] acc2_2 = ctx.mmaFragment(0.0f);
        float[] acc2_3 = ctx.mmaFragment(0.0f);
        float[] acc2_4 = ctx.mmaFragment(0.0f);
        float[] acc2_5 = ctx.mmaFragment(0.0f);
        float[] acc2_6 = ctx.mmaFragment(0.0f);
        float[] acc2_7 = ctx.mmaFragment(0.0f);
        float[] acc3_0 = ctx.mmaFragment(0.0f);
        float[] acc3_1 = ctx.mmaFragment(0.0f);
        float[] acc3_2 = ctx.mmaFragment(0.0f);
        float[] acc3_3 = ctx.mmaFragment(0.0f);
        float[] acc3_4 = ctx.mmaFragment(0.0f);
        float[] acc3_5 = ctx.mmaFragment(0.0f);
        float[] acc3_6 = ctx.mmaFragment(0.0f);
        float[] acc3_7 = ctx.mmaFragment(0.0f);
        int numK = n / 16;
        { int idx = tid + 0;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 1024 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 256;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 1024 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 512;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 1024 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 768;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 1024 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 1024;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
        }
        { int idx = tid + 1280;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
        }
        { int idx = tid + 1536;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
        }
        { int idx = tid + 1792;
          ctx.asyncCopyToLocal(aSmem, 0 * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
        }
        ctx.asyncCopyCommit();
        for (int kt = 0; kt < numK; kt++) {
            int cur = kt % 2;
            if (kt + 1 < numK) {
                int nxt = 1 - cur;
                int kNext = (kt + 1) * 16;
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 512;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 768;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 1024 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 1024;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                }
                { int idx = tid + 1280;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                }
                { int idx = tid + 1536;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                }
                { int idx = tid + 1792;
                  ctx.asyncCopyToLocal(aSmem, nxt * 2048 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                }
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(1);
            } else {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();
            int aBase0 = cur * 8192 + 0 + warpRow * 2048;
            int bBase0 = cur * 4096 + 0 + warpCol * 2048;
            HalfFloat[] fa0_0 = ctx.mmaLoadA(aSmem, 16, aBase0 + 0);
            HalfFloat[] fa0_1 = ctx.mmaLoadA(aSmem, 16, aBase0 + 512);
            HalfFloat[] fa0_2 = ctx.mmaLoadA(aSmem, 16, aBase0 + 1024);
            HalfFloat[] fa0_3 = ctx.mmaLoadA(aSmem, 16, aBase0 + 1536);
            HalfFloat[] fb0_0 = ctx.mmaLoadB(bSmem, 16, bBase0 + 0);
            HalfFloat[] fb0_1 = ctx.mmaLoadB(bSmem, 16, bBase0 + 256);
            HalfFloat[] fb0_2 = ctx.mmaLoadB(bSmem, 16, bBase0 + 512);
            HalfFloat[] fb0_3 = ctx.mmaLoadB(bSmem, 16, bBase0 + 768);
            HalfFloat[] fb0_4 = ctx.mmaLoadB(bSmem, 16, bBase0 + 1024);
            HalfFloat[] fb0_5 = ctx.mmaLoadB(bSmem, 16, bBase0 + 1280);
            HalfFloat[] fb0_6 = ctx.mmaLoadB(bSmem, 16, bBase0 + 1536);
            HalfFloat[] fb0_7 = ctx.mmaLoadB(bSmem, 16, bBase0 + 1792);
            acc0_0 = ctx.mma(fa0_0, fb0_0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa0_0, fb0_1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa0_0, fb0_2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa0_0, fb0_3, acc0_3, MMAShape.M16N8K16);
            acc0_4 = ctx.mma(fa0_0, fb0_4, acc0_4, MMAShape.M16N8K16);
            acc0_5 = ctx.mma(fa0_0, fb0_5, acc0_5, MMAShape.M16N8K16);
            acc0_6 = ctx.mma(fa0_0, fb0_6, acc0_6, MMAShape.M16N8K16);
            acc0_7 = ctx.mma(fa0_0, fb0_7, acc0_7, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa0_1, fb0_0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa0_1, fb0_1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa0_1, fb0_2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa0_1, fb0_3, acc1_3, MMAShape.M16N8K16);
            acc1_4 = ctx.mma(fa0_1, fb0_4, acc1_4, MMAShape.M16N8K16);
            acc1_5 = ctx.mma(fa0_1, fb0_5, acc1_5, MMAShape.M16N8K16);
            acc1_6 = ctx.mma(fa0_1, fb0_6, acc1_6, MMAShape.M16N8K16);
            acc1_7 = ctx.mma(fa0_1, fb0_7, acc1_7, MMAShape.M16N8K16);
            acc2_0 = ctx.mma(fa0_2, fb0_0, acc2_0, MMAShape.M16N8K16);
            acc2_1 = ctx.mma(fa0_2, fb0_1, acc2_1, MMAShape.M16N8K16);
            acc2_2 = ctx.mma(fa0_2, fb0_2, acc2_2, MMAShape.M16N8K16);
            acc2_3 = ctx.mma(fa0_2, fb0_3, acc2_3, MMAShape.M16N8K16);
            acc2_4 = ctx.mma(fa0_2, fb0_4, acc2_4, MMAShape.M16N8K16);
            acc2_5 = ctx.mma(fa0_2, fb0_5, acc2_5, MMAShape.M16N8K16);
            acc2_6 = ctx.mma(fa0_2, fb0_6, acc2_6, MMAShape.M16N8K16);
            acc2_7 = ctx.mma(fa0_2, fb0_7, acc2_7, MMAShape.M16N8K16);
            acc3_0 = ctx.mma(fa0_3, fb0_0, acc3_0, MMAShape.M16N8K16);
            acc3_1 = ctx.mma(fa0_3, fb0_1, acc3_1, MMAShape.M16N8K16);
            acc3_2 = ctx.mma(fa0_3, fb0_2, acc3_2, MMAShape.M16N8K16);
            acc3_3 = ctx.mma(fa0_3, fb0_3, acc3_3, MMAShape.M16N8K16);
            acc3_4 = ctx.mma(fa0_3, fb0_4, acc3_4, MMAShape.M16N8K16);
            acc3_5 = ctx.mma(fa0_3, fb0_5, acc3_5, MMAShape.M16N8K16);
            acc3_6 = ctx.mma(fa0_3, fb0_6, acc3_6, MMAShape.M16N8K16);
            acc3_7 = ctx.mma(fa0_3, fb0_7, acc3_7, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        int row0 = blockRow + warpRow * 64;
        int col0 = blockCol + warpCol * 64;
        ctx.mmaStore(acc0_0, c, row0 + 0, col0 + 0, n);
        ctx.mmaStore(acc0_1, c, row0 + 0, col0 + 8, n);
        ctx.mmaStore(acc0_2, c, row0 + 0, col0 + 16, n);
        ctx.mmaStore(acc0_3, c, row0 + 0, col0 + 24, n);
        ctx.mmaStore(acc0_4, c, row0 + 0, col0 + 32, n);
        ctx.mmaStore(acc0_5, c, row0 + 0, col0 + 40, n);
        ctx.mmaStore(acc0_6, c, row0 + 0, col0 + 48, n);
        ctx.mmaStore(acc0_7, c, row0 + 0, col0 + 56, n);
        ctx.mmaStore(acc1_0, c, row0 + 16, col0 + 0, n);
        ctx.mmaStore(acc1_1, c, row0 + 16, col0 + 8, n);
        ctx.mmaStore(acc1_2, c, row0 + 16, col0 + 16, n);
        ctx.mmaStore(acc1_3, c, row0 + 16, col0 + 24, n);
        ctx.mmaStore(acc1_4, c, row0 + 16, col0 + 32, n);
        ctx.mmaStore(acc1_5, c, row0 + 16, col0 + 40, n);
        ctx.mmaStore(acc1_6, c, row0 + 16, col0 + 48, n);
        ctx.mmaStore(acc1_7, c, row0 + 16, col0 + 56, n);
        ctx.mmaStore(acc2_0, c, row0 + 32, col0 + 0, n);
        ctx.mmaStore(acc2_1, c, row0 + 32, col0 + 8, n);
        ctx.mmaStore(acc2_2, c, row0 + 32, col0 + 16, n);
        ctx.mmaStore(acc2_3, c, row0 + 32, col0 + 24, n);
        ctx.mmaStore(acc2_4, c, row0 + 32, col0 + 32, n);
        ctx.mmaStore(acc2_5, c, row0 + 32, col0 + 40, n);
        ctx.mmaStore(acc2_6, c, row0 + 32, col0 + 48, n);
        ctx.mmaStore(acc2_7, c, row0 + 32, col0 + 56, n);
        ctx.mmaStore(acc3_0, c, row0 + 48, col0 + 0, n);
        ctx.mmaStore(acc3_1, c, row0 + 48, col0 + 8, n);
        ctx.mmaStore(acc3_2, c, row0 + 48, col0 + 16, n);
        ctx.mmaStore(acc3_3, c, row0 + 48, col0 + 24, n);
        ctx.mmaStore(acc3_4, c, row0 + 48, col0 + 32, n);
        ctx.mmaStore(acc3_5, c, row0 + 48, col0 + 40, n);
        ctx.mmaStore(acc3_6, c, row0 + 48, col0 + 48, n);
        ctx.mmaStore(acc3_7, c, row0 + 48, col0 + 56, n);
    }

    public static void p128x128_2x4_k64(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {
        int tid = ctx.localIdx;
        int blocksPerRow = n / 128;
        int blockRow = (ctx.groupIdx / blocksPerRow) * 128;
        int blockCol = (ctx.groupIdx % blocksPerRow) * 128;
        int warp = tid / 32;
        int warpRow = warp / 4;
        int warpCol = warp % 4;
        int[] aSmem = ctx.allocateIntLocalArray(8192);
        int[] bSmem = ctx.allocateIntLocalArray(8192);
        float[] acc0_0 = ctx.mmaFragment(0.0f);
        float[] acc0_1 = ctx.mmaFragment(0.0f);
        float[] acc0_2 = ctx.mmaFragment(0.0f);
        float[] acc0_3 = ctx.mmaFragment(0.0f);
        float[] acc1_0 = ctx.mmaFragment(0.0f);
        float[] acc1_1 = ctx.mmaFragment(0.0f);
        float[] acc1_2 = ctx.mmaFragment(0.0f);
        float[] acc1_3 = ctx.mmaFragment(0.0f);
        float[] acc2_0 = ctx.mmaFragment(0.0f);
        float[] acc2_1 = ctx.mmaFragment(0.0f);
        float[] acc2_2 = ctx.mmaFragment(0.0f);
        float[] acc2_3 = ctx.mmaFragment(0.0f);
        float[] acc3_0 = ctx.mmaFragment(0.0f);
        float[] acc3_1 = ctx.mmaFragment(0.0f);
        float[] acc3_2 = ctx.mmaFragment(0.0f);
        float[] acc3_3 = ctx.mmaFragment(0.0f);
        int numK = n / 64;
        { int idx = tid + 0;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 256;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 512;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 768;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 0 + idx, a, (blockRow + idx / 8) * n + 0 + 0 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 0 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 0;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 256;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 512;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 768;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 1024 + idx, a, (blockRow + idx / 8) * n + 0 + 16 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 16 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 0;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 2048 + idx, a, (blockRow + idx / 8) * n + 0 + 32 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 2048 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 32 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 256;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 2048 + idx, a, (blockRow + idx / 8) * n + 0 + 32 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 2048 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 32 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 512;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 2048 + idx, a, (blockRow + idx / 8) * n + 0 + 32 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 2048 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 32 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 768;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 2048 + idx, a, (blockRow + idx / 8) * n + 0 + 32 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 2048 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 32 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 0;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 3072 + idx, a, (blockRow + idx / 8) * n + 0 + 48 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 3072 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 48 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 256;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 3072 + idx, a, (blockRow + idx / 8) * n + 0 + 48 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 3072 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 48 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 512;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 3072 + idx, a, (blockRow + idx / 8) * n + 0 + 48 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 3072 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 48 + kRow) * n + blockCol + rem * 2);
        }
        { int idx = tid + 768;
          ctx.asyncCopyToLocal(aSmem, 0 * 4096 + 3072 + idx, a, (blockRow + idx / 8) * n + 0 + 48 + (idx % 8) * 2);
          int kRow = idx / 64; int rem = idx % 64;
          ctx.asyncCopyToLocal(bSmem, 0 * 4096 + 3072 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (0 + 48 + kRow) * n + blockCol + rem * 2);
        }
        ctx.asyncCopyCommit();
        for (int kt = 0; kt < numK; kt++) {
            int cur = kt % 2;
            if (kt + 1 < numK) {
                int nxt = 1 - cur;
                int kNext = (kt + 1) * 64;
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 512;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 768;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 0 + idx, a, (blockRow + idx / 8) * n + kNext + 0 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 0 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 0 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 512;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 768;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 1024 + idx, a, (blockRow + idx / 8) * n + kNext + 16 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 1024 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 16 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 2048 + idx, a, (blockRow + idx / 8) * n + kNext + 32 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 2048 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 32 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 2048 + idx, a, (blockRow + idx / 8) * n + kNext + 32 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 2048 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 32 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 512;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 2048 + idx, a, (blockRow + idx / 8) * n + kNext + 32 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 2048 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 32 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 768;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 2048 + idx, a, (blockRow + idx / 8) * n + kNext + 32 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 2048 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 32 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 0;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 3072 + idx, a, (blockRow + idx / 8) * n + kNext + 48 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 3072 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 48 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 256;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 3072 + idx, a, (blockRow + idx / 8) * n + kNext + 48 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 3072 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 48 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 512;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 3072 + idx, a, (blockRow + idx / 8) * n + kNext + 48 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 3072 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 48 + kRow) * n + blockCol + rem * 2);
                }
                { int idx = tid + 768;
                  ctx.asyncCopyToLocal(aSmem, nxt * 4096 + 3072 + idx, a, (blockRow + idx / 8) * n + kNext + 48 + (idx % 8) * 2);
                  int kRow = idx / 64; int rem = idx % 64;
                  ctx.asyncCopyToLocal(bSmem, nxt * 4096 + 3072 + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, (kNext + 48 + kRow) * n + blockCol + rem * 2);
                }
                ctx.asyncCopyCommit();
                ctx.asyncCopyWaitGroup(1);
            } else {
                ctx.asyncCopyWaitGroup(0);
            }
            ctx.localBarrier();
            int aBase0 = cur * 16384 + 0 + warpRow * 2048;
            int bBase0 = cur * 16384 + 0 + warpCol * 1024;
            HalfFloat[] fa0_0 = ctx.mmaLoadA(aSmem, 16, aBase0 + 0);
            HalfFloat[] fa0_1 = ctx.mmaLoadA(aSmem, 16, aBase0 + 512);
            HalfFloat[] fa0_2 = ctx.mmaLoadA(aSmem, 16, aBase0 + 1024);
            HalfFloat[] fa0_3 = ctx.mmaLoadA(aSmem, 16, aBase0 + 1536);
            HalfFloat[] fb0_0 = ctx.mmaLoadB(bSmem, 16, bBase0 + 0);
            HalfFloat[] fb0_1 = ctx.mmaLoadB(bSmem, 16, bBase0 + 256);
            HalfFloat[] fb0_2 = ctx.mmaLoadB(bSmem, 16, bBase0 + 512);
            HalfFloat[] fb0_3 = ctx.mmaLoadB(bSmem, 16, bBase0 + 768);
            acc0_0 = ctx.mma(fa0_0, fb0_0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa0_0, fb0_1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa0_0, fb0_2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa0_0, fb0_3, acc0_3, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa0_1, fb0_0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa0_1, fb0_1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa0_1, fb0_2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa0_1, fb0_3, acc1_3, MMAShape.M16N8K16);
            acc2_0 = ctx.mma(fa0_2, fb0_0, acc2_0, MMAShape.M16N8K16);
            acc2_1 = ctx.mma(fa0_2, fb0_1, acc2_1, MMAShape.M16N8K16);
            acc2_2 = ctx.mma(fa0_2, fb0_2, acc2_2, MMAShape.M16N8K16);
            acc2_3 = ctx.mma(fa0_2, fb0_3, acc2_3, MMAShape.M16N8K16);
            acc3_0 = ctx.mma(fa0_3, fb0_0, acc3_0, MMAShape.M16N8K16);
            acc3_1 = ctx.mma(fa0_3, fb0_1, acc3_1, MMAShape.M16N8K16);
            acc3_2 = ctx.mma(fa0_3, fb0_2, acc3_2, MMAShape.M16N8K16);
            acc3_3 = ctx.mma(fa0_3, fb0_3, acc3_3, MMAShape.M16N8K16);
            int aBase1 = cur * 16384 + 4096 + warpRow * 2048;
            int bBase1 = cur * 16384 + 4096 + warpCol * 1024;
            HalfFloat[] fa1_0 = ctx.mmaLoadA(aSmem, 16, aBase1 + 0);
            HalfFloat[] fa1_1 = ctx.mmaLoadA(aSmem, 16, aBase1 + 512);
            HalfFloat[] fa1_2 = ctx.mmaLoadA(aSmem, 16, aBase1 + 1024);
            HalfFloat[] fa1_3 = ctx.mmaLoadA(aSmem, 16, aBase1 + 1536);
            HalfFloat[] fb1_0 = ctx.mmaLoadB(bSmem, 16, bBase1 + 0);
            HalfFloat[] fb1_1 = ctx.mmaLoadB(bSmem, 16, bBase1 + 256);
            HalfFloat[] fb1_2 = ctx.mmaLoadB(bSmem, 16, bBase1 + 512);
            HalfFloat[] fb1_3 = ctx.mmaLoadB(bSmem, 16, bBase1 + 768);
            acc0_0 = ctx.mma(fa1_0, fb1_0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa1_0, fb1_1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa1_0, fb1_2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa1_0, fb1_3, acc0_3, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa1_1, fb1_0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa1_1, fb1_1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa1_1, fb1_2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa1_1, fb1_3, acc1_3, MMAShape.M16N8K16);
            acc2_0 = ctx.mma(fa1_2, fb1_0, acc2_0, MMAShape.M16N8K16);
            acc2_1 = ctx.mma(fa1_2, fb1_1, acc2_1, MMAShape.M16N8K16);
            acc2_2 = ctx.mma(fa1_2, fb1_2, acc2_2, MMAShape.M16N8K16);
            acc2_3 = ctx.mma(fa1_2, fb1_3, acc2_3, MMAShape.M16N8K16);
            acc3_0 = ctx.mma(fa1_3, fb1_0, acc3_0, MMAShape.M16N8K16);
            acc3_1 = ctx.mma(fa1_3, fb1_1, acc3_1, MMAShape.M16N8K16);
            acc3_2 = ctx.mma(fa1_3, fb1_2, acc3_2, MMAShape.M16N8K16);
            acc3_3 = ctx.mma(fa1_3, fb1_3, acc3_3, MMAShape.M16N8K16);
            int aBase2 = cur * 16384 + 8192 + warpRow * 2048;
            int bBase2 = cur * 16384 + 8192 + warpCol * 1024;
            HalfFloat[] fa2_0 = ctx.mmaLoadA(aSmem, 16, aBase2 + 0);
            HalfFloat[] fa2_1 = ctx.mmaLoadA(aSmem, 16, aBase2 + 512);
            HalfFloat[] fa2_2 = ctx.mmaLoadA(aSmem, 16, aBase2 + 1024);
            HalfFloat[] fa2_3 = ctx.mmaLoadA(aSmem, 16, aBase2 + 1536);
            HalfFloat[] fb2_0 = ctx.mmaLoadB(bSmem, 16, bBase2 + 0);
            HalfFloat[] fb2_1 = ctx.mmaLoadB(bSmem, 16, bBase2 + 256);
            HalfFloat[] fb2_2 = ctx.mmaLoadB(bSmem, 16, bBase2 + 512);
            HalfFloat[] fb2_3 = ctx.mmaLoadB(bSmem, 16, bBase2 + 768);
            acc0_0 = ctx.mma(fa2_0, fb2_0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa2_0, fb2_1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa2_0, fb2_2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa2_0, fb2_3, acc0_3, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa2_1, fb2_0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa2_1, fb2_1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa2_1, fb2_2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa2_1, fb2_3, acc1_3, MMAShape.M16N8K16);
            acc2_0 = ctx.mma(fa2_2, fb2_0, acc2_0, MMAShape.M16N8K16);
            acc2_1 = ctx.mma(fa2_2, fb2_1, acc2_1, MMAShape.M16N8K16);
            acc2_2 = ctx.mma(fa2_2, fb2_2, acc2_2, MMAShape.M16N8K16);
            acc2_3 = ctx.mma(fa2_2, fb2_3, acc2_3, MMAShape.M16N8K16);
            acc3_0 = ctx.mma(fa2_3, fb2_0, acc3_0, MMAShape.M16N8K16);
            acc3_1 = ctx.mma(fa2_3, fb2_1, acc3_1, MMAShape.M16N8K16);
            acc3_2 = ctx.mma(fa2_3, fb2_2, acc3_2, MMAShape.M16N8K16);
            acc3_3 = ctx.mma(fa2_3, fb2_3, acc3_3, MMAShape.M16N8K16);
            int aBase3 = cur * 16384 + 12288 + warpRow * 2048;
            int bBase3 = cur * 16384 + 12288 + warpCol * 1024;
            HalfFloat[] fa3_0 = ctx.mmaLoadA(aSmem, 16, aBase3 + 0);
            HalfFloat[] fa3_1 = ctx.mmaLoadA(aSmem, 16, aBase3 + 512);
            HalfFloat[] fa3_2 = ctx.mmaLoadA(aSmem, 16, aBase3 + 1024);
            HalfFloat[] fa3_3 = ctx.mmaLoadA(aSmem, 16, aBase3 + 1536);
            HalfFloat[] fb3_0 = ctx.mmaLoadB(bSmem, 16, bBase3 + 0);
            HalfFloat[] fb3_1 = ctx.mmaLoadB(bSmem, 16, bBase3 + 256);
            HalfFloat[] fb3_2 = ctx.mmaLoadB(bSmem, 16, bBase3 + 512);
            HalfFloat[] fb3_3 = ctx.mmaLoadB(bSmem, 16, bBase3 + 768);
            acc0_0 = ctx.mma(fa3_0, fb3_0, acc0_0, MMAShape.M16N8K16);
            acc0_1 = ctx.mma(fa3_0, fb3_1, acc0_1, MMAShape.M16N8K16);
            acc0_2 = ctx.mma(fa3_0, fb3_2, acc0_2, MMAShape.M16N8K16);
            acc0_3 = ctx.mma(fa3_0, fb3_3, acc0_3, MMAShape.M16N8K16);
            acc1_0 = ctx.mma(fa3_1, fb3_0, acc1_0, MMAShape.M16N8K16);
            acc1_1 = ctx.mma(fa3_1, fb3_1, acc1_1, MMAShape.M16N8K16);
            acc1_2 = ctx.mma(fa3_1, fb3_2, acc1_2, MMAShape.M16N8K16);
            acc1_3 = ctx.mma(fa3_1, fb3_3, acc1_3, MMAShape.M16N8K16);
            acc2_0 = ctx.mma(fa3_2, fb3_0, acc2_0, MMAShape.M16N8K16);
            acc2_1 = ctx.mma(fa3_2, fb3_1, acc2_1, MMAShape.M16N8K16);
            acc2_2 = ctx.mma(fa3_2, fb3_2, acc2_2, MMAShape.M16N8K16);
            acc2_3 = ctx.mma(fa3_2, fb3_3, acc2_3, MMAShape.M16N8K16);
            acc3_0 = ctx.mma(fa3_3, fb3_0, acc3_0, MMAShape.M16N8K16);
            acc3_1 = ctx.mma(fa3_3, fb3_1, acc3_1, MMAShape.M16N8K16);
            acc3_2 = ctx.mma(fa3_3, fb3_2, acc3_2, MMAShape.M16N8K16);
            acc3_3 = ctx.mma(fa3_3, fb3_3, acc3_3, MMAShape.M16N8K16);
            ctx.localBarrier();
        }
        int row0 = blockRow + warpRow * 64;
        int col0 = blockCol + warpCol * 32;
        ctx.mmaStore(acc0_0, c, row0 + 0, col0 + 0, n);
        ctx.mmaStore(acc0_1, c, row0 + 0, col0 + 8, n);
        ctx.mmaStore(acc0_2, c, row0 + 0, col0 + 16, n);
        ctx.mmaStore(acc0_3, c, row0 + 0, col0 + 24, n);
        ctx.mmaStore(acc1_0, c, row0 + 16, col0 + 0, n);
        ctx.mmaStore(acc1_1, c, row0 + 16, col0 + 8, n);
        ctx.mmaStore(acc1_2, c, row0 + 16, col0 + 16, n);
        ctx.mmaStore(acc1_3, c, row0 + 16, col0 + 24, n);
        ctx.mmaStore(acc2_0, c, row0 + 32, col0 + 0, n);
        ctx.mmaStore(acc2_1, c, row0 + 32, col0 + 8, n);
        ctx.mmaStore(acc2_2, c, row0 + 32, col0 + 16, n);
        ctx.mmaStore(acc2_3, c, row0 + 32, col0 + 24, n);
        ctx.mmaStore(acc3_0, c, row0 + 48, col0 + 0, n);
        ctx.mmaStore(acc3_1, c, row0 + 48, col0 + 8, n);
        ctx.mmaStore(acc3_2, c, row0 + 48, col0 + 16, n);
        ctx.mmaStore(acc3_3, c, row0 + 48, col0 + 24, n);
    }

    public static void main(String[] args) throws Exception {
        String which = args[0];
        int n = Integer.parseInt(args[1]);
        int reps = Integer.parseInt(args[2]);
        HalfFloatArray a = new HalfFloatArray(n * n), b = new HalfFloatArray(n * n);
        for (int i = 0; i < n * n; i++) {
            a.set(i, new HalfFloat(((i * 7 + 3) % 17) * 0.0625f - 0.5f));
            b.set(i, new HalfFloat(((i * 5 + 11) % 13) * 0.0769f - 0.5f));
        }
        FloatArray c = new FloatArray(n * n);
        TaskGraph g = new TaskGraph("g").transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b);
        GridScheduler s = null;
        switch (which) {
            case "simple": {
                g.task("t", KcProbe::kcSimple, new KernelContext(), a, b, c, n);
                WorkerGrid1D w = new WorkerGrid1D((n / 16) * (n / 16) * WARP); w.setLocalWork(WARP, 1, 1);
                s = new GridScheduler("g.t", w); break;
            }
            case "opt": {
                g.task("t", KcProbe::kcOpt, new KernelContext(), a, b, c, n);
                WorkerGrid1D w = new WorkerGrid1D((n / BM) * (n / BN) * THREADS); w.setLocalWork(THREADS, 1, 1);
                s = new GridScheduler("g.t", w); break;
            }
            case "o64x64_2x2": {
                g.task("t", KcProbe::o64x64_2x2, new KernelContext(), a, b, c, n);
                WorkerGrid1D w = new WorkerGrid1D((n / 64) * (n / 64) * 128); w.setLocalWork(128, 1, 1);
                s = new GridScheduler("g.t", w); break;
            }
            case "o128x128_2x4": {
                g.task("t", KcProbe::o128x128_2x4, new KernelContext(), a, b, c, n);
                WorkerGrid1D w = new WorkerGrid1D((n / 128) * (n / 128) * 256); w.setLocalWork(256, 1, 1);
                s = new GridScheduler("g.t", w); break;
            }
            case "o128x64_2x2": {
                g.task("t", KcProbe::o128x64_2x2, new KernelContext(), a, b, c, n);
                WorkerGrid1D w = new WorkerGrid1D((n / 128) * (n / 64) * 128); w.setLocalWork(128, 1, 1);
                s = new GridScheduler("g.t", w); break;
            }
            case "o128x128_2x2": {
                g.task("t", KcProbe::o128x128_2x2, new KernelContext(), a, b, c, n);
                WorkerGrid1D w = new WorkerGrid1D((n / 128) * (n / 128) * 128); w.setLocalWork(128, 1, 1);
                s = new GridScheduler("g.t", w); break;
            }
            case "o64x128_2x2": {
                g.task("t", KcProbe::o64x128_2x2, new KernelContext(), a, b, c, n);
                WorkerGrid1D w = new WorkerGrid1D((n / 64) * (n / 128) * 128); w.setLocalWork(128, 1, 1);
                s = new GridScheduler("g.t", w); break;
            }
            case "o128x128_4x2": {
                g.task("t", KcProbe::o128x128_4x2, new KernelContext(), a, b, c, n);
                WorkerGrid1D w = new WorkerGrid1D((n / 128) * (n / 128) * 256); w.setLocalWork(256, 1, 1);
                s = new GridScheduler("g.t", w); break;
            }
            case "p128x128_2x4_k32": {
                g.task("t", KcProbe::p128x128_2x4_k32, new KernelContext(), a, b, c, n);
                WorkerGrid1D w = new WorkerGrid1D((n / 128) * (n / 128) * 256); w.setLocalWork(256, 1, 1);
                s = new GridScheduler("g.t", w); break;
            }
            case "p128x128_2x2_k32": {
                g.task("t", KcProbe::p128x128_2x2_k32, new KernelContext(), a, b, c, n);
                WorkerGrid1D w = new WorkerGrid1D((n / 128) * (n / 128) * 128); w.setLocalWork(128, 1, 1);
                s = new GridScheduler("g.t", w); break;
            }
            case "p128x256_2x4_k16": {
                g.task("t", KcProbe::p128x256_2x4_k16, new KernelContext(), a, b, c, n);
                WorkerGrid1D w = new WorkerGrid1D((n / 128) * (n / 256) * 256); w.setLocalWork(256, 1, 1);
                s = new GridScheduler("g.t", w); break;
            }
            case "p256x128_4x2_k16": {
                g.task("t", KcProbe::p256x128_4x2_k16, new KernelContext(), a, b, c, n);
                WorkerGrid1D w = new WorkerGrid1D((n / 256) * (n / 128) * 256); w.setLocalWork(256, 1, 1);
                s = new GridScheduler("g.t", w); break;
            }
            case "p128x128_2x4_k64": {
                g.task("t", KcProbe::p128x128_2x4_k64, new KernelContext(), a, b, c, n);
                WorkerGrid1D w = new WorkerGrid1D((n / 128) * (n / 128) * 256); w.setLocalWork(256, 1, 1);
                s = new GridScheduler("g.t", w); break;
            }
            case "cublas":
                g.libraryTask("t", CuBlas::cublasGemmExFP16FP32, CuBlasOperation.CUBLAS_OP_N.operation(), CuBlasOperation.CUBLAS_OP_N.operation(),
                        n, n, n, 1.0f, b, n, a, n, 0.0f, c, n);
                break;
            default: throw new IllegalArgumentException(which);
        }
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, c);
        try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
            if (s != null) p.withGridScheduler(s);
            for (int r = 0; r < reps; r++) p.execute();
        }
        int bad = 0; float maxErr = 0f; java.util.Random rnd = new java.util.Random(1);
        for (int t = 0; t < 256; t++) {
            int i = rnd.nextInt(n), j = rnd.nextInt(n); float sum = 0f;
            for (int k = 0; k < n; k++) sum += a.get(i * n + k).getFloat32() * b.get(k * n + j).getFloat32();
            float e = Math.abs(sum - c.get(i * n + j)); maxErr = Math.max(maxErr, e);
            if (e > 0.05f * n / 256f * Math.max(1f, Math.abs(sum))) bad++;
        }
        System.out.printf("%s n=%d %s maxErr=%.4f bad=%d/256%n", which, n, bad == 0 ? "PASSED" : "FAILED", maxErr, bad);
    }
}
