# BK = 16*S per cp.async stage: S sub-tiles of k16, stored back to back in each buffer.
def kernel(name, BM, BN, WGM, WGN, S):
    WTM, WTN = BM // WGM, BN // WGN
    FM, FN = WTM // 16, WTN // 8
    T = WGM * WGN * 32
    A1, B1 = BM * 8, BN * 8                 # ints per k16 sub-tile
    A_INTS, B_INTS = A1 * S, B1 * S         # ints per stage
    assert A1 % T == 0 and B1 % T == 0
    la, lb = A1 // T, B1 // T
    P = BN // 2
    L = [f"    public static void {name}(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {{",
         "        int tid = ctx.localIdx;",
         f"        int blocksPerRow = n / {BN};",
         f"        int blockRow = (ctx.groupIdx / blocksPerRow) * {BM};",
         f"        int blockCol = (ctx.groupIdx % blocksPerRow) * {BN};",
         "        int warp = tid / 32;",
         f"        int warpRow = warp / {WGN};",
         f"        int warpCol = warp % {WGN};",
         f"        int[] aSmem = ctx.allocateIntLocalArray({2*A_INTS});",
         f"        int[] bSmem = ctx.allocateIntLocalArray({2*B_INTS});"]
    L += [f"        float[] acc{i}_{j} = ctx.mmaFragment(0.0f);" for i in range(FM) for j in range(FN)]
    L.append(f"        int numK = n / {16*S};")
    def loads(bufexpr, kexpr, ind):
        out=[]
        for s in range(S):
            for q in range(max(la,lb)):
                out.append(f"{ind}{{ int idx = tid + {q*T};")
                if q < la:
                    out.append(f"{ind}  ctx.asyncCopyToLocal(aSmem, {bufexpr}{A_INTS} + {s*A1} + idx, a, (blockRow + idx / 8) * n + {kexpr} + {s*16} + (idx % 8) * 2);")
                if q < lb:
                    out.append(f"{ind}  int kRow = idx / {P}; int rem = idx % {P};")
                    out.append(f"{ind}  ctx.asyncCopyToLocal(bSmem, {bufexpr}{B_INTS} + {s*B1} + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, ({kexpr} + {s*16} + kRow) * n + blockCol + rem * 2);")
                out.append(f"{ind}}}")
        return out
    L += loads("0 * ", "0", "        ")
    L += ["        ctx.asyncCopyCommit();",
          "        for (int kt = 0; kt < numK; kt++) {",
          "            int cur = kt % 2;",
          "            if (kt + 1 < numK) {",
          "                int nxt = 1 - cur;",
          f"                int kNext = (kt + 1) * {16*S};"]
    L += loads("nxt * ", "kNext", "                ")
    L += ["                ctx.asyncCopyCommit();", "                ctx.asyncCopyWaitGroup(1);",
          "            } else {", "                ctx.asyncCopyWaitGroup(0);", "            }",
          "            ctx.localBarrier();"]
    for s in range(S):
        L.append(f"            int aBase{s} = cur * {A_INTS*4} + {s*A1*4} + warpRow * {WTM*32};")
        L.append(f"            int bBase{s} = cur * {B_INTS*4} + {s*B1*4} + warpCol * {FN*256};")
        L += [f"            HalfFloat[] fa{s}_{i} = ctx.mmaLoadA(aSmem, 16, aBase{s} + {i*512});" for i in range(FM)]
        L += [f"            HalfFloat[] fb{s}_{j} = ctx.mmaLoadB(bSmem, 16, bBase{s} + {j*256});" for j in range(FN)]
        L += [f"            acc{i}_{j} = ctx.mma(fa{s}_{i}, fb{s}_{j}, acc{i}_{j}, MMAShape.M16N8K16);" for i in range(FM) for j in range(FN)]
    L += ["            ctx.localBarrier();", "        }",
          f"        int row0 = blockRow + warpRow * {WTM};", f"        int col0 = blockCol + warpCol * {WTN};"]
    L += [f"        ctx.mmaStore(acc{i}_{j}, c, row0 + {i*16}, col0 + {j*8}, n);" for i in range(FM) for j in range(FN)]
    L.append("    }")
    return "\n".join(L), T
cfgs = {"p128x128_2x4_k32":(128,128,2,4,2), "p128x128_2x2_k32":(128,128,2,2,2), "p128x256_2x4_k16":(128,256,2,4,1),
        "p256x128_4x2_k16":(256,128,4,2,1), "p128x128_2x4_k64":(128,128,2,4,4)}
s=open('KcProbe.java').read(); meth=[]; cases=[]
for nm,(bm,bn,wm,wn,S) in cfgs.items():
    k,T=kernel(nm,bm,bn,wm,wn,S); meth.append(k)
    cases.append(f'''            case "{nm}": {{
                g.task("t", KcProbe::{nm}, new KernelContext(), a, b, c, n);
                WorkerGrid1D w = new WorkerGrid1D((n / {bm}) * (n / {bn}) * {T}); w.setLocalWork({T}, 1, 1);
                s = new GridScheduler("g.t", w); break;
            }}''')
s=s.replace("\n    public static void main", "\n"+"\n\n".join(meth)+"\n\n    public static void main",1)
s=s.replace('            case "cublas":', "\n".join(cases)+'\n            case "cublas":',1)
open('KcProbe.java','w').write(s); print("ok")
