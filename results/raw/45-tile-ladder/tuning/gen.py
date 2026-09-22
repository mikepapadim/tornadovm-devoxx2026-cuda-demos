import sys
def kernel(name, BM, BN, WGM, WGN):
    # WGM x WGN warps; each warp owns (BM/WGM) x (BN/WGN); frags FM x FN
    WTM, WTN = BM // WGM, BN // WGN
    FM, FN = WTM // 16, WTN // 8
    T = WGM * WGN * 32
    A_INTS, B_INTS = BM * 8, BN * 8          # BK=16: BM rows x 8 ints ; 16 rows x BN/2 ints
    assert (A_INTS % T == 0) and (B_INTS % T == 0), (name, A_INTS, B_INTS, T)
    la, lb = A_INTS // T, B_INTS // T
    P = BN // 2                               # ints per B k-row
    L = []
    L.append(f"    private static final int {name.upper()}_BM = {BM}, {name.upper()}_BN = {BN}, {name.upper()}_T = {T};")
    L.append(f"    public static void {name}(KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray c, int n) {{")
    L.append(f"        int tid = ctx.localIdx;")
    L.append(f"        int blocksPerRow = n / {BN};")
    L.append(f"        int blockRow = (ctx.groupIdx / blocksPerRow) * {BM};")
    L.append(f"        int blockCol = (ctx.groupIdx % blocksPerRow) * {BN};")
    L.append(f"        int warp = tid / 32;")
    L.append(f"        int warpRow = warp / {WGN};")
    L.append(f"        int warpCol = warp % {WGN};")
    L.append(f"        int[] aSmem = ctx.allocateIntLocalArray({2*A_INTS});")
    L.append(f"        int[] bSmem = ctx.allocateIntLocalArray({2*B_INTS});")
    for i in range(FM):
        for j in range(FN):
            L.append(f"        float[] acc{i}_{j} = ctx.mmaFragment(0.0f);")
    L.append(f"        int numK = n / 16;")
    def loads(buf, kexpr):
        out=[]
        for q in range(max(la,lb)):
            out.append(f"            {{ int idx = tid + {q*T};")
            if q < la:
                out.append(f"              ctx.asyncCopyToLocal(aSmem, {buf}{A_INTS} + idx, a, (blockRow + idx / 8) * n + {kexpr} + (idx % 8) * 2);")
            if q < lb:
                out.append(f"              int kRow = idx / {P}; int rem = idx % {P};")
                out.append(f"              ctx.asyncCopyToLocal(bSmem, {buf}{B_INTS} + (rem / 4) * 64 + kRow * 4 + (rem % 4), b, ({kexpr} + kRow) * n + blockCol + rem * 2);")
            out.append("            }")
        return out
    # stage 0
    L += [l.replace("{buf}","0 * ") for l in loads("{buf}", "0")]
    L.append(f"        ctx.asyncCopyCommit();")
    L.append(f"        for (int kt = 0; kt < numK; kt++) {{")
    L.append(f"            int cur = kt % 2;")
    L.append(f"            if (kt + 1 < numK) {{")
    L.append(f"                int nxt = 1 - cur;")
    L.append(f"                int kNext = (kt + 1) * 16;")
    L += ["    "+l.replace("{buf}","nxt * ") for l in loads("{buf}", "kNext")]
    L.append(f"                ctx.asyncCopyCommit();")
    L.append(f"                ctx.asyncCopyWaitGroup(1);")
    L.append(f"            }} else {{")
    L.append(f"                ctx.asyncCopyWaitGroup(0);")
    L.append(f"            }}")
    L.append(f"            ctx.localBarrier();")
    L.append(f"            int aBase = cur * {A_INTS*4} + warpRow * {WTM*32};")
    L.append(f"            int bBase = cur * {B_INTS*4} + warpCol * {FN*256};")
    for i in range(FM): L.append(f"            HalfFloat[] fa{i} = ctx.mmaLoadA(aSmem, 16, aBase + {i*16*32});")
    for j in range(FN): L.append(f"            HalfFloat[] fb{j} = ctx.mmaLoadB(bSmem, 16, bBase + {j*256});")
    for i in range(FM):
        for j in range(FN):
            L.append(f"            acc{i}_{j} = ctx.mma(fa{i}, fb{j}, acc{i}_{j}, MMAShape.M16N8K16);")
    L.append(f"            ctx.localBarrier();")
    L.append(f"        }}")
    L.append(f"        int row0 = blockRow + warpRow * {WTM};")
    L.append(f"        int col0 = blockCol + warpCol * {WTN};")
    for i in range(FM):
        for j in range(FN):
            L.append(f"        ctx.mmaStore(acc{i}_{j}, c, row0 + {i*16}, col0 + {j*8}, n);")
    L.append("    }")
    return "\n".join(L), T, BM, BN
cfgs = {"o64x64_2x2":(64,64,2,2), "o128x128_2x4":(128,128,2,4), "o128x64_2x2":(128,64,2,2),
        "o128x128_2x2":(128,128,2,2), "o64x128_2x2":(64,128,2,2), "o128x128_4x2":(128,128,4,2)}
src=open('KcProbe.java').read()
meth=[];cases=[]
for nm,(bm,bn,wm,wn) in cfgs.items():
    k,T,BM,BN=kernel(nm,bm,bn,wm,wn); meth.append(k)
    cases.append(f'''            case "{nm}": {{
                g.task("t", KcProbe::{nm}, new KernelContext(), a, b, c, n);
                WorkerGrid1D w = new WorkerGrid1D((n / {BM}) * (n / {BN}) * {T}); w.setLocalWork({T}, 1, 1);
                s = new GridScheduler("g.t", w); break;
            }}''')
src=src.replace("\n    public static void main", "\n"+"\n\n".join(meth)+"\n\n    public static void main",1)
src=src.replace('            case "cublas":', "\n".join(cases)+'\n            case "cublas":',1)
open('KcProbe.java','w').write(src); print("ok", list(cfgs))
