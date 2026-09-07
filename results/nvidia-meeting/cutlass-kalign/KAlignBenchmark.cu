// kAlign 4 vs 8 for TornadoVM's exact CUTLASS FP16 GEMM configuration.
// Everything else is held identical to tornado-cutlass.cu: Sm80, 128x128x32
// threadblock, 64x64x32 warp, 16x8x16 instruction, 3 stages, RowMajor x3,
// identity swizzle, FP32 accumulate. Only kAlign varies.
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#define CK(c) do{cudaError_t e=(c); if(e){printf("err %s\n",cudaGetErrorString(e));return 1;}}while(0)

using ElementHalf = cutlass::half_t;
using ElementAccum = float;
using ElementCompute = float;
using Row = cutlass::layout::RowMajor;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
static constexpr int kStages = 3;

template <int kAlign>
using Gemm = cutlass::gemm::device::GemmUniversal<
    ElementHalf, Row, ElementHalf, Row, ElementHalf, Row,
    ElementAccum, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128,128,32>, cutlass::gemm::GemmShape<64,64,32>,
    cutlass::gemm::GemmShape<16,8,16>,
    cutlass::epilogue::thread::LinearCombination<ElementHalf, kAlign, ElementAccum, ElementCompute>,
    Swizzle, kStages, kAlign, kAlign>;

template <typename G>
static double run(int m,int n,int k, ElementHalf*A, ElementHalf*B, ElementHalf*C, int reps){
    typename G::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k},1,{1.0f,0.0f},
        A,B,C,C, (int64_t)m*k,(int64_t)k*n,(int64_t)m*n,(int64_t)m*n, k,n,n,n);
    G op; size_t ws = G::get_workspace_size(args);
    void* wsp=nullptr; if(ws) cudaMalloc(&wsp, ws);
    if (op.can_implement(args) != cutlass::Status::kSuccess){ if(wsp)cudaFree(wsp); return -1.0; }
    op.initialize(args, wsp);
    for(int i=0;i<5;i++) op();                       // warm-up
    cudaDeviceSynchronize();
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<double> t;
    for(int r=0;r<reps;r++){
        cudaEventRecord(a); op(); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms=0; cudaEventElapsedTime(&ms,a,b); t.push_back(ms);
    }
    std::sort(t.begin(),t.end());
    if(wsp)cudaFree(wsp);
    return t[t.size()/2];
}

int main(){
    const int reps=50;
    int shapes[][3] = {{1024,1024,1024},{2048,2048,2048},{4096,4096,4096},{256,256,256},{4096,4096,512},{1024,1020,1020}};
    printf("%-22s %12s %12s %10s\n","shape (m,n,k)","kAlign=4 ms","kAlign=8 ms","speedup");
    for(auto& s: shapes){
        int m=s[0],n=s[1],k=s[2];
        ElementHalf *A,*B,*C;
        CK(cudaMalloc(&A,(size_t)m*k*sizeof(ElementHalf)));
        CK(cudaMalloc(&B,(size_t)k*n*sizeof(ElementHalf)));
        CK(cudaMalloc(&C,(size_t)m*n*sizeof(ElementHalf)));
        CK(cudaMemset(A,0,(size_t)m*k*sizeof(ElementHalf)));
        CK(cudaMemset(B,0,(size_t)k*n*sizeof(ElementHalf)));
        double t4 = run<Gemm<4>>(m,n,k,A,B,C,reps);
        double t8 = run<Gemm<8>>(m,n,k,A,B,C,reps);
        char buf[32]; snprintf(buf,sizeof buf,"%d,%d,%d",m,n,k);
        if(t8<0) printf("%-22s %12.4f %12s %10s\n",buf,t4,"rejected","n/a");
        else     printf("%-22s %12.4f %12.4f %9.3fx\n",buf,t4,t8,t4/t8);
        cudaFree(A);cudaFree(B);cudaFree(C);
    }
    return 0;
}
