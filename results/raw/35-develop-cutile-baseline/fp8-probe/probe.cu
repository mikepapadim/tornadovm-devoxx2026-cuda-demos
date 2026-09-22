#include <cuda_fp16.h>
#include "cuda_tile.h"
namespace ct = cuda::tiles;
using namespace ct::literals;

extern "C" __tile_global__ void probeGemm(__half *a, __half *b, float *c, int m, int n, int k) {
    auto av = ct::partition_view{ct::tensor_span{a, ct::extents{m, k}}, ct::shape<64, 32>{}};
    auto bv = ct::partition_view{ct::tensor_span{b, ct::extents{k, n}}, ct::shape<32, 64>{}};
    auto cv = ct::partition_view{ct::tensor_span{c, ct::extents{m, n}}, ct::shape<64, 64>{}};

    auto acc = ct::zeros<ct::tile<float, ct::shape<64, 64>>>();
    for (int step = 0; step < k / 32; step++) {
        acc = ct::mma(av.load(ct::bid().x, step), bv.load(step, ct::bid().y), acc);
    }
    cv.store(acc, ct::bid().x, ct::bid().y);
}
