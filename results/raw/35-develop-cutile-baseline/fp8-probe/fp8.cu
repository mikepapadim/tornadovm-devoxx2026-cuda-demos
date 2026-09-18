#include <cuda_fp8.h>
#include "cuda_tile.h"
namespace ct = cuda::tiles;
using namespace ct::literals;

extern "C" __tile_global__ void fp8Add(__nv_fp8_e4m3 *a, __nv_fp8_e4m3 *b, __nv_fp8_e4m3 *c, int n) {
    auto av = ct::partition_view{ct::tensor_span{a, ct::extents{n}}, ct::shape<128>{}};
    auto bv = ct::partition_view{ct::tensor_span{b, ct::extents{n}}, ct::shape<128>{}};
    auto cv = ct::partition_view{ct::tensor_span{c, ct::extents{n}}, ct::shape<128>{}};
    cv.store(av.load(ct::bid().x) + bv.load(ct::bid().x), ct::bid().x);
}
