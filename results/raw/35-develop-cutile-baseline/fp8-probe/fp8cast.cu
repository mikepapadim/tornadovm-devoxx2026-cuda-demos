#include <cuda_fp8.h>
#include "cuda_tile.h"
namespace ct = cuda::tiles;
using namespace ct::literals;

// Same op, but converting to fp32 for the arithmetic and back to fp8 to store.
extern "C" __tile_global__ void fp8AddViaF32(__nv_fp8_e4m3 *a, __nv_fp8_e4m3 *b, __nv_fp8_e4m3 *c, int n) {
    auto av = ct::partition_view{ct::tensor_span{a, ct::extents{n}}, ct::shape<128>{}};
    auto bv = ct::partition_view{ct::tensor_span{b, ct::extents{n}}, ct::shape<128>{}};
    auto cv = ct::partition_view{ct::tensor_span{c, ct::extents{n}}, ct::shape<128>{}};
    auto x = ct::element_cast<float>(av.load(ct::bid().x));
    auto y = ct::element_cast<float>(bv.load(ct::bid().x));
    cv.store(ct::element_cast<__nv_fp8_e4m3>(x + y), ct::bid().x);
}
