
from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torchinductor.codecache import CppCodeCache, TritonCodeCache

aten = torch.ops.aten


kernel0 = CppCodeCache.load('''
#include "/tmp/torchinductor_jgong5/qr/cqrr7t6pdy7hpf76525uh7ddg65eljkjlgxa5dhk7amq6xb6a3ia.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       const long ks0,
                       const long ks1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for
        for(long i0=0; i0<ks0; ++i0)
        {
            {
                {
                    float tmp1 = -std::numeric_limits<float>::infinity();
                    #pragma omp simd simdlen(8) reduction(max:tmp1)
                    for(long i1=0; i1<ks1; ++i1)
                    {
                        {
                            auto tmp0 = in_ptr0[i1 + (i0*ks1)];
                            tmp1 = std::max(tmp1, tmp0);
                        }
                    }
                    out_ptr0[i0] = tmp1;
                }
            }
        }
        #pragma omp for
        for(long i0=0; i0<ks0; ++i0)
        {
            {
                {
                    float tmp4 = 0;
                    #pragma omp simd simdlen(8) reduction(+:tmp4)
                    for(long i1=0; i1<ks1; ++i1)
                    {
                        {
                            auto tmp0 = in_ptr0[i1 + (i0*ks1)];
                            auto tmp1 = out_ptr0[i0];
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = std::exp(tmp2);
                            out_ptr1[i1 + (i0*ks1)] = tmp3;
                            tmp4 += tmp3;
                        }
                    }
                    out_ptr2[i0] = tmp4;
                }
            }
        }
        #pragma omp for
        for(long i0=0; i0<ks0; ++i0)
        {
            #pragma omp simd simdlen(8)
            for(long i1=0; i1<ks1; ++i1)
            {
                {
                    {
                        auto tmp0 = out_ptr1[i1 + (i0*ks1)];
                        auto tmp1 = out_ptr2[i0];
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr3[i1 + (i0*ks1)] = tmp2;
                    }
                }
            }
        }
    }
}
''').kernel


def call(arg0_1):
    arg0_1_size = arg0_1.size()
    s0 = arg0_1_size[0]
    s1 = arg0_1_size[1]
    buf0 = empty_strided((s0, 1), (1, s0), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((s0, s1), (s1, 1), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((s0, 1), (1, s0), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((s0, s1), (s1, 1), device='cpu', dtype=torch.float32)
    kernel0(c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_long(s0), c_long(s1))
    return (buf3, )


def bench_run():
    from torchdynamo.testing import rand_strided
    from torchinductor.utils import bench
    arg0_1 = rand_strided((700, 23258), (23258, 1), device='cpu', dtype=torch.float32)
    return bench(lambda: call(arg0_1))


if __name__ == "__main__":
    from torchdynamo.testing import rand_strided
    from torchinductor.utils import print_performance
    arg0_1 = rand_strided((700, 23258), (23258, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call(arg0_1))
