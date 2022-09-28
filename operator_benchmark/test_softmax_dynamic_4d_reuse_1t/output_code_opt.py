
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
                       const long ks0,
                       const long ks1)
{
    float* out_ptr2 = out_ptr0; float* out_ptr3 = out_ptr1;
    #pragma GCC ivdep
    for(long i0=0; i0<ks0*ks1; ++i0)
    {
        {
            {
                    using Vec = at::vec::Vectorized<float>;
                    Vec tmp1(-std::numeric_limits<float>::infinity());
                    for(long i1=0; i1<ks1/8; ++i1)
                    {
                        {
                            Vec tmp0 = Vec::loadu(in_ptr0 + i1*8 + (i0*ks1));
                            Vec tmp1 = at::vec::maximum(tmp1, tmp0);
                        }
                    }
                    out_ptr0[i0] = at::vec::vec_reduce_all([](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, tmp1);
            }
        }
    //}
    //#pragma GCC ivdep
    //for(long i0=0; i0<ks0*ks1; ++i0)
    //{
        {
            {
                    using Vec = at::vec::Vectorized<float>;
                    Vec tmp4(0);
                    assert(ks1 % 8 == 0);
                    for(long i1=0; i1<ks1 / 8; ++i1)
                    {
                        {
                            Vec tmp0 = Vec::loadu(in_ptr0 + i1 * 8 + (i0*ks1));
                            Vec tmp1 = Vec::loadu(out_ptr0 + i0);
                            Vec tmp2 = tmp0 - tmp1;
                            Vec tmp3 = tmp2.exp();
                            tmp3.store(out_ptr1 + i1 * 8 + (i0*ks1));
                            tmp4 += tmp3;
                        }
                    }
                    out_ptr2[i0] = at::vec::vec_reduce_all([](Vec& x, Vec& y) { return x+y; }, tmp4);
            }
        }
    //}
    //#pragma GCC ivdep
    //for(long i0=0; i0<ks0*ks1; ++i0)
    //{
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
''').kernel


def call(arg0_1):
    arg0_1_size = arg0_1.size()
    s0 = arg0_1_size[1]
    s1 = arg0_1_size[2]
    buf0 = empty_strided((1, s0, s1, 1), (s0*s1, s1, 1, s0*s1), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, s0, s1, s1), (s0*(s1*s1), s1*s1, s1, 1), device='cpu', dtype=torch.float32)
    buf2 = buf0; del buf0 #empty_strided((1, s0, s1, 1), (s0*s1, s1, 1, s0*s1), device='cpu', dtype=torch.float32)
    buf3 = buf1; del buf1 #empty_strided((1, s0, s1, s1), (s0*(s1*s1), s1*s1, s1, 1), device='cpu', dtype=torch.float32)
    kernel0(c_void_p(arg0_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_long(s0), c_long(s1))
    return (buf3, )


def bench_run():
    from torchdynamo.testing import rand_strided
    from torchinductor.utils import bench
    arg0_1 = rand_strided((1, 16, 384, 384), (2359296, 147456, 384, 1), device='cpu', dtype=torch.float32)
    return bench(lambda: call(arg0_1))


if __name__ == "__main__":
    from torchdynamo.testing import rand_strided
    from torchinductor.utils import print_performance
    arg0_1 = rand_strided((1, 16, 384, 384), (2359296, 147456, 384, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call(arg0_1))
