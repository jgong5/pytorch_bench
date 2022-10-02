
from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torchinductor.codecache import CppCodeCache, TritonCodeCache

aten = torch.ops.aten


kernel0 = CppCodeCache.load('''
#include "/tmp/torchinductor_jgong5/qr/cqrr7t6pdy7hpf76525uh7ddg65eljkjlgxa5dhk7amq6xb6a3ia.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       const long ks0,
                       const long ks1)
{
    {
        #pragma omp for
        for(long i0=0; i0<ks0; ++i0)
        {
            {
                {
                    using Vec = at::vec::Vectorized<float>;
                    Vec tmp1(0);
                    for(long i1=0; i1<ks1/8; ++i1)
                    {
                        {
                            Vec tmp0 = Vec::loadu(in_ptr0 + i1*8 + (i0*ks1));
                            Vec tmp1 = tmp1 + tmp0;
                        }
                    }
                    out_ptr0[i0] = at::vec::vec_reduce_all([](Vec& x, Vec& y) { return x+y; }, tmp1);
                }
            }
        }
        #pragma omp for
        for(long i0=0; i0<ks0; ++i0)
        {
            {
                {
                    using Vec = at::vec::Vectorized<float>;
                    Vec tmp6(0);
                    Vec tmp7(0);
                    for(long i1=0; i1<ks1/8; ++i1)
                    {
                        {
                            auto tmp0 = Vec::loadu(in_ptr0 + i1*8 + (i0*ks1));
                            auto tmp1 = Vec::loadu(out_ptr0 + i0);
                            Vec tmp2(static_cast<float>(ks1));
                            auto tmp3 = tmp1 / tmp2;
                            auto tmp4 = tmp0 - tmp3;
                            auto tmp5 = tmp4 * tmp4;
                            tmp6 += tmp5;
                            tmp7 += tmp0;
                        }
                    }
                    out_ptr1[i0] = at::vec::vec_reduce_all([](Vec& x, Vec& y) { return x+y; }, tmp6);
                    out_ptr2[i0] = at::vec::vec_reduce_all([](Vec& x, Vec& y) { return x+y; }, tmp7);
                }
            }
        }
        #pragma omp for
        for(long i0=0; i0<ks0; ++i0)
        {
            for(long i1=0; i1<ks1/8; ++i1)
            {
                {
                    {
                        using Vec = at::vec::Vectorized<float>;
                        auto tmp0 = Vec::loadu(in_ptr0 + i1*8 + (i0*ks1));
                        auto tmp1 = Vec::loadu(out_ptr2 + i0);
                        auto tmp5 = Vec::loadu(out_ptr1 + i0);
                        auto tmp12 = Vec::loadu(in_ptr1 + i1*8);
                        auto tmp14 = Vec::loadu(in_ptr2 + i1*8);
                        Vec tmp2(static_cast<float>(ks1));
                        auto tmp3 = tmp1 / tmp2;
                        auto tmp4 = tmp0 - tmp3;
                        auto tmp6 = tmp5 / tmp2;
                        Vec tmp7(static_cast<float>(1e-05));
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = tmp8.sqrt();
                        auto tmp10 = tmp9.reciprocal();
                        auto tmp11 = tmp4 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr3 + i1*8 + (i0*ks1));
                    }
                }
            }
        }
    }
}
''').kernel

def call(arg0_1, arg1_1, arg2_1):
    arg0_1_size = arg0_1.size()
    s0 = arg0_1_size[0]
    s1 = arg0_1_size[1]
    buf0 = empty_strided((s0, 1), (1, s0), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((s0, 1), (1, s0), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((s0, 1), (1, s0), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((s0, s1), (s1, 1), device='cpu', dtype=torch.float32)
    kernel0(c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_long(s0), c_long(s1))
    return (buf3, )


def bench_run():
    from torchdynamo.testing import rand_strided
    from torchinductor.utils import bench
    arg0_1 = rand_strided((384, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    return bench(lambda: call(arg0_1, arg1_1, arg2_1))


if __name__ == "__main__":
    from torchdynamo.testing import rand_strided
    from torchinductor.utils import print_performance
    arg0_1 = rand_strided((384, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    print_performance(lambda: call(arg0_1, arg1_1, arg2_1))
