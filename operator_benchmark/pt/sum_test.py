import operator_benchmark as op_bench
import torch
import torch._C._te as te

"""Microbenchmarks for sum reduction operator."""

# Configs for PT add operator
sum_configs = op_bench.cross_product_configs(
    R=[64, 256],  # Length of reduced dimension
    V=[32, 512],  # Length of other dimension
    dim=[0, 1],
    contiguous=[True],
    device=['cpu', 'cuda'],
    dtype=[torch.float],
    tags=['short']
) + op_bench.cross_product_configs(
    R=[1024, 8192],
    V=[512, 1024],
    dim=[0, 1],
    contiguous=[True],
    device=['cpu', 'cuda'],
    dtype=[torch.float],
    tags=['long']
) + op_bench.cross_product_configs(
    R=[65, 257],  # Length of reduced dimension
    V=[33, 513],  # Length of other dimension
    dim=[0, 1],
    contiguous=[True],
    device=['cpu', 'cuda'],
    dtype=[torch.float],
    tags=['short_unaligned']
) + op_bench.cross_product_configs(
    R=[1025, 8193],
    V=[513, 1025],
    dim=[0, 1],
    contiguous=[True],
    device=['cpu', 'cuda'],
    dtype=[torch.float],
    tags=['long_unaligned']
) + op_bench.cross_product_configs(
    R=[8192],  # Length of reduced dimension
    V=[1, 32],  # Length of other dimension
    dim=[0, 1],
    contiguous=[True],
    device=['cpu', 'cuda'],
    dtype=[torch.float],
    tags=['rfactor']
) + op_bench.cross_product_configs(
    R=[8193],  # Length of reduced dimension
    V=[1, 32],  # Length of other dimension
    dim=[0, 1],
    contiguous=[True],
    device=['cpu', 'cuda'],
    dtype=[torch.float],
    tags=['rfactor_unaligned']
)

class SumBenchmarkBase(op_bench.TorchBenchmarkBase):
    def init(self, R, V, dim, contiguous, device, dtype):
        shape = (R, V) if dim == 0 else (V, R)
        self.R = R
        self.V = V
        tensor = torch.rand(shape, device=device, dtype=dtype)

        if not contiguous:
            storage = torch.empty([s * 2 for s in shape], device=device, dtype=dtype)
            storage[::2, ::2] = tensor
            self.input_tensor = storage[::2, ::2]
        else:
            self.input_tensor = tensor

        self.inputs = {
            "input_tensor": self.input_tensor,
            "dim": dim
        }
        self.set_module_name("sum")

    def get_compute_characteristics(self):
        comp_ch = {}
        comp_ch["mem_read"] = self.input_tensor.numel() * 4
        comp_ch["mem_write"] = self.input_tensor.size(1) * 4 if self.inputs["dim"] == 0 else self.input_tensor.size(0) * 4
        return comp_ch

class SumBenchmark(SumBenchmarkBase):
    def forward(self, input_tensor, dim: int):
        return input_tensor.sum(dim=dim)

'''
class SumNncBenchmark(SumBenchmarkBase):
    def init(self, R, V, dim, contiguous, device, dtype):
        super(SumNncBenchmark, self).init(R, V, dim, contiguous, device, dtype)
        self.dtype = dtype
        x = te.BufHandle("x", self.input_tensor.shape, self.input_tensor.stride(), dtype)
        y = te.Reduce("y", [V], te.Sum(), x, [R])
        loopnest = te.LoopNest([y])
        loopnest.prepare_for_codegen()
        stmt = te.simplify(loopnest.root_stmt())
        print(stmt)
        self.cg = te.construct_codegen("ir_eval", stmt, [x,y])

    def forward(self, input_tensor, dim: int):
        y = torch.empty([self.V], dtype=self.dtype)
        self.cg.call([input_tensor,y])
        return y
'''

import torchdynamo
from torchinductor import config
config.cpp.simdlen = 8
class SumInductorBenchmark(SumBenchmarkBase):
    def init(self, R, V, dim, contiguous, device, dtype):
        super(SumInductorBenchmark, self).init(R, V, dim, contiguous, device, dtype)

    @torchdynamo.optimize()
    def forward(self, input_tensor, dim: int):
        return input_tensor.sum(dim=dim)

#op_bench.generate_pt_test(sum_configs, SumBenchmark)
#op_bench.generate_pt_test(sum_configs, SumNncBenchmark)
op_bench.generate_pt_test(sum_configs, SumInductorBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
