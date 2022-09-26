
import torch
from torch import tensor, device
import torch.fx as fx
from torchdynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

# torch version: 1.13.0a0+git7df0878
# torch cuda version: None
# torch git version: 7df0878b9936961cc1bde9d20c834ac4331d140a


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        var_mean = torch.ops.aten.var_mean.correction(arg0_1, [1], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        sqrt = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        sub = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, reciprocal);  sub = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(reciprocal, torch.float32);  reciprocal = None
        return (convert_element_type,)
        
args = [((384, 1024), (1024, 1), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu'), ((1024,), (1,), torch.float32, 'cpu')]
args = [rand_strided(shape, stride, dtype, device) for shape, stride, dtype, device in args]
mod = make_fx(Repro())(*args)

from torchinductor.compile_fx import compile_fx_inner

compiled = compile_fx_inner(mod, args)
compiled(*args)
