from __future__ import annotations
import mlx.core as mx
import numpy as np
import torch as tc
import math
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass

class TropicalNumber:
    # 轻量级，只关心TropicalNumber的实现本身
    def __init__(self, value: Union[float, int]):
        self.value = value

    #等价关系
    def __eq__(self, other: TropicalNumber) -> bool:
        return self.value == other.value
    
    def __lt__(self, other: TropicalNumber) -> bool:
        return self.value < other.value
    
    def __le__(self, other: TropicalNumber) -> bool:
        return self.value <= other.value
    
    def __gt__(self, other: TropicalNumber) -> bool:
        return self.value > other.value
    
    def __ge__(self, other: TropicalNumber) -> bool:
        return self.value >= other.value
    
    #运算
    def __add__(self, other: TropicalNumber) -> TropicalNumber:
        return TropicalNumber(max(self.value, other.value))
    
    def __radd__(self, other: TropicalNumber) -> TropicalNumber:
        return TropicalNumber(max(self.value, other))
    
    def __mul__(self, other: TropicalNumber) -> TropicalNumber:
        return TropicalNumber(self.value + other)
    
    def __rmul__(self, other: TropicalNumber) -> TropicalNumber:
        return TropicalNumber(self.value + other.value)
        
    def __pow__(self, other: TropicalNumber) -> TropicalNumber:
        return TropicalNumber(self.value * other.value)
    
    #操作
    def __abs__(self) -> TropicalNumber:
        return TropicalNumber(abs(self.value))
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def float(self) -> float:
        return float(self.value)
    
    def sym(self) -> float:
        return TropicalNumber(-self.value)

#对于后端和部分常用算子，不妨写一个最小数据组
@dataclass
class BackendConfig:
    name: str
    lib: Any
    asarray: Callable
    maximum: Callable
    reduce_max: Callable
    einsum: Callable
    transpose: Callable

BACKENDS = {
    "mlx": BackendConfig(
        name = "mlx",
        lib=mx,
        asarray=lambda x: x if isinstance(x, mx.array) else mx.array(x, dtype=mx.float32),
        maximum=mx.maximum,
        reduce_max=lambda x, axis=None, keepdims=False: \
            mx.max(x, axis=axis, keepdims=keepdims),
        einsum=mx.einsum,
        transpose=mx.transpose
    ),
    "numpy": BackendConfig(
        name = "numpy",
        lib=np,
        asarray=lambda x: x if isinstance(x, np.ndarray) else np.array(x, dtype=np.float32),
        maximum=np.maximum,
        reduce_max=lambda x, axis=None, keepdims=False: \
            np.max(x, axis=axis, keepdims=keepdims),
        einsum=np.einsum,
        transpose=np.transpose
    ),
    "torch": BackendConfig(
        name = "torch",
        lib=tc,
        asarray=lambda x: x if isinstance(x, tc.Tensor) else tc.tensor(x, dtype=tc.float32),
        maximum=tc.maximum,
        reduce_max=lambda x, axis=None, keepdims=False: \
            tc.max(x, dim=axis, keepdim=keepdims).values,
        einsum=tc.einsum,
        transpose=tc.permute
    )
}


class TropicalTensor:

    def __init__(self,value, backend: str = "mlx"):
        self.backend = backend
        self.cfg = BACKENDS[backend]
        self.data = self.cfg.asarray(value)

    def _to_array(self, other):
        if isinstance(other, TropicalTensor):
            if other.backend != self.backend:
                raise TypeError("Backend mismatch")
            return other.data
        return self.cfg.asarray(other)
    
    def __add__(self, other: TropicalTensor) -> TropicalTensor:
        if other.backend != self.backend:
            raise TypeError("Backend mismatch")
        else:
            return self.cfg.maximum(self.data, other.data)
    
    def __radd__(self, other) -> TropicalTensor:
        if isinstance(other, TropicalTensor):
            return self.__add__(other)
        elif isinstance(other, Union(float, int)):
            other_ = TropicalTensor(other, self.backend)
            return self.__add__(other_)
        else:
            raise TypeError("DataType mismatch")
        
    def __mul__(self, other: TropicalTensor) -> TropicalTensor:  #对与array真的有逐元素乘法这个东西
        if other.backend != self.backend:
            raise TypeError("Backend mismatch")
        else:
            return self.data + other.data
        
    def __rmul__(self, other) -> TropicalTensor:
        if isinstance(other, TropicalTensor):
            return self.__mul__(other)
        elif isinstance(other, Union(float, int)):
            other_ = TropicalTensor(other, self.backend)
            return self.__mul__(other_)
        else:
            raise TypeError("DataType mismatch")
        
    def __str__(self) -> str:
        return str(self.data)
    
    def __repr__(self) -> str:
        return self.__str__()
        
    def matmul(self, other: TropicalTensor) -> TropicalTensor:
        if other.backend != self.backend:
            raise TypeError("Backend mismatch")
        else:
            A = self.data[..., :, :, None]
            B = other.data[..., None, :, :]
            S = A + B
            C = self.cfg.reduce_max(S, -2)
        return TropicalTensor(C, self.backend)
    
    def __matmul__(self, other: TropicalTensor) -> TropicalTensor:
        return self.matmul(other)
    
    def shape(self):
        return self.data.shape
    
    def ndim(self) -> int:
        return self.data.ndim
    
    def zeros_like(self) -> TropicalTensor:
        return TropicalTensor(self.cfg.lib.zeros_like(self.data) - math.inf, self.backend)
    
    def ones_like(self) -> TropicalTensor:
        return TropicalTensor(self.cfg.lib.zeros_like(self.data), self.backend)
    
    def reshape(self, indexs) -> TropicalTensor:
        return TropicalTensor(self.data.reshape(indexs), self.backend)
    
    def transpose(self, indexs) -> TropicalTensor:
        return TropicalTensor(self.cfg.transpose(self.data, indexs), self.backend)
        
    def squeeze(self, axis=None) -> TropicalTensor:
        return TropicalTensor(self.cfg.lib.squeeze(self.data, axis), self.backend)
