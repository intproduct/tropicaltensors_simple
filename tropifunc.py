import mlx.core as mx
import numpy as np
import torch as tc
import math
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from tropinum import *
from collections import defaultdict

INDEX_VARABLES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def _equations_read(equation: str, n_operands: int) -> Tuple[List[List[str]], List[str]]:
    if "->" not in equation:
        raise ValueError("自动推断未适配，需要显式写出'->'.")
    
    lhs, rhs = equation.split("->")
    input_specs = lhs.split(",")
    if len(input_specs) != n_operands:
        raise ValueError(f"einsum has{len(input_specs)} strings, but only {n_operands} tensors exits.")
    for spec in input_specs + [rhs]:
        for ch in spec:
            if ch and ch not in INDEX_VARABLES:
                raise ValueError("charter is not supported now.")
            
    input_subscripts = [list(s) for s in input_specs]
    output_subscripts = list(rhs)

    return input_subscripts, output_subscripts

def _bulid_index(inputs: List[List[str]]): #记录了每个字母的位置
    occ = defaultdict(list)
    for op_idx, subs in enumerate(inputs):
        for axis_idx, idx in enumerate(subs):
            occ[idx].append((op_idx, axis_idx))
    return occ

def eye(n: int, backend: str = "mlx") -> TropicalTensor:
    return TropicalTensor(BACKENDS[backend].lib.eye(n))
    
def zeros(shape, backend: str = "mlx") -> TropicalTensor:
    return TropicalTensor(BACKENDS[backend].lib.zeros(shape) - math.inf)

def ones(shape, backend: str = "mlx") -> TropicalTensor:
    return TropicalTensor(BACKENDS[backend].lib.ones(shape))

def trace(x: TropicalTensor, offset, axis1, axis2) -> TropicalTensor:
    x_t = x.cfg.lib.diagonal(x.data, offset, axis1, axis2)
    result = x.cfg.reduce_max(x_t, -1)
    return TropicalTensor(result, backend=x.backend)

def transpose(x: TropicalTensor, perm) -> TropicalTensor:
    return TropicalTensor(x.cfg.transpose(x.data, perm), backend=x.backend)

def tensordot(a: TropicalTensor, b: TropicalTensor, axes=2) -> TropicalTensor:
    assert a.backend == b.backend
    a_d = a.data
    b_d = b.data
    libs = a.cfg.lib

    if isinstance(axes, int):
        axes_a = list(range(a_d.ndim - axes, a_d.ndim))
        axes_b = list(range(axes))
    else:
        axes_a, axes_b = axes

    a_non_contract = [i for i in range(a_d.ndim) if i not in axes_a]
    a_reordered = libs.transpose(a_d, a_non_contract + axes_a)

    b_non_contract = [i for i in range(b_d.ndim) if i not in axes_b]
    b_reordered = libs.transpose(b_d, axes_b + b_non_contract)

    a_m = TropicalTensor(a_reordered.reshape(-1, libs.prod(a_reordered.shape[len(a_non_contract):])), backend=a.backend)
    
    b_m = TropicalTensor(b_reordered.reshape(libs.prod(b_reordered.shape[:len(axes_b)]), -1), backend=b.backend)

    result_matrix = a_m @ b_m

    output_shape = a_reordered.shape[:len(a_non_contract)] + b_reordered.shape[len(axes_b):]
    return result_matrix.reshape(output_shape)

#einsum有点难写，tensordot一般足够用了
def einsum(subscripts, *operands: TropicalTensor) -> TropicalTensor:
    input_sub, output_sub = _equations_read(subscripts, len(operands))

    backends = {op.backend for op in operands}
    if len(backends) != 1:
        raise TypeError("Backend mismatch.")
    backend = operands[0].backend

    for op, subs in zip(operands, input_sub):
        if op.data.ndim != len(subs):
            raise ValueError("the number of subscripts in the equation does not match the number of dimensions and no ellipsis was given.")
        
    operands = list(operands)

    while True:
        occ = _bulid_index(input_sub)

        contract_idx = None
        for idx, loc in occ.items():
            if idx not in output_sub and len(loc) > 0:
                contract_idx = idx
                break
        if contract_idx is None:
            break

        locs = occ[contract_idx]

        #自收缩
        op_ids = {op_id for op_id, _ in locs}
        if len(op_ids) == 1:
            op_id = op_id.pop()
            pos = [axis for (oid, axis) in locs if oid == op_id]
            pos = sorted(pos)

            if len(pos)<2:
                raise NotImplementedError(f"index {contract_idx!r} 只在一个张量中出现一次且不在输出中")\
                
            axis1, axis2 = pos[0], pos[1]

            op_old = operands[op_id]
            traced = trace(op_old, 0, axis1, axis2)

            new_subs = []
            for i, idx in enumerate(input_sub[op_id]):
                if i not in (axis1, axis2):
                    new_subs.append(idx)

            operands[op_id] = traced
            input_sub[op_id] = new_subs

            continue

        if len(loc) == 2:
            (op_a, ax_a), (op_b, ax_b) = loc

            if op_a > op_b:
                (op_a, ax_a), (op_b, ax_b) = (op_b, ax_b), (op_a, ax_a)

            A = operands[op_a]
            B = operands[op_b]
            subs_a = input_sub[op_a]
            subs_b = input_sub[op_b]

            C = tensordot(A, B, axes=([ax_a], [ax_b]))

            new_subs = [idx for i, idx in enumerate(subs_a) if i != ax_a] + \
                        [idx for j, idx in enumerate(subs_b) if j != ax_b]

            operands[op_a] = C
            input_sub[op_a] = new_subs

            del operands[op_b]
            del input_sub[op_b]

            continue

        raise NotImplementedError("einsum is not matching the function now.")

    if len(operands) != 1:
        raise RuntimeError("Some indexes are left with no contraction.")

    result = operands[0]
    final_subs = input_sub[0]

    if set(final_subs) != set(output_sub):
        raise ValueError(f"Output indexes{''.join(output_sub)} is not same with the indexs {''.join(final_subs)} now. ")

    perm = [final_subs.index(idx) for idx in output_sub]
    result = transpose(result, perm)

    return result
