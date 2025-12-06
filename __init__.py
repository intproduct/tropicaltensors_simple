# Tropical/__init__.py
"""
Tropical_mlx: a tiny Tropical algebra Circuit library on MLX.

Public API layout:
- tropinum: TropicalNumber/TropicalTensor
- tropifunc: functions of TropicalTensor
"""

__version__ = "0.1.0"

# Experts
from .tropinum import (
    TropicalNumber,
    TropicalTensor,
)

# Gating & routing
from .tropifunc import (
    eye,
    zeros,
    ones,
    trace,
    transpose,
    tensordot,
    einsum,
)

__all__ = [
    #num
    "TropicaNumber",
    "TropicalTensor",
    #funcs
    "eye",
    "zeros",
    "ones",
    "trace",
    "transpose",
    "tensordot",
    "einsum"
]
