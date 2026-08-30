"""Public Python package for DecodeForge's exact binary32 primitives."""

from .q8 import (
    BLOCK_SIZE,
    FORMAT,
    NUMERIC_MODE,
    Q8Error,
    f32_add_bits,
    f32_bits_to_float,
    f32_div_bits,
    f32_from_int,
    f32_mul_bits,
    float_to_f32_bits,
    is_finite_f32_bits,
)

__all__ = [
    "BLOCK_SIZE",
    "FORMAT",
    "NUMERIC_MODE",
    "Q8Error",
    "__version__",
    "f32_add_bits",
    "f32_bits_to_float",
    "f32_div_bits",
    "f32_from_int",
    "f32_mul_bits",
    "float_to_f32_bits",
    "is_finite_f32_bits",
]

__version__ = "0.1.0"
