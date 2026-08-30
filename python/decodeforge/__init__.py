"""Public Python package for DecodeForge's portable Q8 reference."""

from .q8 import (
    BLOCK_SIZE,
    FORMAT,
    NUMERIC_MODE,
    Q8Error,
    Q8Weights,
    dequantize_f32_bits,
    dequantize_f32_rows_bits,
    f32_add_bits,
    f32_bits_to_float,
    f32_div_bits,
    f32_from_int,
    f32_mul_bits,
    float_to_f32_bits,
    is_finite_f32_bits,
    quantize_f32_bits,
)

__all__ = [
    "BLOCK_SIZE",
    "FORMAT",
    "NUMERIC_MODE",
    "Q8Error",
    "Q8Weights",
    "__version__",
    "dequantize_f32_bits",
    "dequantize_f32_rows_bits",
    "f32_add_bits",
    "f32_bits_to_float",
    "f32_div_bits",
    "f32_from_int",
    "f32_mul_bits",
    "float_to_f32_bits",
    "is_finite_f32_bits",
    "quantize_f32_bits",
]

__version__ = "0.1.0"
