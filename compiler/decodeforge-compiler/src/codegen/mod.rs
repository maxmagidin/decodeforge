//! Deterministic source emitters for the fixed G1 compiler contract.

mod neon_c;
mod scalar_c;

pub use scalar_c::{
    GENERATED_ABI_VERSION_V1, MAX_SCALAR_C_SOURCE_BYTES, SCALAR_C_SOURCE_FORMAT_V1, ScalarCModule,
    emit_scalar_c,
};

pub use neon_c::{MAX_NEON_C_SOURCE_BYTES, NEON_C_SOURCE_FORMAT_V1, NeonCModule, emit_neon_c};
