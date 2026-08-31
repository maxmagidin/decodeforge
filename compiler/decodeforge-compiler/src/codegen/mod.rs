//! Deterministic source emitters for the fixed G1 compiler contract.

mod c_abi_v1;
mod neon_c;
mod scalar_c;

/// Generated ABI version incorporated into every C module identity.
pub const GENERATED_ABI_VERSION_V1: u32 = 1;

pub use scalar_c::{
    MAX_SCALAR_C_SOURCE_BYTES, SCALAR_C_SOURCE_FORMAT_V1, ScalarCModule, emit_scalar_c,
};

pub use neon_c::{MAX_NEON_C_SOURCE_BYTES, NEON_C_SOURCE_FORMAT_V1, NeonCModule, emit_neon_c};
