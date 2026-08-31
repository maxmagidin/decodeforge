//! Deterministic source emitters for the fixed G1 compiler contract.

mod scalar_c;

pub use scalar_c::{
    GENERATED_ABI_VERSION_V1, MAX_SCALAR_C_SOURCE_BYTES, SCALAR_C_SOURCE_FORMAT_V1, ScalarCModule,
    emit_scalar_c,
};
