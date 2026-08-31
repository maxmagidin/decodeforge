#![forbid(unsafe_code)]

//! Verified G1 lowering, OI4 packing, and deterministic scalar code generation.
//!
//! The target-independent contract is an `M=1` Q8 linear region, a verified
//! fixed loop kernel, and the shared `DFQ8_B32_OI4_V1` payload. The first
//! backend deterministically emits strict scalar C. Native compilation,
//! dynamic loading, and execution remain separate follow-on stages.

#[cfg(not(target_pointer_width = "64"))]
compile_error!("decodeforge-compiler requires a 64-bit target");

#[cfg(not(target_endian = "little"))]
compile_error!("decodeforge-compiler requires a little-endian target");

use std::fmt;

pub mod codegen;
pub mod ir;
pub mod lower;
pub mod pack;

pub use codegen::{
    GENERATED_ABI_VERSION_V1, MAX_SCALAR_C_SOURCE_BYTES, SCALAR_C_SOURCE_FORMAT_V1, ScalarCModule,
    emit_scalar_c,
};

pub use ir::{
    ArithmeticContract, KPadding, KernelVariant, LoopKernelV1, NTail, Q8LinearRegion,
    Q8LinearShape, ReductionOrder, VectorAxis,
};
pub use lower::lower_q8_linear;
pub use pack::{
    PACK_ALIGNMENT, PACK_BLOCK_SIZE, PACK_FORMAT, PACK_RECORD_BYTES, PACK_TILE, PackManifestV1,
    PackSpecV1, PackedWeightsV1, expected_payload_bytes,
};

/// Stable error returned by a checked compiler-contract operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompilerError {
    code: &'static str,
    summary: String,
}

impl CompilerError {
    pub(crate) fn new(code: &'static str, summary: impl Into<String>) -> Self {
        Self {
            code,
            summary: summary.into(),
        }
    }

    /// Stable machine-readable error code.
    pub const fn code(&self) -> &'static str {
        self.code
    }

    /// Human-readable error summary.
    pub fn summary(&self) -> &str {
        &self.summary
    }
}

impl fmt::Display for CompilerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.summary)
    }
}

impl std::error::Error for CompilerError {}

/// Result type used by this crate's checked constructors and verifiers.
pub type Result<T> = std::result::Result<T, CompilerError>;

pub(crate) const SCHEMA_VERSION: u32 = 1;
pub(crate) const BLOCK_SIZE: u32 = 32;
pub(crate) const OUTPUT_TILE: u32 = 4;
pub(crate) const RECORD_BYTES: u32 = 144;
pub(crate) const PAYLOAD_ALIGNMENT: u32 = 16;

pub(crate) fn invalid(code: &'static str, summary: impl Into<String>) -> CompilerError {
    CompilerError::new(code, summary)
}

pub(crate) fn checked_usize(value: u64, field: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        invalid(
            "DFE-COMP-002",
            format!("{field} is not representable by usize."),
        )
    })
}

pub(crate) fn checked_mul(left: u64, right: u64, field: &str) -> Result<u64> {
    left.checked_mul(right)
        .ok_or_else(|| invalid("DFE-COMP-002", format!("derived {field} size overflows.")))
}

pub(crate) fn is_sha256_identity(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_shape_is_checked_without_panicking() {
        assert!(is_sha256_identity(
            "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        ));
        assert!(!is_sha256_identity("sha256:XYZ"));
    }
}
