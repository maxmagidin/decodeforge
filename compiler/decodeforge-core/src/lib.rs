#![forbid(unsafe_code)]

//! Versioned contracts and the G0 semantic oracle shared by the compiler.
//!
//! This crate hosts the independent `DFQ8_B32_V1` scalar quantization,
//! evaluation, identity, and fixture gates used to freeze G0 semantics. A
//! future quantization crate will own G1 target packing and related lowering;
//! keeping those concerns separate preserves this crate's target-independent
//! oracle and version contract.

pub mod q8;

/// The user-facing package name.
pub const PACKAGE_NAME: &str = "decodeforge";

/// Version of the compiler package represented by this crate.
pub const COMPILER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Major version of the request/output schemas.
pub const SCHEMA_MAJOR_VERSION: u32 = 1;

/// Version of the generated-module ABI.
pub const GENERATED_ABI_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_contract_is_version_one() {
        assert_eq!(PACKAGE_NAME, "decodeforge");
        assert_eq!(COMPILER_VERSION, "0.1.0");
        assert_eq!(SCHEMA_MAJOR_VERSION, 1);
        assert_eq!(GENERATED_ABI_VERSION, 1);
    }
}
