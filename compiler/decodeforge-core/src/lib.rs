#![forbid(unsafe_code)]

//! Versioned constants shared by the target-independent compiler surface.
//!
//! The foundation deliberately contains no quantization, IR, scheduling, or
//! code generation logic. Later changes may build on these constants without
//! changing the public version contract accidentally.

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
