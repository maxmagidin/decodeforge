#![forbid(unsafe_code)]

//! Version constants for the future versioned runtime C boundary.
//!
//! The runtime intentionally has no dependency on `decodeforge-core`.  The
//! runtime loader and its eventual C API must remain independent of compiler
//! and code-generation policy.

/// Version of this crate's runtime package.
pub const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Version of the future runtime C ABI reservation.
pub const RUNTIME_ABI_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_contract_is_version_one() {
        assert_eq!(PACKAGE_VERSION, "0.1.0");
        assert_eq!(RUNTIME_ABI_VERSION, 1);
    }
}
