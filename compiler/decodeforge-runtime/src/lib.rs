#![deny(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

//! Ownership and safe invocation for verified generated modules.
//!
//! The runtime intentionally has no dependency on `decodeforge-core`.  The
//! loader remains independent of compiler and code-generation policy.  Its
//! one unsafe construction boundary is reserved for a compiler-owned,
//! locally built and audited module; ordinary callers receive only the safe
//! [`ScalarExecutableV1`] owner.

mod abi;

// Every dynamic-loader operation, symbol type assertion, raw pointer read,
// and generated C call is confined to this module.
#[allow(unsafe_code)]
mod dylib;

mod error;

// The public executable type remains available on unsupported hosts so callers
// can compile portable code, but its private construction machinery is only
// reachable from the Apple-arm64 loader.
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
mod scalar;

#[doc(hidden)]
pub use dylib::load_trusted_apple_scalar_v1;
pub use error::RuntimeError;
pub use scalar::ScalarExecutableV1;

pub use abi::ScalarStatusV1;

/// Result type returned by checked runtime operations.
pub type Result<T> = std::result::Result<T, RuntimeError>;

/// Version of this crate's runtime package.
pub const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Version of the generated-module C ABI.
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
