//! Closed runtime failures for the first scalar generated module.

use crate::ScalarStatusV1;
use std::fmt;

/// Checked failure from loading or invoking a scalar generated module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeError {
    UnsupportedHost,
    InvalidLoadContract { field: &'static str },
    AllocationFailed { object: &'static str },
    PrivateCopyFailed,
    DynamicLoaderEnvironment,
    DynamicLoadFailed,
    MissingSymbol { symbol: &'static str },
    AbiVersionMismatch { expected: u32, actual: u32 },
    ModuleIdMismatch,
    InputLength { expected: usize, actual: usize },
    KernelStatus(ScalarStatusV1),
    UnknownKernelStatus(i32),
    InvalidSuccessOutput { index: usize },
}

impl RuntimeError {
    /// Stable diagnostic family for this failure.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedHost => "DFE-NATIVE-001",
            Self::InvalidLoadContract { .. }
            | Self::PrivateCopyFailed
            | Self::DynamicLoaderEnvironment
            | Self::DynamicLoadFailed
            | Self::MissingSymbol { .. }
            | Self::AbiVersionMismatch { .. }
            | Self::ModuleIdMismatch => "DFE-NATIVE-007",
            Self::AllocationFailed { .. } | Self::InputLength { .. } => "DFE-NATIVE-008",
            Self::KernelStatus(_)
            | Self::UnknownKernelStatus(_)
            | Self::InvalidSuccessOutput { .. } => "DFE-NATIVE-009",
        }
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: ", self.code())?;
        match self {
            Self::UnsupportedHost => {
                formatter.write_str("scalar dylib loading requires a macOS arm64 host")
            }
            Self::InvalidLoadContract { field } => {
                write!(
                    formatter,
                    "trusted scalar load contract is invalid for {field}"
                )
            }
            Self::AllocationFailed { object } => {
                write!(formatter, "unable to allocate private {object} storage")
            }
            Self::PrivateCopyFailed => {
                formatter.write_str("unable to construct the private scalar dylib copy")
            }
            Self::DynamicLoaderEnvironment => formatter
                .write_str("refusing to load generated code while a DYLD_* override is visible"),
            Self::DynamicLoadFailed => {
                formatter.write_str("dyld rejected the private scalar dylib")
            }
            Self::MissingSymbol { symbol } => {
                write!(
                    formatter,
                    "scalar dylib is missing required symbol {symbol}"
                )
            }
            Self::AbiVersionMismatch { expected, actual } => write!(
                formatter,
                "scalar dylib ABI version {actual} does not equal expected {expected}"
            ),
            Self::ModuleIdMismatch => {
                formatter.write_str("scalar dylib module identity does not match the expected code")
            }
            Self::InputLength { expected, actual } => {
                write!(
                    formatter,
                    "input length {actual} does not equal expected K={expected}"
                )
            }
            Self::KernelStatus(status) => {
                write!(
                    formatter,
                    "scalar kernel returned status {} ({status:?})",
                    status.as_i32()
                )
            }
            Self::UnknownKernelStatus(status) => {
                write!(formatter, "scalar kernel returned unknown status {status}")
            }
            Self::InvalidSuccessOutput { index } => write!(
                formatter,
                "scalar kernel reported success with a nonfinite output at index {index}"
            ),
        }
    }
}

impl std::error::Error for RuntimeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_codes_distinguish_load_input_and_execution() {
        assert_eq!(RuntimeError::DynamicLoadFailed.code(), "DFE-NATIVE-007");
        assert_eq!(
            RuntimeError::InputLength {
                expected: 4,
                actual: 3
            }
            .code(),
            "DFE-NATIVE-008"
        );
        assert_eq!(
            RuntimeError::KernelStatus(ScalarStatusV1::NonFiniteResult).code(),
            "DFE-NATIVE-009"
        );
    }
}
