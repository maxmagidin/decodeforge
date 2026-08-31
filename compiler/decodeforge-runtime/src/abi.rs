//! Private Rust representation of the frozen generated-module ABI.

use std::mem::{offset_of, size_of};

/// Exact byte count of the `sha256:<hex>` artifact ID and terminating NUL.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) const ARTIFACT_ID_CSTR_BYTES_V1: usize = 72;

/// Frozen results returned by any generated module's `df_run_v1` entrypoint.
#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeneratedStatusV1 {
    Ok = 0,
    NullArgument = 1,
    AbiVersion = 2,
    StructSize = 3,
    Flags = 4,
    Reserved = 5,
    Shape = 6,
    Stride = 7,
    PackedWeightBytes = 8,
    PackedWeightAlignment = 9,
    FpEnvironment = 10,
    NonFiniteInput = 11,
    NonFiniteResult = 12,
}

impl GeneratedStatusV1 {
    /// Decode one frozen C status value.
    pub const fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Ok),
            1 => Some(Self::NullArgument),
            2 => Some(Self::AbiVersion),
            3 => Some(Self::StructSize),
            4 => Some(Self::Flags),
            5 => Some(Self::Reserved),
            6 => Some(Self::Shape),
            7 => Some(Self::Stride),
            8 => Some(Self::PackedWeightBytes),
            9 => Some(Self::PackedWeightAlignment),
            10 => Some(Self::FpEnvironment),
            11 => Some(Self::NonFiniteInput),
            12 => Some(Self::NonFiniteResult),
            _ => None,
        }
    }

    /// The exact signed integer sent across the C ABI.
    pub const fn as_i32(self) -> i32 {
        self as i32
    }
}

/// Source-compatible scalar spelling of [`GeneratedStatusV1`].
///
/// Scalar and NEON modules implement the same frozen generated-module ABI;
/// this alias keeps existing scalar callers source compatible without
/// introducing a second status contract.
pub use GeneratedStatusV1 as ScalarStatusV1;

/// Rust spelling of `df_call_v1` from `include/decodeforge/abi_v1.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DfCallV1 {
    abi_version: u32,
    struct_size: u32,
    flags: u64,
    m: u32,
    n: u32,
    k: u32,
    x_stride: u32,
    y_stride: u32,
    reserved0: u32,
    packed_weight_bytes: u64,
}

impl DfCallV1 {
    pub(crate) const fn new(n: u32, k: u32, packed_weight_bytes: u64) -> Self {
        Self {
            abi_version: crate::RUNTIME_ABI_VERSION,
            struct_size: size_of::<Self>() as u32,
            flags: 0,
            m: 1,
            n,
            k,
            x_stride: k,
            y_stride: n,
            reserved0: 0,
            packed_weight_bytes,
        }
    }
}

const _: () = assert!(size_of::<DfCallV1>() == 48);
const _: () = assert!(offset_of!(DfCallV1, abi_version) == 0);
const _: () = assert!(offset_of!(DfCallV1, struct_size) == 4);
const _: () = assert!(offset_of!(DfCallV1, flags) == 8);
const _: () = assert!(offset_of!(DfCallV1, m) == 16);
const _: () = assert!(offset_of!(DfCallV1, n) == 20);
const _: () = assert!(offset_of!(DfCallV1, k) == 24);
const _: () = assert!(offset_of!(DfCallV1, x_stride) == 28);
const _: () = assert!(offset_of!(DfCallV1, y_stride) == 32);
const _: () = assert!(offset_of!(DfCallV1, reserved0) == 36);
const _: () = assert!(offset_of!(DfCallV1, packed_weight_bytes) == 40);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_frozen_status_round_trips() {
        for raw in 0..=12 {
            let status = GeneratedStatusV1::from_i32(raw).expect("frozen status");
            assert_eq!(status.as_i32(), raw);
        }
        assert_eq!(GeneratedStatusV1::from_i32(-1), None);
        assert_eq!(GeneratedStatusV1::from_i32(13), None);
    }

    #[test]
    fn safe_descriptor_has_the_frozen_m1_contract() {
        let call = DfCallV1::new(5, 33, 576);
        assert_eq!(size_of::<DfCallV1>(), 48);
        assert_eq!(call.abi_version, 1);
        assert_eq!(call.struct_size, 48);
        assert_eq!(call.flags, 0);
        assert_eq!(call.m, 1);
        assert_eq!(call.n, 5);
        assert_eq!(call.k, 33);
        assert_eq!(call.x_stride, 33);
        assert_eq!(call.y_stride, 5);
        assert_eq!(call.reserved0, 0);
        assert_eq!(call.packed_weight_bytes, 576);
    }
}
