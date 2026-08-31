//! Safe generated-module ownership, aligned packing, and invocation.

use crate::abi::DfCallV1;
use crate::dylib::LoadedGeneratedDylib;
use crate::{GeneratedStatusV1, Result, RuntimeError};
use std::mem::{align_of, size_of};

const OUTPUT_TILE: u64 = 4;
const BLOCK_SIZE: u64 = 32;
const RECORD_BYTES: u64 = 144;
const PACK_ALIGNMENT: usize = 16;
const MAX_GENERATED_DYLIB_BYTES: usize = 8 * 1024 * 1024;
const QUIET_NAN_BITS: u32 = 0x7fc0_0000;

#[repr(align(16))]
#[derive(Clone, Copy)]
struct PackChunk([u8; PACK_ALIGNMENT]);

const _: () = assert!(size_of::<PackChunk>() == PACK_ALIGNMENT);
const _: () = assert!(align_of::<PackChunk>() == PACK_ALIGNMENT);

pub(super) struct AlignedPack {
    chunks: Vec<PackChunk>,
    byte_len: usize,
}

impl AlignedPack {
    pub(super) fn copy_from_verified(bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() || !bytes.len().is_multiple_of(PACK_ALIGNMENT) {
            return Err(RuntimeError::InvalidLoadContract {
                field: "packed weight length",
            });
        }
        let chunk_count = bytes.len() / PACK_ALIGNMENT;
        let mut chunks = Vec::new();
        chunks
            .try_reserve_exact(chunk_count)
            .map_err(|_| RuntimeError::AllocationFailed {
                object: "aligned packed-weight",
            })?;
        chunks.resize(chunk_count, PackChunk([0; PACK_ALIGNMENT]));
        let (source_chunks, remainder) = bytes.as_chunks::<PACK_ALIGNMENT>();
        debug_assert!(remainder.is_empty());
        for (destination, source) in chunks.iter_mut().zip(source_chunks) {
            destination.0.copy_from_slice(source);
        }
        let pack = Self {
            chunks,
            byte_len: bytes.len(),
        };
        if !pack.as_ptr().addr().is_multiple_of(PACK_ALIGNMENT) {
            return Err(RuntimeError::InvalidLoadContract {
                field: "packed weight alignment",
            });
        }
        Ok(pack)
    }

    pub(super) fn as_ptr(&self) -> *const u8 {
        self.chunks.as_ptr().cast::<u8>()
    }

    pub(super) const fn len(&self) -> usize {
        self.byte_len
    }

    #[cfg(test)]
    fn to_bytes(&self) -> Vec<u8> {
        self.chunks
            .iter()
            .flat_map(|chunk| chunk.0)
            .take(self.byte_len)
            .collect()
    }
}

pub(super) struct ValidatedLoadSpec {
    pub(super) module_id: String,
    pub(super) packed_identity: String,
    pub(super) n: u32,
    pub(super) k: u32,
    pub(super) packed_weight_bytes: u64,
}

impl ValidatedLoadSpec {
    pub(super) fn new(
        dylib_bytes: &[u8],
        module_id: &str,
        n: u32,
        k: u32,
        packed_bytes: &[u8],
        packed_identity: &str,
    ) -> Result<Self> {
        validate_mach_o_header(dylib_bytes)?;
        if !is_sha256_identity(module_id) {
            return Err(RuntimeError::InvalidLoadContract { field: "module ID" });
        }
        if !is_sha256_identity(packed_identity) {
            return Err(RuntimeError::InvalidLoadContract {
                field: "packed identity",
            });
        }
        let expected = expected_payload_bytes(n, k)?;
        if packed_bytes.len() != expected {
            return Err(RuntimeError::InvalidLoadContract {
                field: "packed weight length",
            });
        }
        let packed_weight_bytes =
            u64::try_from(expected).map_err(|_| RuntimeError::InvalidLoadContract {
                field: "packed weight length",
            })?;
        Ok(Self {
            module_id: module_id.to_owned(),
            packed_identity: packed_identity.to_owned(),
            n,
            k,
            packed_weight_bytes,
        })
    }
}

/// Owner of one loaded generated code module and one exact aligned weight pack.
///
/// This owner is backend-neutral: scalar and NEON modules share the same
/// frozen ABI, private aligned pack, checked invocation, and output policy.
pub struct GeneratedExecutableV1 {
    module: LoadedGeneratedDylib,
    pack: AlignedPack,
    module_id: String,
    packed_identity: String,
    n: u32,
    k: u32,
    packed_weight_bytes: u64,
}

/// One validated, allocation-free invocation of a generated module.
///
/// Preparation borrows the executable, input, and caller-owned output for the
/// lifetime of this object and validates both buffer lengths once. Each
/// [`invoke`](Self::invoke) reuses those exact buffers. Its measured boundary
/// includes the full-output sentinel fill, native ABI call, status handling,
/// and finite-output scan; compilation, loading, weight packing, and output
/// allocation are outside that boundary.
pub struct PreparedCallV1<'executable, 'buffers> {
    executable: &'executable GeneratedExecutableV1,
    input: &'buffers [f32],
    output: &'buffers mut [f32],
    call: DfCallV1,
}

impl PreparedCallV1<'_, '_> {
    /// Invoke the generated module without allocating or changing buffers.
    ///
    /// Before every native call the complete output is overwritten with the
    /// fixed `0x7fc0_0000` quiet-NaN sentinel. Any nonzero status, unknown
    /// status, or nonfinite success output restores that sentinel across the
    /// complete caller buffer before returning an error.
    pub fn invoke(&mut self) -> Result<&[f32]> {
        let executable = self.executable;
        let input = self.input;
        let call = &self.call;
        invoke_with_output_policy(self.output, |output| {
            executable
                .module
                .invoke(call, input, &executable.pack, output)
        })?;
        Ok(self.output)
    }
}

impl GeneratedExecutableV1 {
    pub(super) fn from_trusted_parts(
        module: LoadedGeneratedDylib,
        pack: AlignedPack,
        spec: ValidatedLoadSpec,
    ) -> Self {
        Self {
            module,
            pack,
            module_id: spec.module_id,
            packed_identity: spec.packed_identity,
            n: spec.n,
            k: spec.k,
            packed_weight_bytes: spec.packed_weight_bytes,
        }
    }

    /// Stable identity of the generated code contract.
    pub fn module_id(&self) -> &str {
        &self.module_id
    }

    /// Exact identity of the immutable aligned pack bound to this owner.
    pub fn packed_identity(&self) -> &str {
        &self.packed_identity
    }

    /// Compiled output count.
    pub const fn n(&self) -> u32 {
        self.n
    }

    /// Compiled input count.
    pub const fn k(&self) -> u32 {
        self.k
    }

    /// Validate and prepare one allocation-free generated-module call.
    ///
    /// Both slice lengths are checked here and are fixed by the returned
    /// borrows. Invocation therefore performs no extent validation and no
    /// allocation. The caller retains ownership of `output` and may reuse the
    /// same allocation through repeated [`PreparedCallV1::invoke`] calls.
    pub fn prepare_call<'executable, 'buffers>(
        &'executable self,
        x: &'buffers [f32],
        output: &'buffers mut [f32],
    ) -> Result<PreparedCallV1<'executable, 'buffers>> {
        validate_call_lengths(self.k as usize, self.n as usize, x.len(), output.len())?;
        Ok(PreparedCallV1 {
            executable: self,
            input: x,
            output,
            call: DfCallV1::new(self.n, self.k, self.packed_weight_bytes),
        })
    }

    /// Execute one `M=1` input and return a fresh complete output.
    ///
    /// The output allocation is never exposed on a nonzero or unknown kernel
    /// status.  The private pack, input, and output are distinct Rust-owned
    /// borrows at the point where the raw ABI is entered.
    pub fn run(&self, x: &[f32]) -> Result<Vec<f32>> {
        // Preserve the allocating API's fail-before-allocation behavior for an
        // invalid input. `prepare_call` still owns the authoritative paired
        // input/output validation after the exact output has been allocated.
        let expected_input = self.k as usize;
        if x.len() != expected_input {
            return Err(RuntimeError::InputLength {
                expected: expected_input,
                actual: x.len(),
            });
        }
        let output_len = self.n as usize;
        let mut output = Vec::new();
        output
            .try_reserve_exact(output_len)
            .map_err(|_| RuntimeError::AllocationFailed { object: "output" })?;
        output.resize(output_len, f32::from_bits(QUIET_NAN_BITS));
        self.prepare_call(x, &mut output)?.invoke()?;
        Ok(output)
    }
}

/// Source-compatible scalar spelling of [`GeneratedExecutableV1`].
///
/// This is a direct re-export, not a second owner or execution implementation.
pub use GeneratedExecutableV1 as ScalarExecutableV1;

fn validate_call_lengths(
    expected_input: usize,
    expected_output: usize,
    actual_input: usize,
    actual_output: usize,
) -> Result<()> {
    if actual_input != expected_input {
        return Err(RuntimeError::InputLength {
            expected: expected_input,
            actual: actual_input,
        });
    }
    if actual_output != expected_output {
        return Err(RuntimeError::OutputLength {
            expected: expected_output,
            actual: actual_output,
        });
    }
    Ok(())
}

fn invoke_with_output_policy(
    output: &mut [f32],
    invoke: impl FnOnce(&mut [f32]) -> i32,
) -> Result<()> {
    scrub_output(output);
    let raw_status = invoke(output);
    if raw_status != GeneratedStatusV1::Ok.as_i32() {
        scrub_output(output);
        return match GeneratedStatusV1::from_i32(raw_status) {
            Some(GeneratedStatusV1::Ok) => unreachable!("nonzero status decoded as success"),
            Some(status) => Err(RuntimeError::KernelStatus(status)),
            None => Err(RuntimeError::UnknownKernelStatus(raw_status)),
        };
    }
    if let Some(index) = output
        .iter()
        .position(|value| !is_finite_f32_bits(value.to_bits()))
    {
        scrub_output(output);
        return Err(RuntimeError::InvalidSuccessOutput { index });
    }
    Ok(())
}

fn scrub_output(output: &mut [f32]) {
    output.fill(f32::from_bits(QUIET_NAN_BITS));
}

fn expected_payload_bytes(n: u32, k: u32) -> Result<usize> {
    if n == 0 || k == 0 {
        return Err(RuntimeError::InvalidLoadContract { field: "shape" });
    }
    let panels = u64::from(n).div_ceil(OUTPUT_TILE);
    let blocks = u64::from(k).div_ceil(BLOCK_SIZE);
    let bytes = panels
        .checked_mul(blocks)
        .and_then(|records| records.checked_mul(RECORD_BYTES))
        .ok_or(RuntimeError::InvalidLoadContract {
            field: "packed weight length",
        })?;
    usize::try_from(bytes).map_err(|_| RuntimeError::InvalidLoadContract {
        field: "packed weight length",
    })
}

fn validate_mach_o_header(bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() || bytes.len() > MAX_GENERATED_DYLIB_BYTES {
        return Err(RuntimeError::InvalidLoadContract {
            field: "dylib length",
        });
    }
    let header = bytes.get(..32).ok_or(RuntimeError::InvalidLoadContract {
        field: "Mach-O header",
    })?;
    let word = |offset: usize| {
        u32::from_le_bytes(
            header[offset..offset + 4]
                .try_into()
                .expect("fixed Mach-O word lies inside 32-byte header"),
        )
    };
    const MH_MAGIC_64: u32 = 0xfeed_facf;
    const CPU_TYPE_ARM64: u32 = 0x0100_000c;
    const MH_DYLIB: u32 = 6;
    if word(0) != MH_MAGIC_64 || word(4) != CPU_TYPE_ARM64 || word(12) != MH_DYLIB {
        return Err(RuntimeError::InvalidLoadContract {
            field: "Mach-O header",
        });
    }
    Ok(())
}

fn is_sha256_identity(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

const fn is_finite_f32_bits(bits: u32) -> bool {
    bits & 0x7f80_0000 != 0x7f80_0000
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(digit: char) -> String {
        format!("sha256:{}", digit.to_string().repeat(64))
    }

    fn mach_o_header() -> Vec<u8> {
        let mut bytes = vec![0_u8; 32];
        bytes[0..4].copy_from_slice(&0xfeed_facf_u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&0x0100_000c_u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&6_u32.to_le_bytes());
        bytes
    }

    #[test]
    fn aligned_pack_reconstructs_exact_payload_without_unsafe() {
        for length in [144, 288, 576] {
            let bytes = (0..length).map(|index| index as u8).collect::<Vec<_>>();
            let pack = AlignedPack::copy_from_verified(&bytes).unwrap();
            assert!(pack.as_ptr().addr().is_multiple_of(16));
            assert_eq!(pack.len(), length);
            assert_eq!(pack.to_bytes(), bytes);
        }
    }

    #[test]
    fn shape_arithmetic_is_exact_and_checked() {
        assert_eq!(expected_payload_bytes(1, 1).unwrap(), 144);
        assert_eq!(expected_payload_bytes(4, 32).unwrap(), 144);
        assert_eq!(expected_payload_bytes(5, 33).unwrap(), 576);
        assert!(expected_payload_bytes(0, 1).is_err());
        assert!(expected_payload_bytes(1, 0).is_err());
    }

    #[test]
    fn load_spec_rejects_malformed_identity_shape_and_image() {
        let image = mach_o_header();
        let module_id = identity('a');
        let packed_id = identity('b');
        assert!(ValidatedLoadSpec::new(&image, &module_id, 1, 1, &[0; 144], &packed_id).is_ok());
        assert!(ValidatedLoadSpec::new(&image, "sha256:A", 1, 1, &[0; 144], &packed_id).is_err());
        assert!(ValidatedLoadSpec::new(&image, &module_id, 1, 1, &[0; 143], &packed_id).is_err());
        assert!(ValidatedLoadSpec::new(&[0; 32], &module_id, 1, 1, &[0; 144], &packed_id).is_err());
    }

    #[test]
    fn call_lengths_are_validated_once_at_preparation() {
        assert_eq!(validate_call_lengths(3, 2, 3, 2), Ok(()));
        assert_eq!(
            validate_call_lengths(3, 2, 1, 2),
            Err(RuntimeError::InputLength {
                expected: 3,
                actual: 1,
            })
        );
        assert_eq!(
            validate_call_lengths(3, 2, 3, 1),
            Err(RuntimeError::OutputLength {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            validate_call_lengths(3, 2, 3, 4),
            Err(RuntimeError::OutputLength {
                expected: 2,
                actual: 4,
            })
        );
        assert_eq!(
            validate_call_lengths(3, 2, 1, 4),
            Err(RuntimeError::InputLength {
                expected: 3,
                actual: 1,
            }),
            "input validation remains the first failure"
        );
    }

    #[test]
    fn every_failure_status_and_unknown_status_scrubs_the_complete_output() {
        for raw in 1..=12 {
            let mut output = vec![1.0, 2.0, 3.0];
            let result = invoke_with_output_policy(&mut output, |buffer| {
                assert!(buffer.iter().all(|value| value.to_bits() == QUIET_NAN_BITS));
                buffer[0] = 99.0;
                raw
            });
            assert!(matches!(result, Err(RuntimeError::KernelStatus(_))));
            assert!(output.iter().all(|value| value.to_bits() == QUIET_NAN_BITS));
        }
        let mut output = vec![1.0, 2.0, 3.0];
        assert_eq!(
            invoke_with_output_policy(&mut output, |buffer| {
                buffer[1] = 99.0;
                99
            }),
            Err(RuntimeError::UnknownKernelStatus(99))
        );
        assert!(output.iter().all(|value| value.to_bits() == QUIET_NAN_BITS));
    }

    #[test]
    fn successful_output_must_be_complete_and_finite_or_is_scrubbed() {
        let mut complete = vec![9.0, 9.0];
        invoke_with_output_policy(&mut complete, |output| {
            output.copy_from_slice(&[1.0, -0.0]);
            0
        })
        .unwrap();
        assert_eq!(complete, [1.0, -0.0]);

        let mut incomplete = vec![9.0, 9.0];
        assert_eq!(
            invoke_with_output_policy(&mut incomplete, |output| {
                output[0] = 1.0;
                0
            }),
            Err(RuntimeError::InvalidSuccessOutput { index: 1 })
        );
        assert!(
            incomplete
                .iter()
                .all(|value| value.to_bits() == QUIET_NAN_BITS)
        );

        let mut infinite = vec![9.0, 9.0];
        assert_eq!(
            invoke_with_output_policy(&mut infinite, |output| {
                output.copy_from_slice(&[1.0, f32::INFINITY]);
                0
            }),
            Err(RuntimeError::InvalidSuccessOutput { index: 1 })
        );
        assert!(
            infinite
                .iter()
                .all(|value| value.to_bits() == QUIET_NAN_BITS)
        );
    }

    #[test]
    fn repeat_invocation_refills_the_same_output_allocation() {
        let mut output = vec![1.0, 2.0, 3.0];
        let address = output.as_ptr();
        for expected in [4.0, 5.0, 6.0] {
            invoke_with_output_policy(&mut output, |buffer| {
                assert_eq!(buffer.as_ptr(), address);
                assert!(buffer.iter().all(|value| value.to_bits() == QUIET_NAN_BITS));
                buffer.fill(expected);
                0
            })
            .unwrap();
            assert_eq!(output, [expected; 3]);
            assert_eq!(output.as_ptr(), address);
        }
    }
}
