//! Deterministic target-panel packing for `DFQ8_B32_OI4_V1`.

use crate::ir::Q8LinearShape;
use crate::{
    BLOCK_SIZE, OUTPUT_TILE, PAYLOAD_ALIGNMENT, RECORD_BYTES, Result, SCHEMA_VERSION, checked_mul,
    checked_usize, hex_lower, invalid,
};
use decodeforge_core::q8::{self, Q8Weights};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Exact target-independent packed layout name.
pub const PACK_FORMAT: &str = "DFQ8_B32_OI4_V1";
/// Output tile width in one physical record.
pub const PACK_TILE: u32 = OUTPUT_TILE;
/// One physical K block in one record.
pub const PACK_BLOCK_SIZE: u32 = BLOCK_SIZE;
/// Number of bytes in one `(panel, block)` record.
pub const PACK_RECORD_BYTES: u32 = RECORD_BYTES;
/// Required alignment of a payload allocation at the runtime boundary.
pub const PACK_ALIGNMENT: u32 = PAYLOAD_ALIGNMENT;

/// Verified immutable specification of the fixed OI4 payload.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PackSpecV1 {
    schema_version: u32,
    format: String,
    layout: String,
    tile: u32,
    block_size: u32,
    record_bytes: u32,
    alignment: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PackSpecWire {
    schema_version: u32,
    format: String,
    layout: String,
    tile: u32,
    block_size: u32,
    record_bytes: u32,
    alignment: u32,
}

impl<'de> serde::Deserialize<'de> for PackSpecV1 {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = PackSpecWire::deserialize(deserializer)?;
        let spec = Self {
            schema_version: wire.schema_version,
            format: wire.format,
            layout: wire.layout,
            tile: wire.tile,
            block_size: wire.block_size,
            record_bytes: wire.record_bytes,
            alignment: wire.alignment,
        };
        spec.verify().map_err(serde::de::Error::custom)?;
        Ok(spec)
    }
}

impl PackSpecV1 {
    /// Construct the sole supported G1 pack specification.
    pub fn new() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            format: PACK_FORMAT.to_owned(),
            layout: "output-interleaved".to_owned(),
            tile: PACK_TILE,
            block_size: PACK_BLOCK_SIZE,
            record_bytes: PACK_RECORD_BYTES,
            alignment: PACK_ALIGNMENT,
        }
    }

    /// Recheck the fixed pack specification invariants.
    pub fn verify(&self) -> Result<()> {
        if *self != Self::new() {
            return Err(invalid(
                "DFE-COMP-006",
                "PackSpecV1 violates the fixed DFQ8_B32_OI4_V1 contract.",
            ));
        }
        Ok(())
    }

    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn format(&self) -> &str {
        &self.format
    }

    pub fn layout(&self) -> &str {
        &self.layout
    }

    pub const fn tile(&self) -> u32 {
        self.tile
    }

    pub const fn block_size(&self) -> u32 {
        self.block_size
    }

    pub const fn record_bytes(&self) -> u32 {
        self.record_bytes
    }

    pub const fn alignment(&self) -> u32 {
        self.alignment
    }

    /// Required runtime alignment as a host-sized value.
    pub const fn required_alignment(&self) -> usize {
        self.alignment as usize
    }
}

impl Default for PackSpecV1 {
    fn default() -> Self {
        Self::new()
    }
}

/// Structurally validated compact metadata for an OI4 artifact. The payload is
/// intentionally outside this manifest so metadata can be exchanged without
/// copying bytes; byte identities are verified when both parts are supplied.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PackManifestV1 {
    schema_version: u32,
    spec: PackSpecV1,
    shape: Q8LinearShape,
    logical_weight_identity: String,
    packed_identity: String,
    payload_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PackManifestWire {
    schema_version: u32,
    spec: PackSpecV1,
    shape: Q8LinearShape,
    logical_weight_identity: String,
    packed_identity: String,
    payload_bytes: u64,
}

impl<'de> Deserialize<'de> for PackManifestV1 {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = PackManifestWire::deserialize(deserializer)?;
        let manifest = Self {
            schema_version: wire.schema_version,
            spec: wire.spec,
            shape: wire.shape,
            logical_weight_identity: wire.logical_weight_identity,
            packed_identity: wire.packed_identity,
            payload_bytes: wire.payload_bytes,
        };
        manifest.verify().map_err(serde::de::Error::custom)?;
        Ok(manifest)
    }
}

/// Immutable, verified OI4 packed weights with no binary header.
///
/// The owned `Vec<u8>` is a serialization buffer and makes no pointer-alignment
/// promise. The runtime must place these bytes in storage satisfying
/// [`PackSpecV1::required_alignment`] before entering generated code.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedWeightsV1 {
    manifest: PackManifestV1,
    payload: Vec<u8>,
}

impl PackManifestV1 {
    /// Construct validated metadata for an artifact boundary.
    fn from_parts(
        schema_version: u32,
        spec: PackSpecV1,
        shape: Q8LinearShape,
        logical_weight_identity: String,
        packed_identity: String,
        payload_bytes: usize,
    ) -> Result<Self> {
        let payload_bytes = u64::try_from(payload_bytes)
            .map_err(|_| invalid("DFE-COMP-002", "payload length is not representable."))?;
        let manifest = Self {
            schema_version,
            spec,
            shape,
            logical_weight_identity,
            packed_identity,
            payload_bytes,
        };
        manifest.verify()?;
        Ok(manifest)
    }

    /// Recheck metadata invariants. Hash binding to bytes is checked by the
    /// artifact loader, because a manifest does not contain the payload.
    pub fn verify(&self) -> Result<()> {
        self.shape.verify()?;
        self.spec.verify()?;
        if self.schema_version != SCHEMA_VERSION {
            return Err(invalid(
                "DFE-COMP-001",
                "unsupported pack manifest schema version.",
            ));
        }
        if !crate::is_sha256_identity(&self.logical_weight_identity)
            || !crate::is_sha256_identity(&self.packed_identity)
        {
            return Err(invalid(
                "DFE-COMP-004",
                "weight identities must be sha256:<64 lowercase hex digits>.",
            ));
        }
        if self.payload_bytes != self.shape.payload_bytes()? as u64 {
            return Err(invalid(
                "DFE-COMP-003",
                "manifest payload byte count is incorrect.",
            ));
        }
        Ok(())
    }

    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }
    pub const fn spec(&self) -> &PackSpecV1 {
        &self.spec
    }
    pub const fn shape(&self) -> Q8LinearShape {
        self.shape
    }
    pub fn logical_weight_identity(&self) -> &str {
        &self.logical_weight_identity
    }
    pub fn packed_identity(&self) -> &str {
        &self.packed_identity
    }
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// Deterministic compact metadata JSON, with no payload bytes.
    pub fn canonical_json(&self) -> Result<String> {
        self.verify()?;
        serde_json::to_string(self)
            .map_err(|error| invalid("DFE-COMP-006", format!("manifest JSON failed: {error}")))
    }
}

impl PackedWeightsV1 {
    /// Pack immutable G0 Q8 weights into the exact headerless OI4 payload.
    pub fn pack(weights: &Q8Weights) -> Result<Self> {
        let shape = Q8LinearShape::new(weights.n(), weights.k())?;
        let logical_identity = q8::logical_weight_identity(weights);
        let payload = encode_payload(&shape, weights)?;
        let packed_identity =
            compute_packed_identity(&PackSpecV1::new(), &shape, &logical_identity, &payload);
        let manifest = PackManifestV1::from_parts(
            SCHEMA_VERSION,
            PackSpecV1::new(),
            shape,
            logical_identity,
            packed_identity,
            payload.len(),
        )?;
        Self::from_artifact_parts(manifest, payload)
    }

    /// Load an artifact from separately stored manifest metadata and payload.
    ///
    /// This proves the payload's structural integrity and exact binding to its
    /// logical G0 representation. It does not prove which unquantized source
    /// tensor produced those logical Q8 weights, and it does not turn the
    /// serialization buffer into runtime-aligned storage.
    pub fn from_artifact_parts(manifest: PackManifestV1, payload: Vec<u8>) -> Result<Self> {
        validate_artifact_parts(&manifest, &payload)?;
        Ok(Self { manifest, payload })
    }

    /// Schema major version.
    pub const fn schema_version(&self) -> u32 {
        self.manifest.schema_version()
    }

    /// Fixed pack specification.
    pub const fn spec(&self) -> &PackSpecV1 {
        self.manifest.spec()
    }

    /// Packed logical shape.
    pub const fn shape(&self) -> Q8LinearShape {
        self.manifest.shape()
    }

    /// G0 logical-weight identity from which the payload was packed.
    pub fn logical_weight_identity(&self) -> &str {
        self.manifest.logical_weight_identity()
    }

    /// SHA-256 identity of the packed representation.
    pub fn packed_identity(&self) -> &str {
        self.manifest.packed_identity()
    }

    /// Headerless payload bytes.  The slice starts at byte zero of record 0.
    pub fn bytes(&self) -> &[u8] {
        &self.payload
    }

    pub fn len(&self) -> usize {
        self.payload.len()
    }

    pub fn is_empty(&self) -> bool {
        self.payload.is_empty()
    }

    /// Number of physical `(panel, block)` records.
    pub fn record_count(&self) -> usize {
        self.payload.len() / RECORD_BYTES as usize
    }

    /// Required runtime payload alignment, represented in the pack contract.
    pub const fn required_alignment(&self) -> usize {
        PAYLOAD_ALIGNMENT as usize
    }

    /// Byte offset of a record, or `None` for an invalid panel/block index.
    pub fn record_offset(&self, panel: u32, block: u32) -> Option<usize> {
        self.manifest.shape().record_offset(panel, block)
    }

    /// Borrow one exact 144-byte `(panel, block)` record.
    pub fn record(&self, panel: u32, block: u32) -> Option<&[u8]> {
        let offset = self.record_offset(panel, block)?;
        let end = offset.checked_add(RECORD_BYTES as usize)?;
        self.payload.get(offset..end)
    }

    /// Iterate records in canonical panel-major/block-minor order.
    pub fn records(&self) -> impl Iterator<Item = &[u8]> {
        (0..self.payload.len() / RECORD_BYTES as usize).map(|index| {
            let start = index * RECORD_BYTES as usize;
            &self.payload[start..start + RECORD_BYTES as usize]
        })
    }

    /// Read one little-endian scale word from a record.
    pub fn scale_bits_at(&self, panel: u32, block: u32, lane: u32) -> Option<u32> {
        if lane >= OUTPUT_TILE {
            return None;
        }
        let record = self.record(panel, block)?;
        let start = (lane as usize).checked_mul(4)?;
        let bytes: [u8; 4] = record.get(start..start + 4)?.try_into().ok()?;
        Some(u32::from_le_bytes(bytes))
    }

    /// Read one signed q value from the interleaved record body.
    pub fn q_at(&self, panel: u32, block: u32, lane: u32, output_lane: u32) -> Option<i8> {
        if lane >= BLOCK_SIZE || output_lane >= OUTPUT_TILE {
            return None;
        }
        let record = self.record(panel, block)?;
        let offset = 16usize.checked_add((lane as usize).checked_mul(4)?)?;
        Some(record.get(offset + output_lane as usize).copied()? as i8)
    }

    /// Borrowing verification of lengths, zero padding, q/scales, and hashes.
    pub fn verify(&self) -> Result<()> {
        validate_artifact_parts(&self.manifest, &self.payload)
    }

    /// Borrow the validated metadata associated with this payload.
    pub const fn manifest(&self) -> &PackManifestV1 {
        &self.manifest
    }

    /// Deterministic compact metadata JSON, with no payload bytes.
    pub fn canonical_manifest_json(&self) -> Result<String> {
        self.manifest.canonical_json()
    }
}

impl TryFrom<&Q8Weights> for PackedWeightsV1 {
    type Error = crate::CompilerError;

    fn try_from(weights: &Q8Weights) -> Result<Self> {
        Self::pack(weights)
    }
}

fn validate_artifact_parts(manifest: &PackManifestV1, payload: &[u8]) -> Result<()> {
    manifest.verify()?;
    let expected_len = manifest.shape.payload_bytes()?;
    if payload.len() != expected_len {
        return Err(invalid(
            "DFE-COMP-003",
            format!(
                "packed payload length {} does not equal expected {expected_len}.",
                payload.len()
            ),
        ));
    }
    verify_payload(&manifest.shape, payload)?;
    let streamed_logical = compute_logical_identity_from_payload(&manifest.shape, payload)?;
    if manifest.logical_weight_identity != streamed_logical {
        return Err(invalid(
            "DFE-COMP-004",
            "logical weight identity does not match the OI4 payload.",
        ));
    }
    let expected_identity = compute_packed_identity(
        &manifest.spec,
        &manifest.shape,
        &manifest.logical_weight_identity,
        payload,
    );
    if manifest.packed_identity != expected_identity {
        return Err(invalid(
            "DFE-COMP-005",
            "packed identity does not match the spec, dimensions, identity, and payload.",
        ));
    }
    Ok(())
}

/// Independent artifact verifier for the frozen G0 v1 logical identity
/// preimage. OI4 stores all logical q/scales, so this streams them by index
/// without reconstructing a `Q8Weights` allocation.
fn compute_logical_identity_from_payload(shape: &Q8LinearShape, payload: &[u8]) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"DecodeForge/DFQ8_B32_V1/logical-weight/v1\0");
    hasher.update(shape.n().to_le_bytes());
    hasher.update(shape.k().to_le_bytes());
    hasher.update(shape.blocks().to_le_bytes());
    for row in 0..shape.n() {
        let panel = row / OUTPUT_TILE;
        let output_lane = row % OUTPUT_TILE;
        for block in 0..shape.blocks() {
            let offset = shape.checked_record_offset(panel, block)?;
            for lane in 0..BLOCK_SIZE {
                let q_offset = offset + 16 + lane as usize * 4 + output_lane as usize;
                hasher.update([payload[q_offset]]);
            }
        }
    }
    for row in 0..shape.n() {
        let panel = row / OUTPUT_TILE;
        let output_lane = row % OUTPUT_TILE;
        for block in 0..shape.blocks() {
            let offset = shape.checked_record_offset(panel, block)?;
            let scale_offset = offset + output_lane as usize * 4;
            hasher.update(&payload[scale_offset..scale_offset + 4]);
        }
    }
    Ok(format!("sha256:{}", hex_lower(&hasher.finalize())))
}

fn encode_payload(shape: &Q8LinearShape, weights: &Q8Weights) -> Result<Vec<u8>> {
    if weights.n() != shape.n() || weights.k() != shape.k() {
        return Err(invalid(
            "DFE-COMP-003",
            "Q8 weights dimensions do not match the requested pack shape.",
        ));
    }
    let payload_len = shape.payload_bytes()?;
    let mut payload = Vec::new();
    payload.try_reserve_exact(payload_len).map_err(|_| {
        invalid(
            "DFE-COMP-002",
            "packed payload allocation is not representable or available.",
        )
    })?;
    payload.resize(payload_len, 0);

    for panel in 0..shape.panels() {
        for block in 0..shape.blocks() {
            let offset = shape
                .record_offset(panel, block)
                .ok_or_else(|| invalid("DFE-COMP-002", "packed record offset overflows."))?;
            for lane in 0..OUTPUT_TILE {
                let row = panel
                    .checked_mul(OUTPUT_TILE)
                    .and_then(|base| base.checked_add(lane))
                    .ok_or_else(|| invalid("DFE-COMP-002", "packed row offset overflows."))?;
                let scale = if row < shape.n() {
                    weights.scale_at(row, block).ok_or_else(|| {
                        invalid("DFE-COMP-003", "Q8 scale index is outside logical storage.")
                    })?
                } else {
                    0
                };
                let scale_offset = offset + (lane as usize) * 4;
                payload[scale_offset..scale_offset + 4].copy_from_slice(&scale.to_le_bytes());
            }
            for logical_lane in 0..BLOCK_SIZE {
                let logical_k = block
                    .checked_mul(BLOCK_SIZE)
                    .and_then(|base| base.checked_add(logical_lane))
                    .ok_or_else(|| invalid("DFE-COMP-002", "packed K offset overflows."))?;
                for output_lane in 0..OUTPUT_TILE {
                    let row = panel
                        .checked_mul(OUTPUT_TILE)
                        .and_then(|base| base.checked_add(output_lane))
                        .ok_or_else(|| invalid("DFE-COMP-002", "packed row offset overflows."))?;
                    let q = if row < shape.n() && logical_k < shape.k() {
                        weights
                            .q_at(row, block, logical_lane as usize)
                            .ok_or_else(|| {
                                invalid("DFE-COMP-003", "Q8 q index is outside logical storage.")
                            })? as u8
                    } else {
                        0
                    };
                    let q_offset = offset + 16 + (logical_lane as usize) * 4 + output_lane as usize;
                    payload[q_offset] = q;
                }
            }
        }
    }
    Ok(payload)
}

fn verify_payload(shape: &Q8LinearShape, payload: &[u8]) -> Result<()> {
    let expected_len = shape.payload_bytes()?;
    if payload.len() != expected_len {
        return Err(invalid(
            "DFE-COMP-003",
            "packed payload has the wrong length.",
        ));
    }
    for panel in 0..shape.panels() {
        for block in 0..shape.blocks() {
            let offset = shape
                .record_offset(panel, block)
                .ok_or_else(|| invalid("DFE-COMP-002", "packed record offset overflows."))?;
            let record = payload
                .get(offset..offset + RECORD_BYTES as usize)
                .ok_or_else(|| invalid("DFE-COMP-003", "packed record is truncated."))?;
            for lane in 0..OUTPUT_TILE {
                let scale_offset = lane as usize * 4;
                let bits = u32::from_le_bytes(
                    record[scale_offset..scale_offset + 4]
                        .try_into()
                        .map_err(|_| invalid("DFE-COMP-003", "scale record is truncated."))?,
                );
                if !q8::is_finite_f32_bits(bits) || bits & 0x8000_0000 != 0 {
                    return Err(invalid(
                        "DFE-COMP-005",
                        "packed scale is not finite and non-negative.",
                    ));
                }
                let row = panel * OUTPUT_TILE + lane;
                if row >= shape.n() && bits != 0 {
                    return Err(invalid("DFE-COMP-005", "missing output scale is not +0."));
                }
                for logical_lane in 0..BLOCK_SIZE {
                    let q_offset = 16 + logical_lane as usize * 4 + lane as usize;
                    let raw = record[q_offset];
                    let logical_k = block * BLOCK_SIZE + logical_lane;
                    if raw == 0x80 {
                        return Err(invalid("DFE-COMP-005", "packed q contains forbidden -128."));
                    }
                    if (row >= shape.n() || logical_k >= shape.k()) && raw != 0 {
                        return Err(invalid("DFE-COMP-005", "packed q padding is not zero."));
                    }
                    if bits == 0 && raw != 0 {
                        return Err(invalid("DFE-COMP-005", "zero-scale q lane is not zero."));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Compute the domain-separated packed identity for fixed OI4 bytes.
///
/// The preimage is:
/// `domain || logical_identity || N || K || B || P || T || block_size ||
/// record_bytes || alignment || layout || payload`, with all integer fields
/// little-endian and an explicit NUL after the layout string.
fn compute_packed_identity(
    spec: &PackSpecV1,
    shape: &Q8LinearShape,
    logical_weight_identity: &str,
    payload: &[u8],
) -> String {
    debug_assert_eq!(spec.layout(), "output-interleaved");
    let mut hasher = Sha256::new();
    hasher.update(b"DecodeForge/DFQ8_B32_OI4_V1/packed-weight/v1\0");
    hasher.update(logical_weight_identity.as_bytes());
    hasher.update(shape.n().to_le_bytes());
    hasher.update(shape.k().to_le_bytes());
    hasher.update(shape.blocks().to_le_bytes());
    hasher.update(shape.panels().to_le_bytes());
    hasher.update(PACK_TILE.to_le_bytes());
    hasher.update(PACK_BLOCK_SIZE.to_le_bytes());
    hasher.update(PACK_RECORD_BYTES.to_le_bytes());
    hasher.update(PACK_ALIGNMENT.to_le_bytes());
    hasher.update(spec.layout().as_bytes());
    hasher.update([0]);
    hasher.update(payload);
    format!("sha256:{}", hex_lower(&hasher.finalize()))
}

/// Checked expected payload byte count for callers that only have dimensions.
pub fn expected_payload_bytes(n: u32, k: u32) -> Result<usize> {
    Q8LinearShape::new(n, k)?.payload_bytes()
}

// Keep the checked helpers in this module close to the layout arithmetic.  The
// function pointers prevent accidental changes from silently dropping checks.
const _: fn(u64, u64, &str) -> Result<u64> = checked_mul;
const _: fn(u64, &str) -> Result<usize> = checked_usize;

#[cfg(test)]
mod tests {
    use super::*;
    use decodeforge_core::q8::Q8Weights;

    fn sample_weights(n: u32, k: u32) -> Q8Weights {
        let blocks = k.div_ceil(32);
        let mut q = vec![0u8; n as usize * blocks as usize * 32];
        let mut scales = vec![0u32; n as usize * blocks as usize];
        for row in 0..n as usize {
            for lane in 0..k as usize {
                let block = lane / 32;
                let inner = lane % 32;
                q[(row * blocks as usize + block) * 32 + inner] =
                    ([-127i16, -1, 0, 1, 127][(row + lane) % 5]) as i8 as u8;
                scales[row * blocks as usize + block] = 0x3f80_0000;
            }
        }
        Q8Weights::try_new(n, k, blocks, q, scales).unwrap()
    }

    #[test]
    fn exact_record_offsets_match_hand_labeled_n5_k33() {
        let shape = Q8LinearShape::new(5, 33).unwrap();
        assert_eq!(shape.panels(), 2);
        assert_eq!(shape.blocks(), 2);
        assert_eq!(shape.payload_bytes().unwrap(), 576);
        assert_eq!(shape.record_offset(0, 0), Some(0));
        assert_eq!(shape.record_offset(0, 1), Some(144));
        assert_eq!(shape.record_offset(1, 0), Some(288));
        assert_eq!(shape.record_offset(1, 1), Some(432));
    }

    #[test]
    fn missing_output_and_k_padding_are_zero() {
        let weights = sample_weights(5, 33);
        let packed = PackedWeightsV1::pack(&weights).unwrap();
        for block in 0..2 {
            assert_eq!(packed.scale_bits_at(1, block, 1), Some(0));
            assert_eq!(packed.q_at(1, block, 0, 1), Some(0));
            assert_eq!(packed.q_at(1, block, 1, 3), Some(0));
        }
        for lane in 1..4 {
            assert_eq!(packed.q_at(0, 1, 1, lane), Some(0));
        }
    }

    #[test]
    fn all_non_padding_sign_bytes_are_preserved() {
        let weights = sample_weights(1, 5);
        let packed = PackedWeightsV1::pack(&weights).unwrap();
        assert_eq!(packed.q_at(0, 0, 0, 0), Some(-127));
        assert_eq!(packed.q_at(0, 0, 1, 0), Some(-1));
        assert_eq!(packed.q_at(0, 0, 2, 0), Some(0));
        assert_eq!(packed.q_at(0, 0, 3, 0), Some(1));
        assert_eq!(packed.q_at(0, 0, 4, 0), Some(127));
    }

    #[test]
    fn packed_identity_is_sensitive_to_payload_shape_and_layout() {
        let weights = sample_weights(1, 1);
        let packed = PackedWeightsV1::pack(&weights).unwrap();
        let mut changed = packed.bytes().to_vec();
        changed[0] ^= 1;
        let payload_identity = compute_packed_identity(
            packed.spec(),
            &packed.shape(),
            packed.logical_weight_identity(),
            &changed,
        );
        assert_ne!(payload_identity, packed.packed_identity());

        let other_shape = Q8LinearShape::new(1, 2).unwrap();
        assert_ne!(
            compute_packed_identity(
                packed.spec(),
                &other_shape,
                packed.logical_weight_identity(),
                packed.bytes(),
            ),
            packed.packed_identity()
        );
        assert_eq!(packed.manifest().spec().layout(), "output-interleaved");
    }

    #[test]
    fn artifact_loader_rejects_logical_identity_mixed_with_other_payload() {
        let first = PackedWeightsV1::pack(&sample_weights(1, 1)).unwrap();
        let mut second_q = vec![0_u8; 32];
        second_q[0] = 1;
        let second_weights = Q8Weights::try_new(1, 1, 1, second_q, vec![0x3f80_0000]).unwrap();
        let second = PackedWeightsV1::pack(&second_weights).unwrap();

        assert_ne!(
            first.logical_weight_identity(),
            second.logical_weight_identity()
        );
        assert!(
            PackedWeightsV1::from_artifact_parts(
                second.manifest().clone(),
                first.bytes().to_vec(),
            )
            .is_err()
        );
    }

    #[test]
    fn manifest_serialization_is_compact_and_validated() {
        let packed = PackedWeightsV1::pack(&sample_weights(1, 1)).unwrap();
        let json = packed.canonical_manifest_json().unwrap();
        assert!(!json.contains("\"payload\":"));
        let manifest: PackManifestV1 = serde_json::from_str(&json).unwrap();
        manifest.verify().unwrap();
        assert!(PackedWeightsV1::from_artifact_parts(manifest, packed.bytes().to_vec()).is_ok());
    }

    #[test]
    fn all_requested_n_and_k_tails_have_exact_payload_lengths() {
        for n in [1, 2, 3, 4, 5, 255] {
            for k in [1, 2, 31, 32, 33, 63, 64, 65] {
                let weights = sample_weights(n, k);
                let packed = PackedWeightsV1::pack(&weights).unwrap();
                let expected_records = n.div_ceil(4) * k.div_ceil(32);
                assert_eq!(packed.record_count(), expected_records as usize);
                assert_eq!(packed.len(), expected_records as usize * 144);
                packed.verify().unwrap();
            }
        }
    }

    #[test]
    fn all_committed_fixtures_pack_deterministically() {
        let documents = decodeforge_core::q8::fixture::generated_documents().unwrap();
        assert_eq!(documents.len(), 16);
        for (case_id, bytes) in documents {
            let fixture = decodeforge_core::q8::fixture::parse_quant_fixture(&bytes).unwrap();
            let q_bytes = fixture
                .expected_q_bytes
                .iter()
                .map(|value| *value as i8 as u8)
                .collect();
            let weights = Q8Weights::try_new(
                fixture.n,
                fixture.k,
                fixture.blocks,
                q_bytes,
                fixture.expected_scale_bits.clone(),
            )
            .unwrap();
            let first = PackedWeightsV1::pack(&weights).unwrap();
            assert_eq!(
                first.logical_weight_identity(),
                q8::logical_weight_identity(&weights)
            );
            assert_eq!(
                compute_logical_identity_from_payload(&first.shape(), first.bytes()).unwrap(),
                q8::logical_weight_identity(&weights)
            );
            let second = PackedWeightsV1::pack(&weights).unwrap();
            assert_eq!(first.bytes(), second.bytes(), "fixture {case_id}");
            assert_eq!(
                first.packed_identity(),
                second.packed_identity(),
                "fixture {case_id}"
            );
        }
    }
}
