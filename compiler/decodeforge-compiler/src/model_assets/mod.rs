//! Bounded preparation of model weights for the native Q8 runtime bridge.
//!
//! The G3.1 checkpoint prepares exactly 22 named rank-two BF16/F32 query
//! projections from one pinned safetensors model. The source is memory mapped
//! so a multi-gigabyte model is never copied into a second in-memory buffer.
//! Each selected tensor flows through the frozen G0 quantizer, G1 OI4 packer,
//! generated-module identity, and canonical same-Q8 dequantizer before the
//! complete inventory is atomically published.

use crate::{
    KernelVariant, LoopKernelV1, NEON_C_SOURCE_FORMAT_V1, PackedWeightsV1, Q8LinearRegion, Result,
    emit_neon_c, expected_payload_bytes, invalid,
};
use decodeforge_core::q8::{self, dequantize_f32_bits, quantize_f32_bits};
use memmap2::{Mmap, MmapOptions};
use rustix::fs::{FlockOperation, Mode, OFlags, RenameFlags, flock, open, renameat_with};
use safetensors::{Dtype, SafeTensors};
use serde::de::{IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

#[cfg(not(unix))]
compile_error!("DecodeForge model asset preparation currently requires a Unix host");

/// Closed format of the top-level prepared-asset manifest.
pub const ASSET_BUNDLE_FORMAT_V1: &str = "decodeforge_q8_linear_asset_v1";
/// Closed format of the ordered all-query-projection inventory.
pub const Q_PROJ_INVENTORY_FORMAT_V1: &str = "decodeforge_q_proj_inventory_v1";
/// Pinned Hugging Face model identifier for the first prompt-to-text target.
pub const TINYLLAMA_MODEL_ID: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
/// Exact TinyLlama source revision used by DecodeForge evidence.
pub const TINYLLAMA_MODEL_REVISION: &str = "fe8a4ea1ffedaf415f4da2f062534de366a451e6";

const TINYLLAMA_FILENAME: &str = "model.safetensors";
const TINYLLAMA_SOURCE_BYTES: u64 = 2_200_119_864;
const TINYLLAMA_SOURCE_IDENTITY: &str =
    "sha256:6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933";
const TINYLLAMA_Q_PROJ_0_IDENTITY: &str =
    "sha256:5abf98c51f903941a1592f3df83e2e56ca7149252f5d6665c7662927c83008ac";
const TINYLLAMA_LAYER_COUNT: u32 = 22;
const TINYLLAMA_HIDDEN_SIZE: u32 = 2048;

const MANIFEST_FILENAME: &str = "manifest.json";
const PACK_MANIFEST_FILENAME: &str = "pack-manifest.json";
const PACK_PAYLOAD_FILENAME: &str = "weights.oi4.bin";
const FALLBACK_FILENAME: &str = "fallback.f32.bin";
const INVENTORY_FILENAME: &str = "inventory.json";
const LAYERS_DIRECTORY: &str = "layers";
const FALLBACK_FORMAT_V1: &str = "decodeforge_row_major_f32_le_v1";
const PREPARER_NAME: &str = "decodeforge-compiler";
const MAX_SOURCE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;
const MAX_TENSOR_ELEMENTS: u64 = 16 * 1024 * 1024;
const MAX_PACKED_BYTES: usize = 128 * 1024 * 1024;
const MAX_MANIFEST_BYTES: usize = 64 * 1024;
const MAX_PACK_MANIFEST_BYTES: usize = 16 * 1024;
const MAX_FALLBACK_BYTES: usize = 64 * 1024 * 1024;
const MAX_INVENTORY_BYTES: usize = 256 * 1024;

/// Pinned identity and human-readable provenance for one safetensors file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelSourceSpecV1 {
    model_id: String,
    revision: String,
    filename: String,
    source_bytes: u64,
    source_identity: String,
}

impl ModelSourceSpecV1 {
    /// Construct a checked source specification.
    pub fn new(
        model_id: impl Into<String>,
        revision: impl Into<String>,
        filename: impl Into<String>,
        source_bytes: u64,
        source_identity: impl Into<String>,
    ) -> Result<Self> {
        let result = Self {
            model_id: model_id.into(),
            revision: revision.into(),
            filename: filename.into(),
            source_bytes,
            source_identity: source_identity.into(),
        };
        result.verify()?;
        Ok(result)
    }

    fn verify(&self) -> Result<()> {
        validate_text("model ID", &self.model_id, 256)?;
        validate_lower_hex("revision", &self.revision, 40)?;
        validate_filename(&self.filename)?;
        if !(9..=MAX_SOURCE_BYTES).contains(&self.source_bytes) {
            return Err(invalid(
                "DFE-ASSET-003",
                "source byte count is outside the supported bound.",
            ));
        }
        validate_sha256_identity("source identity", &self.source_identity)
    }
}

/// Expected shape and optional independently pinned raw identity for one
/// Q8Linear weight tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q8LinearTensorSpecV1 {
    name: String,
    n: u32,
    k: u32,
    expected_data_identity: Option<String>,
    allow_f32: bool,
}

impl Q8LinearTensorSpecV1 {
    /// Construct a checked rank-two BF16 tensor specification.
    pub fn bf16(
        name: impl Into<String>,
        n: u32,
        k: u32,
        expected_data_identity: Option<String>,
    ) -> Result<Self> {
        let result = Self {
            name: name.into(),
            n,
            k,
            expected_data_identity,
            allow_f32: false,
        };
        result.verify()?;
        Ok(result)
    }

    /// Construct a checked tensor specification accepting either BF16 or
    /// little-endian binary32 source storage. Both routes produce the same
    /// explicit binary32 words before canonical quantization.
    pub fn f32_or_bf16(
        name: impl Into<String>,
        n: u32,
        k: u32,
        expected_data_identity: Option<String>,
    ) -> Result<Self> {
        let mut result = Self::bf16(name, n, k, expected_data_identity)?;
        result.allow_f32 = true;
        Ok(result)
    }

    fn verify(&self) -> Result<()> {
        validate_tensor_name(&self.name)?;
        let elements = u64::from(self.n)
            .checked_mul(u64::from(self.k))
            .ok_or_else(|| invalid("DFE-ASSET-003", "tensor element count overflows."))?;
        if self.n == 0 || self.k == 0 || elements > MAX_TENSOR_ELEMENTS {
            return Err(invalid(
                "DFE-ASSET-003",
                "tensor shape is empty or exceeds the element bound.",
            ));
        }
        let payload_bytes = expected_payload_bytes(self.n, self.k)?;
        if payload_bytes > MAX_PACKED_BYTES {
            return Err(invalid(
                "DFE-ASSET-003",
                "tensor produces a packed payload above the runtime bridge bound.",
            ));
        }
        if let Some(identity) = &self.expected_data_identity {
            validate_sha256_identity("tensor identity", identity)?;
        }
        Ok(())
    }
}

/// Produce the pinned source and tensor specification for one TinyLlama
/// attention query projection. Layer zero additionally carries the independent
/// raw-tensor hash already frozen by G1; all layers remain transitively pinned
/// by the exact full-model hash.
pub fn tinyllama_q_proj_spec_v1(layer: u32) -> Result<(ModelSourceSpecV1, Q8LinearTensorSpecV1)> {
    if layer >= TINYLLAMA_LAYER_COUNT {
        return Err(invalid(
            "DFE-ASSET-001",
            format!("TinyLlama q_proj layer must be in 0..{TINYLLAMA_LAYER_COUNT}."),
        ));
    }
    let source = ModelSourceSpecV1::new(
        TINYLLAMA_MODEL_ID,
        TINYLLAMA_MODEL_REVISION,
        TINYLLAMA_FILENAME,
        TINYLLAMA_SOURCE_BYTES,
        TINYLLAMA_SOURCE_IDENTITY,
    )?;
    let expected_data_identity = (layer == 0).then(|| TINYLLAMA_Q_PROJ_0_IDENTITY.to_owned());
    let tensor = Q8LinearTensorSpecV1::f32_or_bf16(
        format!("model.layers.{layer}.self_attn.q_proj.weight"),
        TINYLLAMA_HIDDEN_SIZE,
        TINYLLAMA_HIDDEN_SIZE,
        expected_data_identity,
    )?;
    Ok((source, tensor))
}

/// Produce the one pinned source specification and the complete ordered set of
/// 22 TinyLlama query-projection tensor specifications.
pub fn tinyllama_q_proj_specs_v1() -> Result<(ModelSourceSpecV1, Vec<Q8LinearTensorSpecV1>)> {
    let (source, first) = tinyllama_q_proj_spec_v1(0)?;
    let mut tensors = Vec::with_capacity(TINYLLAMA_LAYER_COUNT as usize);
    tensors.push(first);
    for layer in 1..TINYLLAMA_LAYER_COUNT {
        let (candidate_source, tensor) = tinyllama_q_proj_spec_v1(layer)?;
        if candidate_source != source {
            return Err(invalid(
                "DFE-ASSET-001",
                "TinyLlama layer specifications do not share one source identity.",
            ));
        }
        tensors.push(tensor);
    }
    Ok((source, tensors))
}

/// Immutable source provenance carried by every prepared bundle.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceProvenanceV1 {
    pub model_id: String,
    pub revision: String,
    pub filename: String,
    pub bytes: u64,
    pub identity: String,
}

/// Selected tensor provenance carried by every prepared bundle.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorProvenanceV1 {
    pub name: String,
    pub dtype: String,
    pub shape: [u32; 2],
    pub data_bytes: u64,
    pub data_identity: String,
}

/// Frozen quantizer identity carried by every prepared bundle.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationProvenanceV1 {
    pub format: String,
    pub numeric_mode: String,
    pub logical_weight_identity: String,
}

/// File identities and canonical G1 packed identity for a prepared bundle.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackArtifactV1 {
    pub format: String,
    pub manifest_file: String,
    pub manifest_bytes: u64,
    pub manifest_identity: String,
    pub payload_file: String,
    pub payload_bytes: u64,
    pub payload_identity: String,
    pub packed_weight_identity: String,
}

/// Exact generated-module identity shared by shape-compatible prepared assets.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModuleArtifactV1 {
    pub variant: String,
    pub source_format: String,
    pub identity: String,
}

/// Canonically dequantized FP32 tensor used by the G3 same-Q8 fallback.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FallbackArtifactV1 {
    pub format: String,
    pub dtype: String,
    pub file: String,
    pub bytes: u64,
    pub identity: String,
    pub parent_logical_weight_identity: String,
    pub parent_packed_weight_identity: String,
}

/// Version of the canonical preparation implementation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolProvenanceV1 {
    pub name: String,
    pub version: String,
}

/// Deterministic top-level manifest for one Q8Linear weight artifact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetManifestV1 {
    pub schema_version: u32,
    pub format: String,
    pub operator: String,
    pub source: SourceProvenanceV1,
    pub tensor: TensorProvenanceV1,
    pub quantization: QuantizationProvenanceV1,
    pub pack: PackArtifactV1,
    pub module: ModuleArtifactV1,
    pub fallback: FallbackArtifactV1,
    pub tool: ToolProvenanceV1,
}

impl AssetManifestV1 {
    /// Recheck all self-contained manifest invariants.
    pub fn verify(&self) -> Result<()> {
        if self.schema_version != 1
            || self.format != ASSET_BUNDLE_FORMAT_V1
            || self.operator != "Q8Linear"
        {
            return Err(invalid(
                "DFE-ASSET-007",
                "asset manifest version, format, or operator is unsupported.",
            ));
        }
        validate_text("model ID", &self.source.model_id, 256)?;
        validate_lower_hex("revision", &self.source.revision, 40)?;
        validate_filename(&self.source.filename)?;
        validate_sha256_identity("source identity", &self.source.identity)?;
        if !(9..=MAX_SOURCE_BYTES).contains(&self.source.bytes) {
            return Err(invalid(
                "DFE-ASSET-007",
                "source byte count is outside the supported bound.",
            ));
        }
        validate_tensor_name(&self.tensor.name)?;
        let element_bytes = match self.tensor.dtype.as_str() {
            "BF16" => 2,
            "F32" => 4,
            _ => {
                return Err(invalid(
                    "DFE-ASSET-007",
                    "prepared tensor dtype must be BF16 or F32.",
                ));
            }
        };
        let [n, k] = self.tensor.shape;
        let tensor_bytes = u64::from(n)
            .checked_mul(u64::from(k))
            .and_then(|value| value.checked_mul(element_bytes))
            .ok_or_else(|| invalid("DFE-ASSET-003", "tensor byte count overflows."))?;
        let elements = u64::from(n)
            .checked_mul(u64::from(k))
            .ok_or_else(|| invalid("DFE-ASSET-003", "tensor element count overflows."))?;
        if n == 0
            || k == 0
            || elements > MAX_TENSOR_ELEMENTS
            || self.tensor.data_bytes != tensor_bytes
        {
            return Err(invalid(
                "DFE-ASSET-007",
                "tensor shape and byte count do not agree.",
            ));
        }
        validate_sha256_identity("tensor identity", &self.tensor.data_identity)?;
        if self.quantization.format != q8::FORMAT
            || self.quantization.numeric_mode != q8::NUMERIC_MODE
        {
            return Err(invalid(
                "DFE-ASSET-007",
                "quantization format or numeric mode is unsupported.",
            ));
        }
        validate_sha256_identity(
            "logical weight identity",
            &self.quantization.logical_weight_identity,
        )?;
        if self.pack.format != crate::PACK_FORMAT
            || self.pack.manifest_file != PACK_MANIFEST_FILENAME
            || self.pack.payload_file != PACK_PAYLOAD_FILENAME
        {
            return Err(invalid(
                "DFE-ASSET-007",
                "pack format or artifact filename is unsupported.",
            ));
        }
        if self.pack.manifest_bytes == 0
            || self.pack.manifest_bytes > MAX_PACK_MANIFEST_BYTES as u64
            || self.pack.payload_bytes != expected_payload_bytes(n, k)? as u64
            || self.pack.payload_bytes > MAX_PACKED_BYTES as u64
        {
            return Err(invalid(
                "DFE-ASSET-007",
                "pack manifest or payload byte count is invalid.",
            ));
        }
        validate_sha256_identity("pack manifest identity", &self.pack.manifest_identity)?;
        validate_sha256_identity("pack payload identity", &self.pack.payload_identity)?;
        validate_sha256_identity("packed weight identity", &self.pack.packed_weight_identity)?;
        if self.module.variant != "neon" || self.module.source_format != NEON_C_SOURCE_FORMAT_V1 {
            return Err(invalid(
                "DFE-ASSET-007",
                "generated module variant or source format is unsupported.",
            ));
        }
        validate_sha256_identity("module identity", &self.module.identity)?;
        let fallback_bytes = u64::from(n)
            .checked_mul(u64::from(k))
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| invalid("DFE-ASSET-003", "fallback tensor byte count overflows."))?;
        if self.fallback.format != FALLBACK_FORMAT_V1
            || self.fallback.dtype != "F32"
            || self.fallback.file != FALLBACK_FILENAME
            || self.fallback.bytes != fallback_bytes
            || self.fallback.bytes > MAX_FALLBACK_BYTES as u64
            || self.fallback.parent_logical_weight_identity
                != self.quantization.logical_weight_identity
            || self.fallback.parent_packed_weight_identity != self.pack.packed_weight_identity
        {
            return Err(invalid(
                "DFE-ASSET-007",
                "fallback tensor metadata is not bound to the canonical Q8 pack.",
            ));
        }
        validate_sha256_identity("fallback tensor identity", &self.fallback.identity)?;
        if self.tool.name != PREPARER_NAME || self.tool.version != env!("CARGO_PKG_VERSION") {
            return Err(invalid(
                "DFE-ASSET-007",
                "asset preparation tool identity is unsupported.",
            ));
        }
        Ok(())
    }
}

/// Successful preparation result. The manifest does not contain a local input
/// path, so its bytes remain reproducible across machines and checkouts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedAssetV1 {
    pub manifest: AssetManifestV1,
    pub output_directory: PathBuf,
}

/// A prepared asset whose four on-disk files have been bounded, hashed, and
/// cross-checked through the canonical OI4 artifact loader.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedAssetV1 {
    pub manifest: AssetManifestV1,
    pub packed: PackedWeightsV1,
    pub fallback_bits: Vec<u32>,
}

/// One ordered layer record in the all-query-projection inventory.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QProjInventoryEntryV1 {
    pub layer: u32,
    pub directory: String,
    pub manifest_identity: String,
    pub tensor_name: String,
    pub tensor_identity: String,
    pub logical_weight_identity: String,
    pub packed_weight_identity: String,
    pub packed_bytes: u64,
    pub module_identity: String,
    pub fallback_identity: String,
    pub fallback_bytes: u64,
}

/// Deterministic, closed inventory of exactly 22 prepared query projections.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QProjInventoryV1 {
    pub schema_version: u32,
    pub format: String,
    pub source: SourceProvenanceV1,
    pub layer_count: u32,
    pub entries: Vec<QProjInventoryEntryV1>,
    pub total_packed_bytes: u64,
    pub total_fallback_bytes: u64,
    pub aggregate_identity: String,
}

#[derive(Serialize)]
struct QProjInventoryPreimage<'a> {
    schema_version: u32,
    format: &'a str,
    source: &'a SourceProvenanceV1,
    layer_count: u32,
    entries: &'a [QProjInventoryEntryV1],
    total_packed_bytes: u64,
    total_fallback_bytes: u64,
}

impl QProjInventoryV1 {
    /// Recheck ordering, aggregate byte counts, and the domain-separated
    /// identity over every entry.
    pub fn verify(&self) -> Result<()> {
        if self.schema_version != 1
            || self.format != Q_PROJ_INVENTORY_FORMAT_V1
            || self.layer_count != TINYLLAMA_LAYER_COUNT
            || self.entries.len() != TINYLLAMA_LAYER_COUNT as usize
        {
            return Err(invalid(
                "DFE-ASSET-007",
                "q_proj inventory must contain exactly 22 version-one entries.",
            ));
        }
        validate_source_provenance(&self.source)?;
        let mut names = BTreeSet::new();
        let mut directories = BTreeSet::new();
        let mut packed_bytes = 0u64;
        let mut fallback_bytes = 0u64;
        for (index, entry) in self.entries.iter().enumerate() {
            let layer = index as u32;
            let expected_name = q_proj_tensor_name(layer);
            let expected_directory = q_proj_directory(layer);
            if entry.layer != layer
                || entry.tensor_name != expected_name
                || entry.directory != expected_directory
                || !names.insert(&entry.tensor_name)
                || !directories.insert(&entry.directory)
            {
                return Err(invalid(
                    "DFE-ASSET-007",
                    "q_proj inventory entries are not the exact ordered layer set.",
                ));
            }
            for (label, identity) in [
                ("asset manifest identity", &entry.manifest_identity),
                ("tensor identity", &entry.tensor_identity),
                ("logical weight identity", &entry.logical_weight_identity),
                ("packed weight identity", &entry.packed_weight_identity),
                ("module identity", &entry.module_identity),
                ("fallback identity", &entry.fallback_identity),
            ] {
                validate_sha256_identity(label, identity)?;
            }
            packed_bytes = packed_bytes
                .checked_add(entry.packed_bytes)
                .ok_or_else(|| {
                    invalid("DFE-ASSET-003", "aggregate packed byte count overflows.")
                })?;
            fallback_bytes = fallback_bytes
                .checked_add(entry.fallback_bytes)
                .ok_or_else(|| {
                    invalid("DFE-ASSET-003", "aggregate fallback byte count overflows.")
                })?;
        }
        if packed_bytes != self.total_packed_bytes || fallback_bytes != self.total_fallback_bytes {
            return Err(invalid(
                "DFE-ASSET-007",
                "q_proj inventory aggregate byte counts do not match its entries.",
            ));
        }
        validate_sha256_identity("inventory aggregate identity", &self.aggregate_identity)?;
        if self.aggregate_identity != inventory_identity(self)? {
            return Err(invalid(
                "DFE-ASSET-007",
                "q_proj inventory aggregate identity does not match its ordered entries.",
            ));
        }
        Ok(())
    }
}

/// Successful atomic preparation of the ordered all-layer inventory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedQProjInventoryV1 {
    pub inventory: QProjInventoryV1,
    pub output_directory: PathBuf,
}

/// Successfully verified all-layer inventory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedQProjInventoryV1 {
    pub inventory: QProjInventoryV1,
}

struct AssetMaterial {
    manifest: AssetManifestV1,
    manifest_bytes: Vec<u8>,
    pack_manifest: String,
    payload: Vec<u8>,
    fallback: Vec<u8>,
}

/// Prepare one pinned BF16 model tensor as canonical Q8 and OI4 artifacts.
///
/// The output directory must not already exist. All four files are first
/// written and synced in a private sibling directory, then the directory is
/// atomically renamed with no-replace semantics.
pub fn prepare_q8_linear_asset_v1(
    source_path: &Path,
    output_directory: &Path,
    source_spec: &ModelSourceSpecV1,
    tensor_spec: &Q8LinearTensorSpecV1,
) -> Result<PreparedAssetV1> {
    source_spec.verify()?;
    tensor_spec.verify()?;
    validate_output_target(output_directory)?;

    let source = MappedSource::open(source_path, source_spec)?;
    let actual_source_identity = sha256_identity(source.bytes());
    if actual_source_identity != source_spec.source_identity {
        return Err(invalid(
            "DFE-ASSET-004",
            format!(
                "source SHA-256 mismatch: expected {}, got {actual_source_identity}.",
                source_spec.source_identity
            ),
        ));
    }

    validate_header_bound(source.bytes())?;
    let tensors = SafeTensors::deserialize(source.bytes()).map_err(|error| {
        invalid(
            "DFE-ASSET-005",
            format!("safetensors deserialization failed: {error}"),
        )
    })?;
    let material = materialize_asset(&tensors, source_spec, tensor_spec, &actual_source_identity)?;
    source.verify_unchanged()?;
    commit_bundle(
        output_directory,
        &material.manifest_bytes,
        material.pack_manifest.as_bytes(),
        &material.payload,
        &material.fallback,
    )?;
    Ok(PreparedAssetV1 {
        manifest: material.manifest,
        output_directory: output_directory.to_owned(),
    })
}

/// Prepare exactly 22 ordered query-projection assets behind one atomic,
/// no-replace directory commit.
///
/// The tensor specifications must name layers `0..21` in order and share one
/// shape. The source may contain unrelated model tensors, but its q_proj
/// weight/bias namespace must be exactly the required 22 bias-free weights.
pub fn prepare_q_proj_inventory_v1(
    source_path: &Path,
    output_directory: &Path,
    source_spec: &ModelSourceSpecV1,
    tensor_specs: &[Q8LinearTensorSpecV1],
) -> Result<PreparedQProjInventoryV1> {
    source_spec.verify()?;
    validate_q_proj_specs(tensor_specs)?;
    validate_output_target(output_directory)?;
    let source = MappedSource::open(source_path, source_spec)?;
    let actual_source_identity = sha256_identity(source.bytes());
    if actual_source_identity != source_spec.source_identity {
        return Err(invalid(
            "DFE-ASSET-004",
            format!(
                "source SHA-256 mismatch: expected {}, got {actual_source_identity}.",
                source_spec.source_identity
            ),
        ));
    }
    validate_header_bound(source.bytes())?;
    let tensors = SafeTensors::deserialize(source.bytes()).map_err(|error| {
        invalid(
            "DFE-ASSET-005",
            format!("safetensors deserialization failed: {error}"),
        )
    })?;
    validate_q_proj_source_inventory(&tensors, tensor_specs)?;

    let parent = output_directory.parent().unwrap_or_else(|| Path::new("."));
    let staging = private_staging(parent, ".decodeforge-qproj-")?;
    let layers = staging.path().join(LAYERS_DIRECTORY);
    fs::create_dir(&layers).map_err(|error| {
        invalid(
            "DFE-ASSET-002",
            format!("unable to create staged layers directory: {error}"),
        )
    })?;
    fs::set_permissions(&layers, fs::Permissions::from_mode(0o700)).map_err(|error| {
        invalid(
            "DFE-ASSET-002",
            format!("unable to protect staged layers directory: {error}"),
        )
    })?;

    let mut entries = Vec::with_capacity(TINYLLAMA_LAYER_COUNT as usize);
    let mut total_packed_bytes = 0u64;
    let mut total_fallback_bytes = 0u64;
    for (index, tensor_spec) in tensor_specs.iter().enumerate() {
        let layer = index as u32;
        let material =
            materialize_asset(&tensors, source_spec, tensor_spec, &actual_source_identity)?;
        source.verify_unchanged()?;
        let layer_directory = layers.join(format!("{layer:02}"));
        fs::create_dir(&layer_directory).map_err(|error| {
            invalid(
                "DFE-ASSET-002",
                format!("unable to create staged layer directory: {error}"),
            )
        })?;
        fs::set_permissions(&layer_directory, fs::Permissions::from_mode(0o700)).map_err(
            |error| {
                invalid(
                    "DFE-ASSET-002",
                    format!("unable to protect staged layer directory: {error}"),
                )
            },
        )?;
        write_material(&layer_directory, &material)?;
        sync_directory(&layer_directory, "staged layer directory")?;
        let verified = verify_q8_linear_asset_v1(&layer_directory)?;
        if verified.manifest != material.manifest {
            return Err(invalid(
                "DFE-ASSET-007",
                "staged layer verification returned different metadata.",
            ));
        }
        total_packed_bytes = total_packed_bytes
            .checked_add(material.manifest.pack.payload_bytes)
            .ok_or_else(|| invalid("DFE-ASSET-003", "aggregate packed bytes overflow."))?;
        total_fallback_bytes = total_fallback_bytes
            .checked_add(material.manifest.fallback.bytes)
            .ok_or_else(|| invalid("DFE-ASSET-003", "aggregate fallback bytes overflow."))?;
        entries.push(QProjInventoryEntryV1 {
            layer,
            directory: q_proj_directory(layer),
            manifest_identity: sha256_identity(&material.manifest_bytes),
            tensor_name: material.manifest.tensor.name.clone(),
            tensor_identity: material.manifest.tensor.data_identity.clone(),
            logical_weight_identity: material
                .manifest
                .quantization
                .logical_weight_identity
                .clone(),
            packed_weight_identity: material.manifest.pack.packed_weight_identity.clone(),
            packed_bytes: material.manifest.pack.payload_bytes,
            module_identity: material.manifest.module.identity.clone(),
            fallback_identity: material.manifest.fallback.identity.clone(),
            fallback_bytes: material.manifest.fallback.bytes,
        });
    }
    sync_directory(&layers, "staged layers directory")?;
    source.verify_unchanged()?;
    let source_provenance = SourceProvenanceV1 {
        model_id: source_spec.model_id.clone(),
        revision: source_spec.revision.clone(),
        filename: source_spec.filename.clone(),
        bytes: source_spec.source_bytes,
        identity: actual_source_identity,
    };
    let mut inventory = QProjInventoryV1 {
        schema_version: 1,
        format: Q_PROJ_INVENTORY_FORMAT_V1.to_owned(),
        source: source_provenance,
        layer_count: TINYLLAMA_LAYER_COUNT,
        entries,
        total_packed_bytes,
        total_fallback_bytes,
        aggregate_identity:
            "sha256:0000000000000000000000000000000000000000000000000000000000000000".to_owned(),
    };
    inventory.aggregate_identity = inventory_identity(&inventory)?;
    inventory.verify()?;
    let inventory_bytes =
        pretty_json_bytes(&inventory, MAX_INVENTORY_BYTES, "q_proj asset inventory")?;
    write_new_synced(&staging.path().join(INVENTORY_FILENAME), &inventory_bytes)?;
    sync_directory(staging.path(), "staged q_proj inventory")?;
    commit_staging(staging, output_directory)?;
    Ok(PreparedQProjInventoryV1 {
        inventory,
        output_directory: output_directory.to_owned(),
    })
}

/// Prepare the complete pinned TinyLlama query-projection inventory.
pub fn prepare_tinyllama_q_proj_inventory_v1(
    source_path: &Path,
    output_directory: &Path,
) -> Result<PreparedQProjInventoryV1> {
    let (source, tensors) = tinyllama_q_proj_specs_v1()?;
    prepare_q_proj_inventory_v1(source_path, output_directory, &source, &tensors)
}

fn materialize_asset(
    tensors: &SafeTensors<'_>,
    source_spec: &ModelSourceSpecV1,
    tensor_spec: &Q8LinearTensorSpecV1,
    source_identity: &str,
) -> Result<AssetMaterial> {
    let tensor = tensors.tensor(&tensor_spec.name).map_err(|error| {
        invalid(
            "DFE-ASSET-006",
            format!("tensor {:?} was not found: {error}", tensor_spec.name),
        )
    })?;
    validate_tensor_view(&tensor, tensor_spec)?;
    let tensor_identity = sha256_identity(tensor.data());
    if tensor_spec
        .expected_data_identity
        .as_ref()
        .is_some_and(|expected| expected != &tensor_identity)
    {
        return Err(invalid(
            "DFE-ASSET-006",
            format!(
                "tensor SHA-256 mismatch: expected {}, got {tensor_identity}.",
                tensor_spec
                    .expected_data_identity
                    .as_deref()
                    .expect("checked as present")
            ),
        ));
    }
    let source_bits = decode_finite_source(&tensor, tensor_spec)?;
    let weights =
        quantize_f32_bits(tensor_spec.n, tensor_spec.k, &source_bits).map_err(|error| {
            invalid(
                "DFE-ASSET-006",
                format!("canonical Q8 quantization failed: {error}"),
            )
        })?;
    let fallback_bits = dequantize_f32_bits(&weights).map_err(|error| {
        invalid(
            "DFE-ASSET-006",
            format!("canonical Q8 dequantization failed: {error}"),
        )
    })?;
    let fallback = encode_f32_bits(&fallback_bits)?;
    let packed = PackedWeightsV1::pack(&weights)?;
    if packed.len() > MAX_PACKED_BYTES || fallback.len() > MAX_FALLBACK_BYTES {
        return Err(invalid(
            "DFE-ASSET-003",
            "prepared tensor exceeds an asset byte bound.",
        ));
    }
    let shape = packed.shape();
    let region = Q8LinearRegion::new(shape, packed.logical_weight_identity())?;
    let kernel = LoopKernelV1::new(&region, KernelVariant::Neon)?;
    let module = emit_neon_c(&region, &kernel, &packed)?;
    let pack_manifest = packed.canonical_manifest_json()?;
    if pack_manifest.len() > MAX_PACK_MANIFEST_BYTES {
        return Err(invalid(
            "DFE-ASSET-003",
            "canonical pack manifest exceeds its byte bound.",
        ));
    }
    let manifest = AssetManifestV1 {
        schema_version: 1,
        format: ASSET_BUNDLE_FORMAT_V1.to_owned(),
        operator: "Q8Linear".to_owned(),
        source: SourceProvenanceV1 {
            model_id: source_spec.model_id.clone(),
            revision: source_spec.revision.clone(),
            filename: source_spec.filename.clone(),
            bytes: source_spec.source_bytes,
            identity: source_identity.to_owned(),
        },
        tensor: TensorProvenanceV1 {
            name: tensor_spec.name.clone(),
            dtype: match tensor.dtype() {
                Dtype::BF16 => "BF16",
                Dtype::F32 => "F32",
                _ => unreachable!("validated source dtype"),
            }
            .to_owned(),
            shape: [tensor_spec.n, tensor_spec.k],
            data_bytes: tensor.data().len() as u64,
            data_identity: tensor_identity,
        },
        quantization: QuantizationProvenanceV1 {
            format: q8::FORMAT.to_owned(),
            numeric_mode: q8::NUMERIC_MODE.to_owned(),
            logical_weight_identity: packed.logical_weight_identity().to_owned(),
        },
        pack: PackArtifactV1 {
            format: crate::PACK_FORMAT.to_owned(),
            manifest_file: PACK_MANIFEST_FILENAME.to_owned(),
            manifest_bytes: pack_manifest.len() as u64,
            manifest_identity: sha256_identity(pack_manifest.as_bytes()),
            payload_file: PACK_PAYLOAD_FILENAME.to_owned(),
            payload_bytes: packed.len() as u64,
            payload_identity: sha256_identity(packed.bytes()),
            packed_weight_identity: packed.packed_identity().to_owned(),
        },
        module: ModuleArtifactV1 {
            variant: "neon".to_owned(),
            source_format: NEON_C_SOURCE_FORMAT_V1.to_owned(),
            identity: module.module_id().to_owned(),
        },
        fallback: FallbackArtifactV1 {
            format: FALLBACK_FORMAT_V1.to_owned(),
            dtype: "F32".to_owned(),
            file: FALLBACK_FILENAME.to_owned(),
            bytes: fallback.len() as u64,
            identity: sha256_identity(&fallback),
            parent_logical_weight_identity: packed.logical_weight_identity().to_owned(),
            parent_packed_weight_identity: packed.packed_identity().to_owned(),
        },
        tool: ToolProvenanceV1 {
            name: PREPARER_NAME.to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
        },
    };
    manifest.verify()?;
    let manifest_bytes = pretty_json_bytes(&manifest, MAX_MANIFEST_BYTES, "asset manifest")?;
    Ok(AssetMaterial {
        manifest,
        manifest_bytes,
        pack_manifest,
        payload: packed.bytes().to_vec(),
        fallback,
    })
}

fn encode_f32_bits(bits: &[u32]) -> Result<Vec<u8>> {
    let length = bits
        .len()
        .checked_mul(4)
        .ok_or_else(|| invalid("DFE-ASSET-003", "fallback tensor byte count overflows."))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(|_| {
        invalid(
            "DFE-ASSET-003",
            "unable to reserve the bounded fallback tensor buffer.",
        )
    })?;
    for word in bits {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    Ok(bytes)
}

fn decode_f32_bits(bytes: &[u8]) -> Result<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(invalid(
            "DFE-ASSET-007",
            "fallback tensor byte count is not a multiple of binary32 storage.",
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}

fn pretty_json_bytes<T: Serialize>(value: &T, bound: usize, label: &str) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        invalid(
            "DFE-ASSET-007",
            format!("{label} serialization failed: {error}"),
        )
    })?;
    bytes.push(b'\n');
    if bytes.len() > bound {
        return Err(invalid(
            "DFE-ASSET-003",
            format!("{label} exceeds its byte bound."),
        ));
    }
    Ok(bytes)
}

/// Read and fully verify one prepared asset directory.
///
/// This is the safe handoff point for later model integration: it rejects
/// symlinked files, enforces all byte bounds and hashes, and asks
/// [`PackedWeightsV1`] to independently validate the canonical manifest,
/// padding, logical identity, packed identity, and payload.
pub fn verify_q8_linear_asset_v1(directory: &Path) -> Result<VerifiedAssetV1> {
    let metadata = fs::symlink_metadata(directory).map_err(|error| {
        invalid(
            "DFE-ASSET-002",
            format!(
                "unable to inspect asset directory {}: {error}",
                directory.display()
            ),
        )
    })?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return Err(invalid(
            "DFE-ASSET-007",
            "asset path must be a non-symlink directory.",
        ));
    }
    verify_directory_inventory(
        directory,
        &[
            MANIFEST_FILENAME,
            PACK_MANIFEST_FILENAME,
            PACK_PAYLOAD_FILENAME,
            FALLBACK_FILENAME,
        ],
    )?;
    let manifest_bytes = read_regular_bounded(
        &directory.join(MANIFEST_FILENAME),
        MAX_MANIFEST_BYTES,
        "asset manifest",
    )?;
    let manifest: AssetManifestV1 = serde_json::from_slice(&manifest_bytes).map_err(|error| {
        invalid(
            "DFE-ASSET-007",
            format!("asset manifest deserialization failed: {error}"),
        )
    })?;
    manifest.verify()?;

    let pack_manifest_bytes = read_regular_bounded(
        &directory.join(&manifest.pack.manifest_file),
        MAX_PACK_MANIFEST_BYTES,
        "pack manifest",
    )?;
    if pack_manifest_bytes.len() as u64 != manifest.pack.manifest_bytes
        || sha256_identity(&pack_manifest_bytes) != manifest.pack.manifest_identity
    {
        return Err(invalid(
            "DFE-ASSET-007",
            "pack manifest bytes do not match the top-level manifest.",
        ));
    }
    let pack_manifest = serde_json::from_slice(&pack_manifest_bytes).map_err(|error| {
        invalid(
            "DFE-ASSET-007",
            format!("pack manifest deserialization failed: {error}"),
        )
    })?;
    let payload_limit = usize::try_from(manifest.pack.payload_bytes).map_err(|_| {
        invalid(
            "DFE-ASSET-003",
            "pack payload byte count is not representable.",
        )
    })?;
    let payload = read_regular_bounded(
        &directory.join(&manifest.pack.payload_file),
        payload_limit,
        "pack payload",
    )?;
    if payload.len() as u64 != manifest.pack.payload_bytes
        || sha256_identity(&payload) != manifest.pack.payload_identity
    {
        return Err(invalid(
            "DFE-ASSET-007",
            "pack payload bytes do not match the top-level manifest.",
        ));
    }
    let packed = PackedWeightsV1::from_artifact_parts(pack_manifest, payload)?;
    if packed.shape().n() != manifest.tensor.shape[0]
        || packed.shape().k() != manifest.tensor.shape[1]
        || packed.logical_weight_identity() != manifest.quantization.logical_weight_identity
        || packed.packed_identity() != manifest.pack.packed_weight_identity
    {
        return Err(invalid(
            "DFE-ASSET-007",
            "top-level tensor identities do not match the canonical OI4 artifact.",
        ));
    }
    let fallback_limit = usize::try_from(manifest.fallback.bytes).map_err(|_| {
        invalid(
            "DFE-ASSET-003",
            "fallback tensor byte count is not representable.",
        )
    })?;
    let fallback = read_regular_bounded(
        &directory.join(&manifest.fallback.file),
        fallback_limit,
        "fallback tensor",
    )?;
    if fallback.len() as u64 != manifest.fallback.bytes
        || sha256_identity(&fallback) != manifest.fallback.identity
    {
        return Err(invalid(
            "DFE-ASSET-007",
            "fallback tensor bytes do not match the top-level manifest.",
        ));
    }
    let fallback_bits = decode_f32_bits(&fallback)?;
    let logical = packed.logical_weights()?;
    let expected_fallback = dequantize_f32_bits(&logical).map_err(|error| {
        invalid(
            "DFE-ASSET-007",
            format!("unable to verify canonical fallback tensor: {error}"),
        )
    })?;
    if fallback_bits != expected_fallback {
        return Err(invalid(
            "DFE-ASSET-007",
            "fallback tensor is not the canonical dequantization of the bound Q8 pack.",
        ));
    }
    let region = Q8LinearRegion::new(packed.shape(), packed.logical_weight_identity())?;
    let kernel = LoopKernelV1::new(&region, KernelVariant::Neon)?;
    let module = emit_neon_c(&region, &kernel, &packed)?;
    if module.module_id() != manifest.module.identity {
        return Err(invalid(
            "DFE-ASSET-007",
            "module identity does not match the canonical generated NEON module.",
        ));
    }
    Ok(VerifiedAssetV1 {
        manifest,
        packed,
        fallback_bits,
    })
}

/// Verify the closed top-level inventory and every identity-bound layer asset.
pub fn verify_q_proj_inventory_v1(directory: &Path) -> Result<VerifiedQProjInventoryV1> {
    require_plain_directory(directory, "q_proj inventory")?;
    verify_directory_inventory(directory, &[INVENTORY_FILENAME, LAYERS_DIRECTORY])?;
    let layers = directory.join(LAYERS_DIRECTORY);
    require_plain_directory(&layers, "q_proj layers")?;
    let expected_layers = (0..TINYLLAMA_LAYER_COUNT)
        .map(|layer| format!("{layer:02}"))
        .collect::<Vec<_>>();
    let expected_layer_names = expected_layers
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    verify_directory_inventory(&layers, &expected_layer_names)?;
    let inventory_bytes = read_regular_bounded(
        &directory.join(INVENTORY_FILENAME),
        MAX_INVENTORY_BYTES,
        "q_proj inventory",
    )?;
    let inventory: QProjInventoryV1 =
        serde_json::from_slice(&inventory_bytes).map_err(|error| {
            invalid(
                "DFE-ASSET-007",
                format!("q_proj inventory deserialization failed: {error}"),
            )
        })?;
    inventory.verify()?;
    let mut total_packed_bytes = 0u64;
    let mut total_fallback_bytes = 0u64;
    for entry in &inventory.entries {
        let asset_directory = directory.join(&entry.directory);
        let verified = verify_q8_linear_asset_v1(&asset_directory)?;
        let manifest_bytes = read_regular_bounded(
            &asset_directory.join(MANIFEST_FILENAME),
            MAX_MANIFEST_BYTES,
            "asset manifest",
        )?;
        if sha256_identity(&manifest_bytes) != entry.manifest_identity
            || verified.manifest.source != inventory.source
            || verified.manifest.tensor.name != entry.tensor_name
            || verified.manifest.tensor.data_identity != entry.tensor_identity
            || verified.manifest.quantization.logical_weight_identity
                != entry.logical_weight_identity
            || verified.manifest.pack.packed_weight_identity != entry.packed_weight_identity
            || verified.manifest.pack.payload_bytes != entry.packed_bytes
            || verified.manifest.module.identity != entry.module_identity
            || verified.manifest.fallback.identity != entry.fallback_identity
            || verified.manifest.fallback.bytes != entry.fallback_bytes
        {
            return Err(invalid(
                "DFE-ASSET-007",
                "q_proj inventory entry does not match its verified layer asset.",
            ));
        }
        total_packed_bytes = total_packed_bytes
            .checked_add(entry.packed_bytes)
            .ok_or_else(|| invalid("DFE-ASSET-003", "aggregate packed bytes overflow."))?;
        total_fallback_bytes = total_fallback_bytes
            .checked_add(entry.fallback_bytes)
            .ok_or_else(|| invalid("DFE-ASSET-003", "aggregate fallback bytes overflow."))?;
    }
    if total_packed_bytes != inventory.total_packed_bytes
        || total_fallback_bytes != inventory.total_fallback_bytes
    {
        return Err(invalid(
            "DFE-ASSET-007",
            "verified q_proj byte totals do not match the inventory.",
        ));
    }
    Ok(VerifiedQProjInventoryV1 { inventory })
}

fn inventory_identity(inventory: &QProjInventoryV1) -> Result<String> {
    let preimage = QProjInventoryPreimage {
        schema_version: inventory.schema_version,
        format: &inventory.format,
        source: &inventory.source,
        layer_count: inventory.layer_count,
        entries: &inventory.entries,
        total_packed_bytes: inventory.total_packed_bytes,
        total_fallback_bytes: inventory.total_fallback_bytes,
    };
    let bytes = serde_json::to_vec(&preimage).map_err(|error| {
        invalid(
            "DFE-ASSET-007",
            format!("q_proj inventory identity serialization failed: {error}"),
        )
    })?;
    let mut hasher = Sha256::new();
    hasher.update(b"DecodeForge/q-proj-inventory/v1\0");
    hasher.update(bytes);
    Ok(format!("sha256:{}", crate::hex_lower(&hasher.finalize())))
}

fn q_proj_tensor_name(layer: u32) -> String {
    format!("model.layers.{layer}.self_attn.q_proj.weight")
}

fn q_proj_directory(layer: u32) -> String {
    format!("{LAYERS_DIRECTORY}/{layer:02}")
}

fn validate_q_proj_specs(specs: &[Q8LinearTensorSpecV1]) -> Result<()> {
    if specs.len() != TINYLLAMA_LAYER_COUNT as usize {
        return Err(invalid(
            "DFE-ASSET-001",
            "q_proj preparation requires exactly 22 tensor specifications.",
        ));
    }
    let first_shape = (specs[0].n, specs[0].k);
    for (index, spec) in specs.iter().enumerate() {
        spec.verify()?;
        if spec.name != q_proj_tensor_name(index as u32) || (spec.n, spec.k) != first_shape {
            return Err(invalid(
                "DFE-ASSET-001",
                "q_proj tensor specifications must be ordered layers 0..21 with one shape.",
            ));
        }
    }
    Ok(())
}

fn validate_q_proj_source_inventory(
    tensors: &SafeTensors<'_>,
    specs: &[Q8LinearTensorSpecV1],
) -> Result<()> {
    let expected = specs
        .iter()
        .map(|spec| spec.name.as_str())
        .collect::<BTreeSet<_>>();
    let actual = tensors
        .names()
        .into_iter()
        .filter(|name| name.contains(".self_attn.q_proj."))
        .collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(invalid(
            "DFE-ASSET-006",
            "source q_proj namespace must contain exactly 22 expected bias-free weights.",
        ));
    }
    Ok(())
}

fn validate_source_provenance(source: &SourceProvenanceV1) -> Result<()> {
    validate_text("model ID", &source.model_id, 256)?;
    validate_lower_hex("revision", &source.revision, 40)?;
    validate_filename(&source.filename)?;
    validate_sha256_identity("source identity", &source.identity)?;
    if !(9..=MAX_SOURCE_BYTES).contains(&source.bytes) {
        return Err(invalid(
            "DFE-ASSET-007",
            "source byte count is outside the supported bound.",
        ));
    }
    Ok(())
}

fn validate_tensor_view(
    tensor: &safetensors::tensor::TensorView<'_>,
    spec: &Q8LinearTensorSpecV1,
) -> Result<()> {
    let expected_shape = [spec.n as usize, spec.k as usize];
    let element_bytes = match tensor.dtype() {
        Dtype::BF16 => 2,
        Dtype::F32 if spec.allow_f32 => 4,
        _ => 0,
    };
    let expected_bytes = expected_shape[0]
        .checked_mul(expected_shape[1])
        .and_then(|value| value.checked_mul(element_bytes))
        .ok_or_else(|| invalid("DFE-ASSET-003", "tensor byte count overflows."))?;
    if element_bytes == 0
        || tensor.shape() != expected_shape
        || tensor.data().len() != expected_bytes
    {
        return Err(invalid(
            "DFE-ASSET-006",
            format!(
                "tensor {:?} must have an accepted floating dtype and shape [{}, {}].",
                spec.name, spec.n, spec.k
            ),
        ));
    }
    Ok(())
}

fn decode_finite_source(
    tensor: &safetensors::tensor::TensorView<'_>,
    spec: &Q8LinearTensorSpecV1,
) -> Result<Vec<u32>> {
    let elements = (spec.n as usize)
        .checked_mul(spec.k as usize)
        .ok_or_else(|| invalid("DFE-ASSET-003", "tensor element count overflows."))?;
    let mut result = Vec::new();
    result.try_reserve_exact(elements).map_err(|_| {
        invalid(
            "DFE-ASSET-003",
            "unable to reserve the bounded binary32 conversion buffer.",
        )
    })?;
    let source_words = match tensor.dtype() {
        Dtype::BF16 => tensor
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u32::from(u16::from_le_bytes(*pair)) << 16)
            .collect::<Vec<_>>(),
        Dtype::F32 if spec.allow_f32 => tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect::<Vec<_>>(),
        _ => {
            return Err(invalid(
                "DFE-ASSET-006",
                "source tensor dtype is unsupported.",
            ));
        }
    };
    if source_words.len() != elements {
        return Err(invalid(
            "DFE-ASSET-006",
            "source tensor data has the wrong byte count.",
        ));
    }
    for (index, bits) in source_words.into_iter().enumerate() {
        if !q8::is_finite_f32_bits(bits) {
            return Err(invalid(
                "DFE-ASSET-006",
                format!("tensor contains a non-finite value at index {index}."),
            ));
        }
        result.push(bits);
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    mode: u32,
    links: u64,
    length: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl FileIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            mode: metadata.mode(),
            links: metadata.nlink(),
            length: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }
}

struct MappedSource {
    file: File,
    map: Mmap,
    identity: FileIdentity,
}

impl MappedSource {
    fn open(path: &Path, spec: &ModelSourceSpecV1) -> Result<Self> {
        let descriptor = open(
            path,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|error| {
            invalid(
                "DFE-ASSET-002",
                format!("unable to securely open source {}: {error}", path.display()),
            )
        })?;
        let file = File::from(descriptor);
        flock(&file, FlockOperation::NonBlockingLockShared).map_err(|error| {
            invalid(
                "DFE-ASSET-002",
                format!("unable to acquire a shared source lock: {error}"),
            )
        })?;
        let metadata = file.metadata().map_err(|error| {
            invalid(
                "DFE-ASSET-002",
                format!("unable to inspect source {}: {error}", path.display()),
            )
        })?;
        if !metadata.file_type().is_file() {
            return Err(invalid(
                "DFE-ASSET-002",
                "safetensors source must be a regular file.",
            ));
        }
        let identity = FileIdentity::from_metadata(&metadata);
        if identity.length != spec.source_bytes {
            return Err(invalid(
                "DFE-ASSET-004",
                format!(
                    "source byte count mismatch: expected {}, got {}.",
                    spec.source_bytes, identity.length
                ),
            ));
        }
        // SAFETY: the descriptor is a securely opened regular file, remains
        // alive for the map lifetime, and is held under a non-blocking shared
        // advisory lock. The full identity is rechecked after all reads. A
        // non-cooperating process can still truncate a mapped file on Unix;
        // callers must treat the pinned model file as immutable while this
        // function runs. Avoiding that OS-level mmap limitation would require
        // a second multi-gigabyte snapshot and is intentionally deferred.
        let map = unsafe { MmapOptions::new().map(&file) }.map_err(|error| {
            invalid(
                "DFE-ASSET-002",
                format!("unable to memory-map source {}: {error}", path.display()),
            )
        })?;
        let result = Self {
            file,
            map,
            identity,
        };
        result.verify_unchanged()?;
        Ok(result)
    }

    fn bytes(&self) -> &[u8] {
        &self.map
    }

    fn verify_unchanged(&self) -> Result<()> {
        let after = self.file.metadata().map_err(|error| {
            invalid(
                "DFE-ASSET-002",
                format!("unable to recheck mapped source: {error}"),
            )
        })?;
        if FileIdentity::from_metadata(&after) != self.identity
            || self.map.len() != after.len() as usize
        {
            return Err(invalid(
                "DFE-ASSET-004",
                "source identity changed while it was mapped.",
            ));
        }
        Ok(())
    }
}

fn validate_header_bound(bytes: &[u8]) -> Result<()> {
    let prefix: [u8; 8] = bytes
        .get(..8)
        .and_then(|value| value.try_into().ok())
        .ok_or_else(|| invalid("DFE-ASSET-005", "safetensors header length is truncated."))?;
    let header = usize::try_from(u64::from_le_bytes(prefix)).map_err(|_| {
        invalid(
            "DFE-ASSET-003",
            "safetensors header length is not representable.",
        )
    })?;
    let end = 8usize
        .checked_add(header)
        .ok_or_else(|| invalid("DFE-ASSET-003", "safetensors header length overflows."))?;
    if header == 0 || header > MAX_HEADER_BYTES || end > bytes.len() {
        return Err(invalid(
            "DFE-ASSET-003",
            "safetensors header exceeds the fixed bound or is truncated.",
        ));
    }
    let mut deserializer = serde_json::Deserializer::from_slice(&bytes[8..end]);
    UniqueHeaderKeys::deserialize(&mut deserializer).map_err(|error| {
        invalid(
            "DFE-ASSET-005",
            format!("safetensors header has duplicate or invalid top-level keys: {error}"),
        )
    })?;
    deserializer.end().map_err(|error| {
        invalid(
            "DFE-ASSET-005",
            format!("safetensors header has trailing invalid data: {error}"),
        )
    })?;
    Ok(())
}

struct UniqueHeaderKeys;

impl<'de> Deserialize<'de> for UniqueHeaderKeys {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(UniqueHeaderVisitor)
    }
}

struct UniqueHeaderVisitor;

impl<'de> Visitor<'de> for UniqueHeaderVisitor {
    type Value = UniqueHeaderKeys;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a safetensors header with unique top-level keys")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut names = BTreeSet::new();
        while let Some(name) = map.next_key::<String>()? {
            if !names.insert(name) {
                return Err(serde::de::Error::custom("duplicate safetensors header key"));
            }
            map.next_value::<IgnoredAny>()?;
        }
        Ok(UniqueHeaderKeys)
    }
}

fn validate_output_target(output: &Path) -> Result<()> {
    if output.as_os_str().is_empty() || output.file_name().is_none() {
        return Err(invalid(
            "DFE-ASSET-007",
            "output must name an explicit new directory.",
        ));
    }
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(
            "DFE-ASSET-007",
            "output path already exists; prepared assets never overwrite it.",
        ));
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let metadata = fs::symlink_metadata(parent).map_err(|error| {
        invalid(
            "DFE-ASSET-002",
            format!(
                "unable to inspect output parent {}: {error}",
                parent.display()
            ),
        )
    })?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return Err(invalid(
            "DFE-ASSET-007",
            "output parent must be an existing non-symlink directory.",
        ));
    }
    Ok(())
}

fn commit_bundle(
    output: &Path,
    manifest: &[u8],
    pack_manifest: &[u8],
    payload: &[u8],
    fallback: &[u8],
) -> Result<()> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let staging = private_staging(parent, ".decodeforge-asset-")?;
    write_new_synced(&staging.path().join(PACK_PAYLOAD_FILENAME), payload)?;
    write_new_synced(&staging.path().join(PACK_MANIFEST_FILENAME), pack_manifest)?;
    write_new_synced(&staging.path().join(FALLBACK_FILENAME), fallback)?;
    write_new_synced(&staging.path().join(MANIFEST_FILENAME), manifest)?;
    sync_directory(staging.path(), "staged asset directory")?;
    commit_staging(staging, output)
}

fn private_staging(parent: &Path, prefix: &str) -> Result<tempfile::TempDir> {
    let staging = tempfile::Builder::new()
        .prefix(prefix)
        .tempdir_in(parent)
        .map_err(|error| {
            invalid(
                "DFE-ASSET-002",
                format!("unable to create private staging directory: {error}"),
            )
        })?;
    fs::set_permissions(staging.path(), fs::Permissions::from_mode(0o700)).map_err(|error| {
        invalid(
            "DFE-ASSET-002",
            format!("unable to protect staging directory: {error}"),
        )
    })?;
    Ok(staging)
}

fn write_material(directory: &Path, material: &AssetMaterial) -> Result<()> {
    write_new_synced(&directory.join(PACK_PAYLOAD_FILENAME), &material.payload)?;
    write_new_synced(
        &directory.join(PACK_MANIFEST_FILENAME),
        material.pack_manifest.as_bytes(),
    )?;
    write_new_synced(&directory.join(FALLBACK_FILENAME), &material.fallback)?;
    write_new_synced(&directory.join(MANIFEST_FILENAME), &material.manifest_bytes)
}

fn sync_directory(directory: &Path, label: &str) -> Result<()> {
    File::open(directory)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| invalid("DFE-ASSET-002", format!("unable to sync {label}: {error}")))
}

fn commit_staging(staging: tempfile::TempDir, output: &Path) -> Result<()> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    renameat_with(
        rustix::fs::CWD,
        staging.path(),
        rustix::fs::CWD,
        output,
        RenameFlags::NOREPLACE,
    )
    .map_err(|error| {
        invalid(
            "DFE-ASSET-007",
            format!("unable to commit asset directory without replacement: {error}"),
        )
    })?;
    let _staging_path = staging.keep();
    sync_directory(parent, "output parent directory")
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true).mode(0o600);
    let mut file = options.open(path).map_err(|error| {
        invalid(
            "DFE-ASSET-002",
            format!("unable to create staged asset {}: {error}", path.display()),
        )
    })?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| {
            invalid(
                "DFE-ASSET-002",
                format!("unable to write staged asset {}: {error}", path.display()),
            )
        })
}

fn read_regular_bounded(path: &Path, bound: usize, label: &str) -> Result<Vec<u8>> {
    let descriptor = open(
        path,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|error| {
        invalid(
            "DFE-ASSET-002",
            format!(
                "unable to securely open {label} {}: {error}",
                path.display()
            ),
        )
    })?;
    let mut file = File::from(descriptor);
    let before = file.metadata().map_err(|error| {
        invalid(
            "DFE-ASSET-002",
            format!("unable to inspect {label} {}: {error}", path.display()),
        )
    })?;
    if !before.file_type().is_file() || before.len() > bound as u64 {
        return Err(invalid(
            "DFE-ASSET-007",
            format!("{label} must be a bounded regular non-symlink file."),
        ));
    }
    let expected = usize::try_from(before.len()).map_err(|_| {
        invalid(
            "DFE-ASSET-003",
            format!("{label} size is not representable."),
        )
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(expected).map_err(|_| {
        invalid(
            "DFE-ASSET-003",
            format!("unable to reserve bounded {label} buffer."),
        )
    })?;
    file.read_to_end(&mut bytes).map_err(|error| {
        invalid(
            "DFE-ASSET-002",
            format!("unable to read {label} {}: {error}", path.display()),
        )
    })?;
    let after = file.metadata().map_err(|error| {
        invalid(
            "DFE-ASSET-002",
            format!("unable to recheck {label} {}: {error}", path.display()),
        )
    })?;
    if bytes.len() != expected
        || FileIdentity::from_metadata(&before) != FileIdentity::from_metadata(&after)
    {
        return Err(invalid(
            "DFE-ASSET-004",
            format!("{label} changed while it was read."),
        ));
    }
    Ok(bytes)
}

fn require_plain_directory(path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        invalid(
            "DFE-ASSET-002",
            format!("unable to inspect {label} {}: {error}", path.display()),
        )
    })?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return Err(invalid(
            "DFE-ASSET-007",
            format!("{label} must be a non-symlink directory."),
        ));
    }
    Ok(())
}

fn verify_directory_inventory(directory: &Path, expected: &[&str]) -> Result<()> {
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(directory).map_err(|error| {
        invalid(
            "DFE-ASSET-002",
            format!(
                "unable to list asset directory {}: {error}",
                directory.display()
            ),
        )
    })? {
        let entry = entry.map_err(|error| {
            invalid(
                "DFE-ASSET-002",
                format!("unable to inspect asset directory entry: {error}"),
            )
        })?;
        let name = entry.file_name().into_string().map_err(|_| {
            invalid(
                "DFE-ASSET-007",
                "asset directory contains a non-UTF-8 name.",
            )
        })?;
        actual.insert(name);
        if actual.len() > expected.len() {
            return Err(invalid(
                "DFE-ASSET-007",
                "asset directory contains an undeclared entry.",
            ));
        }
    }
    if actual.iter().map(String::as_str).collect::<BTreeSet<_>>() != expected {
        return Err(invalid(
            "DFE-ASSET-007",
            "asset directory does not match its closed inventory.",
        ));
    }
    Ok(())
}

fn sha256_identity(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(71);
    result.push_str("sha256:");
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut result, "{byte:02x}").expect("writing to a String cannot fail");
    }
    result
}

fn validate_sha256_identity(field: &str, value: &str) -> Result<()> {
    if value.len() != 71
        || !value.starts_with("sha256:")
        || !value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(
            "DFE-ASSET-001",
            format!("{field} must be sha256:<64 lowercase hex digits>."),
        ));
    }
    Ok(())
}

fn validate_lower_hex(field: &str, value: &str, length: usize) -> Result<()> {
    if value.len() != length
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(
            "DFE-ASSET-001",
            format!("{field} must contain exactly {length} lowercase hexadecimal digits."),
        ));
    }
    Ok(())
}

fn validate_text(field: &str, value: &str, max: usize) -> Result<()> {
    if value.is_empty()
        || value.len() > max
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && byte != b'\\' && byte != b'\"')
    {
        return Err(invalid(
            "DFE-ASSET-001",
            format!("{field} must be bounded printable ASCII without quotes or backslashes."),
        ));
    }
    Ok(())
}

fn validate_filename(value: &str) -> Result<()> {
    validate_text("source filename", value, 128)?;
    let path = Path::new(value);
    if path.file_name().and_then(|name| name.to_str()) != Some(value) {
        return Err(invalid(
            "DFE-ASSET-001",
            "source filename must be one plain path component.",
        ));
    }
    Ok(())
}

fn validate_tensor_name(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 512
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_'))
    {
        return Err(invalid(
            "DFE-ASSET-001",
            "tensor name must use only bounded ASCII letters, digits, dots, and underscores.",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
