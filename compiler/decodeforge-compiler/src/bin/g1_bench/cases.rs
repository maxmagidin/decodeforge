use super::BenchError;
use super::files::{
    atomic_write, decode_u32_le, encode_u32_le, read_bounded, relative_asset_path, sha256_identity,
};
use super::spec::{
    ACTIVATION_STREAM, CASE_BUNDLE_FORMAT, MAX_CASE_FILE_BYTES, MAX_JSON_BYTES, MODEL_FILENAME,
    MODEL_ID, MODEL_REVISION, MODEL_SHA256, MODEL_SIZE_BYTES, PROTOCOL_ID, REAL_CASE_ID,
    REAL_EXPECTED_IDENTITY, REAL_INPUT_IDENTITY, REAL_LOGICAL_WEIGHT_IDENTITY,
    REAL_PACKED_WEIGHT_IDENTITY, TENSOR_DTYPE, TENSOR_IDENTITY, TENSOR_K, TENSOR_N, TENSOR_NAME,
    TENSOR_SHA256, spec_identity, validate_embedded_spec,
};
use decodeforge_compiler::{PackManifestV1, PackedWeightsV1};
use decodeforge_core::q8::{
    Q8Weights, canonical_linear_f32_bits, logical_weight_identity, quantize_f32_bits,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseBundleManifest {
    pub schema_version: u32,
    pub protocol_id: String,
    pub format: String,
    pub spec_identity: String,
    pub source: SourceProvenance,
    pub cases: Vec<CaseManifest>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceProvenance {
    pub model: String,
    pub revision: String,
    pub model_filename: String,
    pub model_size_bytes: u64,
    pub model_sha256: String,
    pub tensor_name: String,
    pub dtype: String,
    pub shape: [usize; 2],
    pub tensor_sha256: String,
    pub input_semantic_identity: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseManifest {
    pub case_id: String,
    pub kind: String,
    pub n: usize,
    pub k: usize,
    pub blocks: usize,
    pub input_file: String,
    pub expected_file: String,
    pub pack_manifest_file: String,
    pub pack_payload_file: String,
    pub input_identity: String,
    pub expected_identity: String,
    pub logical_weight_identity: String,
    pub packed_weight_identity: String,
    pub activation_stream: String,
}

#[derive(Debug)]
pub struct CaseAsset {
    pub manifest: CaseManifest,
    pub packed: PackedWeightsV1,
    pub input: Vec<f32>,
    pub expected_bits: Vec<u32>,
}

#[derive(Debug)]
pub struct CaseBundle {
    pub manifest: CaseBundleManifest,
    pub manifest_identity: String,
    pub cases: Vec<CaseAsset>,
}

pub fn prepare_cases(weights_path: &Path, output: &Path) -> Result<CaseBundleManifest, BenchError> {
    validate_embedded_spec()?;
    if output.as_os_str().is_empty() || output.file_name().is_none() {
        return Err(BenchError::new(
            "DFE-G1-CLI",
            "case output must be an explicit directory",
        ));
    }
    fs::create_dir_all(output).map_err(|error| {
        BenchError::new(
            "DFE-G1-IO",
            format!("unable to create case output {}: {error}", output.display()),
        )
    })?;
    let source = super::safetensors::read_pinned(weights_path)?;
    let weights = quantize_f32_bits(TENSOR_N as u32, TENSOR_K as u32, &source.fp32_bits)?;
    let input_bits = activation_words(REAL_CASE_ID, TENSOR_K);
    let expected_bits = canonical_linear_f32_bits(&input_bits, &weights)?;
    let packed = PackedWeightsV1::pack(&weights)?;
    let case = CaseManifest {
        case_id: REAL_CASE_ID.to_owned(),
        kind: "real".to_owned(),
        n: TENSOR_N,
        k: TENSOR_K,
        blocks: weights.blocks() as usize,
        input_file: format!("{REAL_CASE_ID}.input.f32le"),
        expected_file: format!("{REAL_CASE_ID}.expected.f32le"),
        pack_manifest_file: format!("{REAL_CASE_ID}.pack.json"),
        pack_payload_file: format!("{REAL_CASE_ID}.pack.bin"),
        input_identity: sha256_identity(&encode_u32_le(&input_bits)),
        expected_identity: sha256_identity(&encode_u32_le(&expected_bits)),
        logical_weight_identity: packed.logical_weight_identity().to_owned(),
        packed_weight_identity: packed.packed_identity().to_owned(),
        activation_stream: ACTIVATION_STREAM.to_owned(),
    };
    validate_pinned_case(&case)?;
    write_case_assets(output, &case, &packed, &input_bits, &expected_bits)?;

    let source_metadata = &source.metadata;
    let source = SourceProvenance {
        model: required_metadata(source_metadata, "model")?,
        revision: required_metadata(source_metadata, "revision")?,
        model_filename: required_metadata(source_metadata, "model_filename")?,
        model_size_bytes: required_metadata(source_metadata, "model_size_bytes")?
            .parse()
            .map_err(|_| {
                BenchError::new("DFE-G1-SAFETENSORS", "model_size_bytes is not numeric")
            })?,
        model_sha256: required_metadata(source_metadata, "model_sha256")?,
        tensor_name: required_metadata(source_metadata, "tensor_name")?,
        dtype: TENSOR_DTYPE.to_owned(),
        shape: [TENSOR_N, TENSOR_K],
        tensor_sha256: required_metadata(source_metadata, "tensor_sha256")?,
        input_semantic_identity: source.semantic_identity,
    };
    if source.model != MODEL_ID
        || source.revision != MODEL_REVISION
        || source.model_filename != MODEL_FILENAME
        || source.model_size_bytes != MODEL_SIZE_BYTES
        || source.model_sha256 != MODEL_SHA256
        || source.tensor_name != TENSOR_NAME
        || source.tensor_sha256 != TENSOR_SHA256
        || source.input_semantic_identity != TENSOR_IDENTITY
    {
        return Err(BenchError::new(
            "DFE-G1-SAFETENSORS",
            "prepared source provenance changed after input validation",
        ));
    }
    let manifest = CaseBundleManifest {
        schema_version: 1,
        protocol_id: PROTOCOL_ID.to_owned(),
        format: CASE_BUNDLE_FORMAT.to_owned(),
        spec_identity: spec_identity(),
        source,
        cases: vec![case],
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    if manifest_bytes.len() > MAX_JSON_BYTES {
        return Err(BenchError::new(
            "DFE-G1-LIMIT",
            "case manifest exceeds the fixed JSON bound",
        ));
    }
    atomic_write(&output.join("manifest.json"), &manifest_bytes)?;
    Ok(manifest)
}

fn required_metadata(metadata: &BTreeMap<String, String>, key: &str) -> Result<String, BenchError> {
    metadata
        .get(key)
        .cloned()
        .ok_or_else(|| BenchError::new("DFE-G1-SAFETENSORS", format!("missing metadata {key}")))
}

fn write_case_assets(
    output: &Path,
    case: &CaseManifest,
    packed: &PackedWeightsV1,
    input_bits: &[u32],
    expected_bits: &[u32],
) -> Result<(), BenchError> {
    let input_bytes = encode_u32_le(input_bits);
    let expected_bytes = encode_u32_le(expected_bits);
    let pack_json = packed.canonical_manifest_json()?;
    atomic_write(&output.join(&case.input_file), &input_bytes)?;
    atomic_write(&output.join(&case.expected_file), &expected_bytes)?;
    atomic_write(&output.join(&case.pack_manifest_file), pack_json.as_bytes())?;
    atomic_write(&output.join(&case.pack_payload_file), packed.bytes())?;
    Ok(())
}

pub fn read_case_bundle(path: &Path) -> Result<CaseBundle, BenchError> {
    validate_embedded_spec()?;
    let manifest_bytes = read_bounded(path, MAX_JSON_BYTES, "case manifest")?;
    let manifest: CaseBundleManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|error| {
            BenchError::new("DFE-G1-ASSET", format!("case manifest is invalid: {error}"))
        })?;
    validate_bundle_manifest(&manifest)?;
    let root = path.parent().unwrap_or_else(|| Path::new(".")).to_owned();
    let mut cases = Vec::with_capacity(manifest.cases.len());
    for case in &manifest.cases {
        cases.push(read_case_asset(&root, case)?);
    }
    Ok(CaseBundle {
        manifest,
        manifest_identity: sha256_identity(&manifest_bytes),
        cases,
    })
}

fn validate_bundle_manifest(manifest: &CaseBundleManifest) -> Result<(), BenchError> {
    if manifest.schema_version != 1
        || manifest.protocol_id != PROTOCOL_ID
        || manifest.format != CASE_BUNDLE_FORMAT
        || manifest.spec_identity != spec_identity()
    {
        return Err(BenchError::new(
            "DFE-G1-ASSET",
            "case manifest does not match the checked benchmark protocol",
        ));
    }
    if manifest.cases.len() != 1 || manifest.cases[0].case_id != REAL_CASE_ID {
        return Err(BenchError::new(
            "DFE-G1-ASSET",
            "this runner accepts exactly one real TinyLlama case",
        ));
    }
    if manifest.source.model != MODEL_ID
        || manifest.source.revision != MODEL_REVISION
        || manifest.source.model_filename != MODEL_FILENAME
        || manifest.source.model_size_bytes != MODEL_SIZE_BYTES
        || manifest.source.model_sha256 != MODEL_SHA256
        || manifest.source.tensor_name != TENSOR_NAME
        || manifest.source.dtype != TENSOR_DTYPE
        || manifest.source.shape != [TENSOR_N, TENSOR_K]
        || manifest.source.tensor_sha256 != TENSOR_SHA256
        || manifest.source.input_semantic_identity != TENSOR_IDENTITY
    {
        return Err(BenchError::new(
            "DFE-G1-ASSET",
            "case manifest source provenance is not pinned to TinyLlama",
        ));
    }
    validate_pinned_case(&manifest.cases[0])?;
    Ok(())
}

fn validate_pinned_case(case: &CaseManifest) -> Result<(), BenchError> {
    if case.case_id != REAL_CASE_ID
        || case.kind != "real"
        || case.n != TENSOR_N
        || case.k != TENSOR_K
        || case.blocks != TENSOR_K.div_ceil(32)
        || case.input_identity != REAL_INPUT_IDENTITY
        || case.expected_identity != REAL_EXPECTED_IDENTITY
        || case.logical_weight_identity != REAL_LOGICAL_WEIGHT_IDENTITY
        || case.packed_weight_identity != REAL_PACKED_WEIGHT_IDENTITY
        || case.activation_stream != ACTIVATION_STREAM
    {
        return Err(BenchError::new(
            "DFE-G1-ASSET",
            "real case identities do not match the pinned TinyLlama protocol",
        ));
    }
    Ok(())
}

fn read_case_asset(root: &Path, case: &CaseManifest) -> Result<CaseAsset, BenchError> {
    validate_pinned_case(case)?;
    let input_path = relative_asset_path(root, &case.input_file)?;
    let expected_path = relative_asset_path(root, &case.expected_file)?;
    let pack_manifest_path = relative_asset_path(root, &case.pack_manifest_file)?;
    let payload_path = relative_asset_path(root, &case.pack_payload_file)?;
    let input_bytes = read_bounded(&input_path, MAX_CASE_FILE_BYTES, "input asset")?;
    let expected_bytes = read_bounded(&expected_path, MAX_CASE_FILE_BYTES, "oracle asset")?;
    let pack_manifest_bytes = read_bounded(&pack_manifest_path, MAX_JSON_BYTES, "pack manifest")?;
    let payload = read_bounded(&payload_path, MAX_CASE_FILE_BYTES, "pack payload")?;
    if sha256_identity(&input_bytes) != case.input_identity
        || sha256_identity(&expected_bytes) != case.expected_identity
    {
        return Err(BenchError::new(
            "DFE-G1-ASSET",
            "case input/oracle identity does not match its manifest",
        ));
    }
    let input_bits = decode_u32_le(&input_bytes, case.k, "input asset")?;
    let expected_bits = decode_u32_le(&expected_bytes, case.n, "oracle asset")?;
    if input_bits
        .iter()
        .any(|bits| !f32::from_bits(*bits).is_finite())
        || expected_bits
            .iter()
            .any(|bits| !f32::from_bits(*bits).is_finite())
    {
        return Err(BenchError::new(
            "DFE-G1-ASSET",
            "input and oracle assets must contain only finite binary32 words",
        ));
    }
    let pack_manifest: PackManifestV1 =
        serde_json::from_slice(&pack_manifest_bytes).map_err(|error| {
            BenchError::new("DFE-G1-ASSET", format!("pack manifest is invalid: {error}"))
        })?;
    let packed = PackedWeightsV1::from_artifact_parts(pack_manifest, payload)?;
    if pack_manifest_bytes != packed.canonical_manifest_json()?.as_bytes() {
        return Err(BenchError::new(
            "DFE-G1-ASSET",
            "pack manifest is not the canonical checked JSON representation",
        ));
    }
    if packed.shape().n() as usize != case.n
        || packed.shape().k() as usize != case.k
        || packed.logical_weight_identity() != case.logical_weight_identity
        || packed.packed_identity() != case.packed_weight_identity
    {
        return Err(BenchError::new(
            "DFE-G1-ASSET",
            "reloaded packed artifact does not match the case manifest",
        ));
    }
    let unpacked = unpack_packed_weights(&packed)?;
    if logical_weight_identity(&unpacked) != REAL_LOGICAL_WEIGHT_IDENTITY {
        return Err(BenchError::new(
            "DFE-G1-ASSET",
            "reconstructed logical weights do not match the pinned source identity",
        ));
    }
    let recomputed = canonical_linear_f32_bits(&input_bits, &unpacked)?;
    if recomputed != expected_bits {
        return Err(BenchError::new(
            "DFE-G1-ASSET",
            "oracle asset does not match the reloaded packed weights and input",
        ));
    }
    let input = input_bits
        .iter()
        .map(|bits| f32::from_bits(*bits))
        .collect();
    Ok(CaseAsset {
        manifest: case.clone(),
        packed,
        input,
        expected_bits,
    })
}

fn unpack_packed_weights(packed: &PackedWeightsV1) -> Result<Q8Weights, BenchError> {
    let shape = packed.shape();
    let n = shape.n();
    let k = shape.k();
    let blocks = shape.blocks();
    let q_length = (n as usize)
        .checked_mul(blocks as usize)
        .and_then(|value| value.checked_mul(32))
        .ok_or_else(|| BenchError::new("DFE-G1-LIMIT", "logical Q8 length overflows"))?;
    let scale_length = (n as usize)
        .checked_mul(blocks as usize)
        .ok_or_else(|| BenchError::new("DFE-G1-LIMIT", "logical scale length overflows"))?;
    let mut q_bytes = Vec::with_capacity(q_length);
    let mut scale_bits = Vec::with_capacity(scale_length);
    for row in 0..n {
        let panel = row / 4;
        let output_lane = row % 4;
        for block in 0..blocks {
            for lane in 0..32 {
                let value = packed
                    .q_at(panel, block, lane, output_lane)
                    .ok_or_else(|| BenchError::new("DFE-G1-ASSET", "packed Q8 lookup failed"))?;
                q_bytes.push(value as u8);
            }
        }
    }
    for row in 0..n {
        let panel = row / 4;
        let output_lane = row % 4;
        for block in 0..blocks {
            scale_bits.push(
                packed
                    .scale_bits_at(panel, block, output_lane)
                    .ok_or_else(|| BenchError::new("DFE-G1-ASSET", "packed scale lookup failed"))?,
            );
        }
    }
    Ok(Q8Weights::try_new(n, k, blocks, q_bytes, scale_bits)?)
}

fn activation_words(case_id: &str, count: usize) -> Vec<u32> {
    let mut words = Vec::with_capacity(count);
    let mut counter = 0_u64;
    while words.len() < count {
        let mut hasher = Sha256::new();
        hasher.update(b"DecodeForge/G1/sha256-counter/v1\0");
        hasher.update((case_id.len() as u64).to_le_bytes());
        hasher.update(case_id.as_bytes());
        hasher.update(counter.to_le_bytes());
        let digest = hasher.finalize();
        for chunk in digest.as_slice().as_chunks::<4>().0 {
            if words.len() == count {
                break;
            }
            let raw = u32::from_le_bytes(*chunk);
            let sign = if raw & 1 == 0 { 0 } else { 0x8000_0000 };
            words.push(sign | 0x3f00_0000 | (raw & 0x007f_ffff));
        }
        counter = counter.saturating_add(1);
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_stream_is_deterministic_and_finite() {
        let first = activation_words("case", 2048);
        assert_eq!(first, activation_words("case", 2048));
        assert_ne!(first, activation_words("other", 2048));
        assert!(first.iter().all(|bits| f32::from_bits(*bits).is_finite()));
    }

    #[test]
    fn protocol_is_explicitly_one_real_case_until_tail_slice_lands() {
        assert_eq!(PROTOCOL_ID, "g1-prepared-call-paired-v1");
        assert_eq!(REAL_CASE_ID, "tinyllama-q-proj-2048x2048");
    }

    #[test]
    fn real_case_hashes_are_protocol_constants_not_manifest_authority() {
        let mut case = CaseManifest {
            case_id: REAL_CASE_ID.to_owned(),
            kind: "real".to_owned(),
            n: TENSOR_N,
            k: TENSOR_K,
            blocks: TENSOR_K.div_ceil(32),
            input_file: "input.bin".to_owned(),
            expected_file: "expected.bin".to_owned(),
            pack_manifest_file: "pack.json".to_owned(),
            pack_payload_file: "pack.bin".to_owned(),
            input_identity: REAL_INPUT_IDENTITY.to_owned(),
            expected_identity: REAL_EXPECTED_IDENTITY.to_owned(),
            logical_weight_identity: REAL_LOGICAL_WEIGHT_IDENTITY.to_owned(),
            packed_weight_identity: REAL_PACKED_WEIGHT_IDENTITY.to_owned(),
            activation_stream: ACTIVATION_STREAM.to_owned(),
        };
        validate_pinned_case(&case).unwrap();
        case.expected_identity = format!("sha256:{}", "0".repeat(64));
        assert!(validate_pinned_case(&case).is_err());
    }
}
