use super::BenchError;
use super::files::sha256_hex;
use super::spec::{
    MAX_HEADER_BYTES, MAX_SAFETENSORS_BYTES, MODEL_FILENAME, MODEL_ID, MODEL_REVISION,
    MODEL_SHA256, MODEL_SIZE_BYTES, TENSOR_BYTES, TENSOR_IDENTITY, TENSOR_K, TENSOR_N, TENSOR_NAME,
    TENSOR_SHA256, validate_embedded_spec,
};
use safetensors::{Dtype, SafeTensors};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

const REQUIRED_METADATA: [&str; 7] = [
    "model",
    "revision",
    "model_filename",
    "model_size_bytes",
    "model_sha256",
    "tensor_name",
    "tensor_sha256",
];

/// The one BF16 tensor accepted by the fixed G1 preparation protocol.
#[derive(Clone, Debug)]
pub struct PinnedBf16Tensor {
    pub semantic_identity: String,
    pub metadata: BTreeMap<String, String>,
    pub fp32_bits: Vec<u32>,
}

pub fn read_pinned(path: &Path) -> Result<PinnedBf16Tensor, BenchError> {
    let bytes = super::files::read_bounded(path, MAX_SAFETENSORS_BYTES, "safetensors input")?;
    parse_pinned(&bytes)
}

/// Deserialize through the safetensors crate, then apply the project's closed
/// single-tensor/provenance policy.
pub fn parse_pinned(bytes: &[u8]) -> Result<PinnedBf16Tensor, BenchError> {
    validate_embedded_spec()?;
    if bytes.len() > MAX_SAFETENSORS_BYTES {
        return Err(BenchError::new(
            "DFE-G1-LIMIT",
            "safetensors input exceeds the fixed size bound",
        ));
    }
    if bytes.len() < 8 {
        return Err(BenchError::new(
            "DFE-G1-SAFETENSORS",
            "safetensors input has no complete header length",
        ));
    }
    let header_length = u64::from_le_bytes(
        bytes[..8]
            .try_into()
            .expect("an eight-byte slice always converts"),
    );
    let header_length = usize::try_from(header_length).map_err(|_| {
        BenchError::new(
            "DFE-G1-LIMIT",
            "safetensors header length is not representable",
        )
    })?;
    let header_end = 8_usize
        .checked_add(header_length)
        .ok_or_else(|| BenchError::new("DFE-G1-LIMIT", "safetensors header length overflows"))?;
    if header_length == 0 || header_length > MAX_HEADER_BYTES || header_end > bytes.len() {
        return Err(BenchError::new(
            "DFE-G1-LIMIT",
            "safetensors header exceeds the fixed one-megabyte bound or is truncated",
        ));
    }
    let (_, metadata) = SafeTensors::read_metadata(bytes).map_err(|error| {
        BenchError::new(
            "DFE-G1-SAFETENSORS",
            format!("safetensors metadata deserialization failed: {error}"),
        )
    })?;
    let tensors = SafeTensors::deserialize(bytes).map_err(|error| {
        BenchError::new(
            "DFE-G1-SAFETENSORS",
            format!("safetensors deserialization failed: {error}"),
        )
    })?;
    let names = tensors.names();
    if names.len() != 1 || names[0] != TENSOR_NAME {
        return Err(BenchError::new(
            "DFE-G1-SAFETENSORS",
            format!("the pinned input must contain only tensor {TENSOR_NAME:?}, found {names:?}"),
        ));
    }
    let metadata = parse_metadata(metadata.metadata())?;
    let tensor = tensors.tensor(TENSOR_NAME).map_err(|error| {
        BenchError::new(
            "DFE-G1-SAFETENSORS",
            format!("pinned tensor lookup failed: {error}"),
        )
    })?;
    if tensor.dtype() != Dtype::BF16
        || tensor.shape() != [TENSOR_N, TENSOR_K]
        || tensor.data().len() != TENSOR_BYTES
    {
        return Err(BenchError::new(
            "DFE-G1-SAFETENSORS",
            "pinned tensor must be BF16 with shape [2048, 2048]",
        ));
    }
    let actual_tensor_hash = sha256_hex(tensor.data());
    if actual_tensor_hash != TENSOR_SHA256 {
        return Err(BenchError::new(
            "DFE-G1-SAFETENSORS",
            format!("tensor SHA-256 mismatch: expected {TENSOR_SHA256}, got {actual_tensor_hash}"),
        ));
    }
    let mut fp32_bits = Vec::with_capacity(TENSOR_N * TENSOR_K);
    for pair in tensor.data().as_chunks::<2>().0 {
        fp32_bits.push(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
    }
    Ok(PinnedBf16Tensor {
        semantic_identity: TENSOR_IDENTITY.to_owned(),
        metadata,
        fp32_bits,
    })
}

fn parse_metadata(
    metadata: &Option<HashMap<String, String>>,
) -> Result<BTreeMap<String, String>, BenchError> {
    let metadata = metadata.as_ref().ok_or_else(|| {
        BenchError::new(
            "DFE-G1-SAFETENSORS",
            "pinned safetensors input must provide provenance metadata",
        )
    })?;
    if metadata.len() != REQUIRED_METADATA.len()
        || REQUIRED_METADATA
            .iter()
            .any(|key| !metadata.contains_key(*key))
    {
        return Err(BenchError::new(
            "DFE-G1-SAFETENSORS",
            "provenance metadata must contain exactly the seven pinned fields",
        ));
    }
    let result = metadata
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<BTreeMap<_, _>>();
    if result.get("model").map(String::as_str) != Some(MODEL_ID)
        || result.get("revision").map(String::as_str) != Some(MODEL_REVISION)
        || result.get("model_filename").map(String::as_str) != Some(MODEL_FILENAME)
        || result.get("model_size_bytes").map(String::as_str)
            != Some(MODEL_SIZE_BYTES.to_string().as_str())
        || result.get("model_sha256").map(String::as_str) != Some(MODEL_SHA256)
        || result.get("tensor_name").map(String::as_str) != Some(TENSOR_NAME)
        || result.get("tensor_sha256").map(String::as_str) != Some(TENSOR_SHA256)
    {
        return Err(BenchError::new(
            "DFE-G1-SAFETENSORS",
            "provenance metadata does not match the pinned TinyLlama source",
        ));
    }
    for key in ["revision", "model_sha256", "tensor_sha256"] {
        let value = result.get(key).expect("required metadata");
        let expected_len = if key == "revision" { 40 } else { 64 };
        if value.len() != expected_len
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(BenchError::new(
                "DFE-G1-SAFETENSORS",
                format!("metadata field {key} is not lowercase hexadecimal"),
            ));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_constants_capture_real_source_provenance() {
        assert_eq!(TENSOR_NAME, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(super::super::spec::TENSOR_DTYPE, "BF16");
        assert_eq!([TENSOR_N, TENSOR_K], [2048, 2048]);
        assert_eq!(TENSOR_SHA256.len(), 64);
        assert_eq!(MODEL_REVISION.len(), 40);
    }

    #[test]
    fn malformed_input_fails_through_crate_deserializer() {
        assert!(parse_pinned(b"bad").is_err());
    }

    #[test]
    fn declared_header_limit_is_enforced_before_deserialization() {
        let bytes = ((super::super::spec::MAX_HEADER_BYTES + 1) as u64)
            .to_le_bytes()
            .to_vec();
        let error = parse_pinned(&bytes).unwrap_err().to_string();
        assert!(error.contains("one-megabyte bound"));
    }
}
