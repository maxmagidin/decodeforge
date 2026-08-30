//! Closed Q8 fixture and manifest parsing.
//!
//! The fixture wire format is intentionally typed and closed.  Serde handles
//! JSON syntax and escaping, `deny_unknown_fields` handles the schema surface,
//! and the small recursive visitor below adds the one policy serde_json does
//! not provide by default: duplicate object keys are rejected.

use super::{
    BLOCK_SIZE, FORMAT, NUMERIC_MODE, Q8Error, Q8Weights, canonical_linear_f32_bits,
    fixture_identity, logical_weight_identity, quantize_f32_bits,
};
use serde::Deserialize as DeriveDeserialize;
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use std::collections::BTreeSet;
use std::fmt;

const MANIFEST_NAME: &str = "manifest.json";
const EXPECTED_ERROR_POLICY: &str = "strict_f32_v1";
const EXPECTED_ERROR_COMPARATOR: &str = "dfq8_forward_v1";
const DUPLICATE_SENTINEL: &str = "decodeforge-private-duplicate-json-key:";

/// A typed artifact record from the closed fixture manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactRecord {
    /// Safe relative POSIX path under the manifest directory.
    pub path: String,
    /// Expected UTF-8 fixture byte length.
    pub bytes: u64,
    /// Lowercase SHA-256 digest without a prefix.
    pub sha256: String,
}

/// The closed root fixture manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixtureManifest {
    /// Pinned schema version.
    pub schema_version: u32,
    /// Pinned storage format.
    pub format: String,
    /// Pinned numeric mode.
    pub numeric_mode: String,
    /// Canonically sorted artifact records.
    pub artifacts: Vec<ArtifactRecord>,
}

/// The fixed policy/comparator declaration carried by every quant fixture.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixtureErrorBound {
    /// Pinned arithmetic policy.
    pub policy: String,
    /// Pinned comparator identifier.
    pub comparator: String,
}

/// A typed quantization fixture document.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantFixture {
    /// Pinned schema version.
    pub schema_version: u32,
    /// Operator name.
    pub operator: String,
    /// Operator version.
    pub operator_version: String,
    /// Storage format.
    pub format: String,
    /// Numeric mode.
    pub numeric_mode: String,
    /// Stable corpus case name.
    pub case_id: String,
    /// Output row count.
    pub n: u32,
    /// Logical input column count.
    pub k: u32,
    /// Physical block count.
    pub blocks: u32,
    /// Original source words, retained for identity inspection.
    pub source_fp32_bits: Vec<u32>,
    /// Serialized scale words.
    pub expected_scale_bits: Vec<u32>,
    /// Signed Q8 values in physical row-major order.
    pub expected_q_bytes: Vec<i16>,
    /// One logical input row.
    pub input_fp32_bits: Vec<u32>,
    /// Canonical scalar output words.
    pub expected_output_fp32_bits: Vec<u32>,
    /// Fixed error policy and comparator declaration.
    pub error_bound: FixtureErrorBound,
    /// Canonical identity of the logical weights.
    pub logical_weight_identity: String,
    /// Canonical identity of the complete fixture.
    pub fixture_identity: String,
}

#[derive(DeriveDeserialize)]
#[serde(deny_unknown_fields)]
struct ErrorBoundWire {
    policy: String,
    comparator: String,
}

#[derive(DeriveDeserialize)]
#[serde(deny_unknown_fields)]
struct FixtureWire {
    schema_version: u32,
    operator: String,
    operator_version: String,
    format: String,
    numeric_mode: String,
    case_id: String,
    n: u32,
    k: u32,
    blocks: u32,
    source_fp32_bits: Vec<u32>,
    expected_scale_bits: Vec<u32>,
    expected_q_bytes: Vec<i16>,
    input_fp32_bits: Vec<u32>,
    expected_output_fp32_bits: Vec<u32>,
    error_bound: ErrorBoundWire,
    logical_weight_identity: String,
    fixture_identity: String,
}

#[derive(DeriveDeserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactWire {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(DeriveDeserialize)]
#[serde(deny_unknown_fields)]
struct ManifestWire {
    schema_version: u32,
    format: String,
    numeric_mode: String,
    artifacts: Vec<ArtifactWire>,
}

/// A serde-only recursive value used to detect duplicate object members.
struct DuplicateValue;

impl<'de> Deserialize<'de> for DuplicateValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct DuplicateVisitor;

        impl<'de> Visitor<'de> for DuplicateVisitor {
            type Value = DuplicateValue;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("any JSON value")
            }

            fn visit_bool<E>(self, _: bool) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(DuplicateValue)
            }

            fn visit_i64<E>(self, _: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(DuplicateValue)
            }

            fn visit_u64<E>(self, _: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(DuplicateValue)
            }

            fn visit_f64<E>(self, _: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(DuplicateValue)
            }

            fn visit_str<E>(self, _: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(DuplicateValue)
            }

            fn visit_string<E>(self, _: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(DuplicateValue)
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(DuplicateValue)
            }

            fn visit_none<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(DuplicateValue)
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                while sequence.next_element::<DuplicateValue>()?.is_some() {}
                Ok(DuplicateValue)
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut keys = BTreeSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !keys.insert(key.clone()) {
                        return Err(de::Error::custom(format!("{DUPLICATE_SENTINEL}{key}")));
                    }
                    map.next_value::<DuplicateValue>()?;
                }
                Ok(DuplicateValue)
            }
        }

        deserializer.deserialize_any(DuplicateVisitor)
    }
}

fn schema_error(code: &'static str, summary: &'static str) -> Q8Error {
    Q8Error::new(code, summary).with("path", Vec::<String>::new())
}

fn schema_error_from_json(error: serde_json::Error) -> Q8Error {
    use serde_json::error::Category;

    let message = error.to_string();
    match error.classify() {
        Category::Syntax | Category::Eof => {
            schema_error("DFE-SCHEMA-001", "The JSON document could not be parsed.")
        }
        Category::Io | Category::Data => {
            if message.starts_with("unknown field") {
                schema_error(
                    "DFE-SCHEMA-005",
                    "The document does not satisfy its pinned schema.",
                )
            } else if message.starts_with("missing field") {
                schema_error(
                    "DFE-SCHEMA-004",
                    "The document does not satisfy its pinned schema.",
                )
            } else if message.starts_with("invalid type")
                || message.starts_with("invalid value")
                || message.starts_with("expected ")
            {
                schema_error(
                    "DFE-SCHEMA-006",
                    "The document does not satisfy its pinned schema.",
                )
            } else {
                schema_error(
                    "DFE-SCHEMA-007",
                    "The document does not satisfy its pinned schema.",
                )
            }
        }
    }
}

fn decode_wire<'de, T>(bytes: &'de [u8]) -> Result<T, Q8Error>
where
    T: Deserialize<'de>,
{
    let mut duplicate = serde_json::Deserializer::from_slice(bytes);
    if let Err(error) = DuplicateValue::deserialize(&mut duplicate) {
        let message = error.to_string();
        if message.starts_with(DUPLICATE_SENTINEL) {
            return Err(schema_error(
                "DFE-SCHEMA-008",
                "A JSON object contains a duplicate key.",
            ));
        }
        return Err(schema_error_from_json(error));
    }
    duplicate.end().map_err(schema_error_from_json)?;
    serde_json::from_slice(bytes).map_err(schema_error_from_json)
}

fn valid_content_identity(identity: &str) -> bool {
    identity.len() == 71
        && identity.starts_with("sha256:")
        && identity[7..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

/// Parse and semantically validate one closed quantization fixture.
pub fn parse_quant_fixture(bytes: &[u8]) -> Result<QuantFixture, Q8Error> {
    let wire: FixtureWire = decode_wire(bytes)?;
    if wire.schema_version != 1 {
        return Err(schema_error(
            "DFE-SCHEMA-002",
            "The document does not satisfy its pinned schema.",
        ));
    }
    if wire.operator != "q8_linear"
        || wire.operator_version != "q8_linear_v1"
        || wire.format != FORMAT
        || wire.numeric_mode != NUMERIC_MODE
        || wire.case_id.is_empty()
        || wire.n == 0
        || wire.k == 0
        || wire.blocks == 0
        || wire.error_bound.policy != EXPECTED_ERROR_POLICY
        || wire.error_bound.comparator != EXPECTED_ERROR_COMPARATOR
        || !valid_content_identity(&wire.logical_weight_identity)
        || !valid_content_identity(&wire.fixture_identity)
    {
        return Err(schema_error(
            "DFE-SCHEMA-007",
            "The document does not satisfy its pinned schema.",
        ));
    }
    if wire
        .expected_q_bytes
        .iter()
        .any(|value| !(-127..=127).contains(value))
    {
        return Err(schema_error(
            "DFE-SCHEMA-007",
            "The document does not satisfy its pinned schema.",
        ));
    }
    let computed = quantize_f32_bits(wire.n, wire.k, &wire.source_fp32_bits)?;
    let expected_source_len = (wire.n as usize)
        .checked_mul(wire.k as usize)
        .ok_or_else(|| Q8Error::new("DFE-QUANT-002", "Derived storage size overflows."))?;
    let expected_scale_len = (wire.n as usize)
        .checked_mul(computed.blocks as usize)
        .ok_or_else(|| Q8Error::new("DFE-QUANT-002", "Derived storage size overflows."))?;
    let expected_q_len = expected_scale_len
        .checked_mul(BLOCK_SIZE)
        .ok_or_else(|| Q8Error::new("DFE-QUANT-002", "Derived storage size overflows."))?;
    if wire.source_fp32_bits.len() != expected_source_len
        || wire.expected_scale_bits.len() != expected_scale_len
        || wire.expected_q_bytes.len() != expected_q_len
        || wire.input_fp32_bits.len() != wire.k as usize
        || wire.expected_output_fp32_bits.len() != wire.n as usize
    {
        return Err(Q8Error::new(
            "DFE-QUANT-003",
            "A fixture array length does not match its shape.",
        ));
    }
    if wire.blocks != computed.blocks {
        return Err(Q8Error::new(
            "DFE-QUANT-001",
            "The fixture block count does not equal ceil(K/32).",
        ));
    }
    if wire.expected_scale_bits != computed.scale_bits
        || wire.expected_q_bytes != computed.q_values()
    {
        return Err(Q8Error::new(
            "DFE-QUANT-011",
            "Fixture q or scale bytes do not match the canonical quantizer.",
        ));
    }
    if wire
        .expected_output_fp32_bits
        .iter()
        .any(|bits| !super::is_finite_f32_bits(*bits))
    {
        return Err(Q8Error::new(
            "DFE-QUANT-010",
            "An expected output word is not finite.",
        ));
    }
    let q_bytes = wire
        .expected_q_bytes
        .iter()
        .map(|value| *value as i8 as u8)
        .collect();
    let weights = Q8Weights::try_new(
        wire.n,
        wire.k,
        wire.blocks,
        q_bytes,
        wire.expected_scale_bits.clone(),
    )?;
    let canonical = canonical_linear_f32_bits(&wire.input_fp32_bits, &weights)?;
    if canonical != wire.expected_output_fp32_bits {
        return Err(Q8Error::new(
            "DFE-QUANT-011",
            "Fixture output bits do not match canonical scalar evaluation.",
        ));
    }
    if logical_weight_identity(&weights) != wire.logical_weight_identity {
        return Err(Q8Error::new(
            "DFE-QUANT-008",
            "The logical-weight identity does not match the fixture bytes.",
        ));
    }
    if fixture_identity(
        wire.n,
        wire.k,
        &wire.source_fp32_bits,
        &weights,
        &wire.input_fp32_bits,
        &wire.expected_output_fp32_bits,
    )? != wire.fixture_identity
    {
        return Err(Q8Error::new(
            "DFE-QUANT-008",
            "The fixture identity does not match the canonical preimage.",
        ));
    }
    Ok(QuantFixture {
        schema_version: wire.schema_version,
        operator: wire.operator,
        operator_version: wire.operator_version,
        format: wire.format,
        numeric_mode: wire.numeric_mode,
        case_id: wire.case_id,
        n: wire.n,
        k: wire.k,
        blocks: wire.blocks,
        source_fp32_bits: wire.source_fp32_bits,
        expected_scale_bits: wire.expected_scale_bits,
        expected_q_bytes: wire.expected_q_bytes,
        input_fp32_bits: wire.input_fp32_bits,
        expected_output_fp32_bits: wire.expected_output_fp32_bits,
        error_bound: FixtureErrorBound {
            policy: wire.error_bound.policy,
            comparator: wire.error_bound.comparator,
        },
        logical_weight_identity: wire.logical_weight_identity,
        fixture_identity: wire.fixture_identity,
    })
}

/// Parse one closed fixture manifest and validate path/canonical-order rules.
pub fn parse_fixture_manifest(bytes: &[u8]) -> Result<FixtureManifest, Q8Error> {
    let wire: ManifestWire = decode_wire(bytes)?;
    if wire.schema_version != 1 || wire.format != FORMAT || wire.numeric_mode != NUMERIC_MODE {
        return Err(schema_error(
            "DFE-SCHEMA-007",
            "The document does not satisfy its pinned schema.",
        ));
    }
    if wire.artifacts.is_empty() {
        return Err(schema_error(
            "DFE-SCHEMA-007",
            "The document does not satisfy its pinned schema.",
        ));
    }
    let mut artifacts = Vec::new();
    artifacts
        .try_reserve_exact(wire.artifacts.len())
        .map_err(|_| Q8Error::new("DFE-QUANT-002", "Derived storage size overflows."))?;
    for artifact in wire.artifacts {
        if !safe_relative_path(&artifact.path) || artifact.path == MANIFEST_NAME {
            return Err(Q8Error::new(
                "DFE-QUANT-012",
                "A fixture manifest path is unsafe or self-referential.",
            )
            .with("artifact", artifact.path));
        }
        if artifact.sha256.len() != 64
            || !artifact
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(schema_error(
                "DFE-SCHEMA-007",
                "The document does not satisfy its pinned schema.",
            ));
        }
        artifacts.push(ArtifactRecord {
            path: artifact.path,
            bytes: artifact.bytes,
            sha256: artifact.sha256,
        });
    }
    let mut seen = BTreeSet::new();
    for artifact in &artifacts {
        if !seen.insert(artifact.path.clone()) {
            return Err(Q8Error::new(
                "DFE-QUANT-012",
                "A fixture manifest path appears more than once.",
            )
            .with("artifact", &artifact.path));
        }
    }
    if artifacts
        .windows(2)
        .any(|pair| pair[0].path >= pair[1].path)
    {
        return Err(Q8Error::new(
            "DFE-QUANT-012",
            "Fixture manifest paths are not in canonical sorted order.",
        ));
    }
    Ok(FixtureManifest {
        schema_version: wire.schema_version,
        format: wire.format,
        numeric_mode: wire.numeric_mode,
        artifacts,
    })
}

fn safe_relative_path(path: &str) -> bool {
    if path.is_empty() || path.starts_with('/') || path.contains('\\') || path.contains("//") {
        return false;
    }
    path.split('/').all(|component| {
        !component.is_empty()
            && component != "."
            && component != ".."
            && component
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn parser_rejects_duplicate_and_unknown_fields() {
        let duplicate = br#"{"schema_version":1,"schema_version":1,"format":"DFQ8_B32_V1","numeric_mode":"strict_f32_v1","artifacts":[]}"#;
        assert_eq!(
            parse_fixture_manifest(duplicate).unwrap_err().code,
            "DFE-SCHEMA-008"
        );
        let unknown = br#"{"schema_version":1,"format":"DFQ8_B32_V1","numeric_mode":"strict_f32_v1","artifacts":[],"metadata":{}}"#;
        assert_eq!(
            parse_fixture_manifest(unknown).unwrap_err().code,
            "DFE-SCHEMA-005"
        );
        let unknown_named_duplicate = br#"{"schema_version":1,"format":"DFQ8_B32_V1","numeric_mode":"strict_f32_v1","artifacts":[],"duplicate key":{}}"#;
        assert_eq!(
            parse_fixture_manifest(unknown_named_duplicate)
                .unwrap_err()
                .code,
            "DFE-SCHEMA-005"
        );
        assert_eq!(
            parse_fixture_manifest(br#"{"schema_version":1"#)
                .unwrap_err()
                .code,
            "DFE-SCHEMA-001"
        );
        assert_eq!(
            parse_fixture_manifest(
                br#"{"schema_version":1,"format":"DFQ8_B32_V1","numeric_mode":"strict_f32_v1"}"#
            )
            .unwrap_err()
            .code,
            "DFE-SCHEMA-004"
        );
        assert_eq!(
            parse_fixture_manifest(br#"{"schema_version":"1","format":"DFQ8_B32_V1","numeric_mode":"strict_f32_v1","artifacts":[]}"#)
                .unwrap_err()
                .code,
            "DFE-SCHEMA-006"
        );
    }

    #[test]
    fn diagnostics_are_typed_closed_and_family_specific() {
        let dimension = quantize_f32_bits(0, 7, &[]).unwrap_err();
        let dimension_json: Value = serde_json::from_str(&dimension.diagnostic_json()).unwrap();
        assert_eq!(dimension_json["code"], "DFE-QUANT-001");
        assert_eq!(dimension_json["component"], "quant");
        assert_eq!(dimension_json["context"]["n"], 0);
        assert_eq!(dimension_json["context"]["k"], 7);

        let nonfinite = quantize_f32_bits(1, 1, &[0x7f80_0000]).unwrap_err();
        let nonfinite_json: Value = serde_json::from_str(&nonfinite.diagnostic_json()).unwrap();
        assert_eq!(nonfinite_json["context"]["field"], "source_fp32_bits");
        assert_eq!(nonfinite_json["context"]["index"], 0);
        assert_eq!(nonfinite_json["context"]["bits"], 0x7f80_0000u32);

        let malformed = parse_fixture_manifest(br#"{"#).unwrap_err();
        let malformed_json: Value = serde_json::from_str(&malformed.diagnostic_json()).unwrap();
        assert_eq!(malformed_json["code"], "DFE-SCHEMA-001");
        assert_eq!(malformed_json["component"], "schema");
        assert_eq!(malformed_json["context"]["path"], json!([]));

        let allowed_top = [
            "schema_version",
            "code",
            "severity",
            "component",
            "summary",
            "context",
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        assert_eq!(
            malformed_json
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            allowed_top
        );
    }
}
