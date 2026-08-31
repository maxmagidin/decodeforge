use super::BenchError;
use super::files::sha256_identity;
use decodeforge_compiler::{APPLE_NEON_CLANG_FLAGS, APPLE_SCALAR_CLANG_FLAGS};
use serde::Deserialize;

pub const SPEC_BYTES: &[u8] = include_bytes!("../../../../../benchmarks/g1/spec.json");
pub const PROTOCOL_ID: &str = "g1-prepared-call-paired-v1";
pub const SESSION_FORMAT: &str = "decodeforge_g1_benchmark_v1";
pub const CASE_BUNDLE_FORMAT: &str = "decodeforge_g1_cases_v1";
pub const NUMERIC_MODE: &str = "strict_f32_v1";
pub const PACK_FORMAT: &str = "DFQ8_B32_OI4_V1";
pub const ACTIVATION_STREAM: &str = "sha256-counter-v1";

pub const MODEL_ID: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
pub const MODEL_REVISION: &str = "fe8a4ea1ffedaf415f4da2f062534de366a451e6";
pub const MODEL_FILENAME: &str = "model.safetensors";
pub const MODEL_SIZE_BYTES: u64 = 2_200_119_864;
pub const MODEL_SHA256: &str = "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933";
pub const TENSOR_NAME: &str = "model.layers.0.self_attn.q_proj.weight";
pub const TENSOR_DTYPE: &str = "BF16";
pub const TENSOR_N: usize = 2048;
pub const TENSOR_K: usize = 2048;
pub const TENSOR_BYTES: usize = TENSOR_N * TENSOR_K * 2;
pub const TENSOR_SHA256: &str = "5abf98c51f903941a1592f3df83e2e56ca7149252f5d6665c7662927c83008ac";
pub const TENSOR_IDENTITY: &str =
    "sha256:5abf98c51f903941a1592f3df83e2e56ca7149252f5d6665c7662927c83008ac";

pub const REAL_CASE_ID: &str = "tinyllama-q-proj-2048x2048";
pub const REAL_INPUT_IDENTITY: &str =
    "sha256:03263339062e7a0839f45a28b256b1e1585e5a35e4281853283049017859c590";
pub const REAL_EXPECTED_IDENTITY: &str =
    "sha256:96d06e866b38c28e2f08acdfb6055515b95dc63aede37dddf3b4315b0e5e2f4a";
pub const REAL_LOGICAL_WEIGHT_IDENTITY: &str =
    "sha256:07c6e1c13a280960451fae4698d09dd48a0d0af2b24f29eebc062d86da3253e2";
pub const REAL_PACKED_WEIGHT_IDENTITY: &str =
    "sha256:75641573aa3deae8fe3754919ab2af644546ee79121a3ee6d06f1cecaa872efc";

pub const MIN_WARMUP_CALLS: u64 = 16;
pub const MIN_WARMUP_NS: u128 = 500_000_000;
pub const CALIBRATION_TARGET_NS: u128 = 25_000_000;
pub const CALIBRATION_MAX_REPETITIONS: u64 = 1_048_576;
pub const PAIRED_ROUNDS: usize = 40;
pub const SCALAR_FIRST_PAIRS: usize = 20;
pub const INDEPENDENT_SESSIONS: usize = 3;
pub const DRIFT_REJECTION_FRACTION: f64 = 0.10;
pub const BOOTSTRAP_REPLICATES: usize = 10_000;
pub const BOOTSTRAP_SEED: &str = "sha256-counter-v1/bootstrap";

pub const MAX_SAFETENSORS_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_HEADER_BYTES: usize = 1024 * 1024;
pub const MAX_CASE_FILE_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_JSON_BYTES: usize = 16 * 1024 * 1024;

const METADATA_KEYS: [&str; 7] = [
    "model",
    "revision",
    "model_filename",
    "model_size_bytes",
    "model_sha256",
    "tensor_name",
    "tensor_sha256",
];
const PLANNED_TAIL_SHAPES: [[usize; 2]; 3] = [[2051, 2048], [2048, 2049], [2051, 2049]];

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ProtocolSpec {
    schema_version: u32,
    protocol_id: String,
    format: String,
    numeric_mode: String,
    pack_format: String,
    primary_host: PrimaryHostSpec,
    activation_stream: String,
    real_tensor: RealTensor,
    real_case: RealCase,
    artifact_policy: ArtifactPolicySpec,
    planned_not_executed_synthetic_tail_shapes: Vec<[usize; 2]>,
    timing: TimingSpec,
    limits: LimitsSpec,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct PrimaryHostSpec {
    os: String,
    os_version: String,
    os_build: String,
    kernel_release: String,
    arch: String,
    pointer_width: u32,
    native_supported: bool,
    cpu_model: String,
    hardware_model: String,
    physical_cores: u32,
    logical_cores: u32,
    features: Vec<String>,
    thread_policy: String,
    affinity_policy: String,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RealTensor {
    model: String,
    tensor_name: String,
    dtype: String,
    shape: [usize; 2],
    revision: String,
    model_file: ModelFile,
    raw_tensor_sha256: String,
    metadata_keys: Vec<String>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ModelFile {
    filename: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RealCase {
    case_id: String,
    input_identity: String,
    expected_identity: String,
    logical_weight_identity: String,
    packed_weight_identity: String,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ArtifactPolicySpec {
    real_case_id: String,
    scalar_source_identity: String,
    neon_source_identity: String,
    scalar_disassembly_identity: String,
    neon_disassembly_identity: String,
    scalar: ArtifactBackendPolicySpec,
    neon: ArtifactBackendPolicySpec,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ArtifactBackendPolicySpec {
    compiler: String,
    compiler_version: String,
    target: String,
    sdk_version: String,
    objdump_version: String,
    abi_header_identity: String,
    dynamic_exports: Vec<String>,
    flags: Vec<String>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct TimingSpec {
    warmup_min_calls: u64,
    warmup_min_ns: u128,
    calibration_target_ns: u128,
    calibration_max_repetitions: u64,
    paired_rounds: usize,
    scalar_first_pairs: usize,
    neon_first_pairs: usize,
    independent_sessions: usize,
    boundary: String,
    degenerate_bca_policy: String,
    confidence_level: f64,
    drift_window_pairs: usize,
    effect_estimator: String,
    claim_rule: String,
    aggregate_policy: String,
    descriptive_latency_methods: Vec<String>,
    drift_rejection_fraction: f64,
    bootstrap_method: String,
    bootstrap_replicates: usize,
    bootstrap_seed: String,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct LimitsSpec {
    max_safetensors_bytes: usize,
    max_header_bytes: usize,
    max_output_bytes: usize,
    max_json_bytes: usize,
}

impl ProtocolSpec {
    fn expected() -> Self {
        Self {
            schema_version: 1,
            protocol_id: PROTOCOL_ID.to_owned(),
            format: SESSION_FORMAT.to_owned(),
            numeric_mode: NUMERIC_MODE.to_owned(),
            pack_format: PACK_FORMAT.to_owned(),
            primary_host: PrimaryHostSpec {
                os: "macos".to_owned(),
                os_version: "15.5".to_owned(),
                os_build: "24F74".to_owned(),
                kernel_release: "24.5.0".to_owned(),
                arch: "aarch64".to_owned(),
                pointer_width: 64,
                native_supported: true,
                cpu_model: "Apple M4".to_owned(),
                hardware_model: "Mac16,13".to_owned(),
                physical_cores: 10,
                logical_cores: 10,
                features: vec!["neon".to_owned()],
                thread_policy:
                    "single calling thread; generated kernels create no workers".to_owned(),
                affinity_policy: "macOS default scheduler; no hard affinity requested".to_owned(),
            },
            activation_stream: ACTIVATION_STREAM.to_owned(),
            real_tensor: RealTensor {
                model: MODEL_ID.to_owned(),
                tensor_name: TENSOR_NAME.to_owned(),
                dtype: TENSOR_DTYPE.to_owned(),
                shape: [TENSOR_N, TENSOR_K],
                revision: MODEL_REVISION.to_owned(),
                model_file: ModelFile {
                    filename: MODEL_FILENAME.to_owned(),
                    size_bytes: MODEL_SIZE_BYTES,
                    sha256: MODEL_SHA256.to_owned(),
                },
                raw_tensor_sha256: TENSOR_SHA256.to_owned(),
                metadata_keys: METADATA_KEYS
                    .iter()
                    .map(|value| (*value).to_owned())
                    .collect(),
            },
            real_case: RealCase {
                case_id: REAL_CASE_ID.to_owned(),
                input_identity: REAL_INPUT_IDENTITY.to_owned(),
                expected_identity: REAL_EXPECTED_IDENTITY.to_owned(),
                logical_weight_identity: REAL_LOGICAL_WEIGHT_IDENTITY.to_owned(),
                packed_weight_identity: REAL_PACKED_WEIGHT_IDENTITY.to_owned(),
            },
            artifact_policy: ArtifactPolicySpec {
                real_case_id: REAL_CASE_ID.to_owned(),
                scalar_source_identity: "sha256:99aca3f9ed5177ce4abc7d1990bfb28374fb14cfa411081a2bc6cf767e28b58c"
                    .to_owned(),
                neon_source_identity: "sha256:d803253668b9568a9a4c0100ada42d05a9045a5ef30d56ba6099b9253c3a26de"
                    .to_owned(),
                scalar_disassembly_identity: "sha256:6f5aae5a5f3623336892b38c298ea7f00236adb5f177567f22f9bc136169c3cd"
                    .to_owned(),
                neon_disassembly_identity: "sha256:05cdb5e3727da34d842e9f2903745bc714e144356c8405a9e9867714e8f173cb"
                    .to_owned(),
                scalar: ArtifactBackendPolicySpec {
                    compiler: "clang".to_owned(),
                    compiler_version: "Apple clang version 17.0.0 (clang-1700.0.13.5) Target: arm64-apple-darwin24.5.0 Thread model: posix InstalledDir: /Library/Developer/CommandLineTools/usr/bin".to_owned(),
                    target: "arm64-apple-darwin24.5.0".to_owned(),
                    sdk_version: "15.5".to_owned(),
                    objdump_version: "Apple LLVM version 17.0.0 Optimized build. Registered Targets: aarch64 - AArch64 (little endian) aarch64_32 - AArch64 (little endian ILP32) aarch64_be - AArch64 (big endian) arm - ARM arm64 - ARM64 (little endian) arm64_32 - ARM64 (little endian ILP32) armeb - ARM (big endian) thumb - Thumb thumbeb - Thumb (big endian) x86 - 32-bit X86: Pentium-Pro and above x86-64 - 64-bit X86: EM64T and AMD64".to_owned(),
                    abi_header_identity: "sha256:7e455301d59979be04e79b83a47e3b1bd7a143e681db01227b86557d989966cc".to_owned(),
                    dynamic_exports: vec![
                        "df_abi_version".to_owned(),
                        "df_artifact_id".to_owned(),
                        "df_run_v1".to_owned(),
                    ],
                    flags: APPLE_SCALAR_CLANG_FLAGS
                        .iter()
                        .map(|value| (*value).to_owned())
                        .collect::<Vec<_>>(),
                },
                neon: ArtifactBackendPolicySpec {
                    compiler: "clang".to_owned(),
                    compiler_version: "Apple clang version 17.0.0 (clang-1700.0.13.5) Target: arm64-apple-darwin24.5.0 Thread model: posix InstalledDir: /Library/Developer/CommandLineTools/usr/bin".to_owned(),
                    target: "arm64-apple-darwin24.5.0".to_owned(),
                    sdk_version: "15.5".to_owned(),
                    objdump_version: "Apple LLVM version 17.0.0 Optimized build. Registered Targets: aarch64 - AArch64 (little endian) aarch64_32 - AArch64 (little endian ILP32) aarch64_be - AArch64 (big endian) arm - ARM arm64 - ARM64 (little endian) arm64_32 - ARM64 (little endian ILP32) armeb - ARM (big endian) thumb - Thumb thumbeb - Thumb (big endian) x86 - 32-bit X86: Pentium-Pro and above x86-64 - 64-bit X86: EM64T and AMD64".to_owned(),
                    abi_header_identity: "sha256:7e455301d59979be04e79b83a47e3b1bd7a143e681db01227b86557d989966cc".to_owned(),
                    dynamic_exports: vec![
                        "df_abi_version".to_owned(),
                        "df_artifact_id".to_owned(),
                        "df_run_v1".to_owned(),
                    ],
                    flags: APPLE_NEON_CLANG_FLAGS
                        .iter()
                        .map(|value| (*value).to_owned())
                        .collect::<Vec<_>>(),
                },
            },
            planned_not_executed_synthetic_tail_shapes: PLANNED_TAIL_SHAPES.to_vec(),
            timing: TimingSpec {
                warmup_min_calls: MIN_WARMUP_CALLS,
                warmup_min_ns: MIN_WARMUP_NS,
                calibration_target_ns: CALIBRATION_TARGET_NS,
                calibration_max_repetitions: CALIBRATION_MAX_REPETITIONS,
                paired_rounds: PAIRED_ROUNDS,
                scalar_first_pairs: SCALAR_FIRST_PAIRS,
                neon_first_pairs: PAIRED_ROUNDS - SCALAR_FIRST_PAIRS,
                independent_sessions: INDEPENDENT_SESSIONS,
                boundary: "PreparedCall::invoke plus fixed Rust repetition loop and timer: sentinel fill + df_run_v1 + status decode + finite scan; no allocation"
                    .to_owned(),
                degenerate_bca_policy: "reject".to_owned(),
                confidence_level: 0.95,
                drift_window_pairs: 10,
                effect_estimator: "exp_median_log_paired_latency_ratio".to_owned(),
                claim_rule: "real_tinyllama_2048x2048_and_all_three_session_lower_ci_bounds_gt_1"
                    .to_owned(),
                aggregate_policy: "pooled_point_descriptive_only_no_ci_no_claim".to_owned(),
                descriptive_latency_methods: vec![
                    "median".to_owned(),
                    "median_absolute_deviation".to_owned(),
                    "nearest_rank_p95".to_owned(),
                ],
                drift_rejection_fraction: DRIFT_REJECTION_FRACTION,
                bootstrap_method: "paired_bca".to_owned(),
                bootstrap_replicates: BOOTSTRAP_REPLICATES,
                bootstrap_seed: BOOTSTRAP_SEED.to_owned(),
            },
            limits: LimitsSpec {
                max_safetensors_bytes: MAX_SAFETENSORS_BYTES,
                max_header_bytes: MAX_HEADER_BYTES,
                max_output_bytes: MAX_CASE_FILE_BYTES,
                max_json_bytes: MAX_JSON_BYTES,
            },
        }
    }
}

pub fn validate_embedded_spec() -> Result<(), BenchError> {
    let observed: ProtocolSpec = serde_json::from_slice(SPEC_BYTES).map_err(|error| {
        BenchError::new(
            "DFE-G1-SPEC",
            format!("embedded benchmark spec is invalid: {error}"),
        )
    })?;
    if observed != ProtocolSpec::expected() {
        return Err(BenchError::new(
            "DFE-G1-SPEC",
            "embedded benchmark spec does not match the executable protocol",
        ));
    }
    Ok(())
}

pub fn spec_identity() -> String {
    sha256_identity(SPEC_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_spec_is_typed_and_matches_every_runner_constant() {
        validate_embedded_spec().unwrap();
        assert!(spec_identity().starts_with("sha256:"));
    }
}
