use super::BenchError;
use super::cases::{CaseAsset, CaseBundle, SourceProvenance};
use super::files::sha256_hex;
use super::spec::{
    CALIBRATION_MAX_REPETITIONS, CALIBRATION_TARGET_NS, MIN_WARMUP_CALLS, MIN_WARMUP_NS,
    NUMERIC_MODE, PACK_FORMAT, PAIRED_ROUNDS, PROTOCOL_ID, SCALAR_FIRST_PAIRS, SESSION_FORMAT,
    validate_embedded_spec,
};
use decodeforge_compiler::{
    AppleNeonDylib, AppleScalarDylib, KernelVariant, LoopKernelV1, build_apple_neon_dylib,
    build_apple_scalar_dylib, emit_neon_c, emit_scalar_c, load_apple_neon_v1, load_apple_scalar_v1,
};
use decodeforge_runtime::{GeneratedExecutableV1, PreparedCallV1};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::hint::black_box;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const PROBE_OUTPUT_LIMIT: usize = 1024 * 1024;

#[derive(Debug, Serialize)]
pub struct SessionResult {
    pub schema_version: u32,
    pub protocol_id: &'static str,
    pub format: &'static str,
    pub session_id: String,
    pub spec_identity: String,
    pub case_bundle_identity: String,
    pub source: SourceProvenance,
    pub numeric_mode: &'static str,
    pub pack_format: &'static str,
    pub checkout: CheckoutInfo,
    pub host: HostInfo,
    pub timing: TimingPolicy,
    pub activation: Vec<ActivationRecord>,
    pub cases: Vec<CaseSessionResult>,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct HostInfo {
    pub os: &'static str,
    pub os_version: String,
    pub os_build: String,
    pub kernel_release: String,
    pub arch: &'static str,
    pub pointer_width: u32,
    pub native_supported: bool,
    pub cpu_model: String,
    pub hardware_model: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub features: Vec<&'static str>,
    pub process_id: u32,
    pub thread_policy: &'static str,
    pub affinity_policy: &'static str,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct CheckoutInfo {
    pub revision: String,
    pub dirty: bool,
}

#[derive(Debug, Serialize)]
pub struct TimingPolicy {
    pub boundary: &'static str,
    pub warmup_min_calls: u64,
    pub warmup_min_ns: u64,
    pub calibration_target_ns: u64,
    pub calibration_max_repetitions: u64,
    pub paired_rounds: usize,
    pub scalar_first_pairs: usize,
    pub neon_first_pairs: usize,
    pub trimming: &'static str,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct ActivationRecord {
    pub pair_index: usize,
    pub first_backend: &'static str,
    pub digest: String,
}

#[derive(Debug, Serialize)]
pub struct CaseSessionResult {
    pub case_id: String,
    pub kind: String,
    pub n: usize,
    pub k: usize,
    pub input_identity: String,
    pub expected_identity: String,
    pub logical_weight_identity: String,
    pub packed_weight_identity: String,
    pub region_ir: String,
    pub scalar_loop_ir: String,
    pub neon_loop_ir: String,
    pub pack_manifest: serde_json::Value,
    pub scalar: BackendSessionResult,
    pub neon: BackendSessionResult,
    pub samples: Vec<RawSample>,
}

#[derive(Debug, Serialize)]
pub struct BackendSessionResult {
    pub artifact: ArtifactEvidence,
    pub correctness: CorrectnessEvidence,
    pub warmup: WarmupRecord,
    pub calibration: CalibrationRecord,
}

#[derive(Debug, Serialize)]
pub struct CorrectnessEvidence {
    pub pre_timing_bit_exact: bool,
    pub post_timing_bit_exact: bool,
    pub expected_identity: String,
}

#[derive(Debug, Serialize)]
pub struct WarmupRecord {
    pub calls: u64,
    pub elapsed_ns: u64,
}

#[derive(Debug, Serialize)]
pub struct CalibrationRecord {
    pub target_ns: u64,
    pub max_repetitions: u64,
    pub selected_repetitions: u64,
    pub attempts: Vec<BatchRecord>,
}

#[derive(Debug, Serialize)]
pub struct BatchRecord {
    pub elapsed_ns: u64,
    pub repetitions: u64,
}

#[derive(Debug, Serialize)]
pub struct RawSample {
    pub pair_index: usize,
    pub backend: &'static str,
    pub position: &'static str,
    pub elapsed_ns: u64,
    pub repetitions: u64,
}

#[derive(Debug, Serialize)]
pub struct ArtifactEvidence {
    pub module_id: String,
    pub source_hash: String,
    pub abi_header_hash: String,
    pub dylib_hash: String,
    pub flags: Vec<String>,
    pub compiler: String,
    pub compiler_version: String,
    pub target: String,
    pub sdk_version: String,
    pub objdump_version: String,
    pub dynamic_exports: Vec<String>,
    pub source: String,
    pub disassembly: String,
    pub audit: AuditEvidence,
}

#[derive(Debug, Serialize)]
#[serde(tag = "backend", rename_all = "lowercase")]
pub enum AuditEvidence {
    Scalar {
        helper_symbol: String,
        scalar_scvtf_count: usize,
        scalar_fmul_count: usize,
        scalar_fadd_count: usize,
        return_count: usize,
        conditional_branch_count: usize,
        comparison_count: usize,
        logical_lane_loop_observed: bool,
    },
    Neon {
        helper_symbol: String,
        vector_path_observed: bool,
        scalar_tail_observed: bool,
        signed_widen_8_to_16_count: usize,
        signed_widen_16_to_32_count: usize,
        signed_q8_to_i32_count: usize,
        vector_scvtf_count: usize,
        vector_fmul_count: usize,
        vector_fadd_count: usize,
        vector_broadcast_count: usize,
        vector_store_count: usize,
        return_count: usize,
        conditional_branch_count: usize,
        logical_vector_lane_loop_observed: bool,
    },
}

struct BuiltBackend {
    executable: GeneratedExecutableV1,
    evidence: ArtifactEvidence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Backend {
    Scalar,
    Neon,
}

impl Backend {
    const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
        }
    }
}

pub fn run_session(bundle: &CaseBundle, session_id: &str) -> Result<SessionResult, BenchError> {
    validate_embedded_spec()?;
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Err(BenchError::new(
            "DFE-NATIVE-001",
            "run-session requires a macOS arm64 host; preparation and parsing remain portable",
        ));
    }
    let checkout = capture_checkout()?;
    if checkout.dirty {
        return Err(BenchError::new(
            "DFE-G1-HOST",
            "run-session requires the compiler's anchored checkout to be clean",
        ));
    }
    let host = capture_host()?;
    let first_case = bundle.cases.first().ok_or_else(|| {
        BenchError::new("DFE-G1-ASSET", "case bundle must contain one prepared case")
    })?;
    let activation = activation_order(session_id, &first_case.manifest.case_id);
    let mut cases = Vec::with_capacity(bundle.cases.len());
    for asset in &bundle.cases {
        cases.push(run_case(asset, &activation)?);
    }
    let final_checkout = capture_checkout()?;
    let final_host = capture_host()?;
    if final_checkout != checkout || final_host != host {
        return Err(BenchError::new(
            "DFE-G1-HOST",
            "checkout or host identity changed during the benchmark session",
        ));
    }
    Ok(SessionResult {
        schema_version: 1,
        protocol_id: PROTOCOL_ID,
        format: SESSION_FORMAT,
        session_id: session_id.to_owned(),
        spec_identity: bundle.manifest.spec_identity.clone(),
        case_bundle_identity: bundle.manifest_identity.clone(),
        source: bundle.manifest.source.clone(),
        numeric_mode: NUMERIC_MODE,
        pack_format: PACK_FORMAT,
        checkout,
        host,
        timing: TimingPolicy {
            boundary: "PreparedCall::invoke plus fixed Rust repetition loop and timer: sentinel fill + df_run_v1 + status decode + finite scan; no allocation",
            warmup_min_calls: MIN_WARMUP_CALLS,
            warmup_min_ns: MIN_WARMUP_NS as u64,
            calibration_target_ns: CALIBRATION_TARGET_NS as u64,
            calibration_max_repetitions: CALIBRATION_MAX_REPETITIONS,
            paired_rounds: PAIRED_ROUNDS,
            scalar_first_pairs: SCALAR_FIRST_PAIRS,
            neon_first_pairs: PAIRED_ROUNDS - SCALAR_FIRST_PAIRS,
            trimming: "none; every raw sample is retained",
        },
        activation,
        cases,
    })
}

fn run_case(
    asset: &CaseAsset,
    activation: &[ActivationRecord],
) -> Result<CaseSessionResult, BenchError> {
    let shape =
        decodeforge_compiler::Q8LinearShape::new(asset.manifest.n as u32, asset.manifest.k as u32)?;
    let region = decodeforge_compiler::Q8LinearRegion::new(
        shape,
        asset.manifest.logical_weight_identity.clone(),
    )?;
    let scalar_kernel = LoopKernelV1::new(&region, KernelVariant::Scalar)?;
    let neon_kernel = LoopKernelV1::new(&region, KernelVariant::Neon)?;
    let region_ir = region.canonical_json()?;
    let scalar_loop_ir = scalar_kernel.canonical_json()?;
    let neon_loop_ir = neon_kernel.canonical_json()?;
    let pack_manifest = serde_json::from_str(&asset.packed.canonical_manifest_json()?)?;
    let scalar = build_scalar(&region, &scalar_kernel, &asset.packed)?;
    let neon = build_neon(&region, &neon_kernel, &asset.packed)?;
    let expected_identity = asset.manifest.expected_identity.clone();
    let scalar_pre = exact_run(&scalar.executable, asset, "scalar")?;
    let neon_pre = exact_run(&neon.executable, asset, "neon")?;

    let mut scalar_output = vec![f32::NAN; asset.manifest.n];
    let mut neon_output = vec![f32::NAN; asset.manifest.n];
    let (scalar_warmup, neon_warmup, scalar_calibration, neon_calibration, samples) = {
        let mut scalar_call = scalar
            .executable
            .prepare_call(&asset.input, &mut scalar_output)
            .map_err(|error| context_error("scalar prepare_call", error))?;
        let mut neon_call = neon
            .executable
            .prepare_call(&asset.input, &mut neon_output)
            .map_err(|error| context_error("neon prepare_call", error))?;
        let scalar_warmup = warmup(&mut scalar_call, "scalar")?;
        let neon_warmup = warmup(&mut neon_call, "neon")?;
        let scalar_calibration = calibrate(&mut scalar_call, "scalar")?;
        let neon_calibration = calibrate(&mut neon_call, "neon")?;
        let scalar_repetitions = scalar_calibration.selected_repetitions;
        let neon_repetitions = neon_calibration.selected_repetitions;
        let mut samples = Vec::with_capacity(PAIRED_ROUNDS * 2);
        for record in activation {
            let (first, second) = if record.first_backend == "scalar" {
                (Backend::Scalar, Backend::Neon)
            } else {
                (Backend::Neon, Backend::Scalar)
            };
            measure_ordered(
                record.pair_index,
                first,
                &mut scalar_call,
                &mut neon_call,
                scalar_repetitions,
                neon_repetitions,
                &mut samples,
            )?;
            measure_ordered(
                record.pair_index,
                second,
                &mut scalar_call,
                &mut neon_call,
                scalar_repetitions,
                neon_repetitions,
                &mut samples,
            )?;
        }
        (
            scalar_warmup,
            neon_warmup,
            scalar_calibration,
            neon_calibration,
            samples,
        )
    };
    let scalar_post = exact_buffer(&scalar_output, asset, "scalar")?;
    let neon_post = exact_buffer(&neon_output, asset, "neon")?;

    Ok(CaseSessionResult {
        case_id: asset.manifest.case_id.clone(),
        kind: asset.manifest.kind.clone(),
        n: asset.manifest.n,
        k: asset.manifest.k,
        input_identity: asset.manifest.input_identity.clone(),
        expected_identity: expected_identity.clone(),
        logical_weight_identity: asset.manifest.logical_weight_identity.clone(),
        packed_weight_identity: asset.manifest.packed_weight_identity.clone(),
        region_ir,
        scalar_loop_ir,
        neon_loop_ir,
        pack_manifest,
        scalar: BackendSessionResult {
            artifact: scalar.evidence,
            correctness: CorrectnessEvidence {
                pre_timing_bit_exact: scalar_pre,
                post_timing_bit_exact: scalar_post,
                expected_identity: expected_identity.clone(),
            },
            warmup: scalar_warmup,
            calibration: scalar_calibration,
        },
        neon: BackendSessionResult {
            artifact: neon.evidence,
            correctness: CorrectnessEvidence {
                pre_timing_bit_exact: neon_pre,
                post_timing_bit_exact: neon_post,
                expected_identity,
            },
            warmup: neon_warmup,
            calibration: neon_calibration,
        },
        samples,
    })
}

fn build_scalar(
    region: &decodeforge_compiler::Q8LinearRegion,
    kernel: &LoopKernelV1,
    packed: &decodeforge_compiler::PackedWeightsV1,
) -> Result<BuiltBackend, BenchError> {
    let module = emit_scalar_c(region, kernel, packed)?;
    let source = module.source().to_owned();
    let artifact = build_apple_scalar_dylib(&module)?;
    let evidence = scalar_evidence(&artifact, source);
    let executable = load_apple_scalar_v1(artifact, region, kernel, packed)?;
    Ok(BuiltBackend {
        executable,
        evidence,
    })
}

fn build_neon(
    region: &decodeforge_compiler::Q8LinearRegion,
    kernel: &LoopKernelV1,
    packed: &decodeforge_compiler::PackedWeightsV1,
) -> Result<BuiltBackend, BenchError> {
    let module = emit_neon_c(region, kernel, packed)?;
    let source = module.source().to_owned();
    let artifact = build_apple_neon_dylib(&module)?;
    let evidence = neon_evidence(&artifact, source);
    let executable = load_apple_neon_v1(artifact, region, kernel, packed)?;
    Ok(BuiltBackend {
        executable,
        evidence,
    })
}

fn exact_run(
    executable: &GeneratedExecutableV1,
    asset: &CaseAsset,
    backend: &str,
) -> Result<bool, BenchError> {
    let actual = executable
        .run(&asset.input)
        .map_err(|error| context_error(backend, error))?;
    let actual_bits = actual
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    if actual_bits != asset.expected_bits {
        let mismatch = actual_bits
            .iter()
            .zip(&asset.expected_bits)
            .position(|(actual, expected)| actual != expected)
            .unwrap_or(0);
        return Err(BenchError::new(
            "DFE-G1-CORRECTNESS",
            format!("{backend} output differs at index {mismatch}"),
        ));
    }
    Ok(true)
}

fn exact_buffer(output: &[f32], asset: &CaseAsset, backend: &str) -> Result<bool, BenchError> {
    let actual_bits = output
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    if actual_bits != asset.expected_bits {
        let mismatch = actual_bits
            .iter()
            .zip(&asset.expected_bits)
            .position(|(actual, expected)| actual != expected)
            .unwrap_or(0);
        return Err(BenchError::new(
            "DFE-G1-CORRECTNESS",
            format!("{backend} prepared output differs at index {mismatch}"),
        ));
    }
    Ok(true)
}

fn warmup(call: &mut PreparedCallV1<'_, '_>, backend: &str) -> Result<WarmupRecord, BenchError> {
    let start = Instant::now();
    let mut calls = 0_u64;
    let deadline = start + Duration::from_secs(30);
    while calls < MIN_WARMUP_CALLS || start.elapsed().as_nanos() < MIN_WARMUP_NS {
        if Instant::now() >= deadline {
            return Err(BenchError::new(
                "DFE-G1-TIMING",
                format!("{backend} warmup exceeded the 30-second safety bound"),
            ));
        }
        let output = call
            .invoke()
            .map_err(|error| context_error(format!("{backend} warmup"), error))?;
        black_box(output.as_ptr());
        calls = calls.saturating_add(1);
    }
    Ok(WarmupRecord {
        calls,
        elapsed_ns: elapsed_ns(start.elapsed())?,
    })
}

fn calibrate(
    call: &mut PreparedCallV1<'_, '_>,
    backend: &str,
) -> Result<CalibrationRecord, BenchError> {
    let mut attempts = Vec::new();
    let mut repetitions = 1_u64;
    loop {
        let (elapsed, _) = measure_batch(call, repetitions, backend)?;
        let elapsed_ns = elapsed_ns(elapsed)?;
        attempts.push(BatchRecord {
            elapsed_ns,
            repetitions,
        });
        if elapsed.as_nanos() >= CALIBRATION_TARGET_NS {
            return Ok(CalibrationRecord {
                target_ns: CALIBRATION_TARGET_NS as u64,
                max_repetitions: CALIBRATION_MAX_REPETITIONS,
                selected_repetitions: repetitions,
                attempts,
            });
        }
        if repetitions >= CALIBRATION_MAX_REPETITIONS {
            return Err(BenchError::new(
                "DFE-G1-TIMING",
                format!("{backend} calibration did not reach 25ms before the repetition cap"),
            ));
        }
        repetitions = repetitions
            .checked_mul(2)
            .unwrap_or(CALIBRATION_MAX_REPETITIONS)
            .min(CALIBRATION_MAX_REPETITIONS);
    }
}

fn measure_ordered(
    pair_index: usize,
    backend: Backend,
    scalar: &mut PreparedCallV1<'_, '_>,
    neon: &mut PreparedCallV1<'_, '_>,
    scalar_repetitions: u64,
    neon_repetitions: u64,
    samples: &mut Vec<RawSample>,
) -> Result<(), BenchError> {
    let repetitions = match backend {
        Backend::Scalar => scalar_repetitions,
        Backend::Neon => neon_repetitions,
    };
    let (elapsed, _) = match backend {
        Backend::Scalar => measure_batch(scalar, repetitions, backend.name())?,
        Backend::Neon => measure_batch(neon, repetitions, backend.name())?,
    };
    samples.push(RawSample {
        pair_index,
        backend: backend.name(),
        position: if samples
            .iter()
            .any(|sample| sample.pair_index == pair_index && sample.backend != backend.name())
        {
            "second"
        } else {
            "first"
        },
        elapsed_ns: elapsed_ns(elapsed)?,
        repetitions,
    });
    Ok(())
}

fn measure_batch(
    call: &mut PreparedCallV1<'_, '_>,
    repetitions: u64,
    backend: &str,
) -> Result<(Duration, u64), BenchError> {
    let start = Instant::now();
    let mut last_output_address = 0_usize;
    for _ in 0..repetitions {
        let output = call
            .invoke()
            .map_err(|error| context_error(format!("{backend} timing"), error))?;
        last_output_address = output.as_ptr() as usize;
    }
    let elapsed = start.elapsed();
    black_box(last_output_address);
    Ok((elapsed, repetitions))
}

fn elapsed_ns(duration: Duration) -> Result<u64, BenchError> {
    u64::try_from(duration.as_nanos())
        .map_err(|_| BenchError::new("DFE-G1-TIMING", "elapsed duration does not fit u64 ns"))
}

fn activation_order(session_id: &str, case_id: &str) -> Vec<ActivationRecord> {
    let mut ranks = (0..PAIRED_ROUNDS)
        .map(|pair_index| {
            (
                activation_digest(session_id, case_id, pair_index),
                pair_index,
            )
        })
        .collect::<Vec<_>>();
    ranks.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
    let mut scalar_first = [false; PAIRED_ROUNDS];
    for (_, pair_index) in ranks.iter().take(SCALAR_FIRST_PAIRS) {
        scalar_first[*pair_index] = true;
    }
    (0..PAIRED_ROUNDS)
        .map(|pair_index| {
            let digest = activation_digest(session_id, case_id, pair_index);
            ActivationRecord {
                pair_index,
                first_backend: if scalar_first[pair_index] {
                    "scalar"
                } else {
                    "neon"
                },
                digest: sha256_hex(&digest),
            }
        })
        .collect()
}

fn activation_digest(session_id: &str, case_id: &str, pair_index: usize) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"DecodeForge/G1/sha256-counter/v1/activation\0");
    hasher.update((session_id.len() as u64).to_le_bytes());
    hasher.update(session_id.as_bytes());
    hasher.update((case_id.len() as u64).to_le_bytes());
    hasher.update(case_id.as_bytes());
    hasher.update((pair_index as u64).to_le_bytes());
    hasher.finalize().into()
}

fn probe_in(
    current_dir: Option<&Path>,
    program: &str,
    arguments: &[&str],
    label: &str,
) -> Result<String, BenchError> {
    let mut command = Command::new(program);
    command
        .args(arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    if let Some(directory) = current_dir {
        command.current_dir(directory);
    }
    let mut child = command.spawn().map_err(|error| {
        BenchError::new(
            "DFE-G1-HOST",
            format!("unable to start {label} probe: {error}"),
        )
    })?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| BenchError::new("DFE-G1-HOST", "probe stdout was unavailable"))?;
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let count = match stdout.read(&mut buffer) {
            Ok(count) => count,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(BenchError::new(
                    "DFE-G1-HOST",
                    format!("unable to read {label} probe: {error}"),
                ));
            }
        };
        if count == 0 {
            break;
        }
        if bytes.len().saturating_add(count) > PROBE_OUTPUT_LIMIT {
            let _ = child.kill();
            let _ = child.wait();
            return Err(BenchError::new(
                "DFE-G1-HOST",
                format!("{label} probe exceeded its output bound"),
            ));
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    let status = child.wait().map_err(|error| {
        BenchError::new(
            "DFE-G1-HOST",
            format!("unable to wait for {label} probe: {error}"),
        )
    })?;
    if !status.success() {
        return Err(BenchError::new(
            "DFE-G1-HOST",
            format!("{label} probe failed"),
        ));
    }
    String::from_utf8(bytes)
        .map(|value| value.trim().to_owned())
        .map_err(|_| BenchError::new("DFE-G1-HOST", format!("{label} probe was not UTF-8")))
}

fn probe(program: &str, arguments: &[&str], label: &str) -> Result<String, BenchError> {
    probe_in(None, program, arguments, label)
}

fn parse_probe_u32(program: &str, arguments: &[&str], label: &str) -> Result<u32, BenchError> {
    let value = probe(program, arguments, label)?;
    value.parse::<u32>().map_err(|_| {
        BenchError::new(
            "DFE-G1-HOST",
            format!("{label} probe did not return an unsigned integer"),
        )
    })
}

fn capture_checkout() -> Result<CheckoutInfo, BenchError> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .map_err(|error| {
            BenchError::new(
                "DFE-G1-HOST",
                format!("compiler workspace root is unavailable: {error}"),
            )
        })?;
    let reported_root = probe_in(
        Some(&root),
        "git",
        &["rev-parse", "--show-toplevel"],
        "Git workspace root",
    )?;
    let reported_root = PathBuf::from(reported_root)
        .canonicalize()
        .map_err(|error| {
            BenchError::new(
                "DFE-G1-HOST",
                format!("reported Git workspace root is unavailable: {error}"),
            )
        })?;
    if reported_root != root {
        return Err(BenchError::new(
            "DFE-G1-HOST",
            "benchmark binary is not anchored to its compiler workspace",
        ));
    }
    let revision = probe_in(Some(&root), "git", &["rev-parse", "HEAD"], "Git revision")?;
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(BenchError::new(
            "DFE-G1-HOST",
            "Git revision probe did not return one full lowercase object ID",
        ));
    }
    let status = probe_in(
        Some(&root),
        "git",
        &["status", "--porcelain=v1", "--untracked-files=normal"],
        "Git status",
    )?;
    Ok(CheckoutInfo {
        revision,
        dirty: !status.is_empty(),
    })
}

fn capture_host() -> Result<HostInfo, BenchError> {
    if probe("sysctl", &["-n", "hw.optional.neon"], "NEON feature")? != "1" {
        return Err(BenchError::new(
            "DFE-G1-HOST",
            "required Apple NEON feature is unavailable",
        ));
    }
    Ok(HostInfo {
        os: std::env::consts::OS,
        os_version: probe("sw_vers", &["-productVersion"], "macOS version")?,
        os_build: probe("sw_vers", &["-buildVersion"], "macOS build")?,
        kernel_release: probe("uname", &["-r"], "kernel release")?,
        arch: std::env::consts::ARCH,
        pointer_width: usize::BITS,
        native_supported: true,
        cpu_model: probe("sysctl", &["-n", "machdep.cpu.brand_string"], "CPU model")?,
        hardware_model: probe("sysctl", &["-n", "hw.model"], "hardware model")?,
        physical_cores: parse_probe_u32(
            "sysctl",
            &["-n", "hw.physicalcpu"],
            "physical core count",
        )?,
        logical_cores: parse_probe_u32("sysctl", &["-n", "hw.logicalcpu"], "logical core count")?,
        features: vec!["neon"],
        process_id: std::process::id(),
        thread_policy: "single calling thread; generated kernels create no workers",
        affinity_policy: "macOS default scheduler; no hard affinity requested",
    })
}

fn context_error(context: impl Into<String>, error: impl std::fmt::Display) -> BenchError {
    BenchError::new("DFE-G1-NATIVE", format!("{}: {error}", context.into()))
}

fn scalar_evidence(artifact: &AppleScalarDylib, source: String) -> ArtifactEvidence {
    let toolchain = artifact.toolchain();
    let audit = artifact.audit_report();
    ArtifactEvidence {
        module_id: artifact.module_id().to_owned(),
        source_hash: artifact.source_hash().to_owned(),
        abi_header_hash: artifact.abi_header_hash().to_owned(),
        dylib_hash: artifact.dylib_hash().to_owned(),
        flags: artifact.flags().to_vec(),
        compiler: toolchain.compiler().to_owned(),
        compiler_version: toolchain.compiler_version().to_owned(),
        target: toolchain.target().to_owned(),
        sdk_version: toolchain.sdk_version().to_owned(),
        objdump_version: toolchain.objdump_version().to_owned(),
        dynamic_exports: artifact.dynamic_exports().to_vec(),
        source,
        disassembly: artifact.disassembly().to_owned(),
        audit: AuditEvidence::Scalar {
            helper_symbol: audit.helper_symbol().to_owned(),
            scalar_scvtf_count: audit.scalar_scvtf_count(),
            scalar_fmul_count: audit.scalar_fmul_count(),
            scalar_fadd_count: audit.scalar_fadd_count(),
            return_count: audit.return_count(),
            conditional_branch_count: audit.conditional_branch_count(),
            comparison_count: audit.comparison_count(),
            logical_lane_loop_observed: audit.logical_lane_loop_observed(),
        },
    }
}

fn neon_evidence(artifact: &AppleNeonDylib, source: String) -> ArtifactEvidence {
    let toolchain = artifact.toolchain();
    let audit = artifact.audit_report();
    ArtifactEvidence {
        module_id: artifact.module_id().to_owned(),
        source_hash: artifact.source_hash().to_owned(),
        abi_header_hash: artifact.abi_header_hash().to_owned(),
        dylib_hash: artifact.dylib_hash().to_owned(),
        flags: artifact.flags().to_vec(),
        compiler: toolchain.compiler().to_owned(),
        compiler_version: toolchain.compiler_version().to_owned(),
        target: toolchain.target().to_owned(),
        sdk_version: toolchain.sdk_version().to_owned(),
        objdump_version: toolchain.objdump_version().to_owned(),
        dynamic_exports: artifact.dynamic_exports().to_vec(),
        source,
        disassembly: artifact.disassembly().to_owned(),
        audit: AuditEvidence::Neon {
            helper_symbol: audit.helper_symbol().to_owned(),
            vector_path_observed: audit.vector_path_observed(),
            scalar_tail_observed: audit.scalar_tail_observed(),
            signed_widen_8_to_16_count: audit.signed_widen_8_to_16_count(),
            signed_widen_16_to_32_count: audit.signed_widen_16_to_32_count(),
            signed_q8_to_i32_count: audit.signed_q8_to_i32_count(),
            vector_scvtf_count: audit.vector_scvtf_count(),
            vector_fmul_count: audit.vector_fmul_count(),
            vector_fadd_count: audit.vector_fadd_count(),
            vector_broadcast_count: audit.vector_broadcast_count(),
            vector_store_count: audit.vector_store_count(),
            return_count: audit.return_count(),
            conditional_branch_count: audit.conditional_branch_count(),
            logical_vector_lane_loop_observed: audit.logical_vector_lane_loop_observed(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_has_exactly_twenty_pairs_in_each_order() {
        let records = activation_order("session-a", "case");
        assert_eq!(records.len(), PAIRED_ROUNDS);
        assert_eq!(
            records
                .iter()
                .filter(|record| record.first_backend == "scalar")
                .count(),
            SCALAR_FIRST_PAIRS
        );
        assert_eq!(
            records
                .iter()
                .filter(|record| record.first_backend == "neon")
                .count(),
            PAIRED_ROUNDS - SCALAR_FIRST_PAIRS
        );
        assert_eq!(records, activation_order("session-a", "case"));
        assert_ne!(records, activation_order("session-b", "case"));
    }
}
