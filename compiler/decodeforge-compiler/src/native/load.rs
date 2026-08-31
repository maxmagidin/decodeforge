//! Safe binding of one compiler-owned dylib to one verified packed payload.

use super::{AppleNeonDylib, AppleScalarDylib};
use crate::{
    LoopKernelV1, PackedWeightsV1, Q8LinearRegion, Result, emit_neon_c, emit_scalar_c, invalid,
};

/// Safe owner returned after a checked scalar compiler-to-runtime handoff.
///
/// This remains an exact alias of the backend-neutral generated executable.
pub type AppleScalarExecutableV1 = decodeforge_runtime::GeneratedExecutableV1;

/// Safe owner returned after a checked NEON compiler-to-runtime handoff.
pub type AppleNeonExecutableV1 = decodeforge_runtime::GeneratedExecutableV1;

/// Exact generated `df_run_v1` status vocabulary shared by every backend.
pub use decodeforge_runtime::GeneratedStatusV1;
/// Structured failures returned by either generated executable.
pub use decodeforge_runtime::RuntimeError as GeneratedRuntimeError;

/// Source-compatible scalar spelling of [`GeneratedRuntimeError`].
pub use decodeforge_runtime::RuntimeError as ScalarRuntimeError;
/// Source-compatible scalar spelling of [`GeneratedStatusV1`].
pub use decodeforge_runtime::ScalarStatusV1;

/// Private proof surface implemented only by compiler-owned, audited images.
///
/// Keeping this trait private prevents an ordinary caller from manufacturing
/// a value that can reach the runtime's trusted-image constructor.
trait AuditedAppleGeneratedDylibV1 {
    const BACKEND_NAME: &'static str;

    fn module_id(&self) -> &str;
    fn dylib_bytes(&self) -> &[u8];
}

impl AuditedAppleGeneratedDylibV1 for AppleScalarDylib {
    const BACKEND_NAME: &'static str = "scalar";

    fn module_id(&self) -> &str {
        AppleScalarDylib::module_id(self)
    }

    fn dylib_bytes(&self) -> &[u8] {
        AppleScalarDylib::dylib_bytes(self)
    }
}

impl AuditedAppleGeneratedDylibV1 for AppleNeonDylib {
    const BACKEND_NAME: &'static str = "NEON";

    fn module_id(&self) -> &str {
        AppleNeonDylib::module_id(self)
    }

    fn dylib_bytes(&self) -> &[u8] {
        AppleNeonDylib::dylib_bytes(self)
    }
}

/// Bind a locally built scalar module to an exact verified pack and load it.
///
/// The build owner is consumed. The runtime synchronously reconstructs the
/// pack in aligned storage and makes its own byte-verified private dylib copy,
/// so neither the compiler build directory nor `packed` has to outlive the
/// returned executable.
///
/// This safe handoff requires the native builder to retain and report the
/// exact snapshot on which all pre-load Mach-O audits were performed. It must
/// not be integrated on top of a builder that audits a different pathname.
/// Loading rejects every visible `DYLD_*` override. The supported process
/// boundary excludes C or unsafe code that hides such an override after dyld
/// cached it and hostile same-UID in-place mutation of executable storage.
pub fn load_apple_scalar_v1(
    artifact: AppleScalarDylib,
    region: &Q8LinearRegion,
    kernel: &LoopKernelV1,
    packed: &PackedWeightsV1,
) -> Result<AppleScalarExecutableV1> {
    // Re-emission is the existing complete verification of region, kernel,
    // pack shape, logical identity, pack bytes, and code identity. Because
    // code is shape/schedule-specific rather than weight-specific, another
    // valid same-shape pack intentionally produces the same module ID.
    let expected_module = emit_scalar_c(region, kernel, packed)?;
    load_verified_apple_generated_v1(artifact, expected_module.module_id(), region, packed)
}

/// Bind a locally built strict NEON module to an exact verified pack and load it.
///
/// The build owner is consumed. The runtime synchronously reconstructs the
/// pack in aligned storage and makes its own byte-verified private dylib copy,
/// so neither the compiler build directory nor `packed` has to outlive the
/// returned executable. The same supported process boundary documented by
/// [`load_apple_scalar_v1`] applies.
pub fn load_apple_neon_v1(
    artifact: AppleNeonDylib,
    region: &Q8LinearRegion,
    kernel: &LoopKernelV1,
    packed: &PackedWeightsV1,
) -> Result<AppleNeonExecutableV1> {
    // Re-emission proves that this is the fixed NEON schedule and rechecks the
    // complete region/kernel/pack binding before the trusted-image boundary.
    let expected_module = emit_neon_c(region, kernel, packed)?;
    load_verified_apple_generated_v1(artifact, expected_module.module_id(), region, packed)
}

fn load_verified_apple_generated_v1<A: AuditedAppleGeneratedDylibV1>(
    artifact: A,
    expected_module_id: &str,
    region: &Q8LinearRegion,
    packed: &PackedWeightsV1,
) -> Result<decodeforge_runtime::GeneratedExecutableV1> {
    if artifact.module_id() != expected_module_id {
        return Err(invalid(
            "DFE-NATIVE-007",
            format!(
                "{} dylib code identity does not match the verified binding.",
                A::BACKEND_NAME
            ),
        ));
    }
    let shape = region.shape();

    // SAFETY: The private trait is implemented only for unforgeable compiler
    // owners returned by the fixed scalar and NEON builders. Each public
    // wrapper reaches this helper only after backend-specific re-emission has
    // proven the module/shape/pack binding. Both builders retain the exact
    // locally generated snapshot on which export, helper, initializer,
    // dependency, Mach-O, and machine-code audits succeeded. The runtime
    // copies the image and pack synchronously before `artifact` is dropped.
    let loaded = unsafe {
        decodeforge_runtime::load_trusted_apple_generated_v1(
            artifact.dylib_bytes(),
            expected_module_id,
            shape.n(),
            shape.k(),
            packed.bytes(),
            packed.packed_identity(),
        )
    };
    loaded.map_err(|error| {
        let code = error.code();
        invalid(
            code,
            format!(
                "runtime rejected the verified {} module: {error}",
                A::BACKEND_NAME
            ),
        )
    })
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::{
        KernelVariant, LoopKernelV1, Q8LinearRegion, build_apple_neon_dylib,
        build_apple_scalar_dylib,
    };
    use decodeforge_core::q8::{Q8Weights, canonical_linear_f32_bits, fixture};
    use decodeforge_runtime::{GeneratedStatusV1, RuntimeError, ScalarStatusV1};
    use std::os::unix::ffi::OsStrExt;
    use std::process::Command;

    const CLEAN_DYLD_TEST_CHILD: &str = "DECODEFORGE_CLEAN_DYLD_TEST_CHILD";

    // Cargo launches test binaries with DYLD_FALLBACK_LIBRARY_PATH so they can
    // find Rust dynamic dependencies. Production loading deliberately rejects
    // every DYLD_* override. Re-execute only the native-loading test in a clean
    // child, without putting an environment bypass in production code.
    fn rerun_without_dyld_overrides_if_needed(test_filter: &str) -> bool {
        if std::env::var_os(CLEAN_DYLD_TEST_CHILD).is_some() {
            return false;
        }
        let dyld_keys = std::env::vars_os()
            .map(|(key, _)| key)
            .filter(|key| key.as_bytes().starts_with(b"DYLD_"))
            .collect::<Vec<_>>();
        if dyld_keys.is_empty() {
            return false;
        }

        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .arg(test_filter)
            .arg("--nocapture")
            .env(CLEAN_DYLD_TEST_CHILD, "1");
        for key in dyld_keys {
            command.env_remove(key);
        }
        assert!(command.status().unwrap().success());
        true
    }

    fn fixture_parts_for(
        case_id: &str,
        variant: KernelVariant,
    ) -> (
        Q8LinearRegion,
        LoopKernelV1,
        PackedWeightsV1,
        Vec<f32>,
        Vec<u32>,
    ) {
        let documents = fixture::generated_documents().unwrap();
        let expected_path = format!("fixtures/{case_id}.json");
        let bytes = documents
            .into_iter()
            .find_map(|(candidate, bytes)| (candidate == expected_path).then_some(bytes))
            .unwrap_or_else(|| panic!("missing generated fixture {case_id}"));
        let document = fixture::parse_quant_fixture(&bytes).unwrap();
        let weights = Q8Weights::try_new(
            document.n,
            document.k,
            document.blocks,
            document
                .expected_q_bytes
                .iter()
                .map(|value| *value as i8 as u8)
                .collect(),
            document.expected_scale_bits,
        )
        .unwrap();
        let packed = PackedWeightsV1::pack(&weights).unwrap();
        let region = Q8LinearRegion::from_weights(&weights).unwrap();
        let kernel = LoopKernelV1::new(&region, variant).unwrap();
        let input = document
            .input_fp32_bits
            .into_iter()
            .map(f32::from_bits)
            .collect();
        (
            region,
            kernel,
            packed,
            input,
            document.expected_output_fp32_bits,
        )
    }

    fn fixture_parts(
        case_id: &str,
    ) -> (
        Q8LinearRegion,
        LoopKernelV1,
        PackedWeightsV1,
        Vec<f32>,
        Vec<u32>,
    ) {
        fixture_parts_for(case_id, KernelVariant::Scalar)
    }

    fn neon_fixture_parts(
        case_id: &str,
    ) -> (
        Q8LinearRegion,
        LoopKernelV1,
        PackedWeightsV1,
        Vec<f32>,
        Vec<u32>,
    ) {
        fixture_parts_for(case_id, KernelVariant::Neon)
    }

    fn build_and_load(
        region: &Q8LinearRegion,
        kernel: &LoopKernelV1,
        packed: &PackedWeightsV1,
    ) -> AppleScalarExecutableV1 {
        let module = emit_scalar_c(region, kernel, packed).unwrap();
        let artifact = build_apple_scalar_dylib(&module).unwrap();
        load_apple_scalar_v1(artifact, region, kernel, packed).unwrap()
    }

    fn build_and_load_neon(
        region: &Q8LinearRegion,
        kernel: &LoopKernelV1,
        packed: &PackedWeightsV1,
    ) -> AppleNeonExecutableV1 {
        let module = emit_neon_c(region, kernel, packed).unwrap();
        let artifact = build_apple_neon_dylib(&module).unwrap();
        load_apple_neon_v1(artifact, region, kernel, packed).unwrap()
    }

    fn synthetic_vector_case(
        n: u32,
    ) -> (
        Q8Weights,
        Q8LinearRegion,
        LoopKernelV1,
        PackedWeightsV1,
        Vec<f32>,
        Vec<u32>,
    ) {
        const K: u32 = 33;
        const BLOCKS: u32 = 2;
        let mut q = vec![0_u8; (n * BLOCKS * 32) as usize];
        let mut scale_bits = Vec::with_capacity((n * BLOCKS) as usize);
        for row in 0..n {
            let magnitude = (row + 1) as u8;
            q[(row * BLOCKS * 32) as usize] = magnitude;
            q[((row * BLOCKS + 1) * 32) as usize] = (-(magnitude as i8)) as u8;
            scale_bits.push((1.0_f32 + row as f32 * 0.25).to_bits());
            scale_bits.push((0.5_f32 + row as f32 * 0.125).to_bits());
        }
        let weights = Q8Weights::try_new(n, K, BLOCKS, q, scale_bits).unwrap();
        let packed = PackedWeightsV1::pack(&weights).unwrap();
        let region = Q8LinearRegion::from_weights(&weights).unwrap();
        let kernel = LoopKernelV1::new(&region, KernelVariant::Neon).unwrap();
        let mut input_bits = vec![0_u32; K as usize];
        input_bits[0] = 2.0_f32.to_bits();
        input_bits[32] = 0.5_f32.to_bits();
        let expected = canonical_linear_f32_bits(&input_bits, &weights).unwrap();
        let input = input_bits.into_iter().map(f32::from_bits).collect();
        (weights, region, kernel, packed, input, expected)
    }

    fn assert_fixture(case_id: &str) {
        let (region, kernel, packed, input, expected) = fixture_parts(case_id);
        let executable = build_and_load(&region, &kernel, &packed);
        let actual = executable
            .run(&input)
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "fixture {case_id}");
    }

    fn assert_neon_fixture(case_id: &str) {
        let (region, kernel, packed, input, expected) = neon_fixture_parts(case_id);
        let executable = build_and_load_neon(&region, &kernel, &packed);
        let actual = executable
            .run(&input)
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "NEON fixture {case_id}");
    }

    #[test]
    fn loaded_scalar_matches_every_frozen_fixture_exactly() {
        if rerun_without_dyld_overrides_if_needed(
            "loaded_scalar_matches_every_frozen_fixture_exactly",
        ) {
            return;
        }
        for case_id in [
            "exhaustive-q8",
            "finite-extremes",
            "k-01",
            "k-31",
            "k-32",
            "k-33",
            "k-63",
            "k-64",
            "k-65",
            "mixed-signs-tail",
            "random-sha256-counter",
            "subnormal-clamp",
            "subnormal-scale-min",
            "subnormal-scale-zero",
            "ties-and-extrema",
            "zero-signed-zero",
        ] {
            assert_fixture(case_id);
        }
    }

    #[test]
    fn loaded_neon_matches_every_frozen_fixture_exactly() {
        if rerun_without_dyld_overrides_if_needed(
            "loaded_neon_matches_every_frozen_fixture_exactly",
        ) {
            return;
        }
        for case_id in [
            "exhaustive-q8",
            "finite-extremes",
            "k-01",
            "k-31",
            "k-32",
            "k-33",
            "k-63",
            "k-64",
            "k-65",
            "mixed-signs-tail",
            "random-sha256-counter",
            "subnormal-clamp",
            "subnormal-scale-min",
            "subnormal-scale-zero",
            "ties-and-extrema",
            "zero-signed-zero",
        ] {
            assert_neon_fixture(case_id);
        }
    }

    #[test]
    fn loaded_neon_executes_full_vector_tile_and_scalar_tail_exactly() {
        if rerun_without_dyld_overrides_if_needed(
            "loaded_neon_executes_full_vector_tile_and_scalar_tail_exactly",
        ) {
            return;
        }
        for n in [4, 5] {
            let (_, region, kernel, packed, input, expected) = synthetic_vector_case(n);
            let executable = build_and_load_neon(&region, &kernel, &packed);
            let actual = executable
                .run(&input)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "N={n}, K=33");
        }
    }

    #[test]
    fn shape_mismatch_is_rejected_and_releases_the_builder_snapshot() {
        if rerun_without_dyld_overrides_if_needed(
            "shape_mismatch_is_rejected_and_releases_the_builder_snapshot",
        ) {
            return;
        }
        let (built_region, built_kernel, built_pack, _, _) = fixture_parts("k-32");
        let built_module = emit_scalar_c(&built_region, &built_kernel, &built_pack).unwrap();
        let artifact = build_apple_scalar_dylib(&built_module).unwrap();
        let retained_directory = artifact._temp_dir.path().to_owned();
        assert!(retained_directory.exists());

        let (other_region, other_kernel, other_pack, _, _) = fixture_parts("k-33");
        let error = match load_apple_scalar_v1(artifact, &other_region, &other_kernel, &other_pack)
        {
            Ok(_) => panic!("a shape-specific artifact must not bind to another shape"),
            Err(error) => error,
        };
        assert_eq!(error.code(), "DFE-NATIVE-007");
        assert!(!retained_directory.exists());
    }

    #[test]
    fn neon_shape_mismatch_is_rejected_and_consumes_the_builder_snapshot() {
        if rerun_without_dyld_overrides_if_needed(
            "neon_shape_mismatch_is_rejected_and_consumes_the_builder_snapshot",
        ) {
            return;
        }
        let (_, built_region, built_kernel, built_pack, _, _) = synthetic_vector_case(4);
        let built_module = emit_neon_c(&built_region, &built_kernel, &built_pack).unwrap();
        let artifact = build_apple_neon_dylib(&built_module).unwrap();
        let retained_directory = artifact._temp_dir.path().to_owned();
        assert!(retained_directory.exists());

        let (_, other_region, other_kernel, other_pack, _, _) = synthetic_vector_case(5);
        let error = match load_apple_neon_v1(artifact, &other_region, &other_kernel, &other_pack) {
            Ok(_) => panic!("a shape-specific NEON artifact must not bind to another shape"),
            Err(error) => error,
        };
        assert_eq!(error.code(), "DFE-NATIVE-007");
        assert!(!retained_directory.exists());
    }

    #[test]
    fn distinct_dylibs_execute_interleaved_without_symbol_aliasing() {
        if rerun_without_dyld_overrides_if_needed(
            "distinct_dylibs_execute_interleaved_without_symbol_aliasing",
        ) {
            return;
        }
        let (first_region, first_kernel, first_pack, first_input, first_expected) =
            fixture_parts("k-01");
        let (second_region, second_kernel, second_pack, second_input, second_expected) =
            fixture_parts("k-65");
        let first = build_and_load(&first_region, &first_kernel, &first_pack);
        let second = build_and_load(&second_region, &second_kernel, &second_pack);
        assert_ne!(first.module_id(), second.module_id());

        for _ in 0..4 {
            let first_actual = first
                .run(&first_input)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            let second_actual = second
                .run(&second_input)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            assert_eq!(first_actual, first_expected);
            assert_eq!(second_actual, second_expected);
        }
    }

    #[test]
    fn scalar_and_two_neon_modules_execute_interleaved_without_symbol_aliasing() {
        if rerun_without_dyld_overrides_if_needed(
            "scalar_and_two_neon_modules_execute_interleaved_without_symbol_aliasing",
        ) {
            return;
        }
        let (_, first_region, first_neon_kernel, first_pack, first_input, first_expected) =
            synthetic_vector_case(4);
        let first_scalar_kernel = LoopKernelV1::new(&first_region, KernelVariant::Scalar).unwrap();
        let scalar = build_and_load(&first_region, &first_scalar_kernel, &first_pack);
        let first_neon = build_and_load_neon(&first_region, &first_neon_kernel, &first_pack);

        let (_, second_region, second_kernel, second_pack, second_input, second_expected) =
            synthetic_vector_case(5);
        let second_neon = build_and_load_neon(&second_region, &second_kernel, &second_pack);

        assert_ne!(scalar.module_id(), first_neon.module_id());
        assert_ne!(first_neon.module_id(), second_neon.module_id());
        for _ in 0..4 {
            let second_actual = second_neon
                .run(&second_input)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            let scalar_actual = scalar
                .run(&first_input)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            let first_neon_actual = first_neon
                .run(&first_input)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            assert_eq!(second_actual, second_expected);
            assert_eq!(scalar_actual, first_expected);
            assert_eq!(first_neon_actual, first_expected);
        }
    }

    #[test]
    fn runtime_private_copy_outlives_builder_and_binds_same_shape_weights() {
        if rerun_without_dyld_overrides_if_needed(
            "runtime_private_copy_outlives_builder_and_binds_same_shape_weights",
        ) {
            return;
        }
        let first_weights =
            Q8Weights::try_new(5, 33, 2, vec![0; 5 * 2 * 32], vec![0x3f80_0000; 5 * 2]).unwrap();
        let mut second_q = vec![0_u8; 5 * 2 * 32];
        second_q[0] = 1;
        let second_weights =
            Q8Weights::try_new(5, 33, 2, second_q, vec![0x3f80_0000; 5 * 2]).unwrap();
        let first_packed = PackedWeightsV1::pack(&first_weights).unwrap();
        let first_region = Q8LinearRegion::from_weights(&first_weights).unwrap();
        let first_kernel = LoopKernelV1::new(&first_region, KernelVariant::Scalar).unwrap();
        let first_module = emit_scalar_c(&first_region, &first_kernel, &first_packed).unwrap();

        let second_packed = PackedWeightsV1::pack(&second_weights).unwrap();
        let second_region = Q8LinearRegion::from_weights(&second_weights).unwrap();
        let second_kernel = LoopKernelV1::new(&second_region, KernelVariant::Scalar).unwrap();
        let second_module = emit_scalar_c(&second_region, &second_kernel, &second_packed).unwrap();
        assert_eq!(first_module.module_id(), second_module.module_id());

        let artifact = build_apple_scalar_dylib(&first_module).unwrap();
        let executable =
            load_apple_scalar_v1(artifact, &second_region, &second_kernel, &second_packed).unwrap();
        assert_eq!(executable.module_id(), second_module.module_id());
        assert_eq!(
            executable.packed_identity(),
            second_packed.packed_identity()
        );
        let mut input = vec![0.0; 33];
        input[0] = 2.0;
        assert_eq!(
            executable
                .run(&input)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            [2.0_f32.to_bits(), 0, 0, 0, 0]
        );
    }

    #[test]
    fn neon_private_copy_outlives_builder_and_binds_same_shape_weights() {
        if rerun_without_dyld_overrides_if_needed(
            "neon_private_copy_outlives_builder_and_binds_same_shape_weights",
        ) {
            return;
        }
        let first_weights =
            Q8Weights::try_new(5, 33, 2, vec![0; 5 * 2 * 32], vec![0x3f80_0000; 5 * 2]).unwrap();
        let first_packed = PackedWeightsV1::pack(&first_weights).unwrap();
        let first_region = Q8LinearRegion::from_weights(&first_weights).unwrap();
        let first_kernel = LoopKernelV1::new(&first_region, KernelVariant::Neon).unwrap();
        let first_module = emit_neon_c(&first_region, &first_kernel, &first_packed).unwrap();

        let (_, second_region, second_kernel, second_packed, input, expected) =
            synthetic_vector_case(5);
        let second_module = emit_neon_c(&second_region, &second_kernel, &second_packed).unwrap();
        assert_eq!(first_module.module_id(), second_module.module_id());
        assert_ne!(
            first_packed.packed_identity(),
            second_packed.packed_identity()
        );

        let artifact = build_apple_neon_dylib(&first_module).unwrap();
        let retained_directory = artifact._temp_dir.path().to_owned();
        let executable =
            load_apple_neon_v1(artifact, &second_region, &second_kernel, &second_packed).unwrap();
        assert!(!retained_directory.exists());
        assert_eq!(executable.module_id(), second_module.module_id());
        assert_eq!(
            executable.packed_identity(),
            second_packed.packed_identity()
        );
        let actual = executable
            .run(&input)
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn safe_run_rejects_lengths_and_discards_nonfinite_failures() {
        if rerun_without_dyld_overrides_if_needed(
            "safe_run_rejects_lengths_and_discards_nonfinite_failures",
        ) {
            return;
        }
        let (region, kernel, packed, input, _) = fixture_parts("k-32");
        let executable = build_and_load(&region, &kernel, &packed);
        assert_eq!(
            executable.run(&[]),
            Err(RuntimeError::InputLength {
                expected: 32,
                actual: 0,
            })
        );
        let mut long_input = input.clone();
        long_input.push(0.0);
        assert_eq!(
            executable.run(&long_input),
            Err(RuntimeError::InputLength {
                expected: 32,
                actual: 33,
            })
        );
        let mut nan_input = input;
        nan_input[0] = f32::NAN;
        assert_eq!(
            executable.run(&nan_input),
            Err(RuntimeError::KernelStatus(ScalarStatusV1::NonFiniteInput))
        );

        let overflow_weights = Q8Weights::try_new(
            5,
            33,
            2,
            [127].into_iter().chain([0; 5 * 2 * 32 - 1]).collect(),
            [0x7f7f_ffff].into_iter().chain([0x3f80_0000; 9]).collect(),
        )
        .unwrap();
        let overflow_packed = PackedWeightsV1::pack(&overflow_weights).unwrap();
        let overflow_region = Q8LinearRegion::from_weights(&overflow_weights).unwrap();
        let overflow_kernel = LoopKernelV1::new(&overflow_region, KernelVariant::Scalar).unwrap();
        let overflow_executable =
            build_and_load(&overflow_region, &overflow_kernel, &overflow_packed);
        assert_eq!(
            overflow_executable.run(&[f32::MAX].into_iter().chain([0.0; 32]).collect::<Vec<_>>()),
            Err(RuntimeError::KernelStatus(ScalarStatusV1::NonFiniteResult))
        );
    }

    #[test]
    fn safe_neon_run_rejects_lengths_and_discards_nonfinite_failures() {
        if rerun_without_dyld_overrides_if_needed(
            "safe_neon_run_rejects_lengths_and_discards_nonfinite_failures",
        ) {
            return;
        }
        let (_, region, kernel, packed, input, _) = synthetic_vector_case(5);
        let executable = build_and_load_neon(&region, &kernel, &packed);
        assert_eq!(
            executable.run(&input[..32]),
            Err(RuntimeError::InputLength {
                expected: 33,
                actual: 32,
            })
        );
        let mut long_input = input.clone();
        long_input.push(0.0);
        assert_eq!(
            executable.run(&long_input),
            Err(RuntimeError::InputLength {
                expected: 33,
                actual: 34,
            })
        );
        for nonfinite in [f32::NAN, f32::INFINITY] {
            let mut invalid_input = input.clone();
            invalid_input[0] = nonfinite;
            assert_eq!(
                executable.run(&invalid_input),
                Err(RuntimeError::KernelStatus(
                    GeneratedStatusV1::NonFiniteInput
                ))
            );
        }

        let overflow_weights = Q8Weights::try_new(
            5,
            33,
            2,
            [127].into_iter().chain([0; 5 * 2 * 32 - 1]).collect(),
            [0x7f7f_ffff].into_iter().chain([0x3f80_0000; 9]).collect(),
        )
        .unwrap();
        let overflow_packed = PackedWeightsV1::pack(&overflow_weights).unwrap();
        let overflow_region = Q8LinearRegion::from_weights(&overflow_weights).unwrap();
        let overflow_kernel = LoopKernelV1::new(&overflow_region, KernelVariant::Neon).unwrap();
        let overflow_executable =
            build_and_load_neon(&overflow_region, &overflow_kernel, &overflow_packed);
        assert_eq!(
            overflow_executable.run(&[f32::MAX].into_iter().chain([0.0; 32]).collect::<Vec<_>>()),
            Err(RuntimeError::KernelStatus(
                GeneratedStatusV1::NonFiniteResult
            ))
        );
    }

    #[derive(Clone, Copy)]
    struct FloatingEnvironment {
        fpcr: u64,
        fpsr: u64,
    }

    impl FloatingEnvironment {
        fn capture() -> Self {
            // SAFETY: Reading the current thread's AArch64 FP status/control
            // registers has no memory effects.
            unsafe { Self::capture_unchecked() }
        }

        unsafe fn capture_unchecked() -> Self {
            let fpcr: u64;
            let fpsr: u64;
            // SAFETY: These MRS operations only copy the current thread's
            // architectural FP registers into general-purpose registers.
            unsafe {
                std::arch::asm!("mrs {value}, fpcr", value = out(reg) fpcr, options(nomem, nostack, preserves_flags));
                std::arch::asm!("mrs {value}, fpsr", value = out(reg) fpsr, options(nomem, nostack, preserves_flags));
            }
            Self { fpcr, fpsr }
        }

        unsafe fn install(self) {
            // SAFETY: These MSR operations update only the current thread's
            // FP status/control registers; the caller owns restoration.
            unsafe {
                std::arch::asm!("msr fpcr, {value}", value = in(reg) self.fpcr, options(nomem, nostack, preserves_flags));
                std::arch::asm!("msr fpsr, {value}", value = in(reg) self.fpsr, options(nomem, nostack, preserves_flags));
                std::arch::asm!("isb", options(nomem, nostack, preserves_flags));
            }
        }
    }

    struct RestoreFloatingEnvironment(FloatingEnvironment);

    impl RestoreFloatingEnvironment {
        fn new() -> Self {
            Self(FloatingEnvironment::capture())
        }
    }

    impl Drop for RestoreFloatingEnvironment {
        fn drop(&mut self) {
            // SAFETY: This reinstalls the state captured from the same test
            // thread before any temporary modification.
            unsafe { self.0.install() }
        }
    }

    fn assert_fp_environment_contract(
        executable: &decodeforge_runtime::GeneratedExecutableV1,
        input: &[f32],
    ) {
        const ROUNDING_MODE_MASK: u64 = 3 << 22;
        const ROUND_UP: u64 = 1 << 22;
        const FLUSH_TO_ZERO: u64 = 1 << 24;
        const FLUSH_INPUTS_TO_ZERO: u64 = 1;
        const INVALID_TRAP_ENABLE: u64 = 1 << 8;
        const INVALID_FLAG: u64 = 1;

        let restore = RestoreFloatingEnvironment::new();
        let original = restore.0;
        let compatible_fpcr =
            original.fpcr & !(ROUNDING_MODE_MASK | FLUSH_TO_ZERO | FLUSH_INPUTS_TO_ZERO);

        for fpcr_bits in [ROUND_UP, FLUSH_TO_ZERO, FLUSH_INPUTS_TO_ZERO] {
            let requested = FloatingEnvironment {
                fpcr: compatible_fpcr | fpcr_bits,
                fpsr: original.fpsr | INVALID_FLAG,
            };
            // SAFETY: `restore` is live and will restore the original state.
            unsafe { requested.install() };
            let installed = FloatingEnvironment::capture();
            if fpcr_bits == FLUSH_INPUTS_TO_ZERO && installed.fpcr & FLUSH_INPUTS_TO_ZERO == 0 {
                continue;
            }
            assert_eq!(
                executable.run(input),
                Err(RuntimeError::KernelStatus(GeneratedStatusV1::FpEnvironment)),
                "FPCR mode {fpcr_bits:#x} must be rejected"
            );
            let after = FloatingEnvironment::capture();
            assert_eq!(after.fpcr, installed.fpcr);
            assert_eq!(after.fpsr, installed.fpsr);
        }

        let trap_state = FloatingEnvironment {
            fpcr: compatible_fpcr | INVALID_TRAP_ENABLE,
            fpsr: original.fpsr | INVALID_FLAG,
        };
        // SAFETY: The generated function calls feholdexcept before floating
        // work; `restore` remains live for panic-safe cleanup.
        unsafe { trap_state.install() };
        assert!(executable.run(input).is_ok());
        let after = FloatingEnvironment::capture();
        assert_eq!(after.fpcr, trap_state.fpcr);
        assert_eq!(after.fpsr, trap_state.fpsr);
    }

    #[test]
    fn scalar_generated_entrypoint_preserves_the_fp_environment_contract() {
        if rerun_without_dyld_overrides_if_needed(
            "scalar_generated_entrypoint_preserves_the_fp_environment_contract",
        ) {
            return;
        }
        let (region, kernel, packed, input, _) = fixture_parts("k-32");
        let executable = build_and_load(&region, &kernel, &packed);
        assert_fp_environment_contract(&executable, &input);
    }

    #[test]
    fn neon_generated_entrypoint_preserves_the_fp_environment_contract() {
        if rerun_without_dyld_overrides_if_needed(
            "neon_generated_entrypoint_preserves_the_fp_environment_contract",
        ) {
            return;
        }
        let (_, region, kernel, packed, input, _) = synthetic_vector_case(4);
        let executable = build_and_load_neon(&region, &kernel, &packed);
        assert_fp_environment_contract(&executable, &input);
    }
}
