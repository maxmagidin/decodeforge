//! Deterministic strict ARM64 NEON C emission for one Q8 linear region.
//!
//! The emitter deliberately has one schedule.  It vectors four independent
//! output rows (the OI4 panel), while each lane retains the scalar
//! block/lane reduction order.  This keeps the numerical contract identical
//! to the scalar emitter without asking a compiler to rediscover the layout.

use super::GENERATED_ABI_VERSION_V1;
use super::c_abi_v1::{
    AbiV1Spec, render_abi_version_entrypoint, render_artifact_id_entrypoint,
    render_artifact_id_storage, render_run_v1_entrypoint,
};
use crate::ir::{KernelVariant, LoopKernelV1, Q8LinearRegion};
use crate::pack::{PACK_BLOCK_SIZE, PACK_RECORD_BYTES, PackManifestV1, PackedWeightsV1};
use crate::{Result, hex_lower, invalid};
use sha2::{Digest, Sha256};

/// Frozen source format for the strict output-vector NEON emitter.
pub const NEON_C_SOURCE_FORMAT_V1: &str = "decodeforge_neon_c_v1";
/// Maximum permitted generated NEON source size.
pub const MAX_NEON_C_SOURCE_BYTES: usize = 192 * 1024;

const MODULE_ID_DOMAIN: &[u8] = b"DecodeForge/generated-module/neon-c/v1\0";

/// Immutable result of deterministic NEON C generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NeonCModule {
    module_id: String,
    hidden_kernel_symbol: String,
    /// Frozen logical output size compiled into the helper.
    n: u32,
    /// Frozen logical reduction size compiled into the helper.
    k: u32,
    source: String,
}

impl NeonCModule {
    pub fn module_id(&self) -> &str {
        &self.module_id
    }
    pub fn hidden_kernel_symbol(&self) -> &str {
        &self.hidden_kernel_symbol
    }
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Logical output size compiled into this module.
    pub const fn n(&self) -> u32 {
        self.n
    }

    /// Logical reduction size compiled into this module.
    pub const fn k(&self) -> u32 {
        self.k
    }
}

/// Emit the single fixed strict ARM64 NEON schedule.
pub fn emit_neon_c(
    region: &Q8LinearRegion,
    kernel: &LoopKernelV1,
    packed: &PackedWeightsV1,
) -> Result<NeonCModule> {
    region.verify()?;
    kernel.verify()?;
    packed.verify()?;
    validate_bindings(region, kernel, packed.manifest())?;

    let module_id = neon_c_module_id(region, kernel)?;
    let hash = module_id
        .strip_prefix("sha256:")
        .ok_or_else(|| invalid("DFE-COMP-006", "module identity is not a SHA-256 identity."))?;
    let hidden_kernel_symbol = format!("df_kernel_neon_v1_{hash}");
    let source = render_source(region, &hidden_kernel_symbol, &module_id)?;
    if source.len() > MAX_NEON_C_SOURCE_BYTES {
        return Err(invalid(
            "DFE-COMP-002",
            "generated NEON C source exceeds 192 KiB.",
        ));
    }
    if !source.is_ascii() || !source.ends_with('\n') || source.ends_with("\n\n") {
        return Err(invalid(
            "DFE-COMP-006",
            "generated source must be ASCII with exactly one terminal newline.",
        ));
    }
    Ok(NeonCModule {
        module_id,
        hidden_kernel_symbol,
        n: region.shape().n(),
        k: region.shape().k(),
        source,
    })
}

fn neon_c_module_id(region: &Q8LinearRegion, kernel: &LoopKernelV1) -> Result<String> {
    region.verify()?;
    kernel.verify()?;
    if kernel.variant() != KernelVariant::Neon
        || kernel.shape() != region.shape()
        || kernel.numeric_mode() != region.numeric_mode()
    {
        return Err(invalid(
            "DFE-COMP-006",
            "region and NEON loop contracts do not match the module identity contract.",
        ));
    }
    let schedule_json = kernel.schedule_json()?;
    let mut hasher = Sha256::new();
    hasher.update(MODULE_ID_DOMAIN);
    frame(&mut hasher, region.operator().as_bytes());
    frame(&mut hasher, schedule_json.as_bytes());
    frame(&mut hasher, region.numeric_mode().as_bytes());
    frame(&mut hasher, &GENERATED_ABI_VERSION_V1.to_le_bytes());
    frame(&mut hasher, NEON_C_SOURCE_FORMAT_V1.as_bytes());
    Ok(format!("sha256:{}", hex_lower(&hasher.finalize())))
}

fn frame(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn validate_bindings(
    region: &Q8LinearRegion,
    kernel: &LoopKernelV1,
    manifest: &PackManifestV1,
) -> Result<()> {
    if kernel.variant() != KernelVariant::Neon {
        return Err(invalid(
            "DFE-COMP-007",
            "NEON C emitter requires a NEON LoopKernelV1.",
        ));
    }
    if kernel.shape() != region.shape() || manifest.shape() != region.shape() {
        return Err(invalid(
            "DFE-COMP-003",
            "region, loop kernel, and pack shapes do not match.",
        ));
    }
    if kernel.logical_weight_identity() != region.logical_weight_identity()
        || manifest.logical_weight_identity() != region.logical_weight_identity()
    {
        return Err(invalid(
            "DFE-COMP-004",
            "region, loop kernel, and pack logical identities do not match.",
        ));
    }
    if kernel.numeric_mode() != region.numeric_mode()
        || manifest.spec() != kernel.pack()
        || manifest.spec().format() != "DFQ8_B32_OI4_V1"
        || kernel.vector_lanes() != 4
        || kernel.n_tile() != 4
        || kernel.k_block() != 32
        || kernel.k_unroll() != 1
        || kernel.accumulators() != 1
        || kernel.uses_fma()
        || kernel.uses_horizontal_reduction()
        || !kernel.separate_mul_add()
    {
        return Err(invalid(
            "DFE-COMP-006",
            "loop and pack contracts do not match the fixed NEON schedule.",
        ));
    }
    Ok(())
}

fn render_source(region: &Q8LinearRegion, helper: &str, module_id: &str) -> Result<String> {
    let shape = region.shape();
    let n = shape.n();
    let k = shape.k();
    let panels = shape.panels();
    let blocks = shape.blocks();
    let abi = AbiV1Spec::new(module_id, helper, n, k, panels, blocks, PACK_RECORD_BYTES);
    let mut source = String::new();
    source.push_str("/* DecodeForge generated source format: decodeforge_neon_c_v1. */\n");
    source.push_str(
        "/* Fixed schedule: four output rows, ascending block/lane, separate mul/add. */\n",
    );
    source.push_str("/* OI4 payload and all buffer extents are safe-caller obligations. */\n");
    source.push_str("#include \"decodeforge/abi_v1.h\"\n#include <arm_neon.h>\n#include <fenv.h>\n#include <float.h>\n#include <math.h>\n#include <stddef.h>\n#include <stdint.h>\n#include <string.h>\n\n");
    source.push_str("#if !defined(__aarch64__) || !defined(__ARM_NEON)\n#error \"decodeforge_neon_c_v1 requires AArch64 NEON\"\n#endif\n");
    source.push_str("#if !defined(__BYTE_ORDER__) || !defined(__ORDER_LITTLE_ENDIAN__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__\n#error \"decodeforge_neon_c_v1 requires little-endian AArch64\"\n#endif\n");
    source.push_str(
        "#if !defined(__clang__)\n#error \"decodeforge_neon_c_v1 requires Clang\"\n#endif\n\n",
    );
    source.push_str("#pragma STDC FENV_ACCESS ON\n#pragma STDC FP_CONTRACT OFF\n\n");
    source.push_str("_Static_assert(sizeof(float) == 4, \"binary32 required\");\n_Static_assert(FLT_RADIX == 2, \"binary32 radix required\");\n_Static_assert(FLT_MANT_DIG == 24, \"binary32 mantissa required\");\n_Static_assert(FLT_MAX_EXP == 128, \"binary32 exponent required\");\n_Static_assert(FLT_MIN_EXP == -125, \"binary32 exponent required\");\n_Static_assert(FLT_EVAL_METHOD == 0, \"no excess precision required\");\n\n");
    source.push_str("#define DF_PUBLIC_V1 __attribute__((visibility(\"default\")))\n#define DF_HIDDEN_V1 __attribute__((visibility(\"hidden\")))\n#define DF_USED_V1 __attribute__((used))\n#define DF_NOINLINE_V1 __attribute__((noinline))\n\n");
    render_artifact_id_storage(&mut source, &abi);
    source.push_str("DF_HIDDEN_V1 DF_USED_V1 DF_NOINLINE_V1 int ");
    source.push_str(helper);
    source.push_str("(const float *x, const uint8_t *packed_weight, float *y) {\n");
    source.push_str(&format!("    for (uint32_t panel = 0; panel < UINT32_C({panels}); ++panel) {{\n        const uint32_t row_base = panel * UINT32_C(4);\n        if (row_base + UINT32_C(4) <= UINT32_C({n})) {{\n            float32x4_t accumulator = vdupq_n_f32(0.0f);\n            for (uint32_t block = 0; block < UINT32_C({blocks}); ++block) {{\n                const uint8_t *record = packed_weight + ((size_t)panel * (size_t){blocks} + (size_t)block) * (size_t){PACK_RECORD_BYTES};\n                float32x4_t block_sum = vdupq_n_f32(0.0f);\n                const uint32_t block_start = block * UINT32_C({PACK_BLOCK_SIZE});\n                const uint32_t lane_count = (UINT32_C({k}) - block_start < UINT32_C({PACK_BLOCK_SIZE}) ? UINT32_C({k}) - block_start : UINT32_C({PACK_BLOCK_SIZE}));\n                for (uint32_t lane = 0; lane < lane_count; ++lane) {{\n                    const size_t q_offset = 16u + (size_t)lane * 4u;\n                    const uint64_t q_word = (uint64_t)record[q_offset] | ((uint64_t)record[q_offset + 1u] << 8) | ((uint64_t)record[q_offset + 2u] << 16) | ((uint64_t)record[q_offset + 3u] << 24);\n                    const int8x8_t q8 = vreinterpret_s8_u64(vcreate_u64(q_word));\n                    const int16x8_t q16 = vmovl_s8(q8);\n                    const int32x4_t q32 = vmovl_s16(vget_low_s16(q16));\n                    const float32x4_t qf = vcvtq_f32_s32(q32);\n                    const float32x4_t xv = vdupq_n_f32(x[(size_t)block_start + (size_t)lane]);\n                    block_sum = vaddq_f32(block_sum, vmulq_f32(qf, xv));\n                }}\n                float scales[4];\n                for (uint32_t output_lane = 0; output_lane < UINT32_C(4); ++output_lane) {{\n                    const size_t scale_offset = (size_t)output_lane * 4u;\n                    const uint32_t scale_bits = (uint32_t)record[scale_offset] | ((uint32_t)record[scale_offset + 1u] << 8) | ((uint32_t)record[scale_offset + 2u] << 16) | ((uint32_t)record[scale_offset + 3u] << 24);\n                    memcpy(&scales[output_lane], &scale_bits, sizeof(float));\n                }}\n                block_sum = vmulq_f32(block_sum, vld1q_f32(scales));\n                accumulator = vaddq_f32(accumulator, block_sum);\n            }}\n            vst1q_f32(y + row_base, accumulator);\n        }} else {{\n            for (uint32_t output_lane = 0; output_lane < UINT32_C(4); ++output_lane) {{\n                const uint32_t row = row_base + output_lane;\n                if (row >= UINT32_C({n})) break;\n                float accumulator = 0.0f;\n                for (uint32_t block = 0; block < UINT32_C({blocks}); ++block) {{\n                    const uint8_t *record = packed_weight + ((size_t)panel * (size_t){blocks} + (size_t)block) * (size_t){PACK_RECORD_BYTES};\n                    const size_t scale_offset = (size_t)output_lane * 4u;\n                    const uint32_t scale_bits = (uint32_t)record[scale_offset] | ((uint32_t)record[scale_offset + 1u] << 8) | ((uint32_t)record[scale_offset + 2u] << 16) | ((uint32_t)record[scale_offset + 3u] << 24);\n                    float scale;\n                    memcpy(&scale, &scale_bits, sizeof(scale));\n                    const uint32_t block_start = block * UINT32_C({PACK_BLOCK_SIZE});\n                    const uint32_t lane_count = (UINT32_C({k}) - block_start < UINT32_C({PACK_BLOCK_SIZE}) ? UINT32_C({k}) - block_start : UINT32_C({PACK_BLOCK_SIZE}));\n                    float block_sum = 0.0f;\n                    for (uint32_t lane = 0; lane < lane_count; ++lane) {{\n                        const uint8_t q_raw = record[16u + (size_t)lane * 4u + (size_t)output_lane];\n                        const float q_value = (float)((q_raw >= UINT8_C(128)) ? (int)q_raw - 256 : (int)q_raw);\n                        const float product = q_value * x[(size_t)block_start + (size_t)lane];\n                        block_sum = block_sum + product;\n                    }}\n                    block_sum = block_sum * scale;\n                    accumulator = accumulator + block_sum;\n                }}\n                y[row] = accumulator;\n            }}\n        }}\n    }}\n    return 0;\n}}\n\n"));
    render_abi_version_entrypoint(&mut source);
    render_artifact_id_entrypoint(&mut source);
    render_run_v1_entrypoint(&mut source, &abi);
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KernelVariant, Q8LinearShape};
    use decodeforge_core::q8::Q8Weights;

    fn sample(n: u32, k: u32) -> (Q8LinearRegion, LoopKernelV1, PackedWeightsV1) {
        let shape = Q8LinearShape::new(n, k).unwrap();
        let blocks = k.div_ceil(32);
        let weights = Q8Weights::try_new(
            n,
            k,
            blocks,
            vec![0; n as usize * blocks as usize * 32],
            vec![0x3f80_0000; n as usize * blocks as usize],
        )
        .unwrap();
        let packed = PackedWeightsV1::pack(&weights).unwrap();
        let region = Q8LinearRegion::new(shape, packed.logical_weight_identity()).unwrap();
        let kernel = LoopKernelV1::new(&region, KernelVariant::Neon).unwrap();
        (region, kernel, packed)
    }

    #[test]
    fn emission_is_repeatable_and_free_of_host_state() {
        let (region, kernel, packed) = sample(5, 33);
        let first = emit_neon_c(&region, &kernel, &packed).unwrap();
        let second = emit_neon_c(&region, &kernel, &packed).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.module_id(),
            "sha256:5293e51d03577c2143e5e6305f05d0bc464686f8f8b939ae2bd7bfac797e86c9"
        );
        assert_eq!(
            hex_lower(&Sha256::digest(first.source().as_bytes())),
            "6b674d163c736359ef8276617f602ce556e843239927150e9b53b39537128baa"
        );
        assert_eq!(first.n(), 5);
        assert_eq!(first.k(), 33);
        assert!(first.source.is_ascii());
        assert!(first.source.ends_with('\n'));
        assert!(!first.source.ends_with("\n\n"));
        assert!(first.source.contains("#include <arm_neon.h>"));
        assert!(
            first
                .source
                .contains("#error \"decodeforge_neon_c_v1 requires AArch64 NEON\"")
        );
        assert!(first.source.contains("vget_low_s16"));
        assert!(first.source.contains("vmovl_s8"));
        assert!(first.source.contains("vcvtq_f32_s32"));
        assert!(first.source.contains("vmulq_f32"));
        assert!(first.source.contains("vaddq_f32"));
        assert!(first.source.contains("if (row_base + UINT32_C(4)"));
        assert!(first.source.contains("output_lane < UINT32_C(4)"));
        assert!(
            first
                .source
                .contains("df_probe_true_min_bits != UINT32_C(0x00000001)")
        );
        assert!(
            first
                .source
                .contains("df_probe_half_min_bits != UINT32_C(0x00400000)")
        );
        assert!(!first.source.contains("df_probe_true_min * df_probe_one !="));
        assert!(!first.source.contains("df_probe_min * df_probe_half !="));
        assert!(!first.source.contains("0x1p-127f"));
        assert!(!first.source.contains("vfmaq"));
        assert!(!first.source.contains("vaddvq"));
        assert!(!first.source.contains("CARGO_MANIFEST_DIR"));
        assert!(first.source.len() <= MAX_NEON_C_SOURCE_BYTES);
    }

    #[test]
    fn neon_identity_is_distinct_from_scalar_and_weight_independent() {
        let (region, kernel, packed) = sample(5, 33);
        let first = emit_neon_c(&region, &kernel, &packed).unwrap();
        let scalar_kernel = LoopKernelV1::new(&region, KernelVariant::Scalar).unwrap();
        let scalar = crate::codegen::emit_scalar_c(&region, &scalar_kernel, &packed).unwrap();
        assert_ne!(first.module_id(), scalar.module_id());
        assert!(first.module_id().starts_with("sha256:"));
        assert!(
            first
                .hidden_kernel_symbol()
                .starts_with("df_kernel_neon_v1_")
        );

        let mut changed_q = vec![0_u8; 5 * 2 * 32];
        changed_q[0] = 1;
        let changed_weights =
            Q8Weights::try_new(5, 33, 2, changed_q, vec![0x3f80_0000; 5 * 2]).unwrap();
        let changed_packed = PackedWeightsV1::pack(&changed_weights).unwrap();
        let changed_region = Q8LinearRegion::from_weights(&changed_weights).unwrap();
        let changed_kernel = LoopKernelV1::new(&changed_region, KernelVariant::Neon).unwrap();
        let changed = emit_neon_c(&changed_region, &changed_kernel, &changed_packed).unwrap();
        assert_eq!(first.module_id(), changed.module_id());
        assert_eq!(first.source(), changed.source());

        let (different_region, different_kernel, different_packed) = sample(5, 34);
        let different =
            emit_neon_c(&different_region, &different_kernel, &different_packed).unwrap();
        assert_ne!(first.module_id(), different.module_id());
        assert_ne!(first.source(), different.source());
        assert_eq!(different.n(), 5);
        assert_eq!(different.k(), 34);
    }

    #[test]
    fn every_committed_fixture_emits_for_its_exact_shape() {
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
                fixture.expected_scale_bits,
            )
            .unwrap();
            let packed = PackedWeightsV1::pack(&weights).unwrap();
            let region = Q8LinearRegion::from_weights(&weights).unwrap();
            let kernel = LoopKernelV1::new(&region, KernelVariant::Neon).unwrap();
            let module = emit_neon_c(&region, &kernel, &packed)
                .unwrap_or_else(|error| panic!("fixture {case_id}: {error}"));
            assert_eq!(module.n(), fixture.n, "fixture {case_id}");
            assert_eq!(module.k(), fixture.k, "fixture {case_id}");
            assert_eq!(module.module_id().len(), 71, "fixture {case_id}");
            assert_eq!(module.hidden_kernel_symbol().len(), 82, "fixture {case_id}");
        }
    }

    #[test]
    fn rejects_scalar_kernel() {
        let (region, _, packed) = sample(4, 32);
        let scalar = LoopKernelV1::new(&region, KernelVariant::Scalar).unwrap();
        assert!(emit_neon_c(&region, &scalar, &packed).is_err());
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn generated_source_compiles_for_native_arm64() {
        use std::io::Write;
        use std::process::{Command, Stdio};
        let (region, kernel, packed) = sample(5, 33);
        let module = emit_neon_c(&region, &kernel, &packed).unwrap();
        let include_dir = format!("{}/../../include", env!("CARGO_MANIFEST_DIR"));
        let mut command = Command::new("/usr/bin/xcrun");
        command
            .args([
                "--sdk",
                "macosx",
                "clang",
                "-std=c11",
                "-fsyntax-only",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                "-fno-fast-math",
                "-ffp-model=strict",
                "-ffp-contract=off",
                "-fdenormal-fp-math=ieee",
                "-I",
            ])
            .arg(include_dir)
            .args(["-x", "c", "-"])
            .env_clear()
            .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
            .env("LANG", "C")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .expect("/usr/bin/xcrun could not start the pinned Apple Clang test");
        child
            .stdin
            .take()
            .unwrap()
            .write_all(module.source.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "clang rejected generated NEON C: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
