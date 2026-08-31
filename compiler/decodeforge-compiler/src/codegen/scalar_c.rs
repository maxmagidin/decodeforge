//! The portable, strict C11 emitter for one Q8 linear region.
//!
//! This module intentionally emits one fixed implementation.  It is not a
//! backend framework: the Loop IR and OI4 pack contract are checked at this
//! boundary and all source text is derived from those checked values.

use crate::ir::{KernelVariant, LoopKernelV1, Q8LinearRegion};
use crate::pack::{PACK_BLOCK_SIZE, PACK_RECORD_BYTES, PackManifestV1, PackedWeightsV1};
use crate::{Result, hex_lower, invalid};
use sha2::{Digest, Sha256};

/// Frozen source format used by this emitter and its module identity.
pub const SCALAR_C_SOURCE_FORMAT_V1: &str = "decodeforge_scalar_c_v1";
/// Generated ABI version incorporated into module identities.
pub const GENERATED_ABI_VERSION_V1: u32 = 1;
/// Maximum permitted generated source size.
pub const MAX_SCALAR_C_SOURCE_BYTES: usize = 128 * 1024;

const MODULE_ID_DOMAIN: &[u8] = b"DecodeForge/generated-module/scalar-c/v1\0";

/// Immutable result of deterministic scalar C generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScalarCModule {
    /// Stable SHA-256 identity of the generated code contract.
    module_id: String,
    /// Hidden helper symbol used by the public ABI entrypoint.
    hidden_kernel_symbol: String,
    /// Frozen logical output size compiled into the helper.
    n: u32,
    /// Frozen logical reduction size compiled into the helper.
    k: u32,
    /// Complete ASCII C11 translation unit, ending in one LF.
    source: String,
}

impl ScalarCModule {
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

/// Emit deterministic strict scalar C from already validated IR and packing.
pub fn emit_scalar_c(
    region: &Q8LinearRegion,
    kernel: &LoopKernelV1,
    packed: &PackedWeightsV1,
) -> Result<ScalarCModule> {
    region.verify()?;
    kernel.verify()?;
    packed.verify()?;
    validate_bindings(region, kernel, packed.manifest())?;

    let module_id = scalar_c_module_id(region, kernel)?;
    let hash = module_id
        .strip_prefix("sha256:")
        .ok_or_else(|| invalid("DFE-COMP-006", "module identity is not a SHA-256 identity."))?;
    let hidden_kernel_symbol = format!("df_kernel_scalar_v1_{hash}");
    let source = render_source(region, &hidden_kernel_symbol, module_id.as_str())?;
    if source.len() > MAX_SCALAR_C_SOURCE_BYTES {
        return Err(invalid(
            "DFE-COMP-002",
            "generated scalar C source exceeds 128 KiB.",
        ));
    }
    if !source.is_ascii() || !source.ends_with('\n') || source.ends_with("\n\n") {
        return Err(invalid(
            "DFE-COMP-006",
            "generated source must be ASCII with exactly one terminal newline.",
        ));
    }
    Ok(ScalarCModule {
        module_id,
        hidden_kernel_symbol,
        n: region.shape().n(),
        k: region.shape().k(),
        source,
    })
}

/// Compute the pure module identity without rendering source or observing any
/// host state.  Every variable-length component is length-framed in LE.
fn scalar_c_module_id(region: &Q8LinearRegion, kernel: &LoopKernelV1) -> Result<String> {
    region.verify()?;
    kernel.verify()?;
    if kernel.variant() != KernelVariant::Scalar
        || kernel.shape() != region.shape()
        || kernel.numeric_mode() != region.numeric_mode()
    {
        return Err(invalid(
            "DFE-COMP-006",
            "region and scalar loop contracts do not match the module identity contract.",
        ));
    }

    let schedule_json = kernel.schedule_json()?;
    let mut hasher = Sha256::new();
    hasher.update(MODULE_ID_DOMAIN);
    frame(&mut hasher, region.operator().as_bytes());
    frame(&mut hasher, schedule_json.as_bytes());
    frame(&mut hasher, region.numeric_mode().as_bytes());
    frame(&mut hasher, &GENERATED_ABI_VERSION_V1.to_le_bytes());
    frame(&mut hasher, SCALAR_C_SOURCE_FORMAT_V1.as_bytes());
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
    if kernel.variant() != KernelVariant::Scalar {
        return Err(invalid(
            "DFE-COMP-007",
            "scalar C emitter rejects a non-scalar LoopKernelV1.",
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
    {
        return Err(invalid(
            "DFE-COMP-006",
            "loop and pack contracts do not match the scalar C contract.",
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
    let mut source = String::new();
    source.push_str("/* DecodeForge generated source format: decodeforge_scalar_c_v1. */\n");
    source.push_str("/* Buffer extents, non-aliasing, and packed-weight identity are safe-caller obligations. */\n");
    source.push_str("#include \"decodeforge/abi_v1.h\"\n");
    source.push_str("#include <fenv.h>\n#include <float.h>\n#include <math.h>\n#include <stddef.h>\n#include <stdint.h>\n#include <string.h>\n\n");
    source.push_str("#pragma STDC FENV_ACCESS ON\n");
    source.push_str("#pragma STDC FP_CONTRACT OFF\n\n");
    source.push_str("_Static_assert(sizeof(float) == 4, \"binary32 required\");\n");
    source.push_str("_Static_assert(FLT_RADIX == 2, \"binary32 radix required\");\n");
    source.push_str("_Static_assert(FLT_MANT_DIG == 24, \"binary32 mantissa required\");\n");
    source.push_str("_Static_assert(FLT_MAX_EXP == 128, \"binary32 exponent required\");\n");
    source.push_str("_Static_assert(FLT_MIN_EXP == -125, \"binary32 exponent required\");\n");
    source.push_str("_Static_assert(FLT_EVAL_METHOD == 0, \"no excess precision required\");\n\n");
    source.push_str("#if defined(__clang__)\n");
    source.push_str("#define DF_PUBLIC_V1 __attribute__((visibility(\"default\")))\n");
    source.push_str("#define DF_HIDDEN_V1 __attribute__((visibility(\"hidden\")))\n");
    source.push_str("#define DF_USED_V1 __attribute__((used))\n");
    source.push_str("#define DF_NOINLINE_V1 __attribute__((noinline))\n");
    source.push_str("#else\n#error \"decodeforge_scalar_c_v1 requires Clang\"\n#endif\n\n");
    source.push_str("static const char df_artifact_id_cstr_v1[] = \"");
    source.push_str(module_id);
    source.push_str("\";\n_Static_assert(sizeof(df_artifact_id_cstr_v1) == DF_ARTIFACT_ID_CSTR_BYTES_V1, \"artifact identity size\");\n\n");
    source.push_str("DF_HIDDEN_V1 DF_USED_V1 DF_NOINLINE_V1 int ");
    source.push_str(helper);
    source.push_str("(const float *x, const uint8_t *packed_weight, float *y) {\n");
    source.push_str("    for (uint32_t panel = 0; panel < UINT32_C(");
    source.push_str(&panels.to_string());
    source.push_str("); ++panel) {\n        for (uint32_t output_lane = 0; output_lane < UINT32_C(4); ++output_lane) {\n            const uint32_t row = panel * UINT32_C(4) + output_lane;\n            if (row >= UINT32_C(");
    source.push_str(&n.to_string());
    source.push_str(") ) break;\n            float accumulator = 0.0f;\n            for (uint32_t block = 0; block < UINT32_C(");
    source.push_str(&blocks.to_string());
    source.push_str("); ++block) {\n                const uint8_t *record = packed_weight + ((size_t)panel * (size_t)");
    source.push_str(&blocks.to_string());
    source.push_str(" + (size_t)block) * (size_t)");
    source.push_str(&PACK_RECORD_BYTES.to_string());
    source.push_str(";\n                const size_t scale_offset = (size_t)output_lane * 4u;\n                const uint32_t scale_bits = (uint32_t)record[scale_offset] | ((uint32_t)record[scale_offset + 1u] << 8) | ((uint32_t)record[scale_offset + 2u] << 16) | ((uint32_t)record[scale_offset + 3u] << 24);\n                float scale;\n                memcpy(&scale, &scale_bits, sizeof(scale));\n                const uint32_t block_start = block * UINT32_C(");
    source.push_str(&PACK_BLOCK_SIZE.to_string());
    source.push_str(");\n                const uint32_t lane_count = (UINT32_C(");
    source.push_str(&k.to_string());
    source.push_str(") - block_start < UINT32_C(");
    source.push_str(&PACK_BLOCK_SIZE.to_string());
    source.push_str(") ? UINT32_C(");
    source.push_str(&k.to_string());
    source.push_str(") - block_start : UINT32_C(");
    source.push_str(&PACK_BLOCK_SIZE.to_string());
    source.push_str("));\n                float block_sum = 0.0f;\n                for (uint32_t lane = 0; lane < lane_count; ++lane) {\n                    const uint8_t q_raw = record[16u + (size_t)lane * 4u + (size_t)output_lane];\n                    const float q_value = (float)((q_raw >= UINT8_C(128)) ? (int)q_raw - 256 : (int)q_raw);\n                    const float product = q_value * x[(size_t)block_start + (size_t)lane];\n                    block_sum = block_sum + product;\n                }\n                block_sum = block_sum * scale;\n                accumulator = accumulator + block_sum;\n            }\n            y[row] = accumulator;\n        }\n    }\n    return 0;\n}\n\n");
    source.push_str("DF_PUBLIC_V1 uint32_t df_abi_version(void) {\n    return DF_GENERATED_ABI_VERSION_V1;\n}\n\n");
    source.push_str("DF_PUBLIC_V1 const char *df_artifact_id(void) {\n    return df_artifact_id_cstr_v1;\n}\n\n");
    source.push_str("DF_PUBLIC_V1 int32_t df_run_v1(const df_call_v1 *call, const float *x, const uint8_t *packed_weight, float *y) {\n");
    source.push_str("    if (call == NULL) return DF_STATUS_NULL_ARGUMENT_V1;\n");
    source.push_str("    if (call->abi_version != DF_GENERATED_ABI_VERSION_V1) return DF_STATUS_ABI_VERSION_V1;\n");
    source.push_str("    if (call->struct_size != (uint32_t)sizeof(df_call_v1)) return DF_STATUS_STRUCT_SIZE_V1;\n");
    source.push_str("    if (call->flags != UINT64_C(0)) return DF_STATUS_FLAGS_V1;\n");
    source.push_str("    if (call->reserved0 != UINT32_C(0)) return DF_STATUS_RESERVED_V1;\n");
    source.push_str("    if (call->m != UINT32_C(1) || call->n != UINT32_C(");
    source.push_str(&n.to_string());
    source.push_str(") || call->k != UINT32_C(");
    source.push_str(&k.to_string());
    source.push_str(") ) return DF_STATUS_SHAPE_V1;\n");
    source.push_str("    if (call->x_stride != UINT32_C(");
    source.push_str(&k.to_string());
    source.push_str(") || call->y_stride != UINT32_C(");
    source.push_str(&n.to_string());
    source.push_str(") ) return DF_STATUS_STRIDE_V1;\n");
    source.push_str("    if (call->packed_weight_bytes != UINT64_C(");
    source.push_str(&panels.to_string());
    source.push_str(") * UINT64_C(");
    source.push_str(&blocks.to_string());
    source.push_str(") * UINT64_C(");
    source.push_str(&PACK_RECORD_BYTES.to_string());
    source.push_str(") ) return DF_STATUS_PACKED_WEIGHT_BYTES_V1;\n");
    source.push_str("    if (x == NULL || packed_weight == NULL || y == NULL) return DF_STATUS_NULL_ARGUMENT_V1;\n");
    source.push_str("    if (((uintptr_t)packed_weight & (uintptr_t)15u) != (uintptr_t)0u) return DF_STATUS_PACKED_WEIGHT_ALIGNMENT_V1;\n");
    source.push_str("#if defined(FE_TONEAREST) && defined(FLT_HAS_SUBNORM) && FLT_HAS_SUBNORM == 1 && defined(FLT_TRUE_MIN)\n");
    source.push_str("    fenv_t df_saved_environment;\n");
    source.push_str("    int32_t df_status = DF_STATUS_OK_V1;\n");
    source.push_str(
        "    if (feholdexcept(&df_saved_environment) != 0) return DF_STATUS_FP_ENVIRONMENT_V1;\n",
    );
    source.push_str("    if (fegetround() != FE_TONEAREST) {\n        df_status = DF_STATUS_FP_ENVIRONMENT_V1;\n        goto df_restore_environment;\n    }\n");
    source.push_str("    {\n        volatile float df_probe_true_min = FLT_TRUE_MIN;\n        volatile float df_probe_one = 1.0f;\n        volatile float df_probe_min = FLT_MIN;\n        volatile float df_probe_half = 0.5f;\n        const float df_probe_true_min_result = df_probe_true_min * df_probe_one;\n        const float df_probe_half_min_result = df_probe_min * df_probe_half;\n        uint32_t df_probe_true_min_bits = UINT32_C(0);\n        uint32_t df_probe_half_min_bits = UINT32_C(0);\n        memcpy(&df_probe_true_min_bits, &df_probe_true_min_result, sizeof(df_probe_true_min_bits));\n        memcpy(&df_probe_half_min_bits, &df_probe_half_min_result, sizeof(df_probe_half_min_bits));\n        if (df_probe_true_min_bits != UINT32_C(0x00000001) || df_probe_half_min_bits != UINT32_C(0x00400000)) {\n            df_status = DF_STATUS_FP_ENVIRONMENT_V1;\n            goto df_restore_environment;\n        }\n    }\n");
    source.push_str("    for (uint32_t input = 0; input < UINT32_C(");
    source.push_str(&k.to_string());
    source.push_str("); ++input) {\n        if (!isfinite(x[input])) {\n            df_status = DF_STATUS_NONFINITE_INPUT_V1;\n            goto df_restore_environment;\n        }\n    }\n");
    source.push_str("    (void)");
    source.push_str(helper);
    source.push_str("(x, packed_weight, y);\n    for (uint32_t output = 0; output < UINT32_C(");
    source.push_str(&n.to_string());
    source.push_str("); ++output) {\n        if (!isfinite(y[output])) {\n            df_status = DF_STATUS_NONFINITE_RESULT_V1;\n            goto df_restore_environment;\n        }\n    }\n");
    source.push_str("df_restore_environment:\n    if (fesetenv(&df_saved_environment) != 0) return DF_STATUS_FP_ENVIRONMENT_V1;\n    return df_status;\n#else\n    return DF_STATUS_FP_ENVIRONMENT_V1;\n#endif\n}\n");
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
        let kernel = LoopKernelV1::new(&region, KernelVariant::Scalar).unwrap();
        (region, kernel, packed)
    }

    #[test]
    fn emission_is_repeatable_and_has_strict_tail_forms() {
        let (region, kernel, packed) = sample(5, 33);
        let first = emit_scalar_c(&region, &kernel, &packed).unwrap();
        let second = emit_scalar_c(&region, &kernel, &packed).unwrap();
        assert_eq!(first, second);
        assert!(first.source.is_ascii());
        assert!(first.source.ends_with('\n'));
        assert!(!first.source.ends_with("\n\n"));
        assert!(first.source.contains("const uint32_t lane_count"));
        assert!(first.source.contains("block_sum = block_sum * scale"));
        assert!(first.source.contains("q_raw >= UINT8_C(128)"));
        assert!(first.source.contains("#pragma STDC FENV_ACCESS ON"));
        assert!(first.source.contains("feholdexcept(&df_saved_environment)"));
        assert!(first.source.contains("fesetenv(&df_saved_environment)"));
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
        assert!(!first.source.contains("lane < UINT32_C(32)"));
        assert!(!first.source.contains("__m128"));
        assert!(first.source.len() <= MAX_SCALAR_C_SOURCE_BYTES);
    }

    #[test]
    fn rejects_neon_and_bindings() {
        let (region, _, packed) = sample(4, 32);
        let neon = LoopKernelV1::new(&region, KernelVariant::Neon).unwrap();
        assert!(emit_scalar_c(&region, &neon, &packed).is_err());
        let other = sample(5, 32);
        assert!(emit_scalar_c(&region, &other.1, &packed).is_err());
        assert_eq!(
            packed.manifest().logical_weight_identity(),
            region.logical_weight_identity()
        );
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
            let kernel = LoopKernelV1::new(&region, KernelVariant::Scalar).unwrap();
            let module = emit_scalar_c(&region, &kernel, &packed)
                .unwrap_or_else(|error| panic!("fixture {case_id}: {error}"));
            assert_eq!(module.module_id.len(), 71);
            assert_eq!(module.hidden_kernel_symbol.len(), 84);
        }
    }

    #[test]
    fn identity_and_source_change_with_shape_but_not_weight_values() {
        let (region, kernel, packed) = sample(1, 1);
        let first = emit_scalar_c(&region, &kernel, &packed).unwrap();
        let (other_region, other_kernel, other_packed) = sample(1, 2);
        let second = emit_scalar_c(&other_region, &other_kernel, &other_packed).unwrap();
        assert_ne!(first.module_id, second.module_id);

        let mut changed_q = vec![0_u8; 32];
        changed_q[0] = 1;
        let changed_weights = Q8Weights::try_new(1, 1, 1, changed_q, vec![0x3f80_0000]).unwrap();
        let changed_packed = PackedWeightsV1::pack(&changed_weights).unwrap();
        let changed_region = Q8LinearRegion::from_weights(&changed_weights).unwrap();
        let changed_kernel = LoopKernelV1::new(&changed_region, KernelVariant::Scalar).unwrap();
        assert_ne!(
            region.logical_weight_identity(),
            changed_region.logical_weight_identity()
        );
        assert!(emit_scalar_c(&region, &kernel, &changed_packed).is_err());
        let changed = emit_scalar_c(&changed_region, &changed_kernel, &changed_packed).unwrap();
        assert_eq!(first.module_id, changed.module_id);
        assert_eq!(first.source, changed.source);
        assert!(!first.source.contains(env!("CARGO_MANIFEST_DIR")));
        assert!(!first.source.contains("2026-"));
    }

    #[test]
    fn module_golden_identity_and_source_hash_are_stable() {
        let (region, kernel, packed) = sample(5, 33);
        let module = emit_scalar_c(&region, &kernel, &packed).unwrap();
        let mut source_hasher = Sha256::new();
        source_hasher.update(module.source.as_bytes());
        let source_hash = format!("sha256:{}", hex_lower(&source_hasher.finalize()));
        assert_eq!(
            module.module_id,
            "sha256:35abe27c09b78890fd2a62bc1ee7f826edf7b4a76a55d99cf133309c247b9969"
        );
        assert_eq!(
            source_hash,
            "sha256:3b081276e3ded67787b8cb917ed4a072065773c87735b30ffeb8508a61d28e54"
        );
    }
}
