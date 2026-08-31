//! Canonical C rendering for the generated ABI v1 surface.
//!
//! Kernel bodies remain backend-owned. This module only owns the artifact
//! identity storage and the three public ABI entrypoints shared by those
//! bodies.

/// Frozen values needed to bind a generated kernel helper to ABI v1.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct AbiV1Spec<'a> {
    artifact_id: &'a str,
    helper_symbol: &'a str,
    n: u32,
    k: u32,
    panels: u32,
    blocks: u32,
    record_bytes: u32,
}

impl<'a> AbiV1Spec<'a> {
    pub(super) const fn new(
        artifact_id: &'a str,
        helper_symbol: &'a str,
        n: u32,
        k: u32,
        panels: u32,
        blocks: u32,
        record_bytes: u32,
    ) -> Self {
        Self {
            artifact_id,
            helper_symbol,
            n,
            k,
            panels,
            blocks,
            record_bytes,
        }
    }
}

/// Append the storage read by `df_artifact_id`.
pub(super) fn render_artifact_id_storage(source: &mut String, spec: &AbiV1Spec<'_>) {
    source.push_str("static const char df_artifact_id_cstr_v1[] = \"");
    source.push_str(spec.artifact_id);
    source.push_str("\";\n_Static_assert(sizeof(df_artifact_id_cstr_v1) == DF_ARTIFACT_ID_CSTR_BYTES_V1, \"artifact identity size\");\n\n");
}

/// Append the generated-ABI version entrypoint.
pub(super) fn render_abi_version_entrypoint(source: &mut String) {
    source.push_str("DF_PUBLIC_V1 uint32_t df_abi_version(void) {\n    return DF_GENERATED_ABI_VERSION_V1;\n}\n\n");
}

/// Append the artifact-identity entrypoint.
pub(super) fn render_artifact_id_entrypoint(source: &mut String) {
    source.push_str("DF_PUBLIC_V1 const char *df_artifact_id(void) {\n    return df_artifact_id_cstr_v1;\n}\n\n");
}

/// Append the checked execution entrypoint around a backend-owned helper.
pub(super) fn render_run_v1_entrypoint(source: &mut String, spec: &AbiV1Spec<'_>) {
    source.push_str("DF_PUBLIC_V1 int32_t df_run_v1(const df_call_v1 *call, const float *x, const uint8_t *packed_weight, float *y) {\n");
    source.push_str("    if (call == NULL) return DF_STATUS_NULL_ARGUMENT_V1;\n");
    source.push_str("    if (call->abi_version != DF_GENERATED_ABI_VERSION_V1) return DF_STATUS_ABI_VERSION_V1;\n");
    source.push_str("    if (call->struct_size != (uint32_t)sizeof(df_call_v1)) return DF_STATUS_STRUCT_SIZE_V1;\n");
    source.push_str("    if (call->flags != UINT64_C(0)) return DF_STATUS_FLAGS_V1;\n");
    source.push_str("    if (call->reserved0 != UINT32_C(0)) return DF_STATUS_RESERVED_V1;\n");
    source.push_str("    if (call->m != UINT32_C(1) || call->n != UINT32_C(");
    source.push_str(&spec.n.to_string());
    source.push_str(") || call->k != UINT32_C(");
    source.push_str(&spec.k.to_string());
    source.push_str(") ) return DF_STATUS_SHAPE_V1;\n");
    source.push_str("    if (call->x_stride != UINT32_C(");
    source.push_str(&spec.k.to_string());
    source.push_str(") || call->y_stride != UINT32_C(");
    source.push_str(&spec.n.to_string());
    source.push_str(") ) return DF_STATUS_STRIDE_V1;\n");
    source.push_str("    if (call->packed_weight_bytes != UINT64_C(");
    source.push_str(&spec.panels.to_string());
    source.push_str(") * UINT64_C(");
    source.push_str(&spec.blocks.to_string());
    source.push_str(") * UINT64_C(");
    source.push_str(&spec.record_bytes.to_string());
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
    source.push_str(&spec.k.to_string());
    source.push_str("); ++input) {\n        if (!isfinite(x[input])) {\n            df_status = DF_STATUS_NONFINITE_INPUT_V1;\n            goto df_restore_environment;\n        }\n    }\n");
    source.push_str("    (void)");
    source.push_str(spec.helper_symbol);
    source.push_str("(x, packed_weight, y);\n    for (uint32_t output = 0; output < UINT32_C(");
    source.push_str(&spec.n.to_string());
    source.push_str("); ++output) {\n        if (!isfinite(y[output])) {\n            df_status = DF_STATUS_NONFINITE_RESULT_V1;\n            goto df_restore_environment;\n        }\n    }\n");
    source.push_str("df_restore_environment:\n    if (fesetenv(&df_saved_environment) != 0) return DF_STATUS_FP_ENVIRONMENT_V1;\n    return df_status;\n#else\n    return DF_STATUS_FP_ENVIRONMENT_V1;\n#endif\n}\n");
}
