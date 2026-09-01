//! Export one small, deterministic fixture for the foreign-function smoke test.
//!
//! The packed bytes are deliberately generated from the checked-in Q8 fixture
//! at runtime instead of being copied into another repository artifact.  The
//! JSON on stdout is a transport envelope only; the bridge remains responsible
//! for validating the canonical manifest and OI4 payload.

use decodeforge_compiler::{
    KernelVariant, LoopKernelV1, PackedWeightsV1, Q8LinearRegion, emit_neon_c,
};
use decodeforge_core::q8::{Q8Weights, fixture};
use serde_json::json;
use std::fs;
use std::path::PathBuf;

const CASE_ID: &str = "exhaustive-q8";
const EXPECTED_N: u32 = 255;
const EXPECTED_K: u32 = 2;

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn main() {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/v1/fixtures/exhaustive-q8.json");
    let fixture_bytes = fs::read(&fixture_path)
        .unwrap_or_else(|error| panic!("unable to read {}: {error}", fixture_path.display()));
    let document = fixture::parse_quant_fixture(&fixture_bytes)
        .unwrap_or_else(|error| panic!("fixture {} is invalid: {error}", fixture_path.display()));
    assert_eq!(document.case_id, CASE_ID);
    assert_eq!(document.n, EXPECTED_N);
    assert_eq!(document.k, EXPECTED_K);

    let q_bytes = document
        .expected_q_bytes
        .iter()
        .map(|value| *value as i8 as u8)
        .collect();
    let weights = Q8Weights::try_new(
        document.n,
        document.k,
        document.blocks,
        q_bytes,
        document.expected_scale_bits,
    )
    .expect("fixture Q8 storage must be valid");
    let packed = PackedWeightsV1::pack(&weights).expect("fixture OI4 packing must be valid");
    assert_eq!(packed.len(), 9_216);
    let region = Q8LinearRegion::from_weights(&weights).expect("fixture region must be valid");
    let kernel =
        LoopKernelV1::new(&region, KernelVariant::Neon).expect("fixture NEON kernel must be valid");
    let module = emit_neon_c(&region, &kernel, &packed).expect("fixture NEON C must be valid");
    let manifest_json = packed
        .canonical_manifest_json()
        .expect("fixture manifest JSON must be canonical");

    let envelope = json!({
        "case_id": document.case_id,
        "n": document.n,
        "k": document.k,
        "module_id": module.module_id(),
        "packed_weight_id": packed.packed_identity(),
        "pack_manifest_json": manifest_json,
        "packed_weight_hex": hex_encode(packed.bytes()),
        "input_fp32_bits": document.input_fp32_bits,
        "expected_output_fp32_bits": document.expected_output_fp32_bits,
    });
    println!(
        "{}",
        serde_json::to_string(&envelope).expect("fixture envelope is serializable")
    );
}
