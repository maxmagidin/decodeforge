use super::*;
use crate::PackManifestV1;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, serialize};
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use tempfile::TempDir;

const TENSOR_NAME: &str = "model.layers.0.self_attn.q_proj.weight";

fn bf16_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn write_safetensors(
    directory: &TempDir,
    filename: &str,
    name: &str,
    dtype: Dtype,
    shape: Vec<usize>,
    data: &[u8],
) -> (PathBuf, Vec<u8>) {
    let view = TensorView::new(dtype, shape, data).unwrap();
    let bytes = serialize([(name.to_owned(), view)], None).unwrap();
    let path = directory.path().join(filename);
    fs::write(&path, &bytes).unwrap();
    (path, bytes)
}

fn source_spec(filename: &str, bytes: &[u8]) -> ModelSourceSpecV1 {
    ModelSourceSpecV1::new(
        "test/model",
        "0123456789abcdef0123456789abcdef01234567",
        filename,
        bytes.len() as u64,
        sha256_identity(bytes),
    )
    .unwrap()
}

fn tensor_spec(n: u32, k: u32, data: &[u8]) -> Q8LinearTensorSpecV1 {
    Q8LinearTensorSpecV1::bf16(TENSOR_NAME, n, k, Some(sha256_identity(data))).unwrap()
}

fn sample() -> (TempDir, PathBuf, Vec<u8>, Vec<u8>) {
    let directory = tempfile::tempdir().unwrap();
    let data = bf16_bytes(&[
        0x3f80, 0xc000, 0x3f00, 0x0000, 0xbf80, 0x4000, 0x4040, 0xc080,
    ]);
    let (path, source) = write_safetensors(
        &directory,
        "fixture.safetensors",
        TENSOR_NAME,
        Dtype::BF16,
        vec![2, 4],
        &data,
    );
    (directory, path, source, data)
}

fn write_q_proj_source(
    directory: &TempDir,
    filename: &str,
    omitted_layer: Option<u32>,
    include_bias: bool,
) -> (PathBuf, Vec<u8>) {
    let data = (0..TINYLLAMA_LAYER_COUNT)
        .map(|layer| {
            bf16_bytes(&[
                0x3f80 + layer as u16,
                0xc000,
                0x3f00,
                0x0000,
                0xbf80,
                0x4000 + layer as u16,
                0x4040,
                0xc080,
            ])
        })
        .collect::<Vec<_>>();
    let bias = bf16_bytes(&[0x3f80, 0x4000]);
    let mut views = Vec::new();
    for layer in 0..TINYLLAMA_LAYER_COUNT {
        if omitted_layer == Some(layer) {
            continue;
        }
        views.push((
            q_proj_tensor_name(layer),
            TensorView::new(Dtype::BF16, vec![2, 4], &data[layer as usize]).unwrap(),
        ));
    }
    if include_bias {
        views.push((
            "model.layers.0.self_attn.q_proj.bias".to_owned(),
            TensorView::new(Dtype::BF16, vec![2], &bias).unwrap(),
        ));
    }
    let bytes = serialize(views, None).unwrap();
    let path = directory.path().join(filename);
    fs::write(&path, &bytes).unwrap();
    (path, bytes)
}

fn q_proj_specs() -> Vec<Q8LinearTensorSpecV1> {
    (0..TINYLLAMA_LAYER_COUNT)
        .map(|layer| Q8LinearTensorSpecV1::bf16(q_proj_tensor_name(layer), 2, 4, None).unwrap())
        .collect()
}

#[test]
fn synthetic_bf16_tensor_prepares_a_bridge_consumable_bundle() {
    let (directory, source_path, source_bytes, tensor_bytes) = sample();
    let output = directory.path().join("prepared");
    let prepared = prepare_q8_linear_asset_v1(
        &source_path,
        &output,
        &source_spec("fixture.safetensors", &source_bytes),
        &tensor_spec(2, 4, &tensor_bytes),
    )
    .unwrap();

    assert_eq!(prepared.output_directory, output);
    assert_eq!(prepared.manifest.tensor.shape, [2, 4]);
    assert_eq!(
        prepared.manifest.tensor.data_identity,
        sha256_identity(&tensor_bytes)
    );
    assert_eq!(prepared.manifest.pack.payload_bytes, 144);
    assert_eq!(prepared.manifest.fallback.bytes, 32);
    assert_eq!(
        prepared.manifest.fallback.parent_logical_weight_identity,
        prepared.manifest.quantization.logical_weight_identity
    );
    assert_eq!(
        prepared.manifest.fallback.parent_packed_weight_identity,
        prepared.manifest.pack.packed_weight_identity
    );
    prepared.manifest.verify().unwrap();

    let manifest_bytes = fs::read(output.join(MANIFEST_FILENAME)).unwrap();
    assert_eq!(manifest_bytes.last(), Some(&b'\n'));
    let on_disk: AssetManifestV1 = serde_json::from_slice(&manifest_bytes).unwrap();
    assert_eq!(on_disk, prepared.manifest);
    on_disk.verify().unwrap();

    let pack_manifest_bytes = fs::read(output.join(PACK_MANIFEST_FILENAME)).unwrap();
    let pack_manifest: PackManifestV1 = serde_json::from_slice(&pack_manifest_bytes).unwrap();
    let payload = fs::read(output.join(PACK_PAYLOAD_FILENAME)).unwrap();
    let packed = PackedWeightsV1::from_artifact_parts(pack_manifest, payload).unwrap();
    assert_eq!(
        packed.packed_identity(),
        prepared.manifest.pack.packed_weight_identity
    );
    assert_eq!(
        packed.logical_weight_identity(),
        prepared.manifest.quantization.logical_weight_identity
    );
    let verified = verify_q8_linear_asset_v1(&output).unwrap();
    assert_eq!(verified.fallback_bits.len(), 8);

    assert_eq!(
        fs::metadata(&output).unwrap().permissions().mode() & 0o777,
        0o700
    );
    for filename in [
        MANIFEST_FILENAME,
        PACK_MANIFEST_FILENAME,
        PACK_PAYLOAD_FILENAME,
        FALLBACK_FILENAME,
    ] {
        assert_eq!(
            fs::metadata(output.join(filename))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
}

#[test]
fn preparation_is_byte_deterministic_across_output_directories() {
    let (directory, source_path, source_bytes, tensor_bytes) = sample();
    let source_spec = source_spec("fixture.safetensors", &source_bytes);
    let tensor_spec = tensor_spec(2, 4, &tensor_bytes);
    let first = directory.path().join("first");
    let second = directory.path().join("second");
    prepare_q8_linear_asset_v1(&source_path, &first, &source_spec, &tensor_spec).unwrap();
    prepare_q8_linear_asset_v1(&source_path, &second, &source_spec, &tensor_spec).unwrap();

    for filename in [
        MANIFEST_FILENAME,
        PACK_MANIFEST_FILENAME,
        PACK_PAYLOAD_FILENAME,
        FALLBACK_FILENAME,
    ] {
        assert_eq!(
            fs::read(first.join(filename)).unwrap(),
            fs::read(second.join(filename)).unwrap(),
            "{filename} changed between identical preparations"
        );
    }
}

#[test]
fn exactly_twenty_two_layers_prepare_in_order_with_bound_fallbacks() {
    let directory = tempfile::tempdir().unwrap();
    let (source_path, source_bytes) =
        write_q_proj_source(&directory, "all-qproj.safetensors", None, false);
    let source_spec = source_spec("all-qproj.safetensors", &source_bytes);
    let specs = q_proj_specs();
    let first = directory.path().join("first-inventory");
    let second = directory.path().join("second-inventory");
    let prepared = prepare_q_proj_inventory_v1(&source_path, &first, &source_spec, &specs).unwrap();
    prepare_q_proj_inventory_v1(&source_path, &second, &source_spec, &specs).unwrap();

    assert_eq!(prepared.inventory.entries.len(), 22);
    assert_eq!(prepared.inventory.total_packed_bytes, 22 * 144);
    assert_eq!(prepared.inventory.total_fallback_bytes, 22 * 2 * 4 * 4);
    for (index, entry) in prepared.inventory.entries.iter().enumerate() {
        assert_eq!(entry.layer, index as u32);
        assert_eq!(entry.tensor_name, q_proj_tensor_name(index as u32));
        assert_eq!(entry.directory, q_proj_directory(index as u32));
        let first_layer = first.join(&entry.directory);
        let second_layer = second.join(&entry.directory);
        for filename in [
            MANIFEST_FILENAME,
            PACK_MANIFEST_FILENAME,
            PACK_PAYLOAD_FILENAME,
            FALLBACK_FILENAME,
        ] {
            assert_eq!(
                fs::read(first_layer.join(filename)).unwrap(),
                fs::read(second_layer.join(filename)).unwrap(),
                "layer {index} asset {filename} was not deterministic"
            );
        }
    }
    assert_eq!(
        fs::read(first.join(INVENTORY_FILENAME)).unwrap(),
        fs::read(second.join(INVENTORY_FILENAME)).unwrap()
    );
    let verified = verify_q_proj_inventory_v1(&first).unwrap();
    assert_eq!(verified.inventory, prepared.inventory);
}

#[test]
fn all_layer_preparation_rejects_incomplete_or_biased_source_atomically() {
    let directory = tempfile::tempdir().unwrap();
    let specs = q_proj_specs();
    let (missing_path, missing_bytes) =
        write_q_proj_source(&directory, "missing.safetensors", Some(7), false);
    let missing_output = directory.path().join("missing-output");
    let error = prepare_q_proj_inventory_v1(
        &missing_path,
        &missing_output,
        &source_spec("missing.safetensors", &missing_bytes),
        &specs,
    )
    .unwrap_err();
    assert_eq!(error.code(), "DFE-ASSET-006");
    assert!(!missing_output.exists());

    let (bias_path, bias_bytes) = write_q_proj_source(&directory, "biased.safetensors", None, true);
    let bias_output = directory.path().join("bias-output");
    let error = prepare_q_proj_inventory_v1(
        &bias_path,
        &bias_output,
        &source_spec("biased.safetensors", &bias_bytes),
        &specs,
    )
    .unwrap_err();
    assert_eq!(error.code(), "DFE-ASSET-006");
    assert!(!bias_output.exists());
}

#[test]
fn fallback_and_inventory_tampering_fail_closed() {
    let directory = tempfile::tempdir().unwrap();
    let (source_path, source_bytes) =
        write_q_proj_source(&directory, "all-qproj.safetensors", None, false);
    let output = directory.path().join("inventory");
    prepare_q_proj_inventory_v1(
        &source_path,
        &output,
        &source_spec("all-qproj.safetensors", &source_bytes),
        &q_proj_specs(),
    )
    .unwrap();
    let fallback = output.join(q_proj_directory(3)).join(FALLBACK_FILENAME);
    let mut bytes = fs::read(&fallback).unwrap();
    bytes[0] ^= 1;
    fs::write(&fallback, bytes).unwrap();
    let error = verify_q_proj_inventory_v1(&output).unwrap_err();
    assert_eq!(error.code(), "DFE-ASSET-007");
}

#[test]
fn source_must_be_regular_non_symlink_and_match_size_and_hash() {
    let (directory, source_path, source_bytes, tensor_bytes) = sample();
    let output = directory.path().join("prepared");
    let tensor_spec = tensor_spec(2, 4, &tensor_bytes);

    let link = directory.path().join("linked.safetensors");
    std::os::unix::fs::symlink(&source_path, &link).unwrap();
    let error = prepare_q8_linear_asset_v1(
        &link,
        &output,
        &source_spec("fixture.safetensors", &source_bytes),
        &tensor_spec,
    )
    .unwrap_err();
    assert_eq!(error.code(), "DFE-ASSET-002");

    let wrong_size = ModelSourceSpecV1::new(
        "test/model",
        "0123456789abcdef0123456789abcdef01234567",
        "fixture.safetensors",
        source_bytes.len() as u64 + 1,
        sha256_identity(&source_bytes),
    )
    .unwrap();
    let error =
        prepare_q8_linear_asset_v1(&source_path, &output, &wrong_size, &tensor_spec).unwrap_err();
    assert_eq!(error.code(), "DFE-ASSET-004");

    let wrong_hash = ModelSourceSpecV1::new(
        "test/model",
        "0123456789abcdef0123456789abcdef01234567",
        "fixture.safetensors",
        source_bytes.len() as u64,
        "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    )
    .unwrap();
    let error =
        prepare_q8_linear_asset_v1(&source_path, &output, &wrong_hash, &tensor_spec).unwrap_err();
    assert_eq!(error.code(), "DFE-ASSET-004");
}

#[test]
fn source_mutation_and_truncation_during_snapshot_fail_closed() {
    for truncate in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("mutable.safetensors");
        let source_bytes = vec![0x5a; 2 * 1024 * 1024 + 17];
        fs::write(&source_path, &source_bytes).unwrap();
        let spec = source_spec("mutable.safetensors", &source_bytes);
        let mut changed = false;
        let result = SnapshottedSource::open_with_progress_hook(&source_path, &spec, |copied| {
            if changed {
                return;
            }
            assert_eq!(copied, 1024 * 1024);
            changed = true;
            if truncate {
                OpenOptions::new()
                    .write(true)
                    .open(&source_path)
                    .unwrap()
                    .set_len(copied + 7)
                    .unwrap();
            } else {
                let mut writer = OpenOptions::new().write(true).open(&source_path).unwrap();
                writer.seek(SeekFrom::Start(copied + 7)).unwrap();
                writer
                    .write_all(&[source_bytes[(copied + 7) as usize] ^ 1])
                    .unwrap();
                writer.flush().unwrap();
            }
        });
        let error = match result {
            Ok(_) => panic!("changed source unexpectedly produced a snapshot"),
            Err(error) => error,
        };
        assert!(changed);
        assert_eq!(error.code(), "DFE-ASSET-004");
        assert!(error.summary().contains("changed"));
    }
}

#[test]
fn tensor_dtype_shape_identity_and_finiteness_are_strict() {
    let directory = tempfile::tempdir().unwrap();
    let f32_data = [1.0_f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let (f32_path, f32_source) = write_safetensors(
        &directory,
        "f32.safetensors",
        TENSOR_NAME,
        Dtype::F32,
        vec![2, 2],
        &f32_data,
    );
    let error = prepare_q8_linear_asset_v1(
        &f32_path,
        &directory.path().join("f32-output"),
        &source_spec("f32.safetensors", &f32_source),
        &Q8LinearTensorSpecV1::bf16(TENSOR_NAME, 2, 2, None).unwrap(),
    )
    .unwrap_err();
    assert_eq!(error.code(), "DFE-ASSET-006");
    let flexible_output = directory.path().join("f32-flex-output");
    let prepared = prepare_q8_linear_asset_v1(
        &f32_path,
        &flexible_output,
        &source_spec("f32.safetensors", &f32_source),
        &Q8LinearTensorSpecV1::f32_or_bf16(TENSOR_NAME, 2, 2, None).unwrap(),
    )
    .unwrap();
    assert_eq!(prepared.manifest.tensor.dtype, "F32");
    verify_q8_linear_asset_v1(&flexible_output).unwrap();

    let nonfinite = bf16_bytes(&[0x3f80, 0x7f80, 0x4000, 0x4040]);
    let (nonfinite_path, nonfinite_source) = write_safetensors(
        &directory,
        "nonfinite.safetensors",
        TENSOR_NAME,
        Dtype::BF16,
        vec![2, 2],
        &nonfinite,
    );
    let error = prepare_q8_linear_asset_v1(
        &nonfinite_path,
        &directory.path().join("nonfinite-output"),
        &source_spec("nonfinite.safetensors", &nonfinite_source),
        &Q8LinearTensorSpecV1::bf16(TENSOR_NAME, 2, 2, None).unwrap(),
    )
    .unwrap_err();
    assert_eq!(error.code(), "DFE-ASSET-006");
    assert!(error.summary().contains("non-finite"));

    let finite = bf16_bytes(&[0x3f80, 0x4000, 0x4040, 0x4080]);
    let (finite_path, finite_source) = write_safetensors(
        &directory,
        "finite.safetensors",
        TENSOR_NAME,
        Dtype::BF16,
        vec![2, 2],
        &finite,
    );
    let wrong_identity = Q8LinearTensorSpecV1::bf16(
        TENSOR_NAME,
        2,
        2,
        Some("sha256:0000000000000000000000000000000000000000000000000000000000000000".to_owned()),
    )
    .unwrap();
    let error = prepare_q8_linear_asset_v1(
        &finite_path,
        &directory.path().join("identity-output"),
        &source_spec("finite.safetensors", &finite_source),
        &wrong_identity,
    )
    .unwrap_err();
    assert_eq!(error.code(), "DFE-ASSET-006");

    let wrong_shape = Q8LinearTensorSpecV1::bf16(TENSOR_NAME, 1, 4, None).unwrap();
    let error = prepare_q8_linear_asset_v1(
        &finite_path,
        &directory.path().join("shape-output"),
        &source_spec("finite.safetensors", &finite_source),
        &wrong_shape,
    )
    .unwrap_err();
    assert_eq!(error.code(), "DFE-ASSET-006");
}

#[test]
fn tensor_lookup_and_header_bound_fail_closed() {
    let directory = tempfile::tempdir().unwrap();
    let data = bf16_bytes(&[0x3f80, 0x4000, 0x4040, 0x4080]);
    let (path, source) = write_safetensors(
        &directory,
        "other.safetensors",
        "other.weight",
        Dtype::BF16,
        vec![2, 2],
        &data,
    );
    let error = prepare_q8_linear_asset_v1(
        &path,
        &directory.path().join("missing-output"),
        &source_spec("other.safetensors", &source),
        &Q8LinearTensorSpecV1::bf16(TENSOR_NAME, 2, 2, None).unwrap(),
    )
    .unwrap_err();
    assert_eq!(error.code(), "DFE-ASSET-006");

    let mut oversized_header = ((MAX_HEADER_BYTES + 1) as u64).to_le_bytes().to_vec();
    oversized_header.push(0);
    let malformed_path = directory.path().join("malformed.safetensors");
    fs::write(&malformed_path, &oversized_header).unwrap();
    let error = prepare_q8_linear_asset_v1(
        &malformed_path,
        &directory.path().join("malformed-output"),
        &source_spec("malformed.safetensors", &oversized_header),
        &Q8LinearTensorSpecV1::bf16(TENSOR_NAME, 2, 2, None).unwrap(),
    )
    .unwrap_err();
    assert_eq!(error.code(), "DFE-ASSET-003");

    let tensor_info = r#"{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}"#;
    let mut header = format!(r#"{{"{TENSOR_NAME}":{tensor_info},"{TENSOR_NAME}":{tensor_info}}}"#);
    while !(8 + header.len()).is_multiple_of(8) {
        header.push(' ');
    }
    let mut duplicate = (header.len() as u64).to_le_bytes().to_vec();
    duplicate.extend_from_slice(header.as_bytes());
    duplicate.extend_from_slice(&data);
    let duplicate_path = directory.path().join("duplicate.safetensors");
    fs::write(&duplicate_path, &duplicate).unwrap();
    let error = prepare_q8_linear_asset_v1(
        &duplicate_path,
        &directory.path().join("duplicate-output"),
        &source_spec("duplicate.safetensors", &duplicate),
        &Q8LinearTensorSpecV1::bf16(TENSOR_NAME, 2, 2, None).unwrap(),
    )
    .unwrap_err();
    assert_eq!(error.code(), "DFE-ASSET-005");
}

#[test]
fn output_is_no_clobber_and_parent_must_not_be_a_symlink() {
    let (directory, source_path, source_bytes, tensor_bytes) = sample();
    let source_spec = source_spec("fixture.safetensors", &source_bytes);
    let tensor_spec = tensor_spec(2, 4, &tensor_bytes);
    let output = directory.path().join("existing");
    fs::create_dir(&output).unwrap();
    fs::write(output.join("keep.txt"), b"preserve me").unwrap();
    let error =
        prepare_q8_linear_asset_v1(&source_path, &output, &source_spec, &tensor_spec).unwrap_err();
    assert_eq!(error.code(), "DFE-ASSET-007");
    assert_eq!(fs::read(output.join("keep.txt")).unwrap(), b"preserve me");

    let real_parent = directory.path().join("real-parent");
    fs::create_dir(&real_parent).unwrap();
    let linked_parent = directory.path().join("linked-parent");
    std::os::unix::fs::symlink(&real_parent, &linked_parent).unwrap();
    let error = prepare_q8_linear_asset_v1(
        &source_path,
        &linked_parent.join("output"),
        &source_spec,
        &tensor_spec,
    )
    .unwrap_err();
    assert_eq!(error.code(), "DFE-ASSET-007");
}

#[test]
fn specs_are_closed_and_tinyllama_layer_names_scale_to_twenty_two() {
    assert!(ModelSourceSpecV1::new("", "0".repeat(40), "model", 9, "x").is_err());
    assert!(Q8LinearTensorSpecV1::bf16("../weight", 1, 1, None).is_err());
    assert!(Q8LinearTensorSpecV1::bf16("weight", 0, 1, None).is_err());

    let (source, first) = tinyllama_q_proj_spec_v1(0).unwrap();
    let (_, last) = tinyllama_q_proj_spec_v1(21).unwrap();
    assert_eq!(source.source_bytes, TINYLLAMA_SOURCE_BYTES);
    assert_eq!(first.name, "model.layers.0.self_attn.q_proj.weight");
    assert!(first.expected_data_identity.is_some());
    assert_eq!(last.name, "model.layers.21.self_attn.q_proj.weight");
    assert!(last.expected_data_identity.is_none());
    assert!(tinyllama_q_proj_spec_v1(22).is_err());
}

#[test]
fn required_real_shape_byte_extents_are_recomputed() {
    let packed_per_layer = expected_payload_bytes(2048, 2048).unwrap();
    let fallback_per_layer = 2048usize * 2048 * std::mem::size_of::<f32>();
    assert_eq!(packed_per_layer, 4_718_592);
    assert_eq!(
        packed_per_layer * TINYLLAMA_LAYER_COUNT as usize,
        103_809_024
    );
    assert_eq!(fallback_per_layer, 16_777_216);
    assert_eq!(
        fallback_per_layer * TINYLLAMA_LAYER_COUNT as usize,
        369_098_752
    );
}
