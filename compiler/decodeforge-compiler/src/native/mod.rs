//! Minimal Apple-only scalar dylib construction.
//!
//! This is intentionally not a runtime loader or backend framework. It takes
//! the already-verified scalar C module, compiles one fixed arm64 Mach-O dylib
//! in a private temporary directory, and audits the hidden helper before the
//! owner is returned.

mod audit;
mod macho;
mod toolchain;

use crate::{Result, ScalarCModule, invalid};
use std::fs::File;
use tempfile::TempDir;

pub use audit::ScalarDylibAuditReport;

/// Upper bound for a generated dylib retained by the build owner.
pub const MAX_SCALAR_DYLIB_BYTES: usize = 8 * 1024 * 1024;

/// Fixed Apple Clang arguments, in the exact order passed to the compiler.
pub const APPLE_SCALAR_CLANG_FLAGS: &[&str] = &[
    "-std=c11",
    "--no-default-config",
    "-O2",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Werror",
    "-fno-fast-math",
    "-ffp-model=strict",
    "-ffp-contract=off",
    "-fdenormal-fp-math=ieee",
    "-fno-vectorize",
    "-fno-slp-vectorize",
    "-fno-unroll-loops",
    "-fvisibility=hidden",
    "-fPIC",
    "-dynamiclib",
    "-arch",
    "arm64",
    "-mmacosx-version-min=15.0",
    "-Wl,-install_name,@rpath/decodeforge_scalar_v1.dylib",
    // Undefined symbols are errors by default for this link mode. Apple ld 17
    // deprecates the redundant `-undefined error` spelling, which would turn
    // into a build failure under the following fatal-warning policy.
    "-Wl,-fatal_warnings",
    "-Wl,-exported_symbol,_df_abi_version",
    "-Wl,-exported_symbol,_df_artifact_id",
    "-Wl,-exported_symbol,_df_run_v1",
];

/// Normalized, bounded provenance of the Apple tools that built the dylib.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppleToolchainProvenance {
    compiler: String,
    compiler_version: String,
    target: String,
    sdk_version: String,
    objdump_version: String,
}

impl AppleToolchainProvenance {
    /// Fixed compiler family selected through `/usr/bin/xcrun`.
    pub fn compiler(&self) -> &str {
        &self.compiler
    }

    /// Normalized Apple Clang version output.
    pub fn compiler_version(&self) -> &str {
        &self.compiler_version
    }

    /// Normalized compiler target triple.
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Normalized macOS SDK version.
    pub fn sdk_version(&self) -> &str {
        &self.sdk_version
    }

    /// Normalized LLVM objdump version output.
    pub fn objdump_version(&self) -> &str {
        &self.objdump_version
    }
}

/// Owner of a checked scalar dylib and its private retained backing file.
///
/// The original build directory is removed before this value is returned.
/// Dropping it removes the separate audit directory. The crate does not load
/// the dylib or retain a dynamic-library handle.
pub struct AppleScalarDylib {
    // Fields are ordered so the descriptor closes before its backing directory
    // is removed.
    pub(crate) _dylib_file: File,
    pub(crate) _temp_dir: TempDir,
    pub(crate) module_id: String,
    pub(crate) source_hash: String,
    pub(crate) abi_header_hash: String,
    pub(crate) dylib_hash: String,
    pub(crate) dylib_bytes: Vec<u8>,
    pub(crate) toolchain: AppleToolchainProvenance,
    pub(crate) flags: Vec<String>,
    pub(crate) disassembly: String,
    pub(crate) audit_report: ScalarDylibAuditReport,
    pub(crate) dynamic_exports: Vec<String>,
}

impl AppleScalarDylib {
    /// Stable identity supplied by the verified generated C module.
    pub fn module_id(&self) -> &str {
        &self.module_id
    }

    /// SHA-256 identity of the exact source file compiled.
    pub fn source_hash(&self) -> &str {
        &self.source_hash
    }

    /// SHA-256 identity of the ABI header snapshot compiled with the source.
    pub fn abi_header_hash(&self) -> &str {
        &self.abi_header_hash
    }

    /// SHA-256 identity of this particular linker output.
    ///
    /// This records the observed output; it does not claim byte-for-byte
    /// reproducibility across Apple linker versions or installations.
    pub fn dylib_hash(&self) -> &str {
        &self.dylib_hash
    }

    /// The retained Mach-O bytes, bounded by [`MAX_SCALAR_DYLIB_BYTES`].
    pub fn dylib_bytes(&self) -> &[u8] {
        &self.dylib_bytes
    }

    /// Normalized compiler and SDK provenance.
    pub fn toolchain(&self) -> &AppleToolchainProvenance {
        &self.toolchain
    }

    /// Fixed code-generation and linker policy flags.
    ///
    /// The private source/include/output path arguments are intentionally not
    /// exposed as portable policy.
    pub fn flags(&self) -> &[String] {
        &self.flags
    }

    /// Bounded raw `llvm-objdump` output used by the scalar audit.
    pub fn disassembly(&self) -> &str {
        &self.disassembly
    }

    /// Structural report from auditing the named hidden helper.
    pub fn audit_report(&self) -> &ScalarDylibAuditReport {
        &self.audit_report
    }

    /// The three required dynamically exported ABI symbols.
    pub fn dynamic_exports(&self) -> &[String] {
        &self.dynamic_exports
    }
}

/// Build and audit one fixed Apple arm64 scalar dylib.
///
/// On any non-macOS-arm64 host this returns a stable unsupported-host error
/// without invoking a compiler. Portable parser and runner tests remain
/// available on those hosts.
pub fn build_apple_scalar_dylib(module: &ScalarCModule) -> Result<AppleScalarDylib> {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Err(invalid(
            "DFE-NATIVE-001",
            "Apple scalar dylib builds require a macOS arm64 host.",
        ));
    }
    toolchain::build_apple_scalar_dylib(module)
}

#[cfg(all(test, not(all(target_os = "macos", target_arch = "aarch64"))))]
mod unsupported_host_tests {
    use super::*;
    use crate::{KernelVariant, LoopKernelV1, PackedWeightsV1, Q8LinearRegion, emit_scalar_c};
    use decodeforge_core::q8::Q8Weights;

    #[test]
    fn apple_build_api_reports_an_unsupported_host() {
        let weights = Q8Weights::try_new(1, 1, 1, vec![0_u8; 32], vec![0x3f80_0000]).unwrap();
        let packed = PackedWeightsV1::pack(&weights).unwrap();
        let region = Q8LinearRegion::from_weights(&weights).unwrap();
        let kernel = LoopKernelV1::new(&region, KernelVariant::Scalar).unwrap();
        let module = emit_scalar_c(&region, &kernel, &packed).unwrap();
        let error = match build_apple_scalar_dylib(&module) {
            Ok(_) => panic!("non-macOS hosts must not invoke an Apple toolchain"),
            Err(error) => error,
        };
        assert_eq!(error.code(), "DFE-NATIVE-001");
        assert!(error.summary().contains("macOS arm64"));
    }
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod tests {
    use super::macho::audit_scalar_macho;
    use super::*;
    use crate::{KernelVariant, LoopKernelV1, PackedWeightsV1, Q8LinearRegion, emit_scalar_c};
    use decodeforge_core::q8::Q8Weights;
    use sha2::{Digest, Sha256};
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    fn sha256_identity(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("sha256:{}", crate::hex_lower(&hasher.finalize()))
    }

    fn shape_module(n: u32, k: u32) -> ScalarCModule {
        let blocks = k.div_ceil(32);
        let weights = Q8Weights::try_new(
            n,
            k,
            blocks,
            vec![0_u8; (n * blocks * 32) as usize],
            vec![0x3f80_0000; (n * blocks) as usize],
        )
        .unwrap();
        let packed = PackedWeightsV1::pack(&weights).unwrap();
        let region = Q8LinearRegion::from_weights(&weights).unwrap();
        let kernel = LoopKernelV1::new(&region, KernelVariant::Scalar).unwrap();
        emit_scalar_c(&region, &kernel, &packed).unwrap()
    }

    fn load_command_offset(bytes: &[u8], wanted: u32) -> usize {
        let command_count = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let mut offset = 32_usize;
        for _ in 0..command_count {
            let command = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            let size =
                u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
            if command == wanted {
                return offset;
            }
            offset += size;
        }
        panic!("expected Mach-O load command {wanted:#x}");
    }

    #[test]
    fn builds_and_audits_n5_k33_apple_scalar_dylib() {
        let first = shape_module(5, 33);
        let second = shape_module(5, 33);
        assert_eq!(first.module_id(), second.module_id());
        assert_eq!(first.source(), second.source());

        let build = build_apple_scalar_dylib(&first).unwrap_or_else(|error| {
            panic!("Apple scalar dylib build failed: {error}");
        });
        assert_eq!(build.module_id(), first.module_id());
        assert_eq!(
            build.source_hash(),
            sha256_identity(first.source().as_bytes())
        );
        assert_eq!(
            build.abi_header_hash(),
            sha256_identity(include_str!("../../../../include/decodeforge/abi_v1.h").as_bytes())
        );
        assert_eq!(build.dylib_hash(), sha256_identity(build.dylib_bytes()));
        assert!(build.dylib_bytes().starts_with(&[0xcf, 0xfa, 0xed, 0xfe]));
        let retained_path = build
            ._temp_dir
            .path()
            .join("out/decodeforge_scalar_v1.dylib");
        assert!(retained_path.is_file());
        assert_eq!(
            fs::metadata(build._temp_dir.path())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(&retained_path).unwrap().permissions().mode() & 0o777,
            0o400
        );
        assert_eq!(
            build.dynamic_exports(),
            ["df_abi_version", "df_artifact_id", "df_run_v1"]
        );
        assert!(build.audit_report().scalar_fmul_count() > 0);
        assert!(build.audit_report().scalar_fadd_count() > 0);
        assert!(build.audit_report().logical_lane_loop_observed());
        assert!(build.toolchain().compiler_version().contains("clang"));
        assert!(build.toolchain().target().contains("arm64"));
        assert!(!build.flags().iter().any(|flag| flag.contains("march")));
        assert_eq!(
            build.flags(),
            APPLE_SCALAR_CLANG_FLAGS
                .iter()
                .map(|flag| (*flag).to_owned())
                .collect::<Vec<_>>()
        );
        let private_directory = build._temp_dir.path().to_string_lossy();
        assert!(!first.source().contains(private_directory.as_ref()));
        assert!(!build.disassembly().contains(private_directory.as_ref()));

        let build_directory = build._temp_dir.path().to_owned();
        drop(build);
        assert!(!build_directory.exists());
    }

    #[test]
    fn builds_degenerate_and_tail_apple_scalar_dylibs() {
        for (n, k) in [(1, 1), (1, 32), (1, 33), (3, 33)] {
            let module = shape_module(n, k);
            build_apple_scalar_dylib(&module)
                .unwrap_or_else(|error| panic!("native build failed for N={n}, K={k}: {error}"));
        }
    }

    #[test]
    fn macho_audit_rejects_security_contract_mutations() {
        let module = shape_module(5, 33);
        let build = build_apple_scalar_dylib(&module).unwrap();
        audit_scalar_macho(build.dylib_bytes(), module.hidden_kernel_symbol()).unwrap();

        let mut wrong_subtype = build.dylib_bytes().to_vec();
        wrong_subtype[8..12].copy_from_slice(&object::macho::CPU_SUBTYPE_ARM64E.0.to_le_bytes());
        assert!(audit_scalar_macho(&wrong_subtype, module.hidden_kernel_symbol()).is_err());

        let mut zero_uuid = build.dylib_bytes().to_vec();
        let uuid = load_command_offset(&zero_uuid, object::macho::LC_UUID.0);
        zero_uuid[uuid + 8..uuid + 24].fill(0);
        assert!(audit_scalar_macho(&zero_uuid, module.hidden_kernel_symbol()).is_err());

        let mut wrong_platform = build.dylib_bytes().to_vec();
        let version = load_command_offset(&wrong_platform, object::macho::LC_BUILD_VERSION.0);
        wrong_platform[version + 8..version + 12]
            .copy_from_slice(&object::macho::PLATFORM_IOS.0.to_le_bytes());
        assert!(audit_scalar_macho(&wrong_platform, module.hidden_kernel_symbol()).is_err());

        let mut wrong_id = build.dylib_bytes().to_vec();
        let id = b"@rpath/decodeforge_scalar_v1.dylib";
        let id_offset = wrong_id
            .windows(id.len())
            .position(|window| window == id)
            .expect("fixed dylib ID must be present");
        wrong_id[id_offset] = b'/';
        assert!(audit_scalar_macho(&wrong_id, module.hidden_kernel_symbol()).is_err());
    }
}
