//! Fixed Apple toolchain invocation and bounded subprocess support.
//!
//! All production commands in this module invoke the absolute xcrun path with
//! a closed argument list. The resolved tools are used only for provenance and
//! validation; they are never executed directly.

use super::{
    APPLE_NEON_CLANG_FLAGS, APPLE_SCALAR_CLANG_FLAGS, AppleNeonDylib, AppleScalarDylib,
    AppleToolchainProvenance, MAX_APPLE_GENERATED_DYLIB_BYTES,
};
use crate::native::audit::{audit_neon_helper_disassembly, audit_scalar_helper_disassembly};
use crate::native::macho::{audit_neon_macho, audit_scalar_macho};
use crate::{
    MAX_NEON_C_SOURCE_BYTES, MAX_SCALAR_C_SOURCE_BYTES, NeonCModule, Result, ScalarCModule,
    hex_lower, invalid,
};
use rustix::fs::{Mode, OFlags, fcntl_getfl, fcntl_setfl, open};
use rustix::process::geteuid;
use sha2::{Digest, Sha256};
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::Builder;

#[cfg(unix)]
use rustix::process::{Pid, Signal, kill_process_group, test_kill_process_group};
#[cfg(unix)]
use std::os::unix::{
    fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    process::CommandExt,
};

const XCRUN_PATH: &str = "/usr/bin/xcrun";
const FIXED_PATH: &str = "/usr/bin:/bin";
const FIXED_LOCALE: &str = "C";
const SOURCE_FILE: &str = "src/module.c";
const ABI_HEADER_FILE: &str = "include/decodeforge/abi_v1.h";
const SCALAR_DYLIB_FILE: &str = "out/decodeforge_scalar_v1.dylib";
const NEON_DYLIB_FILE: &str = "out/decodeforge_neon_v1.dylib";
const TEMP_SUBDIRECTORY: &str = "tmp";
const ABI_V1_HEADER: &str = include_str!("../../../../include/decodeforge/abi_v1.h");

const PROBE_DEADLINE: Duration = Duration::from_secs(5);
const COMPILE_DEADLINE: Duration = Duration::from_secs(30);
const AUDIT_DEADLINE: Duration = Duration::from_secs(10);
const POLL_INTERVAL: Duration = Duration::from_millis(10);
const TERMINATE_GRACE: Duration = Duration::from_millis(250);
const REAP_GRACE: Duration = Duration::from_secs(5);
const PIPE_PUMP_BYTES: usize = 64 * 1024;
const MAX_DIAGNOSTIC_BYTES: usize = 8 * 1024;
const MAX_TOOL_OUTPUT_BYTES: usize = 128 * 1024;
const MAX_DISASSEMBLY_BYTES: usize = 2 * 1024 * 1024;
const MAX_XCRUN_ARGUMENTS: usize = 40;
const MAX_XCRUN_ARGUMENT_BYTES: usize = 8 * 1024;
const MAX_XCRUN_ENVIRONMENT: usize = 7;
const MAX_XCRUN_ENVIRONMENT_BYTES: usize = 8 * 1024;

#[derive(Clone, Copy)]
struct RunLimits {
    deadline: Duration,
    stdout_limit: usize,
    stderr_limit: usize,
}

const PROBE_LIMITS: RunLimits = RunLimits {
    deadline: PROBE_DEADLINE,
    stdout_limit: MAX_TOOL_OUTPUT_BYTES,
    stderr_limit: MAX_TOOL_OUTPUT_BYTES,
};
const COMPILE_LIMITS: RunLimits = RunLimits {
    deadline: COMPILE_DEADLINE,
    stdout_limit: MAX_TOOL_OUTPUT_BYTES,
    stderr_limit: MAX_TOOL_OUTPUT_BYTES,
};
const AUDIT_LIMITS: RunLimits = RunLimits {
    deadline: AUDIT_DEADLINE,
    stdout_limit: MAX_DISASSEMBLY_BYTES,
    stderr_limit: MAX_TOOL_OUTPUT_BYTES,
};
struct DiscoveredToolchain {
    sdk_path: PathBuf,
    developer_dir: PathBuf,
    provenance: AppleToolchainProvenance,
}

#[derive(Clone, Copy)]
struct XcrunEnvironment<'a> {
    build_root: &'a Path,
    sdk_path: Option<&'a Path>,
    developer_dir: Option<&'a Path>,
    deployment_target: bool,
}

impl<'a> XcrunEnvironment<'a> {
    const fn probe(build_root: &'a Path) -> Self {
        Self {
            build_root,
            sdk_path: None,
            developer_dir: None,
            deployment_target: false,
        }
    }

    const fn with_toolchain(
        build_root: &'a Path,
        sdk_path: &'a Path,
        developer_dir: &'a Path,
    ) -> Self {
        Self {
            build_root,
            sdk_path: Some(sdk_path),
            developer_dir: Some(developer_dir),
            deployment_target: false,
        }
    }

    const fn compile(build_root: &'a Path, sdk_path: &'a Path, developer_dir: &'a Path) -> Self {
        Self {
            build_root,
            sdk_path: Some(sdk_path),
            developer_dir: Some(developer_dir),
            deployment_target: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    mode: u32,
    links: u64,
    owner: u32,
    length: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl FileIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            mode: metadata.mode(),
            links: metadata.nlink(),
            owner: metadata.uid(),
            length: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }
}

#[derive(Debug)]
struct SecureDylibSnapshot {
    file: File,
    identity: FileIdentity,
    bytes: Vec<u8>,
}

/// Build the fixed source in one private directory and audit its actual
/// Mach-O output. The public wrapper has already rejected unsupported hosts.
pub(crate) fn build_apple_scalar_dylib(module: &ScalarCModule) -> Result<AppleScalarDylib> {
    verify_scalar_module(module)?;
    let (common, audit_report) = build_apple_generated_dylib(
        AppleBuildSpec {
            backend_name: "scalar",
            build_prefix: "decodeforge-scalar-build-",
            artifact_prefix: "decodeforge-scalar-artifact-",
            dylib_file: SCALAR_DYLIB_FILE,
            flags: APPLE_SCALAR_CLANG_FLAGS,
            module_id: module.module_id(),
            hidden_symbol: module.hidden_kernel_symbol(),
            source: module.source(),
        },
        audit_scalar_macho,
        |disassembly| {
            audit_scalar_helper_disassembly(
                module.hidden_kernel_symbol(),
                disassembly,
                module.k() > 1,
            )
        },
    )?;

    Ok(AppleScalarDylib {
        _dylib_file: common.dylib_file,
        _temp_dir: common.temp_dir,
        module_id: common.module_id,
        source_hash: common.source_hash,
        abi_header_hash: common.abi_header_hash,
        dylib_hash: common.dylib_hash,
        dylib_bytes: common.dylib_bytes,
        toolchain: common.toolchain,
        flags: common.flags,
        disassembly: common.disassembly,
        audit_report,
        dynamic_exports: common.dynamic_exports,
    })
}

/// Build the fixed strict NEON source and audit its actual Mach-O output.
pub(crate) fn build_apple_neon_dylib(module: &NeonCModule) -> Result<AppleNeonDylib> {
    verify_neon_module(module)?;
    let (common, audit_report) = build_apple_generated_dylib(
        AppleBuildSpec {
            backend_name: "NEON",
            build_prefix: "decodeforge-neon-build-",
            artifact_prefix: "decodeforge-neon-artifact-",
            dylib_file: NEON_DYLIB_FILE,
            flags: APPLE_NEON_CLANG_FLAGS,
            module_id: module.module_id(),
            hidden_symbol: module.hidden_kernel_symbol(),
            source: module.source(),
        },
        audit_neon_macho,
        |disassembly| {
            audit_neon_helper_disassembly(
                module.hidden_kernel_symbol(),
                disassembly,
                module.n(),
                module.k(),
            )
        },
    )?;

    Ok(AppleNeonDylib {
        _dylib_file: common.dylib_file,
        _temp_dir: common.temp_dir,
        module_id: common.module_id,
        source_hash: common.source_hash,
        abi_header_hash: common.abi_header_hash,
        dylib_hash: common.dylib_hash,
        dylib_bytes: common.dylib_bytes,
        toolchain: common.toolchain,
        flags: common.flags,
        disassembly: common.disassembly,
        audit_report,
        dynamic_exports: common.dynamic_exports,
    })
}

struct AppleBuildSpec<'a> {
    backend_name: &'static str,
    build_prefix: &'static str,
    artifact_prefix: &'static str,
    dylib_file: &'static str,
    flags: &'static [&'static str],
    module_id: &'a str,
    hidden_symbol: &'a str,
    source: &'a str,
}

struct CommonAppleDylib {
    dylib_file: File,
    temp_dir: tempfile::TempDir,
    module_id: String,
    source_hash: String,
    abi_header_hash: String,
    dylib_hash: String,
    dylib_bytes: Vec<u8>,
    toolchain: AppleToolchainProvenance,
    flags: Vec<String>,
    disassembly: String,
    dynamic_exports: Vec<String>,
}

fn build_apple_generated_dylib<Report>(
    spec: AppleBuildSpec<'_>,
    audit_macho: fn(&[u8], &str) -> Result<Vec<String>>,
    audit_disassembly: impl FnOnce(&str) -> Result<Report>,
) -> Result<(CommonAppleDylib, Report)> {
    let build_directory = private_tempdir(spec.build_prefix)?;
    let build_root = build_directory.path();
    write_fixed_build_inputs(build_root, spec.source)?;

    let toolchain = discover_toolchain(build_root)?;
    let flags = spec
        .flags
        .iter()
        .map(|flag| (*flag).to_owned())
        .collect::<Vec<_>>();
    compile_dylib(
        build_root,
        &toolchain,
        &flags,
        spec.dylib_file,
        spec.backend_name,
    )?;

    let compiler_output = secure_dylib_snapshot(&build_root.join(spec.dylib_file))?;
    let retained_directory = private_tempdir(spec.artifact_prefix)?;
    prepare_artifact_directory(retained_directory.path())?;
    let dylib_path = retained_directory.path().join(spec.dylib_file);
    write_private_dylib(&dylib_path, &compiler_output.bytes)?;
    let mut retained = secure_dylib_snapshot(&dylib_path)?;
    if retained.bytes != compiler_output.bytes {
        return Err(invalid(
            "DFE-NATIVE-005",
            "private audit copy differs from the compiler output snapshot.",
        ));
    }
    let dynamic_exports = audit_macho(&retained.bytes, spec.hidden_symbol)?;
    let disassembly = disassemble_helper(
        retained_directory.path(),
        &toolchain,
        spec.hidden_symbol,
        spec.dylib_file,
        spec.backend_name,
    )?;
    let audit_report = audit_disassembly(&disassembly)?;
    revalidate_retained_snapshot(&mut retained, &dylib_path)?;

    Ok((
        CommonAppleDylib {
            dylib_file: retained.file,
            temp_dir: retained_directory,
            module_id: spec.module_id.to_owned(),
            source_hash: sha256_identity(spec.source.as_bytes()),
            abi_header_hash: sha256_identity(ABI_V1_HEADER.as_bytes()),
            dylib_hash: sha256_identity(&retained.bytes),
            dylib_bytes: retained.bytes,
            toolchain: toolchain.provenance,
            flags,
            disassembly,
            dynamic_exports,
        },
        audit_report,
    ))
}

fn verify_scalar_module(module: &ScalarCModule) -> Result<()> {
    let module_id = module.module_id();
    let Some(hash) = module_id.strip_prefix("sha256:") else {
        return Err(invalid(
            "DFE-NATIVE-002",
            "scalar C module identity is not a SHA-256 identity.",
        ));
    };
    if hash.len() != 64
        || !hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(
            "DFE-NATIVE-002",
            "scalar C module identity is not a lowercase SHA-256 identity.",
        ));
    }
    let expected_symbol = format!("df_kernel_scalar_v1_{hash}");
    if module.hidden_kernel_symbol() != expected_symbol {
        return Err(invalid(
            "DFE-NATIVE-002",
            "scalar C module hidden helper symbol does not match its identity.",
        ));
    }
    let source = module.source();
    if source.len() > MAX_SCALAR_C_SOURCE_BYTES
        || !source.is_ascii()
        || !source.ends_with('\n')
        || source.ends_with("\n\n")
    {
        return Err(invalid(
            "DFE-NATIVE-002",
            "scalar C module source violates the bounded source contract.",
        ));
    }
    if !source.contains(module.hidden_kernel_symbol()) {
        return Err(invalid(
            "DFE-NATIVE-002",
            "scalar C module source does not contain its hidden helper symbol.",
        ));
    }
    Ok(())
}

fn verify_neon_module(module: &NeonCModule) -> Result<()> {
    let module_id = module.module_id();
    let Some(hash) = module_id.strip_prefix("sha256:") else {
        return Err(invalid(
            "DFE-NATIVE-002",
            "NEON C module identity is not a SHA-256 identity.",
        ));
    };
    if hash.len() != 64
        || !hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(
            "DFE-NATIVE-002",
            "NEON C module identity is not a lowercase SHA-256 identity.",
        ));
    }
    let expected_symbol = format!("df_kernel_neon_v1_{hash}");
    if module.hidden_kernel_symbol() != expected_symbol {
        return Err(invalid(
            "DFE-NATIVE-002",
            "NEON C module hidden helper symbol does not match its identity.",
        ));
    }
    let source = module.source();
    if source.len() > MAX_NEON_C_SOURCE_BYTES
        || !source.is_ascii()
        || !source.ends_with('\n')
        || source.ends_with("\n\n")
    {
        return Err(invalid(
            "DFE-NATIVE-002",
            "NEON C module source violates the bounded source contract.",
        ));
    }
    if !source.contains(module.hidden_kernel_symbol()) {
        return Err(invalid(
            "DFE-NATIVE-002",
            "NEON C module source does not contain its hidden helper symbol.",
        ));
    }
    Ok(())
}

fn private_tempdir(prefix: &str) -> Result<tempfile::TempDir> {
    let mut builder = Builder::new();
    builder.prefix(prefix);
    #[cfg(unix)]
    builder.permissions(fs::Permissions::from_mode(0o700));
    let directory = builder.tempdir().map_err(|_| {
        invalid(
            "DFE-NATIVE-002",
            "unable to create private native directory.",
        )
    })?;
    require_private_directory(directory.path())?;
    Ok(directory)
}

fn require_private_directory(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path).map_err(|_| {
        invalid(
            "DFE-NATIVE-002",
            "private native directory metadata is unavailable.",
        )
    })?;
    if !metadata.is_dir() || metadata.permissions().mode() & 0o777 != 0o700 {
        return Err(invalid(
            "DFE-NATIVE-002",
            "private native directory does not have owner-only permissions.",
        ));
    }
    Ok(())
}

fn create_private_directory(path: &Path) -> Result<()> {
    fs::create_dir(path).map_err(|_| {
        invalid(
            "DFE-NATIVE-002",
            "unable to create a fixed private native subdirectory.",
        )
    })?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(|_| {
        invalid(
            "DFE-NATIVE-002",
            "unable to restrict a private native subdirectory.",
        )
    })?;
    require_private_directory(path)
}

fn write_create_new(path: &Path, bytes: &[u8], mode: u32, summary: &'static str) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true).mode(mode);
    let mut file = options
        .open(path)
        .map_err(|_| invalid("DFE-NATIVE-002", summary))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .and_then(|_| file.set_permissions(fs::Permissions::from_mode(mode)))
        .map_err(|_| invalid("DFE-NATIVE-002", summary))?;
    Ok(file)
}

fn write_fixed_build_inputs(build_root: &Path, source: &str) -> Result<()> {
    for relative in [
        "src",
        "include",
        "include/decodeforge",
        "out",
        TEMP_SUBDIRECTORY,
    ] {
        create_private_directory(&build_root.join(relative))?;
    }
    drop(write_create_new(
        &build_root.join(SOURCE_FILE),
        source.as_bytes(),
        0o600,
        "unable to create the generated C source privately.",
    )?);
    drop(write_create_new(
        &build_root.join(ABI_HEADER_FILE),
        ABI_V1_HEADER.as_bytes(),
        0o600,
        "unable to create the ABI header snapshot privately.",
    )?);
    Ok(())
}

fn prepare_artifact_directory(root: &Path) -> Result<()> {
    create_private_directory(&root.join("out"))?;
    create_private_directory(&root.join(TEMP_SUBDIRECTORY))
}

fn write_private_dylib(path: &Path, bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() || bytes.len() > MAX_APPLE_GENERATED_DYLIB_BYTES {
        return Err(invalid(
            "DFE-NATIVE-005",
            "compiler output violates the fixed dylib byte bound.",
        ));
    }
    let mut file = write_create_new(
        path,
        bytes,
        0o400,
        "unable to create the private dylib audit copy.",
    )?;
    file.seek(SeekFrom::Start(0)).map_err(|_| {
        invalid(
            "DFE-NATIVE-005",
            "unable to verify private dylib audit copy.",
        )
    })?;
    let mut readback = Vec::with_capacity(bytes.len());
    (&mut file)
        .take((MAX_APPLE_GENERATED_DYLIB_BYTES + 1) as u64)
        .read_to_end(&mut readback)
        .map_err(|_| {
            invalid(
                "DFE-NATIVE-005",
                "unable to verify private dylib audit copy.",
            )
        })?;
    if readback != bytes {
        return Err(invalid(
            "DFE-NATIVE-005",
            "private dylib audit copy failed exact readback.",
        ));
    }
    Ok(())
}

fn discover_toolchain(build_root: &Path) -> Result<DiscoveredToolchain> {
    let probe_environment = XcrunEnvironment::probe(build_root);
    let sdk_path = parse_absolute_path(
        "macOS SDK path",
        &successful_xcrun(
            "discovering the macOS SDK",
            &["--show-sdk-path"],
            probe_environment,
            PROBE_LIMITS,
        )?
        .stdout,
    )?;
    let sdk_path = fs::canonicalize(&sdk_path).map_err(|_| {
        invalid(
            "DFE-NATIVE-003",
            "selected macOS SDK path is not accessible.",
        )
    })?;
    if !sdk_path.is_dir() {
        return Err(invalid(
            "DFE-NATIVE-003",
            "selected macOS SDK path is not a directory.",
        ));
    }
    let developer_dir = developer_dir_from_sdk(&sdk_path)?;
    let pinned_environment =
        XcrunEnvironment::with_toolchain(build_root, &sdk_path, &developer_dir);
    let clang_path = find_tool("clang", pinned_environment)?;
    let objdump_path = find_tool("llvm-objdump", pinned_environment)?;
    validate_resolved_tool(&clang_path, "clang", &developer_dir)?;
    validate_resolved_tool(&objdump_path, "llvm-objdump", &developer_dir)?;

    let compiler_version = normalize_probe(
        "Apple Clang version",
        &successful_xcrun(
            "probing Apple Clang version",
            &["clang", "--version"],
            pinned_environment,
            PROBE_LIMITS,
        )?
        .stdout,
    )?;
    let target = normalize_probe(
        "Apple Clang target",
        &successful_xcrun(
            "probing Apple Clang target",
            &["clang", "-print-target-triple"],
            pinned_environment,
            PROBE_LIMITS,
        )?
        .stdout,
    )?;
    let sdk_version = normalize_probe(
        "macOS SDK version",
        &successful_xcrun(
            "probing macOS SDK version",
            &["--show-sdk-version"],
            pinned_environment,
            PROBE_LIMITS,
        )?
        .stdout,
    )?;
    let objdump_version = normalize_probe(
        "LLVM objdump version",
        &successful_xcrun(
            "probing LLVM objdump version",
            &["llvm-objdump", "--version"],
            pinned_environment,
            PROBE_LIMITS,
        )?
        .stdout,
    )?;

    Ok(DiscoveredToolchain {
        sdk_path,
        developer_dir,
        provenance: AppleToolchainProvenance {
            compiler: "clang".to_owned(),
            compiler_version,
            target,
            sdk_version,
            objdump_version,
        },
    })
}

fn find_tool(tool: &str, environment: XcrunEnvironment<'_>) -> Result<PathBuf> {
    let output = successful_xcrun(
        "discovering Apple developer tool",
        &["--find", tool],
        environment,
        PROBE_LIMITS,
    )?;
    parse_absolute_path("Apple developer tool", &output.stdout)
}

fn parse_absolute_path(label: &str, bytes: &[u8]) -> Result<PathBuf> {
    let text = std::str::from_utf8(bytes).map_err(|_| {
        invalid(
            "DFE-NATIVE-003",
            format!("{label} output is not valid UTF-8."),
        )
    })?;
    let text = text.trim();
    if text.is_empty() || text.contains(['\r', '\n', '\0']) {
        return Err(invalid(
            "DFE-NATIVE-003",
            format!("{label} output is not one newline-free absolute path."),
        ));
    }
    let path = PathBuf::from(text);
    if !path.is_absolute() {
        return Err(invalid(
            "DFE-NATIVE-003",
            format!("{label} is not an absolute path."),
        ));
    }
    Ok(path)
}

fn developer_dir_from_sdk(sdk_path: &Path) -> Result<PathBuf> {
    let mut prefix = PathBuf::new();
    let mut previous_was_contents = false;
    for component in sdk_path.components() {
        prefix.push(component.as_os_str());
        let name = component.as_os_str();
        if name == OsStr::new("CommandLineTools") {
            return Ok(prefix);
        }
        if previous_was_contents && name == OsStr::new("Developer") {
            return Ok(prefix);
        }
        previous_was_contents = name == OsStr::new("Contents");
    }
    Err(invalid(
        "DFE-NATIVE-003",
        "selected macOS SDK is not beneath an Apple developer directory.",
    ))
}

fn validate_resolved_tool(path: &Path, expected_name: &str, developer_dir: &Path) -> Result<()> {
    let canonical = fs::canonicalize(path).map_err(|_| {
        invalid(
            "DFE-NATIVE-003",
            "resolved Apple developer tool is not accessible.",
        )
    })?;
    let expected_file_name = canonical.file_name() == Some(OsStr::new(expected_name))
        || (expected_name == "nm" && canonical.file_name() == Some(OsStr::new("llvm-nm")));
    if !expected_file_name || !canonical.starts_with(developer_dir) {
        return Err(invalid(
            "DFE-NATIVE-003",
            "resolved Apple developer tool is outside the selected developer directory.",
        ));
    }
    let metadata = fs::metadata(&canonical).map_err(|_| {
        invalid(
            "DFE-NATIVE-003",
            "resolved Apple developer tool metadata is unavailable.",
        )
    })?;
    if !metadata.is_file() || !is_executable(&metadata) {
        return Err(invalid(
            "DFE-NATIVE-003",
            "resolved Apple developer tool is not a regular executable file.",
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn is_executable(metadata: &fs::Metadata) -> bool {
    metadata.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn is_executable(_metadata: &fs::Metadata) -> bool {
    true
}

fn compile_dylib(
    build_root: &Path,
    toolchain: &DiscoveredToolchain,
    flags: &[String],
    dylib_file: &str,
    backend_name: &str,
) -> Result<()> {
    let mut arguments = Vec::with_capacity(1 + flags.len() + 5);
    arguments.push("clang".to_owned());
    arguments.extend(flags.iter().cloned());
    arguments.extend(
        ["-I", "include", "-o", dylib_file, SOURCE_FILE]
            .into_iter()
            .map(str::to_owned),
    );
    let output = successful_xcrun(
        &format!("compiling generated {backend_name} C"),
        &arguments_as_strs(&arguments),
        XcrunEnvironment::compile(build_root, &toolchain.sdk_path, &toolchain.developer_dir),
        COMPILE_LIMITS,
    )?;
    if !output.stdout.is_empty() || !output.stderr.is_empty() {
        // Apple Clang generally stays quiet on a successful fixed build. The
        // output has still been bounded and retained only for this check; it
        // is intentionally not made part of the artifact identity.
    }
    Ok(())
}

fn disassemble_helper(
    build_root: &Path,
    toolchain: &DiscoveredToolchain,
    hidden_symbol: &str,
    dylib_file: &str,
    backend_name: &str,
) -> Result<String> {
    let mach_o_symbol = format!("_{hidden_symbol}");
    let arguments = [
        "llvm-objdump",
        "--macho",
        "--disassemble",
        "--no-show-raw-insn",
        "--dis-symname",
        mach_o_symbol.as_str(),
        dylib_file,
    ];
    let output = successful_xcrun(
        &format!("auditing generated {backend_name} helper disassembly"),
        &arguments,
        XcrunEnvironment::with_toolchain(build_root, &toolchain.sdk_path, &toolchain.developer_dir),
        AUDIT_LIMITS,
    )?;
    String::from_utf8(output.stdout).map_err(|_| {
        invalid(
            "DFE-NATIVE-006",
            "llvm-objdump disassembly is not valid UTF-8.",
        )
    })
}

fn secure_dylib_snapshot(path: &Path) -> Result<SecureDylibSnapshot> {
    let descriptor = open(
        path,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|_| invalid("DFE-NATIVE-005", "unable to securely open generated dylib."))?;
    let mut file = File::from(descriptor);
    let before = file
        .metadata()
        .map_err(|_| invalid("DFE-NATIVE-005", "generated dylib metadata is unavailable."))?;
    validate_dylib_metadata(&before)?;
    let identity = FileIdentity::from_metadata(&before);

    let mut bytes = Vec::with_capacity(
        usize::try_from(identity.length)
            .unwrap_or(MAX_APPLE_GENERATED_DYLIB_BYTES)
            .min(MAX_APPLE_GENERATED_DYLIB_BYTES),
    );
    (&mut file)
        .take((MAX_APPLE_GENERATED_DYLIB_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| invalid("DFE-NATIVE-005", "unable to read generated dylib snapshot."))?;
    let after = file.metadata().map_err(|_| {
        invalid(
            "DFE-NATIVE-005",
            "generated dylib metadata became unavailable.",
        )
    })?;
    if bytes.len() > MAX_APPLE_GENERATED_DYLIB_BYTES
        || bytes.len() as u64 != identity.length
        || FileIdentity::from_metadata(&after) != identity
    {
        return Err(invalid(
            "DFE-NATIVE-005",
            "generated dylib changed while its snapshot was read.",
        ));
    }
    require_path_identity(path, identity)?;
    file.seek(SeekFrom::Start(0)).map_err(|_| {
        invalid(
            "DFE-NATIVE-005",
            "unable to rewind generated dylib snapshot.",
        )
    })?;
    Ok(SecureDylibSnapshot {
        file,
        identity,
        bytes,
    })
}

fn validate_dylib_metadata(metadata: &fs::Metadata) -> Result<()> {
    if !metadata.file_type().is_file()
        || metadata.nlink() != 1
        || metadata.uid() != geteuid().as_raw()
        || metadata.len() == 0
        || metadata.len() > MAX_APPLE_GENERATED_DYLIB_BYTES as u64
    {
        return Err(invalid(
            "DFE-NATIVE-005",
            "generated dylib is not one bounded owner-controlled regular file.",
        ));
    }
    Ok(())
}

fn require_path_identity(path: &Path, expected: FileIdentity) -> Result<()> {
    let metadata = fs::symlink_metadata(path).map_err(|_| {
        invalid(
            "DFE-NATIVE-005",
            "private dylib path metadata is unavailable.",
        )
    })?;
    validate_dylib_metadata(&metadata)?;
    if FileIdentity::from_metadata(&metadata) != expected {
        return Err(invalid(
            "DFE-NATIVE-005",
            "private dylib path no longer names the retained snapshot.",
        ));
    }
    Ok(())
}

fn revalidate_retained_snapshot(snapshot: &mut SecureDylibSnapshot, path: &Path) -> Result<()> {
    require_path_identity(path, snapshot.identity)?;
    snapshot
        .file
        .seek(SeekFrom::Start(0))
        .map_err(|_| invalid("DFE-NATIVE-005", "unable to rewind retained dylib."))?;
    let mut bytes = Vec::with_capacity(snapshot.bytes.len());
    (&mut snapshot.file)
        .take((MAX_APPLE_GENERATED_DYLIB_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| invalid("DFE-NATIVE-005", "unable to re-read retained dylib."))?;
    let metadata = snapshot.file.metadata().map_err(|_| {
        invalid(
            "DFE-NATIVE-005",
            "retained dylib metadata became unavailable.",
        )
    })?;
    if bytes != snapshot.bytes || FileIdentity::from_metadata(&metadata) != snapshot.identity {
        return Err(invalid(
            "DFE-NATIVE-005",
            "retained dylib changed during structural audits.",
        ));
    }
    require_path_identity(path, snapshot.identity)?;
    snapshot
        .file
        .seek(SeekFrom::Start(0))
        .map_err(|_| invalid("DFE-NATIVE-005", "unable to rewind retained dylib."))?;
    Ok(())
}

fn successful_xcrun(
    phase: &str,
    arguments: &[&str],
    environment: XcrunEnvironment<'_>,
    limits: RunLimits,
) -> Result<BoundedOutput> {
    let command = fixed_xcrun_command(arguments, environment)?;
    let output = run_bounded(command, limits).map_err(|failure| {
        invalid(
            "DFE-NATIVE-004",
            format!(
                "{phase}: {}",
                safe_runner_diagnostic(&failure, environment.build_root)
            ),
        )
    })?;
    if output.status.success() {
        Ok(output)
    } else {
        let diagnostic = sanitized_process_diagnostic(&output, environment.build_root);
        Err(invalid(
            "DFE-NATIVE-004",
            format!("{phase} failed: {diagnostic}"),
        ))
    }
}

fn fixed_xcrun_command(arguments: &[&str], environment: XcrunEnvironment<'_>) -> Result<Command> {
    if arguments.is_empty()
        || arguments.len() > MAX_XCRUN_ARGUMENTS
        || arguments.iter().any(|argument| {
            argument.is_empty() || !argument.is_ascii() || argument.as_bytes().contains(&0)
        })
        || arguments
            .iter()
            .map(|argument| argument.len())
            .sum::<usize>()
            > MAX_XCRUN_ARGUMENT_BYTES
    {
        return Err(invalid(
            "DFE-NATIVE-002",
            "fixed xcrun argument contract was exceeded.",
        ));
    }
    let temporary_directory = environment.build_root.join(TEMP_SUBDIRECTORY);
    let temporary_directory = temporary_directory.to_str().ok_or_else(|| {
        invalid(
            "DFE-NATIVE-002",
            "private temporary directory is not UTF-8 representable.",
        )
    })?;
    let mut variables = vec![
        ("PATH", FIXED_PATH.to_owned()),
        ("LANG", FIXED_LOCALE.to_owned()),
        ("LC_ALL", FIXED_LOCALE.to_owned()),
        ("TMPDIR", temporary_directory.to_owned()),
    ];
    if let Some(sdk_path) = environment.sdk_path {
        let sdk_path = sdk_path.to_str().ok_or_else(|| {
            invalid(
                "DFE-NATIVE-002",
                "selected macOS SDK path is not UTF-8 representable.",
            )
        })?;
        variables.push(("SDKROOT", sdk_path.to_owned()));
    }
    if let Some(developer_dir) = environment.developer_dir {
        let developer_dir = developer_dir.to_str().ok_or_else(|| {
            invalid(
                "DFE-NATIVE-002",
                "selected Apple developer directory is not UTF-8 representable.",
            )
        })?;
        variables.push(("DEVELOPER_DIR", developer_dir.to_owned()));
    }
    if environment.deployment_target {
        variables.push(("MACOSX_DEPLOYMENT_TARGET", "15.0".to_owned()));
    }
    if variables.len() > MAX_XCRUN_ENVIRONMENT
        || variables
            .iter()
            .map(|(key, value)| key.len() + value.len())
            .sum::<usize>()
            > MAX_XCRUN_ENVIRONMENT_BYTES
    {
        return Err(invalid(
            "DFE-NATIVE-002",
            "fixed xcrun environment contract was exceeded.",
        ));
    }

    let mut command = Command::new(XCRUN_PATH);
    command
        .args(arguments)
        .current_dir(environment.build_root)
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in variables {
        command.env(key, value);
    }
    Ok(command)
}

fn arguments_as_strs(arguments: &[String]) -> Vec<&str> {
    arguments.iter().map(String::as_str).collect()
}

#[derive(Debug)]
struct BoundedOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RunnerFailureKind {
    Spawn,
    Timeout,
    OutputLimit,
    Pipe,
    Termination,
}

#[derive(Debug)]
struct RunnerFailure {
    kind: RunnerFailureKind,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

struct CapturedStream {
    bytes: Vec<u8>,
    overflow: bool,
    read_error: bool,
    eof: bool,
}

/// Run a configured command with bounded concurrent pipe drains.
///
/// `process_group(0)` isolates the tool and every descendant. Failure paths use
/// rustix's safe process-group signal API and nonblocking drains, so every
/// failure path can close over the complete process group without detached
/// reader threads.
fn run_bounded(
    mut command: Command,
    limits: RunLimits,
) -> std::result::Result<BoundedOutput, RunnerFailure> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    command.process_group(0);

    let mut child = command.spawn().map_err(|_| RunnerFailure {
        kind: RunnerFailureKind::Spawn,
        stdout: Vec::new(),
        stderr: Vec::new(),
    })?;
    #[cfg(unix)]
    let process_group = Pid::from_child(&child);
    let mut stdout_reader = child
        .stdout
        .take()
        .expect("piped child stdout must be available");
    let mut stderr_reader = child
        .stderr
        .take()
        .expect("piped child stderr must be available");
    if set_nonblocking(&stdout_reader).is_err() || set_nonblocking(&stderr_reader).is_err() {
        drop(stdout_reader);
        drop(stderr_reader);
        cancel_unconfigured_child(&mut child, process_group);
        return Err(RunnerFailure {
            kind: RunnerFailureKind::Pipe,
            stdout: Vec::new(),
            stderr: Vec::new(),
        });
    }
    let started = Instant::now();
    let mut stdout = CapturedStream::new(limits.stdout_limit);
    let mut stderr = CapturedStream::new(limits.stderr_limit);
    let mut status = None;
    loop {
        pump_stream(&mut stdout_reader, &mut stdout, limits.stdout_limit);
        pump_stream(&mut stderr_reader, &mut stderr, limits.stderr_limit);
        if stdout.overflow || stderr.overflow {
            return terminate_after_failure(
                &mut child,
                process_group,
                RunnerFailureKind::OutputLimit,
                &mut stdout_reader,
                &mut stderr_reader,
                stdout,
                stderr,
            );
        }
        if stdout.read_error || stderr.read_error {
            return terminate_after_failure(
                &mut child,
                process_group,
                RunnerFailureKind::Pipe,
                &mut stdout_reader,
                &mut stderr_reader,
                stdout,
                stderr,
            );
        }
        if status.is_none() {
            match child.try_wait() {
                Ok(Some(exit_status)) => status = Some(exit_status),
                Ok(None) => {}
                Err(_) => {
                    return terminate_after_failure(
                        &mut child,
                        process_group,
                        RunnerFailureKind::Pipe,
                        &mut stdout_reader,
                        &mut stderr_reader,
                        stdout,
                        stderr,
                    );
                }
            }
        }
        if status.is_some() && stdout.eof && stderr.eof {
            break;
        }
        if started.elapsed() >= limits.deadline {
            return terminate_after_failure(
                &mut child,
                process_group,
                RunnerFailureKind::Timeout,
                &mut stdout_reader,
                &mut stderr_reader,
                stdout,
                stderr,
            );
        }
        thread::sleep(POLL_INTERVAL);
    }

    if process_group_exists(process_group) {
        return terminate_after_failure(
            &mut child,
            process_group,
            RunnerFailureKind::Termination,
            &mut stdout_reader,
            &mut stderr_reader,
            stdout,
            stderr,
        );
    }
    Ok(BoundedOutput {
        status: status.expect("successful completion observed an exit status"),
        stdout: stdout.bytes,
        stderr: stderr.bytes,
    })
}

impl CapturedStream {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(limit.min(8 * 1024)),
            overflow: false,
            read_error: false,
            eof: false,
        }
    }
}

fn set_nonblocking<F: std::os::fd::AsFd>(stream: &F) -> std::io::Result<()> {
    let flags = fcntl_getfl(stream)?;
    Ok(fcntl_setfl(stream, flags | OFlags::NONBLOCK)?)
}

fn pump_stream<R: Read>(reader: &mut R, captured: &mut CapturedStream, limit: usize) {
    if captured.eof || captured.read_error {
        return;
    }
    let mut buffer = [0_u8; 8 * 1024];
    let mut pumped = 0_usize;
    while pumped < PIPE_PUMP_BYTES {
        match reader.read(&mut buffer) {
            Ok(0) => {
                captured.eof = true;
                break;
            }
            Ok(count) => {
                pumped += count;
                let available = limit.saturating_sub(captured.bytes.len());
                captured
                    .bytes
                    .extend_from_slice(&buffer[..count.min(available)]);
                if count > available {
                    captured.overflow = true;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => {
                captured.read_error = true;
                break;
            }
        }
    }
}

fn terminate_after_failure<Stdout: Read, Stderr: Read>(
    child: &mut std::process::Child,
    process_group: Pid,
    kind: RunnerFailureKind,
    stdout_reader: &mut Stdout,
    stderr_reader: &mut Stderr,
    mut stdout: CapturedStream,
    mut stderr: CapturedStream,
) -> std::result::Result<BoundedOutput, RunnerFailure> {
    let _ = kill_process_group(process_group, Signal::TERM);
    let terminate_deadline = Instant::now() + TERMINATE_GRACE;
    while Instant::now() < terminate_deadline {
        let stdout_limit = stdout.bytes.len();
        let stderr_limit = stderr.bytes.len();
        pump_stream(stdout_reader, &mut stdout, stdout_limit);
        pump_stream(stderr_reader, &mut stderr, stderr_limit);
        if !process_group_exists(process_group) {
            break;
        }
        thread::sleep(POLL_INTERVAL);
    }
    if process_group_exists(process_group) {
        let _ = kill_process_group(process_group, Signal::KILL);
    }
    let _ = child.kill();
    let deadline = Instant::now() + REAP_GRACE;
    let mut reaped = false;
    while Instant::now() < deadline {
        let stdout_limit = stdout.bytes.len();
        let stderr_limit = stderr.bytes.len();
        pump_stream(stdout_reader, &mut stdout, stdout_limit);
        pump_stream(stderr_reader, &mut stderr, stderr_limit);
        if !reaped {
            match child.try_wait() {
                Ok(Some(_)) => reaped = true,
                Ok(None) => {}
                Err(_) => break,
            }
        }
        if reaped
            && !process_group_exists(process_group)
            && (stdout.eof || stdout.read_error)
            && (stderr.eof || stderr.read_error)
        {
            break;
        }
        thread::sleep(POLL_INTERVAL);
    }
    let group_gone = !process_group_exists(process_group);
    let drains_closed = (stdout.eof || stdout.read_error) && (stderr.eof || stderr.read_error);
    let failure_kind = if reaped && group_gone && drains_closed {
        kind
    } else {
        RunnerFailureKind::Termination
    };
    Err(RunnerFailure {
        kind: failure_kind,
        stdout: stdout.bytes,
        stderr: stderr.bytes,
    })
}

fn cancel_unconfigured_child(child: &mut std::process::Child, process_group: Pid) {
    let _ = kill_process_group(process_group, Signal::KILL);
    let _ = child.kill();
    let deadline = Instant::now() + REAP_GRACE;
    let mut reaped = false;
    while Instant::now() < deadline {
        if !reaped {
            match child.try_wait() {
                Ok(Some(_)) => reaped = true,
                Ok(None) => {}
                Err(_) => break,
            }
        }
        if reaped && !process_group_exists(process_group) {
            break;
        }
        thread::sleep(POLL_INTERVAL);
    }
}

fn process_group_exists(process_group: Pid) -> bool {
    test_kill_process_group(process_group).is_ok()
}

fn safe_runner_diagnostic(failure: &RunnerFailure, build_root: &Path) -> String {
    let reason = match failure.kind {
        RunnerFailureKind::Spawn => "unable to start fixed xcrun command",
        RunnerFailureKind::Timeout => "fixed xcrun command exceeded its deadline",
        RunnerFailureKind::OutputLimit => "fixed xcrun command exceeded an output limit",
        RunnerFailureKind::Pipe => "fixed xcrun command pipe capture failed",
        RunnerFailureKind::Termination => {
            "fixed xcrun command did not terminate after cancellation"
        }
    };
    let output = diagnostic_text(&failure.stderr, &failure.stdout, build_root);
    if output.is_empty() {
        reason.to_owned()
    } else {
        format!("{reason}: {output}")
    }
}

fn sanitized_process_diagnostic(output: &BoundedOutput, build_root: &Path) -> String {
    let diagnostic = diagnostic_text(&output.stderr, &output.stdout, build_root);
    if diagnostic.is_empty() {
        "fixed xcrun command returned a nonzero status".to_owned()
    } else {
        diagnostic
    }
}

fn diagnostic_text(primary: &[u8], secondary: &[u8], build_root: &Path) -> String {
    let mut bytes = Vec::with_capacity(MAX_DIAGNOSTIC_BYTES);
    for stream in [primary, secondary] {
        let remaining = MAX_DIAGNOSTIC_BYTES.saturating_sub(bytes.len());
        if remaining == 0 {
            break;
        }
        bytes.extend_from_slice(&stream[..stream.len().min(remaining)]);
    }
    let text = String::from_utf8_lossy(&bytes);
    scrub_private_paths(text.as_ref(), build_root)
}

fn scrub_private_paths(text: &str, build_root: &Path) -> String {
    let build_root = build_root.to_string_lossy();
    let mut scrubbed = text.replace(build_root.as_ref(), "<private-build-dir>");
    if let Some(home) = std::env::var_os("HOME") {
        let home = home.to_string_lossy();
        scrubbed = scrubbed.replace(home.as_ref(), "<home>");
    }
    if let Some(user) = std::env::var_os("USER") {
        let user = user.to_string_lossy();
        scrubbed = scrubbed.replace(user.as_ref(), "<user>");
    }
    let mut result = String::with_capacity(scrubbed.len());
    let bytes = scrubbed.as_bytes();
    let mut index = 0_usize;
    while index < bytes.len() {
        if bytes[index] != b'/' {
            result.push(bytes[index] as char);
            index += 1;
            continue;
        }
        result.push_str("<path>");
        index += 1;
        while index < bytes.len()
            && !bytes[index].is_ascii_whitespace()
            && !matches!(
                bytes[index],
                b':' | b',' | b')' | b']' | b'}' | b'\"' | b'\''
            )
        {
            index += 1;
        }
    }
    result.trim().chars().take(MAX_DIAGNOSTIC_BYTES).collect()
}

fn normalize_probe(label: &str, bytes: &[u8]) -> Result<String> {
    let text = std::str::from_utf8(bytes).map_err(|_| {
        invalid(
            "DFE-NATIVE-003",
            format!("{label} output is not valid UTF-8."),
        )
    })?;
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() || normalized.len() > MAX_DIAGNOSTIC_BYTES {
        return Err(invalid(
            "DFE-NATIVE-003",
            format!("{label} output is empty or exceeds its fixed bound."),
        ));
    }
    Ok(normalized)
}

fn sha256_identity(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{}", hex_lower(&hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn test_command(program: &str, arguments: &[&str]) -> Command {
        let mut command = Command::new(program);
        command.args(arguments).stdin(Stdio::null());
        command
    }

    #[cfg(unix)]
    #[test]
    fn bounded_runner_handles_success_and_nonzero_statuses() {
        let output = run_bounded(
            test_command("/usr/bin/true", &[]),
            RunLimits {
                deadline: Duration::from_secs(1),
                stdout_limit: 1024,
                stderr_limit: 1024,
            },
        )
        .unwrap();
        assert!(output.status.success());

        let output = run_bounded(
            test_command("/usr/bin/false", &[]),
            RunLimits {
                deadline: Duration::from_secs(1),
                stdout_limit: 1024,
                stderr_limit: 1024,
            },
        )
        .unwrap();
        assert!(!output.status.success());
    }

    #[cfg(unix)]
    #[test]
    fn bounded_runner_terminates_timeout_and_floods() {
        let timeout = run_bounded(
            test_command("/bin/sh", &["-c", "sleep 5 & wait"]),
            RunLimits {
                deadline: Duration::from_millis(30),
                stdout_limit: 1024,
                stderr_limit: 1024,
            },
        )
        .expect_err("sleep must time out");
        assert_eq!(timeout.kind, RunnerFailureKind::Timeout);

        let inherited_pipe = run_bounded(
            test_command("/bin/sh", &["-c", "sleep 5 &"]),
            RunLimits {
                deadline: Duration::from_millis(30),
                stdout_limit: 1024,
                stderr_limit: 1024,
            },
        )
        .expect_err("a descendant retaining pipes must not outlive the deadline");
        assert_eq!(inherited_pipe.kind, RunnerFailureKind::Timeout);

        let closed_pipes = run_bounded(
            test_command("/bin/sh", &["-c", "sleep 5 >/dev/null 2>&1 &"]),
            RunLimits {
                deadline: Duration::from_secs(1),
                stdout_limit: 1024,
                stderr_limit: 1024,
            },
        )
        .expect_err("a background descendant must not outlive a successful leader");
        assert_eq!(closed_pipes.kind, RunnerFailureKind::Termination);

        let stdout_flood = run_bounded(
            test_command("/usr/bin/yes", &["x"]),
            RunLimits {
                deadline: Duration::from_secs(1),
                stdout_limit: 1024,
                stderr_limit: 1024,
            },
        )
        .expect_err("yes must exceed stdout cap");
        assert_eq!(stdout_flood.kind, RunnerFailureKind::OutputLimit);
        assert!(stdout_flood.stdout.len() <= 1024);

        let stderr_flood = run_bounded(
            test_command("/bin/sh", &["-c", "while :; do printf x >&2; done"]),
            RunLimits {
                deadline: Duration::from_secs(1),
                stdout_limit: 1024,
                stderr_limit: 1024,
            },
        )
        .expect_err("shell must exceed stderr cap");
        assert_eq!(stderr_flood.kind, RunnerFailureKind::OutputLimit);
        assert!(stderr_flood.stderr.len() <= 1024);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn fixed_link_policy_rejects_undefined_symbols() {
        const UNRESOLVED_SOURCE: &str = "\
#include <stdint.h>\n\
#define DF_TEST_EXPORT __attribute__((visibility(\"default\")))\n\
extern int decodeforge_intentionally_missing_symbol(void);\n\
DF_TEST_EXPORT uint32_t df_abi_version(void) { return UINT32_C(1); }\n\
DF_TEST_EXPORT const char *df_artifact_id(void) { return \"test\"; }\n\
DF_TEST_EXPORT int32_t df_run_v1(void) { return decodeforge_intentionally_missing_symbol(); }\n";

        let directory = private_tempdir("decodeforge-undefined-test-").unwrap();
        for relative in [
            "src",
            "include",
            "include/decodeforge",
            "out",
            TEMP_SUBDIRECTORY,
        ] {
            create_private_directory(&directory.path().join(relative)).unwrap();
        }
        drop(
            write_create_new(
                &directory.path().join(SOURCE_FILE),
                UNRESOLVED_SOURCE.as_bytes(),
                0o600,
                "test",
            )
            .unwrap(),
        );
        let toolchain = discover_toolchain(directory.path()).unwrap();
        let flags = APPLE_SCALAR_CLANG_FLAGS
            .iter()
            .map(|flag| (*flag).to_owned())
            .collect::<Vec<_>>();
        let error = compile_dylib(
            directory.path(),
            &toolchain,
            &flags,
            SCALAR_DYLIB_FILE,
            "scalar",
        )
        .expect_err("the fixed link mode must reject unresolved symbols by default");
        assert_eq!(error.code(), "DFE-NATIVE-004");
    }

    #[test]
    fn private_tempdir_owner_cleans_up() {
        let path = {
            let directory = tempfile::TempDir::new().unwrap();
            let path = directory.path().to_owned();
            assert!(path.is_dir());
            path
        };
        assert!(!path.exists());
    }

    #[test]
    fn absolute_toolchain_paths_may_contain_spaces() {
        let parsed = parse_absolute_path(
            "test path",
            b"/Applications/Xcode Preview.app/Contents/Developer\n",
        )
        .unwrap();
        assert_eq!(
            parsed,
            PathBuf::from("/Applications/Xcode Preview.app/Contents/Developer")
        );
    }

    #[cfg(unix)]
    #[test]
    fn secure_snapshot_rejects_symlinks_and_multiply_linked_files() {
        use std::os::unix::fs::symlink;

        let directory = private_tempdir("decodeforge-snapshot-test-").unwrap();
        let original = directory.path().join("original.dylib");
        drop(write_create_new(&original, b"bytes", 0o400, "test").unwrap());

        let symlink_path = directory.path().join("symlink.dylib");
        symlink(&original, &symlink_path).unwrap();
        let error = secure_dylib_snapshot(&symlink_path)
            .expect_err("O_NOFOLLOW must reject a dylib symlink");
        assert_eq!(error.code(), "DFE-NATIVE-005");

        let hardlink_path = directory.path().join("hardlink.dylib");
        fs::hard_link(&original, &hardlink_path).unwrap();
        let error =
            secure_dylib_snapshot(&original).expect_err("a multiply linked dylib must be rejected");
        assert_eq!(error.code(), "DFE-NATIVE-005");
    }

    #[test]
    fn private_artifact_copy_is_exact_and_owner_only() {
        let directory = private_tempdir("decodeforge-artifact-test-").unwrap();
        prepare_artifact_directory(directory.path()).unwrap();
        let path = directory.path().join(SCALAR_DYLIB_FILE);
        write_private_dylib(&path, b"fixed dylib bytes").unwrap();
        let mut snapshot = secure_dylib_snapshot(&path).unwrap();
        assert_eq!(snapshot.bytes, b"fixed dylib bytes");
        assert_eq!(
            fs::metadata(directory.path()).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o400
        );
        revalidate_retained_snapshot(&mut snapshot, &path).unwrap();
    }

    #[test]
    fn retained_snapshot_rejects_path_replacement() {
        let directory = private_tempdir("decodeforge-replacement-test-").unwrap();
        prepare_artifact_directory(directory.path()).unwrap();
        let path = directory.path().join(SCALAR_DYLIB_FILE);
        write_private_dylib(&path, b"original dylib bytes").unwrap();
        let mut snapshot = secure_dylib_snapshot(&path).unwrap();

        let displaced = directory.path().join("out/displaced.dylib");
        fs::rename(&path, &displaced).unwrap();
        drop(write_create_new(&path, b"original dylib bytes", 0o400, "test").unwrap());
        let error = revalidate_retained_snapshot(&mut snapshot, &path)
            .expect_err("same bytes at a replacement inode must be rejected");
        assert_eq!(error.code(), "DFE-NATIVE-005");
    }
}
