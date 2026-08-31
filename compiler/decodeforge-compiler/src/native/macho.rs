//! Structural validation of the exact Mach-O bytes retained for loading.

use crate::{Result, invalid};
use object::macho;
use object::read::macho::{MachHeader, MachOFile64};
use object::{
    Architecture, ExportTarget, FileKind, ImportLibraryFlags, NameOrOrdinal, Object, ObjectKind,
    ObjectSection, ObjectSymbol, SectionFlags, SectionKind, SymbolKind,
};

const REQUIRED_EXPORTS: [&[u8]; 3] = [b"_df_abi_version", b"_df_artifact_id", b"_df_run_v1"];
const REQUIRED_LIBRARY: &[u8] = b"/usr/lib/libSystem.B.dylib";
const SCALAR_DYLIB_ID: &[u8] = b"@rpath/decodeforge_scalar_v1.dylib";
const NEON_DYLIB_ID: &[u8] = b"@rpath/decodeforge_neon_v1.dylib";

/// Validate one thin Apple ARM64 dylib and return normalized public exports.
pub(crate) fn audit_scalar_macho(bytes: &[u8], hidden_helper: &str) -> Result<Vec<String>> {
    audit_macho(bytes, hidden_helper, SCALAR_DYLIB_ID, "scalar")
}

/// Validate one thin Apple ARM64 NEON dylib and return normalized public exports.
pub(crate) fn audit_neon_macho(bytes: &[u8], hidden_helper: &str) -> Result<Vec<String>> {
    audit_macho(bytes, hidden_helper, NEON_DYLIB_ID, "NEON")
}

fn audit_macho(
    bytes: &[u8],
    hidden_helper: &str,
    required_dylib_id: &[u8],
    helper_kind: &str,
) -> Result<Vec<String>> {
    if FileKind::parse(bytes).ok() != Some(FileKind::MachO64) {
        return Err(audit_error(
            "native artifact is not one thin 64-bit Mach-O image.",
        ));
    }
    let file = MachOFile64::<object::Endianness>::parse(bytes)
        .map_err(|_| audit_error("native artifact has malformed Mach-O structure."))?;
    let endian = file.endian();
    if !file.is_64()
        || !file.is_little_endian()
        || file.architecture() != Architecture::Aarch64
        || file.kind() != ObjectKind::Dynamic
        || file.macho_header().filetype(endian) != macho::MH_DYLIB
        || file.macho_header().cpusubtype(endian)
            != macho::CpuSubtype::from(macho::CPU_SUBTYPE_ARM64_ALL)
    {
        return Err(audit_error(
            "native artifact is not a little-endian ARM64 MH_DYLIB.",
        ));
    }
    let uuid = file
        .mach_uuid()
        .map_err(|_| audit_error("native artifact UUID metadata is malformed."))?
        .ok_or_else(|| audit_error("native artifact lacks the LC_UUID required by dyld."))?;
    if uuid.iter().all(|byte| *byte == 0) {
        return Err(audit_error("native artifact contains an all-zero UUID."));
    }

    let mut dylib_id_count = 0_usize;
    let mut build_version_count = 0_usize;
    let mut uuid_count = 0_usize;
    let mut commands = file
        .macho_load_commands()
        .map_err(|_| audit_error("native artifact load commands are malformed."))?;
    while let Some(command) = commands
        .next()
        .map_err(|_| audit_error("native artifact load command is malformed."))?
    {
        if matches!(
            command.cmd(),
            macho::LC_RPATH
                | macho::LC_ROUTINES
                | macho::LC_ROUTINES_64
                | macho::LC_DYLD_ENVIRONMENT
                | macho::LC_THREAD
                | macho::LC_UNIXTHREAD
        ) {
            return Err(audit_error(
                "native artifact contains a forbidden rpath or initializer command.",
            ));
        }
        if command.cmd() == macho::LC_ID_DYLIB {
            let dylib = command
                .data::<macho::DylibCommand<object::Endianness>>()
                .map_err(|_| audit_error("native artifact dylib identity is malformed."))?;
            let name = command
                .string(endian, dylib.dylib.name)
                .map_err(|_| audit_error("native artifact dylib identity is malformed."))?;
            if name != required_dylib_id {
                return Err(audit_error(
                    "native artifact does not use the fixed dylib identity.",
                ));
            }
            dylib_id_count += 1;
        }
        if command.cmd() == macho::LC_UUID {
            command
                .uuid()
                .map_err(|_| audit_error("native artifact UUID metadata is malformed."))?
                .ok_or_else(|| audit_error("native artifact UUID metadata is malformed."))?;
            uuid_count += 1;
        }
        if let Some((build_version, _tools)) = command
            .build_version(endian)
            .map_err(|_| audit_error("native artifact build version is malformed."))?
        {
            if build_version.platform.get(endian) != macho::PLATFORM_MACOS
                || build_version.minos.get(endian) != macho::Version::new(15, 0, 0)
            {
                return Err(audit_error(
                    "native artifact does not match the fixed macOS 15.0 deployment contract.",
                ));
            }
            build_version_count += 1;
        }
    }
    if dylib_id_count != 1 {
        return Err(audit_error(
            "native artifact does not contain exactly one fixed dylib identity.",
        ));
    }
    if build_version_count != 1 {
        return Err(audit_error(
            "native artifact does not contain exactly one macOS build version.",
        ));
    }
    if uuid_count != 1 {
        return Err(audit_error(
            "native artifact does not contain exactly one UUID command.",
        ));
    }

    for section in file.sections() {
        let name = section
            .name_bytes()
            .map_err(|_| audit_error("native artifact section name is malformed."))?;
        let forbidden_type = matches!(
            section.flags(),
            SectionFlags::MachO { flags, .. }
                if matches!(
                    flags.typ(),
                    macho::S_MOD_INIT_FUNC_POINTERS
                        | macho::S_MOD_TERM_FUNC_POINTERS
                        | macho::S_INIT_FUNC_OFFSETS
                        | macho::S_THREAD_LOCAL_INIT_FUNCTION_POINTERS
                        | macho::S_INTERPOSING
                )
        );
        if forbidden_type || matches!(name, b"__mod_init_func" | b"__mod_term_func") {
            return Err(audit_error(
                "native artifact contains an initializer or terminator section.",
            ));
        }
    }

    let libraries = file
        .import_libraries()
        .map_err(|_| audit_error("native artifact dependencies are malformed."))?
        .map(|library| library.map_err(|_| audit_error("native artifact dependency is malformed.")))
        .collect::<Result<Vec<_>>>()?;
    let direct_libsystem = libraries.len() == 1
        && libraries[0].name() == REQUIRED_LIBRARY
        && matches!(
            libraries[0].flags(),
            ImportLibraryFlags::MachO {
                ordinal: 1,
                cmd: macho::LC_LOAD_DYLIB,
                use_flags,
                ..
            } if use_flags.is_none_or(|flags| flags.0 == 0)
        );
    if !direct_libsystem {
        return Err(audit_error(
            "native artifact dependencies are not exactly the fixed libSystem policy.",
        ));
    }

    let mut exports = file
        .exports()
        .map_err(|_| audit_error("native artifact exports are malformed."))?
        .map(|export| {
            let export = export.map_err(|_| audit_error("native artifact export is malformed."))?;
            if export.is_weak() {
                return Err(audit_error("native artifact contains a weak export."));
            }
            let address = match export.target() {
                ExportTarget::Address { address } => address,
                _ => {
                    return Err(audit_error(
                        "native artifact contains a non-address export target.",
                    ));
                }
            };
            let name = match export.name() {
                NameOrOrdinal::Name(name) => name,
                NameOrOrdinal::Ordinal(_) => {
                    return Err(audit_error(
                        "native artifact contains an ordinal-only export.",
                    ));
                }
            };
            let mut definitions = file.symbols().filter(|symbol| {
                symbol.name_bytes().ok() == Some(name)
                    && symbol.is_definition()
                    && symbol.kind() == SymbolKind::Text
                    && symbol.address() == address
                    && symbol
                        .section_index()
                        .and_then(|index| file.section_by_index(index).ok())
                        .is_some_and(|section| section.kind() == SectionKind::Text)
            });
            if definitions.next().is_none() || definitions.next().is_some() {
                return Err(audit_error(
                    "native artifact export is not backed by one text definition.",
                ));
            }
            Ok(name.to_vec())
        })
        .collect::<Result<Vec<_>>>()?;
    exports.sort();
    let mut expected_exports = REQUIRED_EXPORTS.map(<[u8]>::to_vec).to_vec();
    expected_exports.sort();
    if exports != expected_exports {
        return Err(audit_error(
            "native artifact does not expose exactly the three required ABI symbols.",
        ));
    }

    let expected_helper = format!("_{hidden_helper}");
    let mut helpers = file.symbols().filter(|symbol| {
        symbol.name_bytes().ok() == Some(expected_helper.as_bytes())
            && symbol.is_definition()
            && symbol.is_local()
            && symbol.kind() == SymbolKind::Text
            && symbol
                .section_index()
                .and_then(|index| file.section_by_index(index).ok())
                .is_some_and(|section| section.kind() == SectionKind::Text)
    });
    if helpers.next().is_none() || helpers.next().is_some() {
        return Err(audit_error(format!(
            "native artifact does not contain exactly one local {helper_kind} text helper."
        )));
    }

    Ok(exports
        .into_iter()
        .map(|name| {
            String::from_utf8(name)
                .expect("required exports are ASCII")
                .trim_start_matches('_')
                .to_owned()
        })
        .collect())
}

fn audit_error(summary: impl Into<String>) -> crate::CompilerError {
    invalid("DFE-NATIVE-006", summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_macho_and_truncated_macho() {
        for bytes in [&b"not a Mach-O image"[..], &b"\xcf\xfa\xed\xfe"[..]] {
            let error = audit_scalar_macho(bytes, "df_kernel_scalar_v1_test")
                .expect_err("malformed bytes must be rejected");
            assert_eq!(error.code(), "DFE-NATIVE-006");
        }
    }

    #[test]
    fn both_backend_wrappers_reject_non_macho_bytes() {
        let scalar = audit_scalar_macho(b"bad", "df_kernel_scalar_v1_test").unwrap_err();
        let neon = audit_neon_macho(b"bad", "df_kernel_neon_v1_test").unwrap_err();
        assert_eq!(scalar.code(), "DFE-NATIVE-006");
        assert_eq!(neon.code(), "DFE-NATIVE-006");
    }
}
