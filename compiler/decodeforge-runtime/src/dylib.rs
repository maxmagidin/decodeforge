//! The sole unsafe boundary for trusted generated-module loading and calls.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod platform {
    use crate::abi::{ARTIFACT_ID_CSTR_BYTES_V1, DfCallV1};
    use crate::scalar::{AlignedPack, ScalarExecutableV1, ValidatedLoadSpec};
    use crate::{RUNTIME_ABI_VERSION, Result, RuntimeError};
    use libloading::os::unix::{Library, RTLD_LOCAL, RTLD_NOW};
    use std::ffi::{OsStr, c_char};
    use std::fs::{self, File, OpenOptions, Permissions};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::fd::AsRawFd;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
    use std::path::{Path, PathBuf};
    use tempfile::{Builder, TempDir};

    const PRIVATE_DYLIB_FILE: &str = "module.dylib";
    const COPY_BUFFER_BYTES: usize = 8 * 1024;

    type AbiVersionFn = unsafe extern "C" fn() -> u32;
    type ArtifactIdFn = unsafe extern "C" fn() -> *const c_char;
    type RunFn = unsafe extern "C" fn(*const DfCallV1, *const f32, *const u8, *mut f32) -> i32;

    struct LibraryBacking {
        // Fields drop in declaration order: dyld is closed before either
        // backing filesystem owner is released.
        library: Library,
        _file: File,
        _directory: TempDir,
    }

    pub(crate) struct LoadedScalarDylib {
        run_v1: RunFn,
        backing: LibraryBacking,
    }

    impl LoadedScalarDylib {
        unsafe fn open(
            mut private: PrivateDylibCopy,
            expected_module_id: &str,
            expected_bytes: &[u8],
        ) -> Result<Self> {
            ensure_safe_dynamic_loader_environment()?;
            verify_path_identity(
                &private.path,
                &private.file,
                private.device,
                private.inode,
                private.length,
            )?;
            verify_file_bytes(&mut private.file, expected_bytes)?;
            // SAFETY: The public unsafe constructor requires this to be the
            // exact locally generated and audited image. The private copy was
            // byte-checked above and the builder prerequisite excludes module
            // initializers, terminators, and unexpected dependencies.
            let descriptor_path = private.descriptor_path()?;
            let library = unsafe {
                Library::open(
                    Some(descriptor_path.as_path()),
                    RTLD_NOW | RTLD_LOCAL | libc::RTLD_FIRST,
                )
                .map_err(|_| RuntimeError::DynamicLoadFailed)?
            };
            verify_path_identity(
                &private.path,
                &private.file,
                private.device,
                private.inode,
                private.length,
            )?;
            verify_file_bytes(&mut private.file, expected_bytes)?;

            // SAFETY: The audited image exports these exact unmangled C
            // symbols, and the types are frozen by abi_v1.h.
            let abi_version = unsafe {
                load_symbol::<AbiVersionFn>(&library, b"df_abi_version\0", "df_abi_version")?
            };
            // SAFETY: Same symbol/type invariant as above.
            let artifact_id = unsafe {
                load_symbol::<ArtifactIdFn>(&library, b"df_artifact_id\0", "df_artifact_id")?
            };
            // SAFETY: Same symbol/type invariant as above.
            let run_v1 = unsafe { load_symbol::<RunFn>(&library, b"df_run_v1\0", "df_run_v1")? };

            // SAFETY: The copied function pointer has the audited C type and
            // the `library` owner remains alive for this call.
            let actual_abi = unsafe { abi_version() };
            if actual_abi != RUNTIME_ABI_VERSION {
                return Err(RuntimeError::AbiVersionMismatch {
                    expected: RUNTIME_ABI_VERSION,
                    actual: actual_abi,
                });
            }

            // SAFETY: The trusted emitter defines a static 72-byte object for
            // this function. We deliberately avoid an unbounded C-string
            // scan; arbitrary dylibs do not satisfy this precondition.
            let id_pointer = unsafe { artifact_id() };
            if id_pointer.is_null() {
                return Err(RuntimeError::ModuleIdMismatch);
            }
            // SAFETY: The exact locally generated module guarantees this
            // pointer references all 72 bytes for the lifetime of `library`.
            let id_bytes = unsafe {
                std::slice::from_raw_parts(id_pointer.cast::<u8>(), ARTIFACT_ID_CSTR_BYTES_V1)
            };
            if &id_bytes[..ARTIFACT_ID_CSTR_BYTES_V1 - 1] != expected_module_id.as_bytes()
                || id_bytes[ARTIFACT_ID_CSTR_BYTES_V1 - 1] != 0
            {
                return Err(RuntimeError::ModuleIdMismatch);
            }

            Ok(Self {
                run_v1,
                backing: LibraryBacking {
                    library,
                    _file: private.file,
                    _directory: private.directory,
                },
            })
        }

        pub(crate) fn invoke(
            &self,
            call: &DfCallV1,
            x: &[f32],
            pack: &AlignedPack,
            y: &mut [f32],
        ) -> i32 {
            debug_assert!(!x.is_empty());
            debug_assert!(!y.is_empty());
            debug_assert!(pack.len() > 0);
            debug_assert!(pack.as_ptr().addr().is_multiple_of(16));
            let _keep_library_alive = &self.backing.library;
            // SAFETY: `call` has the frozen layout; slices provide valid
            // extents; safe Rust borrows make output non-aliasing; `pack` is
            // immutable, exact-length, and 16-byte aligned; the library owner
            // outlives its copied function pointer.
            unsafe { (self.run_v1)(call, x.as_ptr(), pack.as_ptr(), y.as_mut_ptr()) }
        }
    }

    fn ensure_safe_dynamic_loader_environment() -> Result<()> {
        if std::env::vars_os().any(|(key, _)| is_dyld_environment_key(&key)) {
            return Err(RuntimeError::DynamicLoaderEnvironment);
        }
        Ok(())
    }

    fn is_dyld_environment_key(key: &OsStr) -> bool {
        key.as_bytes().starts_with(b"DYLD_")
    }

    unsafe fn load_symbol<T: Copy>(
        library: &Library,
        name: &[u8],
        display_name: &'static str,
    ) -> Result<T> {
        // SAFETY: The caller supplies the frozen function type for this exact
        // audited symbol name.
        let symbol =
            unsafe { library.get::<T>(name) }.map_err(|_| RuntimeError::MissingSymbol {
                symbol: display_name,
            })?;
        Ok(*symbol)
    }

    struct PrivateDylibCopy {
        // File precedes directory so it is closed before TempDir cleanup.
        file: File,
        directory: TempDir,
        path: PathBuf,
        device: u64,
        inode: u64,
        length: u64,
    }

    impl PrivateDylibCopy {
        fn new(bytes: &[u8]) -> Result<Self> {
            let directory = Builder::new()
                .prefix("decodeforge-scalar-load-")
                .tempdir()
                .map_err(|_| RuntimeError::PrivateCopyFailed)?;
            fs::set_permissions(directory.path(), Permissions::from_mode(0o700))
                .map_err(|_| RuntimeError::PrivateCopyFailed)?;
            let directory_metadata = fs::symlink_metadata(directory.path())
                .map_err(|_| RuntimeError::PrivateCopyFailed)?;
            if !directory_metadata.file_type().is_dir()
                || directory_metadata.mode() & 0o777 != 0o700
            {
                return Err(RuntimeError::PrivateCopyFailed);
            }

            let path = directory.path().join(PRIVATE_DYLIB_FILE);
            if !path.is_absolute() {
                return Err(RuntimeError::PrivateCopyFailed);
            }
            let mut writable = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)
                .map_err(|_| RuntimeError::PrivateCopyFailed)?;
            writable
                .set_permissions(Permissions::from_mode(0o600))
                .map_err(|_| RuntimeError::PrivateCopyFailed)?;
            writable
                .write_all(bytes)
                .map_err(|_| RuntimeError::PrivateCopyFailed)?;
            writable
                .flush()
                .map_err(|_| RuntimeError::PrivateCopyFailed)?;
            writable
                .set_permissions(Permissions::from_mode(0o400))
                .map_err(|_| RuntimeError::PrivateCopyFailed)?;
            drop(writable);

            // Retain no descriptor capable of modifying executable bytes.
            let mut file = OpenOptions::new()
                .read(true)
                .open(&path)
                .map_err(|_| RuntimeError::PrivateCopyFailed)?;
            verify_file_bytes(&mut file, bytes)?;

            let metadata = file
                .metadata()
                .map_err(|_| RuntimeError::PrivateCopyFailed)?;
            if !metadata.file_type().is_file()
                || metadata.nlink() != 1
                || metadata.len() != bytes.len() as u64
                || metadata.mode() & 0o777 != 0o400
            {
                return Err(RuntimeError::PrivateCopyFailed);
            }
            verify_path_identity(&path, &file, metadata.dev(), metadata.ino(), metadata.len())?;

            Ok(Self {
                file,
                directory,
                path,
                device: metadata.dev(),
                inode: metadata.ino(),
                length: metadata.len(),
            })
        }

        fn descriptor_path(&self) -> Result<PathBuf> {
            let descriptor = self.file.as_raw_fd();
            if descriptor < 0 {
                return Err(RuntimeError::PrivateCopyFailed);
            }
            Ok(PathBuf::from(format!("/dev/fd/{descriptor}")))
        }
    }

    fn verify_file_bytes(file: &mut File, expected: &[u8]) -> Result<()> {
        file.seek(SeekFrom::Start(0))
            .map_err(|_| RuntimeError::PrivateCopyFailed)?;
        let mut buffer = [0_u8; COPY_BUFFER_BYTES];
        for expected_chunk in expected.chunks(COPY_BUFFER_BYTES) {
            file.read_exact(&mut buffer[..expected_chunk.len()])
                .map_err(|_| RuntimeError::PrivateCopyFailed)?;
            if &buffer[..expected_chunk.len()] != expected_chunk {
                return Err(RuntimeError::PrivateCopyFailed);
            }
        }
        let mut trailing = [0_u8; 1];
        if file
            .read(&mut trailing)
            .map_err(|_| RuntimeError::PrivateCopyFailed)?
            != 0
        {
            return Err(RuntimeError::PrivateCopyFailed);
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|_| RuntimeError::PrivateCopyFailed)?;
        Ok(())
    }

    fn verify_path_identity(
        path: &Path,
        file: &File,
        device: u64,
        inode: u64,
        length: u64,
    ) -> Result<()> {
        let path_metadata =
            fs::symlink_metadata(path).map_err(|_| RuntimeError::PrivateCopyFailed)?;
        let file_metadata = file
            .metadata()
            .map_err(|_| RuntimeError::PrivateCopyFailed)?;
        if !path_metadata.file_type().is_file()
            || !file_metadata.file_type().is_file()
            || path_metadata.dev() != device
            || path_metadata.ino() != inode
            || file_metadata.dev() != device
            || file_metadata.ino() != inode
            || path_metadata.len() != length
            || file_metadata.len() != length
            || path_metadata.nlink() != 1
            || file_metadata.nlink() != 1
            || path_metadata.mode() & 0o777 != 0o400
            || file_metadata.mode() & 0o777 != 0o400
        {
            return Err(RuntimeError::PrivateCopyFailed);
        }
        Ok(())
    }

    /// Load bytes proven by the caller to be an exact locally generated and
    /// audited Apple scalar module.
    ///
    /// # Safety
    ///
    /// `dylib_bytes` must be the immutable retained snapshot produced by the
    /// hardened DecodeForge builder. That builder must have audited this exact
    /// image, excluded initialization/termination routines and unexpected
    /// dependencies, and proven the three exported symbol types. `n`, `k`, and
    /// `expected_module_id` must describe that code. `verified_pack_bytes` and
    /// `expected_packed_identity` must come from a successfully verified pack
    /// for the same shape. The process must not use C or unsafe code to conceal
    /// a launch-time `DYLD_*` override after dyld cached it; hostile same-UID
    /// in-place mutation of executable storage is also outside this boundary.
    pub unsafe fn load_trusted_apple_scalar_v1(
        dylib_bytes: &[u8],
        expected_module_id: &str,
        n: u32,
        k: u32,
        verified_pack_bytes: &[u8],
        expected_packed_identity: &str,
    ) -> Result<ScalarExecutableV1> {
        let spec = ValidatedLoadSpec::new(
            dylib_bytes,
            expected_module_id,
            n,
            k,
            verified_pack_bytes,
            expected_packed_identity,
        )?;
        let pack = AlignedPack::copy_from_verified(verified_pack_bytes)?;
        let private = PrivateDylibCopy::new(dylib_bytes)?;
        // SAFETY: This function's caller establishes the generated-image
        // provenance precondition; `private` is its exact verified copy.
        let module = unsafe { LoadedScalarDylib::open(private, &spec.module_id, dylib_bytes)? };
        Ok(ScalarExecutableV1::from_trusted_parts(module, pack, spec))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::os::unix::fs::symlink;
        use std::process::Command;

        #[test]
        fn private_copy_is_exact_regular_private_and_removed_on_drop() {
            let bytes = (0..20_000).map(|index| index as u8).collect::<Vec<_>>();
            let path = {
                let mut copy = PrivateDylibCopy::new(&bytes).unwrap();
                assert_eq!(copy.file.metadata().unwrap().len(), bytes.len() as u64);
                assert_eq!(copy.file.metadata().unwrap().mode() & 0o777, 0o400);
                assert!(copy.file.write_all(&[0_u8]).is_err());
                assert_eq!(
                    fs::symlink_metadata(copy.directory.path()).unwrap().mode() & 0o777,
                    0o700
                );
                let mut readback = Vec::new();
                let mut reader = File::open(&copy.path).unwrap();
                reader.read_to_end(&mut readback).unwrap();
                assert_eq!(readback, bytes);
                copy.path.clone()
            };
            assert!(!path.exists());
        }
        #[test]
        fn private_copy_identity_check_rejects_permission_change() {
            let copy = PrivateDylibCopy::new(&[0_u8; 64]).unwrap();
            fs::set_permissions(&copy.path, Permissions::from_mode(0o600)).unwrap();

            assert_eq!(
                verify_path_identity(&copy.path, &copy.file, copy.device, copy.inode, copy.length,),
                Err(RuntimeError::PrivateCopyFailed)
            );
        }

        #[test]
        fn private_copy_identity_check_rejects_path_substitution() {
            let copy = PrivateDylibCopy::new(&[0_u8; 64]).unwrap();
            let displaced = copy.directory.path().join("displaced.dylib");
            fs::rename(&copy.path, &displaced).unwrap();
            symlink(&displaced, &copy.path).unwrap();

            assert_eq!(
                verify_path_identity(&copy.path, &copy.file, copy.device, copy.inode, copy.length,),
                Err(RuntimeError::PrivateCopyFailed)
            );
        }
        #[test]
        fn dyld_key_detection_is_prefix_exact_and_value_independent() {
            assert!(is_dyld_environment_key(OsStr::new("DYLD_LIBRARY_PATH")));
            assert!(is_dyld_environment_key(OsStr::new("DYLD_")));
            assert!(!is_dyld_environment_key(OsStr::new("dyld_LIBRARY_PATH")));
            assert!(!is_dyld_environment_key(OsStr::new(
                "NOT_DYLD_LIBRARY_PATH"
            )));
        }

        #[test]
        fn launch_time_dyld_override_is_rejected() {
            let mut command = Command::new(std::env::current_exe().unwrap());
            for (key, _) in std::env::vars_os() {
                if is_dyld_environment_key(&key) {
                    command.env_remove(key);
                }
            }
            let status = command
                .arg("dyld_launch_environment_helper")
                .arg("--nocapture")
                .env("DECODEFORGE_DYLD_TEST_CHILD", "1")
                .env("DYLD_LIBRARY_PATH", "/decodeforge-intentionally-absent")
                .status()
                .unwrap();
            assert!(status.success());
        }

        #[test]
        fn dyld_launch_environment_helper() {
            if std::env::var_os("DECODEFORGE_DYLD_TEST_CHILD").is_none() {
                return;
            }
            assert_eq!(
                ensure_safe_dynamic_loader_environment(),
                Err(RuntimeError::DynamicLoaderEnvironment)
            );
        }

        #[test]
        fn descriptor_load_path_remains_bound_after_path_substitution() {
            let original = (0..4096).map(|index| index as u8).collect::<Vec<_>>();
            let replacement = vec![0xa5_u8; original.len()];
            let copy = PrivateDylibCopy::new(&original).unwrap();
            let displaced = copy.directory.path().join("displaced.dylib");
            fs::rename(&copy.path, &displaced).unwrap();
            fs::write(&copy.path, &replacement).unwrap();

            let mut descriptor_bytes = Vec::new();
            File::open(copy.descriptor_path().unwrap())
                .unwrap()
                .read_to_end(&mut descriptor_bytes)
                .unwrap();
            assert_eq!(descriptor_bytes, original);
            assert_eq!(fs::read(&copy.path).unwrap(), replacement);
        }
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
mod platform {
    use crate::abi::DfCallV1;
    use crate::scalar::{AlignedPack, ScalarExecutableV1};
    use crate::{Result, RuntimeError};

    pub(crate) struct LoadedScalarDylib;

    impl LoadedScalarDylib {
        pub(crate) fn invoke(
            &self,
            _call: &DfCallV1,
            _x: &[f32],
            _pack: &AlignedPack,
            _y: &mut [f32],
        ) -> i32 {
            unreachable!("unsupported hosts cannot construct a loaded scalar dylib")
        }
    }

    /// Unsupported-host spelling of the trusted loader.
    ///
    /// # Safety
    ///
    /// The same trusted-image requirements as the macOS arm64 implementation
    /// apply, but this target always rejects before observing any arguments.
    pub unsafe fn load_trusted_apple_scalar_v1(
        _dylib_bytes: &[u8],
        _expected_module_id: &str,
        _n: u32,
        _k: u32,
        _verified_pack_bytes: &[u8],
        _expected_packed_identity: &str,
    ) -> Result<ScalarExecutableV1> {
        Err(RuntimeError::UnsupportedHost)
    }
}

pub(crate) use platform::LoadedScalarDylib;
pub use platform::load_trusted_apple_scalar_v1;
