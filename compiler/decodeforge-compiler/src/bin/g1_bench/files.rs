use super::BenchError;
use super::spec::MAX_JSON_BYTES;
use rustix::fs::{Mode, OFlags, open};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    mode: u32,
    links: u64,
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
            length: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }
}

pub fn read_bounded(path: &Path, limit: usize, kind: &'static str) -> Result<Vec<u8>, BenchError> {
    let descriptor = open(
        path,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|error| {
        BenchError::new(
            "DFE-G1-IO",
            format!("unable to securely open {kind} {}: {error}", path.display()),
        )
    })?;
    let mut file = File::from(descriptor);
    let before = file.metadata().map_err(|error| {
        BenchError::new(
            "DFE-G1-IO",
            format!("unable to inspect {kind} {}: {error}", path.display()),
        )
    })?;
    if !before.file_type().is_file() {
        return Err(BenchError::new(
            "DFE-G1-IO",
            format!("{kind} must be a regular file"),
        ));
    }
    let identity = FileIdentity::from_metadata(&before);
    let length = usize::try_from(identity.length)
        .map_err(|_| BenchError::new("DFE-G1-LIMIT", format!("{kind} is too large")))?;
    if length > limit {
        return Err(BenchError::new(
            "DFE-G1-LIMIT",
            format!("{kind} exceeds the {limit}-byte bound"),
        ));
    }
    let mut bytes = Vec::with_capacity(length);
    (&mut file)
        .take((limit as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| {
            BenchError::new(
                "DFE-G1-IO",
                format!("unable to read {kind} {}: {error}", path.display()),
            )
        })?;
    let after = file.metadata().map_err(|error| {
        BenchError::new(
            "DFE-G1-IO",
            format!("unable to recheck {kind} {}: {error}", path.display()),
        )
    })?;
    if bytes.len() > limit
        || bytes.len() != length
        || FileIdentity::from_metadata(&after) != identity
    {
        return Err(BenchError::new(
            "DFE-G1-IO",
            format!("{kind} exceeded its bound or changed while it was read"),
        ));
    }
    Ok(bytes)
}

pub fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), BenchError> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if bytes.len() > MAX_JSON_BYTES {
        return Err(BenchError::new(
            "DFE-G1-LIMIT",
            "JSON output exceeds the fixed bound",
        ));
    }
    atomic_write(path, &bytes)
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), BenchError> {
    if path.as_os_str().is_empty() || path.file_name().is_none() {
        return Err(BenchError::new(
            "DFE-G1-IO",
            format!("output path {} is not an explicit file", path.display()),
        ));
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|error| {
            BenchError::new(
                "DFE-G1-IO",
                format!(
                    "unable to create output directory {}: {error}",
                    parent.display()
                ),
            )
        })?;
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| BenchError::new("DFE-G1-IO", "system clock predates the Unix epoch"))?
        .as_nanos();
    let file_name = path.file_name().unwrap();
    let temp_path = parent.join(format!(
        ".{}.tmp-{}-{}",
        file_name.to_string_lossy(),
        std::process::id(),
        nonce
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temp_path).map_err(|error| {
        BenchError::new(
            "DFE-G1-IO",
            format!(
                "unable to create temporary output {}: {error}",
                temp_path.display()
            ),
        )
    })?;
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temp_path, path)?;
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok::<(), std::io::Error>(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temp_path);
        return Err(BenchError::new(
            "DFE-G1-IO",
            format!("atomic output write failed for {}: {error}", path.display()),
        ));
    }
    Ok(())
}

pub fn sha256_identity(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut text = String::from("sha256:");
    for byte in digest {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut text = String::with_capacity(64);
    for byte in digest {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

pub fn encode_u32_le(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

pub fn decode_u32_le(
    bytes: &[u8],
    expected_words: usize,
    kind: &'static str,
) -> Result<Vec<u32>, BenchError> {
    let expected_bytes = expected_words
        .checked_mul(4)
        .ok_or_else(|| BenchError::new("DFE-G1-LIMIT", format!("{kind} word count overflows")))?;
    if bytes.len() != expected_bytes {
        return Err(BenchError::new(
            "DFE-G1-ASSET",
            format!(
                "{kind} has {} bytes; expected {expected_bytes}",
                bytes.len()
            ),
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(u32::from_le_bytes)
        .collect())
}

pub fn relative_asset_path(root: &Path, value: &str) -> Result<PathBuf, BenchError> {
    let path = Path::new(value);
    if path.is_absolute() || value.len() > 256 {
        return Err(BenchError::new(
            "DFE-G1-ASSET",
            format!("asset path {value:?} must be short and relative"),
        ));
    }
    for component in path.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err(BenchError::new(
                "DFE-G1-ASSET",
                format!("asset path {value:?} contains a forbidden component"),
            ));
        }
    }
    Ok(root.join(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn atomic_write_replaces_only_the_explicit_target() {
        let directory = tempdir().unwrap();
        let target = directory.path().join("result.json");
        atomic_write(&target, b"first").unwrap();
        atomic_write(&target, b"second").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"second");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn relative_assets_reject_absolute_and_parent_paths() {
        let root = Path::new("/tmp/cases");
        assert!(relative_asset_path(root, "case/input.bin").is_ok());
        assert!(relative_asset_path(root, "../input.bin").is_err());
        assert!(relative_asset_path(root, "/tmp/input.bin").is_err());
    }

    #[test]
    fn bounded_reads_reject_symlinks_and_oversized_files() {
        let directory = tempdir().unwrap();
        let regular = directory.path().join("regular.bin");
        std::fs::write(&regular, b"12345").unwrap();
        assert_eq!(read_bounded(&regular, 5, "test").unwrap(), b"12345");
        assert!(read_bounded(&regular, 4, "test").is_err());

        let link = directory.path().join("link.bin");
        std::os::unix::fs::symlink(&regular, &link).unwrap();
        assert!(read_bounded(&link, 5, "test").is_err());
        assert!(read_bounded(directory.path(), 5, "test").is_err());
    }
}
