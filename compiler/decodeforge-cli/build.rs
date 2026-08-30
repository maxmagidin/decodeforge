use std::env;
use std::path::Path;
use std::process::Command;

fn command_stdout(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let value = String::from_utf8(output.stdout).ok()?;
    Some(value.trim().to_owned())
}

fn is_full_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn source_revision() -> String {
    if let Ok(value) = env::var("DECODEFORGE_SOURCE_REVISION") {
        if value == "unknown"
            || is_full_revision(&value)
            || value.strip_prefix("dirty:").is_some_and(is_full_revision)
        {
            return value;
        }
        return "unknown".to_owned();
    }

    let Some(manifest_dir) = env::var_os("CARGO_MANIFEST_DIR") else {
        return "unknown".to_owned();
    };
    let root = Path::new(&manifest_dir).join("../..");
    let Some(revision) = command_stdout(&root, &["rev-parse", "--verify", "HEAD"]) else {
        return "unknown".to_owned();
    };
    if !is_full_revision(&revision) {
        return "unknown".to_owned();
    }

    let Some(status) = command_stdout(&root, &["status", "--porcelain", "--untracked-files=all"])
    else {
        return "unknown".to_owned();
    };
    if status.is_empty() {
        revision
    } else {
        format!("dirty:{revision}")
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=DECODEFORGE_SOURCE_REVISION");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
    println!(
        "cargo:rustc-env=DECODEFORGE_SOURCE_REVISION={}",
        source_revision()
    );
}
