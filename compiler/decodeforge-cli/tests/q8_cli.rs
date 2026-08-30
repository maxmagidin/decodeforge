use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

fn unique_temp_dir(label: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let base = std::env::temp_dir()
        .canonicalize()
        .expect("system temp directory must be accessible");
    for _ in 0..100 {
        let number = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = base.join(format!(
            "decodeforge-cli-{label}-{}-{number}",
            std::process::id()
        ));
        if fs::create_dir(&path).is_ok() {
            return path;
        }
    }
    panic!("unable to allocate a unique CLI test directory");
}

#[test]
fn q8_regenerate_is_a_usage_error_and_cannot_create_the_target() {
    let parent = unique_temp_dir("regenerate-rejected");
    let target = parent.join("must-not-exist");
    let output = Command::new(env!("CARGO_BIN_EXE_decodeforge"))
        .args(["q8", "regenerate", "--output"])
        .arg(&target)
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    assert!(!target.exists());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("q8 verify")
    );
    fs::remove_dir_all(parent).unwrap();
}
