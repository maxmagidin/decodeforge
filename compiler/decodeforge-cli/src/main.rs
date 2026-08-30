use std::env;
use std::process;

use decodeforge_core::{
    COMPILER_VERSION, GENERATED_ABI_VERSION, PACKAGE_NAME, SCHEMA_MAJOR_VERSION,
};
use decodeforge_runtime::RUNTIME_ABI_VERSION;

fn version_line() -> String {
    format!(
        "{PACKAGE_NAME} {} package={PACKAGE_NAME} compiler={COMPILER_VERSION} \
         schema_major={SCHEMA_MAJOR_VERSION} generated_abi={GENERATED_ABI_VERSION} \
         runtime_abi={RUNTIME_ABI_VERSION} source_revision={}",
        env!("CARGO_PKG_VERSION"),
        env!("DECODEFORGE_SOURCE_REVISION"),
    )
}

fn usage(program: &str) -> String {
    format!("Usage: {program} --version")
}

fn main() {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| PACKAGE_NAME.to_owned());

    match (args.next().as_deref(), args.next()) {
        (Some("--version"), None) => println!("{}", version_line()),
        (Some("--help"), None) => println!("{}", usage(&program)),
        (None, None) => println!("{}", usage(&program)),
        _ => {
            eprintln!("{}", usage(&program));
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_line_contains_the_frozen_contract() {
        let line = version_line();
        assert!(line.starts_with("decodeforge 0.1.0"));
        assert!(line.contains("compiler=0.1.0"));
        assert!(line.contains("schema_major=1"));
        assert!(line.contains("generated_abi=1"));
        assert!(line.contains("runtime_abi=1"));
        assert!(line.contains("source_revision="));
    }
}
