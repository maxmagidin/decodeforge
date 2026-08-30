use std::env;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process;

use decodeforge_core::q8::fixture::verify_fixture_root;
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
    format!("Usage: {program} --version | {program} q8 verify [--root PATH | --manifest PATH]")
}

fn q8_usage(program: &str) -> String {
    format!("Usage: {program} q8 verify [--root PATH | --manifest PATH]")
}

#[derive(Debug)]
enum CliError {
    Usage(String),
    Data(String),
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(message) | Self::Data(message) => formatter.write_str(message),
        }
    }
}

fn usage_error(message: impl Into<String>) -> CliError {
    CliError::Usage(message.into())
}

fn data_error(message: impl Into<String>) -> CliError {
    CliError::Data(message.into())
}

fn manifest_root(manifest: &Path) -> Result<PathBuf, CliError> {
    if manifest.file_name().and_then(|name| name.to_str()) != Some("manifest.json") {
        return Err(usage_error("q8 verify --manifest must name manifest.json"));
    }
    let parent = manifest.parent().unwrap_or_else(|| Path::new("."));
    if parent.as_os_str().is_empty() {
        Ok(PathBuf::from("."))
    } else {
        Ok(parent.to_owned())
    }
}

fn q8_command(program: &str, args: &[String]) -> Result<String, CliError> {
    let Some(command) = args.first().map(String::as_str) else {
        return Err(usage_error(q8_usage(program)));
    };
    match command {
        "verify" => {
            let mut root = None;
            let mut manifest = None;
            let mut index = 1;
            while index < args.len() {
                let option = args[index].as_str();
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| usage_error(q8_usage(program)))?;
                match option {
                    "--root" if root.is_none() => root = Some(PathBuf::from(value)),
                    "--manifest" if manifest.is_none() => manifest = Some(PathBuf::from(value)),
                    _ => return Err(usage_error(q8_usage(program))),
                }
                index += 2;
            }
            if root.is_some() && manifest.is_some() {
                return Err(usage_error(q8_usage(program)));
            }
            let root = if let Some(manifest) = manifest {
                manifest_root(&manifest)?
            } else {
                root.unwrap_or_else(|| PathBuf::from("tests/fixtures/v1"))
            };
            let count =
                verify_fixture_root(&root).map_err(|error| data_error(error.to_string()))?;
            Ok(format!(
                "fixture-check: ok ({count} deterministic fixtures)"
            ))
        }
        _ => Err(usage_error(q8_usage(program))),
    }
}

fn main() {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| PACKAGE_NAME.to_owned());

    let remaining = args.collect::<Vec<_>>();
    match remaining.as_slice() {
        [command] if command == "--version" => println!("{}", version_line()),
        [command] if command == "--help" => println!("{}", usage(&program)),
        [] => println!("{}", usage(&program)),
        [command, rest @ ..] if command == "q8" => match q8_command(&program, rest) {
            Ok(message) => println!("{message}"),
            Err(CliError::Usage(message)) => {
                eprintln!("{message}");
                process::exit(2);
            }
            Err(CliError::Data(message)) => {
                eprintln!("{message}");
                process::exit(1);
            }
        },
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

    #[test]
    fn q8_usage_accepts_the_fixture_gate_commands() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/v1");
        let output = q8_command(
            "decodeforge",
            &[
                "verify".to_owned(),
                "--root".to_owned(),
                root.display().to_string(),
            ],
        )
        .unwrap();
        assert!(output.contains("16 deterministic fixtures"));
    }

    #[test]
    fn q8_verify_rejects_ambiguous_or_misnamed_manifest_options() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/v1");
        let root_arg = root.display().to_string();
        let manifest_arg = root.join("manifest.json").display().to_string();
        let both = q8_command(
            "decodeforge",
            &[
                "verify".to_owned(),
                "--root".to_owned(),
                root_arg,
                "--manifest".to_owned(),
                manifest_arg,
            ],
        )
        .unwrap_err();
        assert!(both.to_string().contains("Usage: decodeforge q8 verify"));

        let misnamed = q8_command(
            "decodeforge",
            &[
                "verify".to_owned(),
                "--manifest".to_owned(),
                root.join("fixtures.json").display().to_string(),
            ],
        )
        .unwrap_err();
        assert!(misnamed.to_string().contains("must name manifest.json"));
    }

    #[test]
    fn bare_manifest_normalizes_to_the_current_directory() {
        assert_eq!(
            manifest_root(Path::new("manifest.json")).unwrap(),
            Path::new(".")
        );
    }
}
