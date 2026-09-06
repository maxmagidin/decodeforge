#![deny(unsafe_code)]

//! Prepare all pinned TinyLlama q_proj assets for the DecodeForge native bridge.

use decodeforge_compiler::{prepare_tinyllama_q_proj_inventory_v1, verify_q_proj_inventory_v1};
use std::ffi::OsString;
use std::fmt;
use std::path::PathBuf;

const USAGE: &str = r#"DecodeForge TinyLlama q_proj asset preparer

USAGE:
  decodeforge-prepare-qproj --source PATH --output DIR
  decodeforge-prepare-qproj --verify DIR
  decodeforge-prepare-qproj --help

The source must be the exact pinned TinyLlama model.safetensors revision.
The output directory must not already exist.
"#;

#[derive(Debug, Eq, PartialEq)]
enum Command {
    Prepare { source: PathBuf, output: PathBuf },
    Verify { assets: PathBuf },
}

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

fn parse_options(
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<Option<Command>, CliError> {
    let mut arguments = arguments.into_iter();
    let mut source = None;
    let mut output = None;
    let mut verify = None;
    while let Some(option) = arguments.next() {
        let option = option
            .to_str()
            .ok_or_else(|| CliError("options must be valid UTF-8".to_owned()))?;
        if matches!(option, "--help" | "-h") {
            if arguments.next().is_some()
                || source.is_some()
                || output.is_some()
                || verify.is_some()
            {
                return Err(CliError(
                    "--help does not accept other arguments".to_owned(),
                ));
            }
            return Ok(None);
        }
        let value = arguments
            .next()
            .ok_or_else(|| CliError(format!("option {option:?} requires a value\n\n{USAGE}")))?;
        match option {
            "--source" if source.is_none() => source = Some(path_value("--source", value)?),
            "--output" if output.is_none() => output = Some(path_value("--output", value)?),
            "--verify" if verify.is_none() => verify = Some(path_value("--verify", value)?),
            "--source" | "--output" | "--verify" => {
                return Err(CliError(format!(
                    "option {option} must appear exactly once"
                )));
            }
            _ => return Err(CliError(format!("unknown option {option:?}\n\n{USAGE}"))),
        }
    }
    match (source, output, verify) {
        (Some(source), Some(output), None) => Ok(Some(Command::Prepare { source, output })),
        (None, None, Some(assets)) => Ok(Some(Command::Verify { assets })),
        _ => Err(CliError(format!(
            "choose exactly one mode: --source PATH --output DIR, or --verify DIR\n\n{USAGE}"
        ))),
    }
}

fn path_value(option: &str, value: OsString) -> Result<PathBuf, CliError> {
    let value = value
        .to_str()
        .ok_or_else(|| CliError(format!("{option} must be valid UTF-8")))?;
    if value.is_empty() || matches!(value, "." | "..") {
        return Err(CliError(format!(
            "{option} must name an explicit file or directory"
        )));
    }
    Ok(PathBuf::from(value))
}

fn run() -> Result<(), CliError> {
    let Some(command) = parse_options(std::env::args_os().skip(1))? else {
        print!("{USAGE}");
        return Ok(());
    };
    match command {
        Command::Prepare { source, output } => {
            let prepared = prepare_tinyllama_q_proj_inventory_v1(&source, &output)
                .map_err(|error| CliError(error.to_string()))?;
            println!(
                "asset-preparation: ok layers={} aggregate_identity={} packed_bytes={} fallback_bytes={} output={}",
                prepared.inventory.layer_count,
                prepared.inventory.aggregate_identity,
                prepared.inventory.total_packed_bytes,
                prepared.inventory.total_fallback_bytes,
                prepared.output_directory.display(),
            );
        }
        Command::Verify { assets } => {
            let verified =
                verify_q_proj_inventory_v1(&assets).map_err(|error| CliError(error.to_string()))?;
            println!(
                "asset-verification: ok layers={} aggregate_identity={} packed_bytes={} fallback_bytes={} input={}",
                verified.inventory.layer_count,
                verified.inventory.aggregate_identity,
                verified.inventory.total_packed_bytes,
                verified.inventory.total_fallback_bytes,
                assets.display(),
            );
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(items: &[&str]) -> Vec<OsString> {
        items.iter().map(OsString::from).collect()
    }

    #[test]
    fn exact_options_are_required() {
        let command = parse_options(values(&[
            "--source",
            "model.safetensors",
            "--output",
            "qproj-0",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(
            command,
            Command::Prepare {
                source: PathBuf::from("model.safetensors"),
                output: PathBuf::from("qproj-0"),
            }
        );
        assert_eq!(
            parse_options(values(&["--verify", "qproj-assets"]))
                .unwrap()
                .unwrap(),
            Command::Verify {
                assets: PathBuf::from("qproj-assets")
            }
        );
        assert!(parse_options(values(&["--source", "model.safetensors"])).is_err());
        assert!(
            parse_options(values(&[
                "--source",
                "model.safetensors",
                "--output",
                "qproj-0",
                "--verify",
                "qproj-assets",
            ]))
            .is_err()
        );
    }

    #[test]
    fn duplicate_and_unknown_options_are_rejected() {
        assert!(
            parse_options(values(
                &["--source", "a", "--source", "b", "--output", "o",]
            ))
            .is_err()
        );
        assert!(parse_options(values(&["--wat", "x"])).is_err());
    }

    #[test]
    fn help_is_standalone() {
        assert_eq!(parse_options(values(&["--help"])).unwrap(), None);
        assert!(parse_options(values(&["--help", "extra"])).is_err());
    }
}
