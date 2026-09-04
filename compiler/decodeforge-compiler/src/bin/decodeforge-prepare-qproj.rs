#![deny(unsafe_code)]

//! Prepare all pinned TinyLlama q_proj assets for the DecodeForge native bridge.

use decodeforge_compiler::prepare_tinyllama_q_proj_inventory_v1;
use std::ffi::OsString;
use std::fmt;
use std::path::PathBuf;

const USAGE: &str = r#"DecodeForge TinyLlama q_proj asset preparer

USAGE:
  decodeforge-prepare-qproj --source PATH --output DIR
  decodeforge-prepare-qproj --help

The source must be the exact pinned TinyLlama model.safetensors revision.
The output directory must not already exist.
"#;

#[derive(Debug, Eq, PartialEq)]
struct Options {
    source: PathBuf,
    output: PathBuf,
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
) -> Result<Option<Options>, CliError> {
    let mut arguments = arguments.into_iter();
    let mut source = None;
    let mut output = None;
    while let Some(option) = arguments.next() {
        let option = option
            .to_str()
            .ok_or_else(|| CliError("options must be valid UTF-8".to_owned()))?;
        if matches!(option, "--help" | "-h") {
            if arguments.next().is_some() || source.is_some() || output.is_some() {
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
            "--source" | "--output" => {
                return Err(CliError(format!(
                    "option {option} must appear exactly once"
                )));
            }
            _ => return Err(CliError(format!("unknown option {option:?}\n\n{USAGE}"))),
        }
    }
    Ok(Some(Options {
        source: source.ok_or_else(|| CliError(format!("missing --source\n\n{USAGE}")))?,
        output: output.ok_or_else(|| CliError(format!("missing --output\n\n{USAGE}")))?,
    }))
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
    let Some(options) = parse_options(std::env::args_os().skip(1))? else {
        print!("{USAGE}");
        return Ok(());
    };
    let prepared = prepare_tinyllama_q_proj_inventory_v1(&options.source, &options.output)
        .map_err(|error| CliError(error.to_string()))?;
    println!(
        "asset-preparation: ok layers={} aggregate_identity={} packed_bytes={} fallback_bytes={} output={}",
        prepared.inventory.layer_count,
        prepared.inventory.aggregate_identity,
        prepared.inventory.total_packed_bytes,
        prepared.inventory.total_fallback_bytes,
        prepared.output_directory.display(),
    );
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
        let options = parse_options(values(&[
            "--source",
            "model.safetensors",
            "--output",
            "qproj-0",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(options.source, PathBuf::from("model.safetensors"));
        assert_eq!(options.output, PathBuf::from("qproj-0"));
        assert!(parse_options(values(&["--source", "model.safetensors"])).is_err());
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
