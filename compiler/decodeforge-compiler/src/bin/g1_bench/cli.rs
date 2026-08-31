use super::BenchError;
use std::env;
use std::path::PathBuf;

pub const USAGE: &str = r#"DecodeForge G1 benchmark runner

USAGE:
  decodeforge-g1-bench prepare-cases --weights PATH --output DIR
  decodeforge-g1-bench run-session --cases MANIFEST --output JSON --session-id ID
  decodeforge-g1-bench --help

prepare-cases validates one TinyLlama q_proj BF16 safetensors tensor and writes
immutable local Q8, activation, oracle, and pack-manifest assets.
run-session builds and audits scalar and ARM64 NEON modules on a macOS arm64
host, then writes exactly 40 balanced paired raw timing rounds.
"#;

#[derive(Debug)]
pub enum Command {
    Help,
    PrepareCases {
        weights: PathBuf,
        output: PathBuf,
    },
    RunSession {
        cases: PathBuf,
        output: PathBuf,
        session_id: String,
    },
}

pub fn parse() -> Result<Command, BenchError> {
    let mut args = env::args_os();
    let _program = args.next();
    let Some(command) = args.next() else {
        return Err(BenchError::new("DFE-G1-CLI", USAGE));
    };
    let command = command
        .to_str()
        .ok_or_else(|| BenchError::new("DFE-G1-CLI", "command must be valid UTF-8"))?;
    if matches!(command, "--help" | "-h" | "help") {
        if args.next().is_some() {
            return Err(BenchError::new(
                "DFE-G1-CLI",
                "--help does not accept arguments",
            ));
        }
        return Ok(Command::Help);
    }

    let mut options = Vec::new();
    while let Some(option) = args.next() {
        let option = option
            .to_str()
            .ok_or_else(|| BenchError::new("DFE-G1-CLI", "option must be valid UTF-8"))?;
        if !option.starts_with("--") {
            return Err(BenchError::new(
                "DFE-G1-CLI",
                format!("unexpected positional argument {option:?}"),
            ));
        }
        let value = args.next().ok_or_else(|| {
            BenchError::new("DFE-G1-CLI", format!("option {option} requires a value"))
        })?;
        options.push((option.to_owned(), value));
    }

    match command {
        "prepare-cases" => {
            let weights = required_path(&options, "--weights")?;
            let output = required_path(&options, "--output")?;
            reject_unexpected(&options, &["--weights", "--output"])?;
            Ok(Command::PrepareCases { weights, output })
        }
        "run-session" => {
            let cases = required_path(&options, "--cases")?;
            let output = required_path(&options, "--output")?;
            let session_id = required_string(&options, "--session-id")?;
            validate_session_id(&session_id)?;
            reject_unexpected(&options, &["--cases", "--output", "--session-id"])?;
            Ok(Command::RunSession {
                cases,
                output,
                session_id,
            })
        }
        _ => Err(BenchError::new(
            "DFE-G1-CLI",
            format!("unknown command {command:?}\n\n{USAGE}"),
        )),
    }
}

fn required_path(
    options: &[(String, std::ffi::OsString)],
    name: &str,
) -> Result<PathBuf, BenchError> {
    let value = exactly_one(options, name)?;
    let value = value
        .to_str()
        .ok_or_else(|| BenchError::new("DFE-G1-CLI", format!("{name} must be valid UTF-8")))?;
    if value.is_empty() || value == "." || value == ".." {
        return Err(BenchError::new(
            "DFE-G1-CLI",
            format!("{name} must be an explicit file or directory path"),
        ));
    }
    Ok(PathBuf::from(value))
}

fn required_string(
    options: &[(String, std::ffi::OsString)],
    name: &str,
) -> Result<String, BenchError> {
    let value = exactly_one(options, name)?
        .to_str()
        .ok_or_else(|| BenchError::new("DFE-G1-CLI", format!("{name} must be valid UTF-8")))?;
    Ok(value.to_owned())
}

fn exactly_one<'a>(
    options: &'a [(String, std::ffi::OsString)],
    name: &str,
) -> Result<&'a std::ffi::OsString, BenchError> {
    let mut values = options
        .iter()
        .filter(|(key, _)| key == name)
        .map(|(_, value)| value);
    let Some(value) = values.next() else {
        return Err(BenchError::new(
            "DFE-G1-CLI",
            format!("missing required option {name}"),
        ));
    };
    if values.next().is_some() {
        return Err(BenchError::new(
            "DFE-G1-CLI",
            format!("option {name} must appear exactly once"),
        ));
    }
    Ok(value)
}

fn reject_unexpected(
    options: &[(String, std::ffi::OsString)],
    allowed: &[&str],
) -> Result<(), BenchError> {
    for (key, _) in options {
        if !allowed.contains(&key.as_str()) {
            return Err(BenchError::new(
                "DFE-G1-CLI",
                format!("unknown option {key:?}"),
            ));
        }
    }
    Ok(())
}

fn validate_session_id(session_id: &str) -> Result<(), BenchError> {
    if session_id.is_empty() || session_id.len() > 64 || !session_id.is_ascii() {
        return Err(BenchError::new(
            "DFE-G1-CLI",
            "session ID must be 1..=64 ASCII bytes",
        ));
    }
    if !session_id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(BenchError::new(
            "DFE-G1-CLI",
            "session ID may contain only ASCII letters, digits, '.', '_' or '-'",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_ids_are_closed_and_path_options_are_not_optional() {
        assert!(validate_session_id("m4-session-1").is_ok());
        assert!(validate_session_id("../escape").is_err());
        assert!(validate_session_id("").is_err());
        assert!(required_string(&[], "--session-id").is_err());
    }
}
