//! Implementation modules for the fixed G1 benchmark protocol.

mod cases;
mod cli;
mod error;
mod files;
mod safetensors;
mod session;
mod spec;

pub use error::BenchError;

use cases::{prepare_cases, read_case_bundle};
use cli::Command;
use session::run_session;

/// Parse the closed command line and execute exactly one benchmark phase.
pub fn run() -> Result<(), BenchError> {
    match cli::parse()? {
        Command::Help => {
            println!("{}", cli::USAGE);
            Ok(())
        }
        Command::PrepareCases { weights, output } => {
            let manifest = prepare_cases(&weights, &output)?;
            println!(
                "prepared {} cases at {}",
                manifest.cases.len(),
                output.display()
            );
            Ok(())
        }
        Command::RunSession {
            cases,
            output,
            session_id,
        } => {
            let bundle = read_case_bundle(&cases)?;
            let result = run_session(&bundle, &session_id)?;
            files::write_json_atomic(&output, &result)?;
            println!("wrote benchmark session to {}", output.display());
            Ok(())
        }
    }
}
