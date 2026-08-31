#![deny(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

//! Narrow, reproducible G1 scalar-versus-NEON benchmark runner.
//!
//! The binary deliberately has two phases. prepare-cases validates one
//! pinned BF16 safetensors tensor and materializes all immutable Q8/input/oracle
//! assets. run-session consumes only that closed manifest, builds and audits
//! both generated Apple modules, validates their outputs, and records raw
//! paired timings. It does not download models, tune schedules, or hide a
//! framework/cache layer behind the benchmark boundary.

mod g1_bench;

fn main() {
    if let Err(error) = g1_bench::run() {
        eprintln!("{error}");
        std::process::exit(error.exit_code());
    }
}
