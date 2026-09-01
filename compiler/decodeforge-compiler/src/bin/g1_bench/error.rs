use std::fmt;

/// Closed user-facing failure for preparation or benchmark execution.
#[derive(Debug)]
pub struct BenchError {
    code: &'static str,
    message: String,
    exit_code: i32,
}

impl BenchError {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            exit_code: 2,
        }
    }

    pub fn exit_code(&self) -> i32 {
        self.exit_code
    }
}

impl fmt::Display for BenchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for BenchError {}

impl From<std::io::Error> for BenchError {
    fn from(error: std::io::Error) -> Self {
        Self::new("DFE-G1-IO", error.to_string())
    }
}

impl From<serde_json::Error> for BenchError {
    fn from(error: serde_json::Error) -> Self {
        Self::new("DFE-G1-JSON", error.to_string())
    }
}

impl From<decodeforge_core::q8::Q8Error> for BenchError {
    fn from(error: decodeforge_core::q8::Q8Error) -> Self {
        Self::new("DFE-G1-Q8", error.to_string())
    }
}

impl From<decodeforge_compiler::CompilerError> for BenchError {
    fn from(error: decodeforge_compiler::CompilerError) -> Self {
        Self::new(error.code(), error.summary())
    }
}

impl From<decodeforge_runtime::RuntimeError> for BenchError {
    fn from(error: decodeforge_runtime::RuntimeError) -> Self {
        Self::new(error.code(), error.to_string())
    }
}
