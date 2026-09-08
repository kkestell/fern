//! Fern's compiler driver. See the repository README for native tool prerequisites.
mod backend;
mod diagnostic;
mod frontend;
mod ir;
mod semantic;
mod source;
mod types;

use std::{fmt, fs, path::Path};

/// A source diagnostic or a file, compiler, or native tool failure.
#[derive(Debug)]
pub struct CompileError {
    message: String,
}

impl CompileError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for CompileError {}

/// Compile a Fern source file to a native executable, replacing `output` on success.
///
/// # Errors
/// Returns an error for invalid source, an output aliasing the input, file failures,
/// or failed native tools. Compilation failures preserve an existing output.
pub fn compile(input: &Path, output: &Path) -> Result<(), CompileError> {
    let source = source::Source::load(input)?;
    reject_input_output_alias(input, output)?;
    let syntax = frontend::parse(&source.text).map_err(|e| CompileError::new(e.render(&source)))?;
    let checked = semantic::check(&syntax).map_err(|e| CompileError::new(e.render(&source)))?;
    let entry = ir::lower(checked).verify()?;
    backend::build(&entry, &source, output)
}

fn reject_input_output_alias(input: &Path, output: &Path) -> Result<(), CompileError> {
    let input_path = fs::canonicalize(input)
        .map_err(|e| CompileError::new(format!("cannot resolve {}: {e}", input.display())))?;
    if let Ok(output_path) = fs::canonicalize(output) {
        let mut same = input_path == output_path;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if let (Ok(a), Ok(b)) = (fs::metadata(input), fs::metadata(output)) {
                same |= a.dev() == b.dev() && a.ino() == b.ino();
            }
        }
        if same {
            return Err(CompileError::new("output would overwrite the input source"));
        }
    }
    Ok(())
}
