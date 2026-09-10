//! Fern's compiler driver. See the repository README for native tool prerequisites.
mod backend;
mod diagnostic;
mod frontend;
mod ir;
mod module;
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

/// Compile a Fern root module to a native executable, replacing `output` on success.
///
/// `root` is either a directory whose `.fern` files form one module or a single
/// `.fern` file forming a one-file module. Its imports resolve against the
/// ordered module search roots, which default to the root module directory's
/// parent. `FERNPATH` replaces those roots only when it contains a non-empty
/// entry.
///
/// # Errors
/// Returns an error for invalid source, an unresolved import, a module
/// dependency cycle, an output aliasing a source file, file failures, or
/// failed native tools. Compilation failures preserve an existing output.
pub fn compile(root: &Path, output: &Path) -> Result<(), CompileError> {
    let roots = module::search_roots(root)?;
    let program = module::load(root, &roots).map_err(module::LoadError::into_compile_error)?;
    let (sources, syntax) = (program.sources, program.syntax);
    reject_input_output_alias(&sources, output)?;
    let checked = semantic::namespaces::check(&syntax, &program.modules, &program.files)
        .map_err(|e| CompileError::new(e.render(&sources)))?;
    let entry = ir::lower::lower(checked).verify()?;
    backend::toolchain::build(&entry, &sources, output)
}

fn reject_input_output_alias(
    sources: &source::SourceMap,
    output: &Path,
) -> Result<(), CompileError> {
    let Ok(output_path) = fs::canonicalize(output) else {
        return Ok(());
    };
    for source in sources.files() {
        let input = source.path.as_path();
        let input_path = fs::canonicalize(input)
            .map_err(|e| CompileError::new(format!("cannot resolve {}: {e}", input.display())))?;
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
