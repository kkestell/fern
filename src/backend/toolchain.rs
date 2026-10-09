//! Temporary-file and native-tool orchestration.

use crate::{CompileError, LinkInput, ir::verify::VerifiedProgram};
use std::{env, fs, path::Path, process::Command};

pub(crate) fn build(
    program: &VerifiedProgram,
    output: &Path,
    links: &[LinkInput],
) -> Result<(), CompileError> {
    build_text(&super::emitter::emit(program), output, links)
}

pub(super) fn build_text(
    text: &str,
    output: &Path,
    links: &[LinkInput],
) -> Result<(), CompileError> {
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let temporary = tempfile::Builder::new()
        .prefix(".fern-")
        .tempdir_in(parent)
        .map_err(|error| {
            CompileError::new(format!(
                "cannot create temporary files beside {}: {error}",
                output.display()
            ))
        })?;
    let qbe = temporary.path().join("program.ssa");
    let assembly = temporary.path().join("program.s");
    let executable = temporary.path().join("program");
    fs::write(&qbe, text)
        .map_err(|error| CompileError::new(format!("cannot write QBE input: {error}")))?;
    run(
        Command::new(env::var_os("QBE").unwrap_or_else(|| "qbe".into()))
            .arg("-o")
            .arg(&assembly)
            .arg(&qbe),
    )?;
    let mut cc = Command::new(env::var_os("CC").unwrap_or_else(|| "cc".into()));
    cc.arg(&assembly);
    for link in links {
        match link {
            LinkInput::File(path) => cc.arg(path),
            LinkInput::SearchPath(directory) => cc.arg("-L").arg(directory),
            LinkInput::Library(name) => cc.arg("-l").arg(name),
        };
    }
    run(cc.arg("-o").arg(&executable))?;
    fs::rename(&executable, output)
        .map_err(|error| CompileError::new(format!("cannot publish {}: {error}", output.display())))
}

fn run(command: &mut Command) -> Result<(), CompileError> {
    let tool = command.get_program().to_string_lossy().into_owned();
    let result = command
        .output()
        .map_err(|error| CompileError::new(format!("cannot run {tool}: {error}")))?;
    if result.status.success() {
        return Ok(());
    }
    Err(CompileError::new(format!(
        "{tool} failed ({}):\n{}",
        result.status,
        String::from_utf8_lossy(&result.stderr)
    )))
}
