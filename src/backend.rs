use crate::{CompileError, ir::VerifiedEntry};
use std::{env, fs, path::Path, process::Command};

pub(crate) fn emit(entry: &VerifiedEntry) -> String {
    format!(
        "export function w $main() {{\n@start\n    ret {}\n}}\n",
        entry.status()
    )
}

pub(crate) fn build(entry: &VerifiedEntry, output: &Path) -> Result<(), CompileError> {
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let temp = tempfile::Builder::new()
        .prefix(".fern-")
        .tempdir_in(parent)
        .map_err(|e| {
            CompileError::new(format!(
                "cannot create temporary files beside {}: {e}",
                output.display()
            ))
        })?;
    let qbe = temp.path().join("program.ssa");
    let assembly = temp.path().join("program.s");
    let executable = temp.path().join("program");
    fs::write(&qbe, emit(entry))
        .map_err(|e| CompileError::new(format!("cannot write QBE input: {e}")))?;
    run(
        Command::new(env::var_os("QBE").unwrap_or_else(|| "qbe".into()))
            .arg("-o")
            .arg(&assembly)
            .arg(&qbe),
    )?;
    run(
        Command::new(env::var_os("CC").unwrap_or_else(|| "cc".into()))
            .arg(&assembly)
            .arg("-o")
            .arg(&executable),
    )?;
    fs::rename(&executable, output)
        .map_err(|e| CompileError::new(format!("cannot publish {}: {e}", output.display())))?;
    Ok(())
}

fn run(command: &mut Command) -> Result<(), CompileError> {
    let tool = command.get_program().to_string_lossy().into_owned();
    let result = command
        .output()
        .map_err(|e| CompileError::new(format!("cannot run {tool}: {e}")))?;
    if !result.status.success() {
        return Err(CompileError::new(format!(
            "{tool} failed ({}):\n{}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        )));
    }
    Ok(())
}
