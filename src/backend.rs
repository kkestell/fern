use crate::{
    CompileError,
    ir::{Operand, ValueId, VerifiedEntry},
};
use std::{env, fmt::Write, fs, path::Path, process::Command};

fn operand(operand: Operand) -> String {
    match operand {
        Operand::Integer(value) => value.to_string(),
        Operand::Value(ValueId(id)) => format!("%v{id}"),
    }
}

pub(crate) fn emit(verified: &VerifiedEntry) -> String {
    let entry = verified.entry();
    let mut text = String::from("export function w $main() {\n@start\n");
    for (id, value) in entry.values.iter().enumerate() {
        writeln!(text, "    %v{id} =w copy {}", operand(*value)).unwrap();
    }
    writeln!(text, "    %status =w and {}, 255", operand(entry.exit)).unwrap();
    text.push_str("    ret %status\n}\n");
    text
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Entry;

    #[test]
    fn negative_exit_values_are_masked_before_returning() {
        for (value, expected) in [(-1, 255), (i32::MIN, 0)] {
            for through_copy in [false, true] {
                let entry = Entry {
                    values: if through_copy {
                        vec![Operand::Integer(value), Operand::Value(ValueId(0))]
                    } else {
                        vec![]
                    },
                    exit: if through_copy {
                        Operand::Value(ValueId(1))
                    } else {
                        Operand::Integer(value)
                    },
                }
                .verify()
                .unwrap();
                let qbe = emit(&entry);
                let expected_operand = if through_copy {
                    "%v1".to_owned()
                } else {
                    value.to_string()
                };
                assert!(qbe.contains(&format!(
                    "%status =w and {expected_operand}, 255\n    ret %status"
                )));
                if through_copy {
                    assert!(qbe.contains(&format!("%v0 =w copy {value}\n    %v1 =w copy %v0")));
                }
                let dir = tempfile::tempdir().unwrap();
                let output = dir.path().join("program");
                build(&entry, &output).unwrap();
                assert_eq!(
                    Command::new(output).status().unwrap().code(),
                    Some(expected)
                );
            }
        }
    }
}
