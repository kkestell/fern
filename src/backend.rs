use crate::{
    CompileError,
    ir::{Operand, ValueId, ValueKind, VerifiedEntry},
    semantic::Type,
};
use std::{env, fmt::Write, fs, path::Path, process::Command};

fn operand(operand: Operand) -> String {
    match operand {
        Operand::Integer { value, .. } => value.to_string(),
        Operand::Value(ValueId(id)) => format!("%v{id}"),
    }
}

fn qbe_type(ty: Type) -> char {
    if ty.width() == 64 { 'l' } else { 'w' }
}

pub(crate) fn emit(verified: &VerifiedEntry) -> String {
    let entry = verified.entry();
    let mut text = String::from("export function w $main() {\n@start\n");
    for (id, value) in entry.values.iter().enumerate() {
        let (ValueKind::Copy(source) | ValueKind::Convert(source)) = value.kind;
        let source_ty = match source {
            Operand::Integer { ty, .. } => ty,
            Operand::Value(ValueId(id)) => entry.values[id].ty,
        };
        // Sub-word values occupy QBE words. Verified copies preserve their
        // value; widening extends the source's signed or unsigned bits.
        let instruction = if matches!(value.kind, ValueKind::Convert(_))
            && source_ty.width() < value.ty.width()
        {
            match (source_ty.width(), source_ty.signed()) {
                (8, true) => "extsb",
                (8, false) => "extub",
                (16, true) => "extsh",
                (16, false) => "extuh",
                (32, true) => "extsw",
                (32, false) => "extuw",
                _ => unreachable!("verified integer widening"),
            }
        } else {
            "copy"
        };
        writeln!(
            text,
            "    %v{id} ={} {instruction} {}",
            qbe_type(value.ty),
            operand(source)
        )
        .unwrap();
    }
    writeln!(text, "    %status =w and {}, 255", operand(entry.exit)).unwrap();
    text.push_str("    ret %status\n}\n");
    text
}

pub(crate) fn build(entry: &VerifiedEntry, output: &Path) -> Result<(), CompileError> {
    build_text(&emit(entry), output)
}

fn build_text(text: &str, output: &Path) -> Result<(), CompileError> {
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
    fs::write(&qbe, text).map_err(|e| CompileError::new(format!("cannot write QBE input: {e}")))?;
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
    use crate::ir::{Entry, Value};

    fn integer(value: i32) -> Operand {
        Operand::Integer {
            value: i128::from(value),
            ty: Type::Int,
        }
    }

    fn copy(operand: Operand) -> Value {
        Value {
            ty: Type::Int,
            kind: ValueKind::Copy(operand),
        }
    }

    // Observe every emitted value at its full QBE width before the exit mask.
    // Keeping the comparisons in the native program also prevents unused
    // wide values from disappearing without their representation being tested.
    fn assert_native_values(verified: &VerifiedEntry, expected: &[i128]) {
        assert_eq!(verified.entry().values.len(), expected.len());
        let mut text = emit(verified);
        text.truncate(text.find("    %status =").unwrap());
        text.push_str("    %ok0 =w copy 1\n");
        for (id, (value, expected)) in verified.entry().values.iter().zip(expected).enumerate() {
            let width = if matches!(value.ty, Type::I64 | Type::U64) {
                'l'
            } else {
                'w'
            };
            writeln!(text, "    %check{id} =w ceq{width} %v{id}, {expected}").unwrap();
            writeln!(text, "    %ok{} =w and %ok{id}, %check{id}", id + 1).unwrap();
        }
        writeln!(text, "    ret %ok{}\n}}", expected.len()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("program");
        build_text(&text, &output).unwrap();
        assert_eq!(Command::new(output).status().unwrap().code(), Some(1));
    }

    #[test]
    fn full_width_integer_copies_and_conversions_execute() {
        let ranges = [
            (Type::I8, -128, 127),
            (Type::I16, -32768, 32767),
            (Type::I32, -2147483648, 2147483647),
            (Type::I64, -9223372036854775808, 9223372036854775807),
            (Type::U8, 0, 255),
            (Type::U16, 0, 65535),
            (Type::U32, 0, 4294967295),
            (Type::U64, 0, 18446744073709551615),
            (Type::Int, -2147483648, 2147483647),
            (Type::Uint, 0, 4294967295),
        ];
        let mut entry = Entry {
            values: vec![],
            exit: integer(0),
        };
        let mut expected = Vec::new();
        for (source, min, max) in ranges {
            for number in [min, if min < 0 { -1 } else { 1 }, 0, max / 2 + 1, max] {
                let literal = Operand::Integer {
                    value: number,
                    ty: source,
                };
                let id = entry.values.len();
                entry.values.push(Value {
                    ty: source,
                    kind: ValueKind::Copy(literal),
                });
                expected.push(number);
                for (destination, dest_min, dest_max) in ranges {
                    if (min < 0) != (dest_min < 0) || min < dest_min || max > dest_max {
                        continue;
                    }
                    for operand in [literal, Operand::Value(ValueId(id))] {
                        let kind = if source == destination {
                            ValueKind::Copy(operand)
                        } else {
                            ValueKind::Convert(operand)
                        };
                        entry.values.push(Value {
                            ty: destination,
                            kind,
                        });
                        expected.push(number);
                        let copied = Operand::Value(ValueId(entry.values.len() - 1));
                        entry.values.push(Value {
                            ty: destination,
                            kind: ValueKind::Copy(copied),
                        });
                        expected.push(number);
                    }
                }
            }
        }
        assert_native_values(&entry.verify().unwrap(), &expected);
    }

    #[test]
    fn source_unsigned_high_bits_survive_assignment_copies_and_shadowing() {
        let syntax = crate::frontend::parse(
            "fn main() -> void {
            var x = 255u8;
            const saved: u64 = x;
            { var x = 4294967295u32; const native: uint = x;
              x = 0; const wide: u64 = native; }
            const medium: u16 = x; const word: u32 = medium;
            const wide: u64 = word;
            x = 1;
            var result: u64 = 18446744073709551615;
            const copy = result;
            result = saved;
        }",
        )
        .unwrap();
        let checked = crate::semantic::check(&syntax).unwrap();
        assert_native_values(
            &crate::ir::lower(checked).verify().unwrap(),
            &[
                255,
                255,
                4294967295,
                4294967295,
                0,
                4294967295,
                255,
                255,
                255,
                1,
                18446744073709551615,
                18446744073709551615,
                255,
            ],
        );
    }

    #[test]
    fn negative_values_survive_chained_widening_and_exit() {
        for number in [-128, -1] {
            let entry = Entry {
                values: vec![
                    Value {
                        ty: Type::I8,
                        kind: ValueKind::Copy(Operand::Integer {
                            value: number,
                            ty: Type::I8,
                        }),
                    },
                    Value {
                        ty: Type::I16,
                        kind: ValueKind::Convert(Operand::Value(ValueId(0))),
                    },
                    Value {
                        ty: Type::I32,
                        kind: ValueKind::Convert(Operand::Value(ValueId(1))),
                    },
                    Value {
                        ty: Type::I64,
                        kind: ValueKind::Convert(Operand::Value(ValueId(2))),
                    },
                    Value {
                        ty: Type::Int,
                        kind: ValueKind::Convert(Operand::Value(ValueId(2))),
                    },
                ],
                exit: Operand::Value(ValueId(4)),
            };
            let dir = tempfile::tempdir().unwrap();
            let output = dir.path().join("program");
            let verified = entry.verify().unwrap();
            build(&verified, &output).unwrap();
            assert_eq!(
                Command::new(output).status().unwrap().code(),
                Some((number & 255) as i32)
            );
            assert_native_values(&verified, &[number; 5]);
        }
    }

    #[test]
    fn negative_exit_values_are_masked_before_returning() {
        for (value, expected) in [(-1, 255), (i32::MIN, 0)] {
            for through_copy in [false, true] {
                let entry = Entry {
                    values: if through_copy {
                        vec![copy(integer(value)), copy(Operand::Value(ValueId(0)))]
                    } else {
                        vec![]
                    },
                    exit: if through_copy {
                        Operand::Value(ValueId(1))
                    } else {
                        integer(value)
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
