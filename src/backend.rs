use crate::{
    CompileError,
    diagnostic::Diagnostic,
    ir::{Operand, ValueId, ValueKind, VerifiedEntry},
    semantic::Type,
    source::Source,
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

pub(crate) fn emit(verified: &VerifiedEntry, source_file: Option<&Source>) -> String {
    let entry = verified.entry();
    let mut text = String::new();
    let mut data = String::new();
    text.push_str("export function w $main() {\n@start\n");
    for (id, value) in entry.values.iter().enumerate() {
        match value.kind {
            ValueKind::Copy(source) => {
                writeln!(
                    text,
                    "    %v{id} ={} copy {}",
                    qbe_type(value.ty),
                    operand(source)
                )
                .unwrap();
            }
            ValueKind::Convert {
                operand: source,
                truncating,
            } => {
                let source_ty = operand_type(entry, source);
                if truncating || conversion_fits(source_ty, value.ty) {
                    emit_truncation(&mut text, id, source, source_ty, value.ty);
                } else {
                    let message = format!(
                        "checked integer conversion failed: `{}` to `{}`",
                        source_ty.name(),
                        value.ty.name()
                    );
                    let message = match (&value.span, source_file) {
                        (Some(span), Some(source)) => Diagnostic::new(span.clone(), message)
                            .render(source)
                            .to_string(),
                        _ => format!("{message}\n"),
                    };
                    write!(data, "data $fern_conversion{id}_message = {{ ").unwrap();
                    // Numeric bytes also handle quotes, backslashes, and UTF-8 in source paths.
                    for byte in message.bytes() {
                        write!(data, "b {byte}, ").unwrap();
                    }
                    data.push_str("b 0 }\n");
                    emit_checked_conversion(
                        &mut text,
                        id,
                        source,
                        source_ty,
                        value.ty,
                        message.len(),
                    );
                }
            }
        }
    }
    let exit = operand(entry.exit);
    let exit = if Type::Int.width() == 64 {
        writeln!(text, "    %exit_status =w copy {exit}").unwrap();
        "%exit_status"
    } else {
        exit.as_str()
    };
    writeln!(text, "    %status =w and {exit}, 255").unwrap();
    text.push_str("    ret %status\n}\n");
    data + &text
}

fn conversion_fits(source: Type, destination: Type) -> bool {
    type_minimum(source) >= type_minimum(destination) && source.max() <= destination.max()
}

fn operand_type(entry: &crate::ir::Entry, operand: Operand) -> Type {
    match operand {
        Operand::Integer { ty, .. } => ty,
        Operand::Value(ValueId(id)) => entry.values[id].ty,
    }
}

fn type_minimum(ty: Type) -> i128 {
    if ty.signed() {
        -(1i128 << (ty.width() - 1))
    } else {
        0
    }
}

fn comparison_name(signed: bool, ty: Type) -> String {
    format!("c{}lt{}", if signed { "s" } else { "u" }, qbe_type(ty))
}

fn emit_checked_conversion(
    text: &mut String,
    id: usize,
    source: Operand,
    source_ty: Type,
    destination: Type,
    message_len: usize,
) {
    let source = operand(source);
    let source_minimum = type_minimum(source_ty);
    let source_maximum = i128::from(source_ty.max());
    let destination_minimum = type_minimum(destination);
    let destination_maximum = i128::from(destination.max());
    let failure = format!("conversion{id}_failed");
    let ready = format!("conversion{id}_ready");
    let done = format!("conversion{id}_done");
    let upper = format!("conversion{id}_upper");

    if source_maximum > destination_maximum {
        writeln!(
            text,
            "    %conversion{id}_too_large =w {} {}, {}",
            comparison_name(source_ty.signed(), source_ty),
            destination_maximum,
            source
        )
        .unwrap();
        writeln!(
            text,
            "    jnz %conversion{id}_too_large, @{failure}, @{upper}"
        )
        .unwrap();
    }
    if source_minimum < destination_minimum {
        if source_maximum > destination_maximum {
            writeln!(text, "@{upper}").unwrap();
        }
        writeln!(
            text,
            "    %conversion{id}_too_small =w {} {}, {}",
            comparison_name(true, source_ty),
            source,
            destination_minimum
        )
        .unwrap();
        writeln!(
            text,
            "    jnz %conversion{id}_too_small, @{failure}, @{ready}"
        )
        .unwrap();
    } else if source_maximum > destination_maximum {
        writeln!(text, "@{upper}").unwrap();
    }
    writeln!(text, "@{ready}").unwrap();
    emit_truncation_operand(text, id, &source, source_ty, destination);
    writeln!(text, "    jmp @{done}").unwrap();
    writeln!(text, "@{failure}").unwrap();
    // Write directly to stderr: abort does not reliably flush C stdio buffers.
    writeln!(
        text,
        "    call $write(w 2, l $fern_conversion{id}_message, {} {message_len})",
        qbe_type(Type::Uint)
    )
    .unwrap();
    text.push_str("    call $abort()\n    ret 1\n");
    writeln!(text, "@{done}").unwrap();
}

fn emit_truncation(
    text: &mut String,
    id: usize,
    source: Operand,
    source_ty: Type,
    destination: Type,
) {
    emit_truncation_operand(text, id, &operand(source), source_ty, destination);
}

fn emit_truncation_operand(
    text: &mut String,
    id: usize,
    source: &str,
    source_ty: Type,
    destination: Type,
) {
    let raw = format!("%conversion{id}_raw");
    match destination.width() {
        8 | 16 => {
            let input = if source_ty.width() == 64 {
                writeln!(text, "    {raw} =w copy {source}").unwrap();
                raw.as_str()
            } else {
                source
            };
            let instruction = match (destination.width(), destination.signed()) {
                (8, true) => "extsb",
                (8, false) => "extub",
                (16, true) => "extsh",
                (16, false) => "extuh",
                _ => unreachable!(),
            };
            writeln!(text, "    %v{id} =w {instruction} {input}").unwrap();
        }
        32 => {
            writeln!(text, "    %v{id} =w copy {source}").unwrap();
        }
        64 => {
            if source_ty.width() == 64 {
                writeln!(text, "    %v{id} =l copy {source}").unwrap();
                return;
            }
            let instruction = match (source_ty.width(), source_ty.signed()) {
                (8, true) => "extsb",
                (8, false) => "extub",
                (16, true) => "extsh",
                (16, false) => "extuh",
                (32, true) => "extsw",
                (32, false) => "extuw",
                _ => unreachable!(),
            };
            if source_ty.width() == 32 {
                writeln!(text, "    %v{id} =l {instruction} {source}").unwrap();
            } else {
                writeln!(text, "    {raw} =w {instruction} {source}").unwrap();
                let extend = if source_ty.signed() { "extsw" } else { "extuw" };
                writeln!(text, "    %v{id} =l {extend} {raw}").unwrap();
            }
        }
        _ => unreachable!(),
    }
}

pub(crate) fn build(
    entry: &VerifiedEntry,
    source: &Source,
    output: &Path,
) -> Result<(), CompileError> {
    build_text(&emit(entry, Some(source)), output)
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
            span: None,
            ty: Type::Int,
            kind: ValueKind::Copy(operand),
        }
    }

    // Observe every emitted value at its full QBE width before the exit mask.
    // Keeping the comparisons in the native program also prevents unused
    // wide values from disappearing without their representation being tested.
    fn assert_native_values(verified: &VerifiedEntry, expected: &[i128]) {
        assert_eq!(verified.entry().values.len(), expected.len());
        let mut text = emit(verified, None);
        text.truncate(text.find("    %status =").unwrap());
        text.push_str("    %ok0 =w copy 1\n");
        for (id, (value, expected)) in verified.entry().values.iter().zip(expected).enumerate() {
            let width = qbe_type(value.ty);
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
            (
                Type::Int,
                type_minimum(Type::Int),
                i128::from(Type::Int.max()),
            ),
            (Type::Uint, 0, i128::from(Type::Uint.max())),
        ];
        let mut entry = Entry {
            values: vec![],
            exit: integer(0),
        };
        let mut expected = Vec::new();
        for (source, min, max) in ranges {
            for (destination, dest_min, dest_max) in ranges {
                let lower = min.max(dest_min);
                let upper = max.min(dest_max);
                let mut numbers = vec![lower, upper, 0, 1];
                if lower < 0 {
                    numbers.push(-1);
                }
                numbers.sort_unstable();
                numbers.dedup();
                for number in numbers {
                    let literal = Operand::Integer {
                        value: number,
                        ty: source,
                    };
                    let id = entry.values.len();
                    entry.values.push(Value {
                        span: None,
                        ty: source,
                        kind: ValueKind::Copy(literal),
                    });
                    expected.push(number);
                    for operand in [literal, Operand::Value(ValueId(id))] {
                        entry.values.push(Value {
                            span: None,
                            ty: destination,
                            kind: ValueKind::Convert {
                                operand,
                                truncating: false,
                            },
                        });
                        expected.push(number);
                        let copied = Operand::Value(ValueId(entry.values.len() - 1));
                        entry.values.push(Value {
                            span: None,
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

    fn truncated(value: i128, destination: Type) -> i128 {
        let modulus = 1i128 << destination.width();
        let bits = value.rem_euclid(modulus);
        if destination.signed() && bits >= (1i128 << (destination.width() - 1)) {
            bits - modulus
        } else {
            bits
        }
    }

    #[test]
    fn truncating_conversions_preserve_each_destination_bit_pattern() {
        let types = [
            Type::I8,
            Type::I16,
            Type::I32,
            Type::I64,
            Type::U8,
            Type::U16,
            Type::U32,
            Type::U64,
            Type::Int,
            Type::Uint,
        ];
        let mut entry = Entry {
            values: vec![],
            exit: integer(0),
        };
        let mut expected = Vec::new();
        for source in types {
            let minimum = type_minimum(source);
            let maximum = i128::from(source.max());
            for value in [minimum, if source.signed() { -1 } else { 1 }, maximum] {
                let source_id = entry.values.len();
                entry.values.push(Value {
                    span: None,
                    ty: source,
                    kind: ValueKind::Copy(Operand::Integer { value, ty: source }),
                });
                expected.push(value);
                for destination in types {
                    entry.values.push(Value {
                        span: None,
                        ty: destination,
                        kind: ValueKind::Convert {
                            operand: Operand::Value(ValueId(source_id)),
                            truncating: true,
                        },
                    });
                    expected.push(truncated(value, destination));
                }
            }
        }
        assert_native_values(&entry.verify().unwrap(), &expected);
    }

    #[test]
    fn checked_conversions_trap_outside_each_destination_range() {
        let types = [
            Type::I8,
            Type::I16,
            Type::I32,
            Type::I64,
            Type::U8,
            Type::U16,
            Type::U32,
            Type::U64,
            Type::Int,
            Type::Uint,
        ];
        for source in types {
            let minimum = type_minimum(source);
            let maximum = i128::from(source.max());
            for destination in types {
                let lower = type_minimum(destination);
                let upper = i128::from(destination.max());
                let mut failures = vec![minimum, lower - 1, upper + 1, maximum];
                failures.retain(|value| {
                    (minimum..=maximum).contains(value) && !(lower..=upper).contains(value)
                });
                failures.sort_unstable();
                failures.dedup();
                for value in failures {
                    // Source execution also checks semantic classification and lowering.
                    // Truncation expresses negative values before unary syntax is supported.
                    let bits = value.rem_euclid(1i128 << source.width());
                    let text = format!(
                        "fn main() -> void {{ var value = {}.truncate({bits}); \
                         const result = {}(value); exit(42); }}",
                        source.name(),
                        destination.name(),
                    );
                    let syntax = crate::frontend::parse(&text).unwrap();
                    let entry = crate::ir::lower(crate::semantic::check(&syntax).unwrap())
                        .verify()
                        .unwrap();
                    let dir = tempfile::tempdir().unwrap();
                    let output = dir.path().join("program");
                    build_text(&emit(&entry, None), &output).unwrap();
                    let result = Command::new(output).output().unwrap();
                    assert!(!result.status.success(), "{text}");
                    assert!(
                        String::from_utf8_lossy(&result.stderr).contains(&format!(
                            "checked integer conversion failed: `{}` to `{}`",
                            source.name(),
                            destination.name(),
                        )),
                        "{text}: {:?}",
                        result,
                    );
                }
            }
        }
    }

    #[test]
    fn infallible_conversions_do_not_emit_trap_blocks_or_messages() {
        let syntax = crate::frontend::parse(
            "fn main() -> void { var x: u8 = 42; var y = int(x); exit(y); }",
        )
        .unwrap();
        let entry = crate::ir::lower(crate::semantic::check(&syntax).unwrap())
            .verify()
            .unwrap();
        let text = emit(&entry, None);
        assert!(!text.contains("$abort"));
        assert!(!text.contains("$write"));
        assert!(!text.contains("data $"));
        assert!(!text.contains("jnz"));
    }

    #[test]
    fn source_unsigned_high_bits_survive_assignment_copies_and_shadowing() {
        let syntax = crate::frontend::parse(
            "fn main() -> void {
            var x: u8 = 255;
            const saved = u64(x);
            { var x: u32 = 4294967295; const native = uint(x);
              x = 0; const wide = u64(native); }
            const medium = u16(x); const word = u32(medium);
            const wide = u64(word);
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
                        span: None,
                        ty: Type::I8,
                        kind: ValueKind::Copy(Operand::Integer {
                            value: number,
                            ty: Type::I8,
                        }),
                    },
                    Value {
                        span: None,
                        ty: Type::I16,
                        kind: ValueKind::Convert {
                            operand: Operand::Value(ValueId(0)),
                            truncating: false,
                        },
                    },
                    Value {
                        span: None,
                        ty: Type::I32,
                        kind: ValueKind::Convert {
                            operand: Operand::Value(ValueId(1)),
                            truncating: false,
                        },
                    },
                    Value {
                        span: None,
                        ty: Type::I64,
                        kind: ValueKind::Convert {
                            operand: Operand::Value(ValueId(2)),
                            truncating: false,
                        },
                    },
                    Value {
                        span: None,
                        ty: Type::Int,
                        kind: ValueKind::Convert {
                            operand: Operand::Value(ValueId(2)),
                            truncating: false,
                        },
                    },
                ],
                exit: Operand::Value(ValueId(4)),
            };
            let dir = tempfile::tempdir().unwrap();
            let output = dir.path().join("program");
            let verified = entry.verify().unwrap();
            build_text(&emit(&verified, None), &output).unwrap();
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
                let qbe = emit(&entry, None);
                let expected_operand = if through_copy {
                    "%v1".to_owned()
                } else {
                    value.to_string()
                };
                let expected_mask = if Type::Int.width() == 64 {
                    format!(
                        "%exit_status =w copy {expected_operand}\n    %status =w and %exit_status, 255\n    ret %status"
                    )
                } else {
                    format!("%status =w and {expected_operand}, 255\n    ret %status")
                };
                assert!(qbe.contains(&expected_mask));
                if through_copy {
                    let ty = qbe_type(Type::Int);
                    assert!(
                        qbe.contains(&format!("%v0 ={ty} copy {value}\n    %v1 ={ty} copy %v0"))
                    );
                }
                let dir = tempfile::tempdir().unwrap();
                let output = dir.path().join("program");
                build_text(&emit(&entry, None), &output).unwrap();
                assert_eq!(
                    Command::new(output).status().unwrap().code(),
                    Some(expected)
                );
            }
        }
    }
}
