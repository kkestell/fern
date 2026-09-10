use super::*;

#[test]
fn message_data_uses_quoted_runs_and_numeric_exception_bytes() {
    let message = format!("{}\"\\é\n", "printable text ".repeat(1_000));
    let mut data = String::new();
    emit_message_data(&mut data, "message", &message);
    assert!(data.len() < message.len() + 100);
    assert!(data.contains("b \"printable text printable text"));
    for byte in [b'"', b'\\', 0xc3, 0xa9, b'\n'] {
        assert!(data.contains(&format!("b {byte},")));
    }

    let mut qbe = data;
    qbe.push_str("export function w $main() {\n@start\n    ret 0\n}\n");
    let dir = tempfile::tempdir().unwrap();
    build_text(&qbe, &dir.path().join("program")).unwrap();
}

#[test]
fn many_traps_do_not_expand_qbe_byte_by_byte() {
    let mut text = String::from("fn main() -> void {\nvar left = 1; var right = 2;\n");
    for id in 0..3_000 {
        writeln!(text, "const value{id} = left + right;").unwrap();
    }
    text.push_str("}\n");
    let program = lowered(&text);
    let sources = SourceMap::from_text(&text);
    let qbe = emit(&program, Some(&sources));
    assert!(qbe.len() < 5_000_000, "QBE was {} bytes", qbe.len());
    assert!(qbe.contains("b \"Error"));
}

#[test]
fn full_width_integer_copies_and_conversions_execute() {
    let ranges = Scalar::ALL_INTEGERS.map(|ty| (ty, ty.min(), i128::from(ty.max())));
    let mut values = vec![];
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
                let id = values.len();
                values.push(Value {
                    span: None,
                    ty: source.into(),
                    kind: ValueKind::Convert {
                        operand: literal,
                        truncating: false,
                    },
                });
                expected.push(number);
                for operand in [literal, Operand::Value(ValueId(id))] {
                    // Conversions between types carry the span their range trap reports.
                    values.push(Value {
                        span: Some(0..1),
                        ty: destination.into(),
                        kind: ValueKind::Convert {
                            operand,
                            truncating: false,
                        },
                    });
                    expected.push(number);
                    let copied = Operand::Value(ValueId(values.len() - 1));
                    values.push(Value {
                        span: Some(0..1),
                        ty: destination.into(),
                        kind: ValueKind::Convert {
                            operand: copied,
                            truncating: false,
                        },
                    });
                    expected.push(number);
                }
            }
        }
    }
    assert_native_values(&program(values, integer(0)).verify().unwrap(), &expected);
}

fn truncated(value: i128, destination: Scalar) -> i128 {
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
    let types = Scalar::ALL_INTEGERS;
    let mut values = vec![];
    let mut expected = Vec::new();
    for source in types {
        let minimum = source.min();
        let maximum = i128::from(source.max());
        for value in [minimum, if source.signed() { -1 } else { 1 }, maximum] {
            let source_id = values.len();
            values.push(Value {
                span: None,
                ty: source.into(),
                kind: ValueKind::Convert {
                    operand: Operand::Integer { value, ty: source },
                    truncating: false,
                },
            });
            expected.push(value);
            for destination in types {
                values.push(Value {
                    span: None,
                    ty: destination.into(),
                    kind: ValueKind::Convert {
                        operand: Operand::Value(ValueId(source_id)),
                        truncating: true,
                    },
                });
                expected.push(truncated(value, destination));
            }
        }
    }
    assert_native_values(&program(values, integer(0)).verify().unwrap(), &expected);
}

#[test]
fn native_integer_operations_preserve_values_for_every_type() {
    let types = Scalar::ALL_INTEGERS;
    for ty in types {
        let name = ty.name();
        let minimum = ty.min();
        let maximum = i128::from(ty.max());
        let signed_operations = if ty.signed() {
            format!(
                "const negate = -a; var negative: {name} = -7; const negative_quotient = negative / b; const negative_remainder = negative % b;"
            )
        } else {
            String::new()
        };
        let text = format!(
            "fn main() -> void {{
                    var a: {name} = 20; var b: {name} = 3;
                    const add = a + b; const subtract = a - b;
                    const multiply = a * b; const divide = a / b; const remainder = a % b;
                    const wrapping_add = a +% b; const wrapping_subtract = a -% b;
                    const wrapping_multiply = a *% b;
                    {signed_operations} const wrapping_negate = -%a; const complement = ^a;
                    const and = a & b; const and_not = a & ^b;
                    const xor = a ^ b; const or = a | b;
                    const shift_left = a << b; const shift_right = a >> b;
                    var maximum: {name} = {maximum}; var minimum: {name} = {minimum};
                    var one: {name} = 1;
                    const wrapped_maximum = maximum +% one;
                    const wrapped_minimum = minimum -% one;
                    const wrapped_product = maximum *% b;
                }}",
            name = name,
        );
        let program = lowered(&text);
        let mut expected = vec![];
        for result in [23, 17, 60, 6, 2, 23, 17, 60] {
            expected.extend([20, 3, result]);
        }
        if ty.signed() {
            expected.extend([20, -20, -7, 3, -2, -7, 3, -1]);
        }
        expected.extend([
            20,
            truncated(-20, ty),
            20,
            truncated(!20, ty),
            20,
            3,
            20 & 3,
            20,
            3,
            truncated(!3, ty),
            20 & !3,
            20,
            3,
            20 ^ 3,
            20,
            3,
            20 | 3,
            20,
            3,
            truncated(20 << 3, ty),
            20,
            3,
            20 >> 3,
            maximum,
            1,
            truncated(maximum + 1, ty),
            minimum,
            1,
            truncated(minimum - 1, ty),
            maximum,
            3,
            truncated(maximum * 3, ty),
        ]);
        assert_native_values(&program, &expected);
    }
}

#[test]
fn checked_arithmetic_traps_at_each_integer_width() {
    let types = Scalar::ALL_INTEGERS;
    for ty in types {
        let name = ty.name();
        let minimum = ty.min();
        let maximum = i128::from(ty.max());
        for (operator, left, right) in [("+", maximum, 1), ("-", minimum, 1), ("*", maximum, 2)] {
            assert_native_failure(
                &format!(
                    "var left: {name} = {left}; var right: {name} = {right}; const failed = left {operator} right;"
                ),
                &format!("integer `{operator}` overflowed"),
            );
        }
        if ty.signed() {
            assert_native_failure(
                &format!(
                    "var minimum: {name} = {minimum}; var negative_one: {name} = -1; const failed = minimum * negative_one;"
                ),
                "integer `*` overflowed",
            );
            assert_native_failure(
                &format!("var minimum: {name} = {minimum}; const failed = -minimum;"),
                "integer unary `-` overflowed",
            );
        }
    }
}

#[test]
fn division_remainder_and_shift_failures_are_explicit() {
    let types = Scalar::ALL_INTEGERS;
    for ty in types {
        let name = ty.name();
        for operator in ["/", "%"] {
            assert_native_failure(
                &format!(
                    "var value: {name} = 1; var zero: {name} = 0; const failed = value {operator} zero;"
                ),
                &format!("integer `{operator}` has a zero divisor"),
            );
        }
        assert_native_failure(
            &format!(
                "var value: {name} = 1; var negative: int = -1; const failed = value << negative;"
            ),
            "integer shift count is negative",
        );
        if ty.signed() {
            let minimum = ty.min();
            for operator in ["/", "%"] {
                assert_native_failure(
                    &format!(
                        "var minimum: {name} = {minimum}; var negative_one: {name} = -1; const failed = minimum {operator} negative_one;"
                    ),
                    &format!("integer `{operator}` overflowed"),
                );
            }
        }
    }
}

#[test]
fn runtime_shifts_define_discarded_bits_and_large_counts() {
    let syntax = crate::frontend::parser::parse(&SourceMap::from_text(
        "fn main() -> void {
                var high: u8 = 128; var one: uint = 1; var width: u64 = 8;
                var huge: u64 = 18446744073709551615;
                const discarded = high << one;
                const left_overshift = high << width;
                const right_overshift = high >> width;
                const huge_overshift = high << huge;
                var negative: i8 = -1;
                const sign_fill = negative >> width;
                var zero: int = 0;
                const unchanged = high << zero;
            }",
    ))
    .unwrap();
    let program =
        crate::ir::lower::lower(crate::semantic::namespaces::check_root(&syntax).unwrap())
            .verify()
            .unwrap();
    assert_native_values(
        &program,
        &[
            128,
            1,
            0,
            128,
            8,
            0,
            128,
            8,
            0,
            128,
            18446744073709551615,
            0,
            -1,
            8,
            -1,
            128,
            0,
            128,
        ],
    );
}

#[test]
fn nested_runtime_failures_follow_left_to_right_operand_order() {
    let stderr = assert_native_failure(
        "var one: int = 1; var zero: int = 0; var negative: int = -1;
             const failed = (one / zero) + (one << negative);",
        "integer `/` has a zero divisor",
    );
    assert!(!stderr.contains("shift count"));

    let stderr = assert_native_failure(
        "var wide: u16 = 300; var one: int = 1; var zero: int = 0;
             const failed = int(u8(wide)) + (one / zero);",
        "checked integer conversion failed: `u16` to `u8`",
    );
    assert!(!stderr.contains("zero divisor"));
}

#[test]
fn checked_conversions_trap_outside_each_destination_range() {
    let types = Scalar::ALL_INTEGERS;
    for source in types {
        let minimum = source.min();
        let maximum = i128::from(source.max());
        for destination in types {
            let lower = destination.min();
            let upper = i128::from(destination.max());
            let mut failures = vec![minimum, lower - 1, upper + 1, maximum];
            failures.retain(|value| {
                (minimum..=maximum).contains(value) && !(lower..=upper).contains(value)
            });
            failures.sort_unstable();
            failures.dedup();
            for value in failures {
                // Source execution also checks semantic classification and lowering.
                // Truncation constructs every source bit pattern uniformly.
                let bits = value.rem_euclid(1i128 << source.width());
                let text = format!(
                    "fn main() -> void {{ var value = {}.truncate({bits}); \
                         const result = {}(value); exit(42); }}",
                    source.name(),
                    destination.name(),
                );
                let program = lowered(&text);
                let dir = tempfile::tempdir().unwrap();
                let output = dir.path().join("program");
                build_text(&emit(&program, None), &output).unwrap();
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
    let syntax = crate::frontend::parser::parse(&SourceMap::from_text(
        "fn main() -> void { var x: u8 = 42; var y = int(x); exit(y); }",
    ))
    .unwrap();
    let program =
        crate::ir::lower::lower(crate::semantic::namespaces::check_root(&syntax).unwrap())
            .verify()
            .unwrap();
    let text = emit(&program, None);
    assert!(!text.contains("$abort"));
    assert!(!text.contains("$write"));
    assert!(!text.contains("data $"));
    assert!(!text.contains("jnz"));
}

#[test]
fn source_unsigned_high_bits_survive_assignment_copies_and_shadowing() {
    let syntax = crate::frontend::parser::parse(&SourceMap::from_text(
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
    ))
    .unwrap();
    let checked = crate::semantic::namespaces::check_root(&syntax).unwrap();
    assert_native_values(
        &crate::ir::lower::lower(checked).verify().unwrap(),
        &[
            255,
            255,
            4294967295,
            4294967295,
            4294967295,
            4294967295,
            255,
            255,
            255,
            255,
            255,
            255,
            18446744073709551615,
            255,
        ],
    );
}

#[test]
fn negative_values_survive_chained_widening_and_exit() {
    for number in [-128, -1] {
        let program = program(
            vec![
                Value {
                    span: None,
                    ty: Scalar::I8.into(),
                    kind: ValueKind::Convert {
                        operand: Operand::Integer {
                            value: number,
                            ty: Scalar::I8,
                        },
                        truncating: false,
                    },
                },
                Value {
                    span: None,
                    ty: Scalar::I16.into(),
                    kind: ValueKind::Convert {
                        operand: Operand::Value(ValueId(0)),
                        truncating: false,
                    },
                },
                Value {
                    span: None,
                    ty: Scalar::I32.into(),
                    kind: ValueKind::Convert {
                        operand: Operand::Value(ValueId(1)),
                        truncating: false,
                    },
                },
                Value {
                    span: None,
                    ty: Scalar::I64.into(),
                    kind: ValueKind::Convert {
                        operand: Operand::Value(ValueId(2)),
                        truncating: false,
                    },
                },
                Value {
                    span: None,
                    ty: Scalar::Int.into(),
                    kind: ValueKind::Convert {
                        operand: Operand::Value(ValueId(2)),
                        truncating: false,
                    },
                },
            ],
            Operand::Value(ValueId(4)),
        );
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("program");
        let verified = program.verify().unwrap();
        build_text(&emit(&verified, None), &output).unwrap();
        assert_eq!(
            Command::new(output).status().unwrap().code(),
            Some((number & 255) as i32)
        );
        assert_native_values(&verified, &[number; 5]);
    }
}

#[test]
fn traps_at_one_value_id_in_two_functions_link_separately() {
    let program = lowered(
        "fn divide(left: int, right: int) -> int { return left / right; }
             fn modulo(left: int, right: int) -> int { return left % right; }
             fn main() -> void { exit(divide(84, 2) - modulo(84, 42)); }",
    );
    let qbe = emit(&program, None);
    let symbols: Vec<&str> = qbe
        .lines()
        .filter(|line| line.starts_with("data $fern_"))
        .map(|line| line.split(' ').nth(1).expect("a named data definition"))
        .collect();
    // Both callees trap on the same value ID, so only the function keeps
    // their message symbols apart.
    for symbol in [
        "$fern_function0_operation2_zero_message",
        "$fern_function1_operation2_zero_message",
    ] {
        assert!(symbols.contains(&symbol), "{symbols:?}");
    }
    let unique: BTreeSet<&&str> = symbols.iter().collect();
    assert_eq!(unique.len(), symbols.len(), "{symbols:?}");

    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&qbe, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
}

#[test]
fn negative_exit_values_are_masked_before_returning() {
    for (value, expected) in [(-1, 255), (i32::MIN, 0)] {
        for through_copy in [false, true] {
            let program = program(
                if through_copy {
                    vec![copy(integer(value)), copy(Operand::Value(ValueId(0)))]
                } else {
                    vec![]
                },
                if through_copy {
                    Operand::Value(ValueId(1))
                } else {
                    integer(value)
                },
            )
            .verify()
            .unwrap();
            let qbe = emit(&program, None);
            let expected_operand = if through_copy {
                "%v1".to_owned()
            } else {
                value.to_string()
            };
            let expected_mask = format!(
                "%block0_status =w and {expected_operand}, 255\n    call $exit(w %block0_status)"
            );
            assert!(qbe.contains(&expected_mask));
            if through_copy {
                let ty = qbe_type(Scalar::Int);
                assert!(qbe.contains(&format!("%v0 ={ty} copy {value}\n    %v1 ={ty} copy %v0")));
            }
            let dir = tempfile::tempdir().unwrap();
            let output = dir.path().join("program");
            build_text(&emit(&program, None), &output).unwrap();
            assert_eq!(
                Command::new(output).status().unwrap().code(),
                Some(expected)
            );
        }
    }
}
