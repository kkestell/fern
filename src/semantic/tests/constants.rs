use super::*;

#[test]
fn signed_minima_and_expression_constants_keep_their_contracts() {
    for (name, minimum) in [
        ("i8", 1u128 << 7),
        ("i16", 1u128 << 15),
        ("i32", 1u128 << 31),
        ("i64", 1u128 << 63),
        ("int", 1u128 << (usize::BITS - 1)),
    ] {
        accepts(&format!(
            "const direct: {name} = -{minimum}; const converted = {name}(-{minimum});"
        ));
        let invalid = minimum + 1;
        rejects(
            &format!("const value: {name} = -{invalid};"),
            &format!("-{invalid}"),
            &format!("integer value out of range for `{name}`"),
        );
    }

    let body = "const exact = 1 + 2;
             const copy = exact;
             var runtime = 1;
             const mixed = runtime + 2;
             { const exact = runtime; const shadowed = exact + 1; }
             exit(0);
             const after = copy ^ 1;";
    let text = format!("fn main() -> void {{ {body} }}");
    let syntax = parse(&text).unwrap();
    let checked = check_root(&syntax).unwrap();
    assert_eq!(
        checked
            .bindings
            .iter()
            .map(|(_, binding)| binding.constant.is_some())
            .collect::<Vec<_>>(),
        [true, true, false, false, false, false, true]
    );
}

#[test]
fn constant_evaluation_preserves_exact_and_typed_operations() {
    let text = "fn main() -> void {
            const exact: u8 = (250 + 10) / 2;
            const ordinary = 9 * 5 - 3;
            const quotient = -7 / 3;
            const remainder = -7 % 3;
            const complement = ^0;
            const cleared = 15 & ^3;
            const bits = (12 & 10) ^ 3 | 16;
            const large: i64 = 1 << 40;
            const signed_shift = -8 >> 2;
            const wrapped_add = u8(250) +% 10;
            const wrapped_subtract = u8(1) -% 2;
            const wrapped_multiply = u8(200) *% 2;
            const wrapped_negate = -%u8(1);
            const high: u8 = 128;
            const discarded = high << 1;
            const negative: i8 = -1;
            const sign_fill = negative >> 8;
            const huge = u8.truncate((1 << 255) + 42);
            const distant_bit = u8.truncate(((1 << 1000000) * 2) >> 1000001);
            const copy = exact;
            const combined = copy + u8(1);
            var runtime = 2;
            const saved = runtime;
            const not_constant = saved + 1;
        }";
    let syntax = parse(text).unwrap();
    let checked = check_root(&syntax).unwrap();
    assert_eq!(
        checked
            .bindings
            .iter()
            .map(|(_, binding)| binding.constant.clone())
            .collect::<Vec<_>>(),
        [
            folded(130),
            folded(42),
            folded(-2),
            folded(-1),
            folded(-1),
            folded(12),
            folded(27),
            Some(Constant::Integer(BigInt::from(1u8) << 40)),
            folded(-2),
            folded(4),
            folded(255),
            folded(144),
            folded(255),
            folded(128),
            folded(0),
            folded(-1),
            folded(-1),
            folded(42),
            folded(1),
            folded(130),
            folded(131),
            None,
            None,
            None,
        ]
    );
}

#[test]
fn constant_failures_are_diagnosed_before_runtime_lowering() {
    for (body, offending, message) in [
        (
            "const x = u8(255) + 1;",
            "+",
            "constant `+` on `u8` would overflow",
        ),
        (
            "const x = u8(0) - 1;",
            "-",
            "constant `-` on `u8` would overflow",
        ),
        (
            "const x = u8(128) * 2;",
            "*",
            "constant `*` on `u8` would overflow",
        ),
        (
            "const minimum: i8 = -128; const x = -minimum;",
            "-",
            "constant unary `-` on `i8` would trap",
        ),
        (
            "const x = i8(-128) / -1;",
            "/",
            "constant `/` on `i8` would trap",
        ),
        (
            "const x = i8(-128) % -1;",
            "%",
            "constant `%` on `i8` would trap",
        ),
        (
            "var x: u8 = 1; const y = x / 0;",
            "/",
            "constant `/` divisor is zero",
        ),
        (
            "var x: u8 = 1; const y = x % 0;",
            "%",
            "constant `%` divisor is zero",
        ),
        (
            "var x: u8 = 1; const y = x << -1;",
            "<<",
            "constant shift count is negative",
        ),
        (
            "const x: u8 = 250 + 10;",
            "250 + 10",
            "integer value out of range for `u8`",
        ),
        (
            "const x = u8(250 + 10);",
            "250 + 10",
            "integer value out of range for `u8`",
        ),
    ] {
        rejects(body, offending, message);
        rejects(&format!("exit(0); {body}"), offending, message);
    }
    rejects(
        "var runtime: u8 = 1; const outer = runtime + (u8(255) + 1);",
        "+",
        "constant `+` on `u8` would overflow",
    );
}

#[test]
fn constant_classification_does_not_depend_on_binding_mutability() {
    let text = "fn main() -> void {
            const immutable = 1 + 2;
            var mutable = 1 + 2;
            const immutable_copy = immutable;
            const mutable_copy = mutable;
        }";
    let syntax = parse(text).unwrap();
    let checked = check_root(&syntax).unwrap();
    let expression_constants: Vec<_> = syntax.functions[checked.main]
        .body
        .iter()
        .map(|statement| match syntax.statements[*statement].kind {
            StatementKind::Binding {
                initializer: Some(initializer),
                ..
            } => checked.expressions[initializer].constant.clone(),
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(
        expression_constants,
        [folded(3), folded(3), folded(3), None]
    );
    assert_eq!(
        checked
            .bindings
            .iter()
            .map(|(_, binding)| binding.constant.clone())
            .collect::<Vec<_>>(),
        [folded(3), None, folded(3), None]
    );
}

#[test]
fn precedence_wrapping_and_shift_boundaries_follow_the_integer_contract() {
    let text = "fn main() -> void {
            const precedence_left = 1 + 2 << 1;
            const precedence_right = 1 << 2 + 1;
            const same_level = 15 & ^3 & 6;
            const bit_levels = 1 | 2 ^ 3 & 4;
            const wrapped_left = u8(250) +% 10;
            const wrapped_right = 250 +% u8(10);
            const high: u8 = 128;
            const discarded = high << 1;
            var runtime_high: u8 = 128;
            const runtime_discarded = runtime_high << 1;
            const overshift = u8(1) << 999999999999999999999999999999999999;
            const negative: i8 = -1;
            const sign_fill = negative >> 8;
            const exact: u16 = 1 << 8;
            const reduced: u8 = 256 >> u8(8);
        }";
    let syntax = parse(text).unwrap();
    let checked = check_root(&syntax).unwrap();
    assert_eq!(
        checked
            .bindings
            .iter()
            .map(|(_, binding)| binding.constant.clone())
            .collect::<Vec<_>>(),
        [
            folded(5),
            folded(5),
            folded(4),
            folded(3),
            folded(4),
            folded(4),
            folded(128),
            folded(0),
            None,
            None,
            folded(0),
            folded(-1),
            folded(-1),
            folded(256),
            folded(1),
        ]
    );

    for (body, offending, message) in [
        (
            "const invalid: u8 = 250 +% 10;",
            "+%",
            "wrapping arithmetic requires a typed operand",
        ),
        (
            "const invalid = -%1;",
            "-%",
            "wrapping negation requires a typed operand",
        ),
        (
            "const typed = u8(1) +% 256;",
            "256",
            "integer literal out of range for `u8`",
        ),
        (
            "const invalid: u8 = 1 << 8;",
            "1 << 8",
            "integer value out of range for `u8`",
        ),
    ] {
        rejects(body, offending, message);
        rejects(&format!("exit(0); {body}"), offending, message);
    }
}

#[test]
fn untyped_constant_folding_is_bounded_and_discards_child_values() {
    for body in [
        "const value = (1 << 1000000) * (1 << 1000000);",
        "exit(0); const value = (1 << 1000000) * (1 << 1000000);",
    ] {
        rejects(
            body,
            "*",
            "constant expression exceeds compiler resource limit",
        );
    }

    let syntax = parse("fn main() -> void { const value = (1 + 2) * (3 + 4); }").unwrap();
    let checked = check_root(&syntax).unwrap();
    let root = match syntax.statements[syntax.functions[checked.main].body[0]].kind {
        StatementKind::Binding {
            initializer: Some(initializer),
            ..
        } => initializer,
        _ => unreachable!(),
    };
    assert_eq!(checked.expressions[root].constant, folded(21));
    assert!(
        checked
            .expressions
            .iter()
            .all(|(id, expression)| id == root || expression.constant.is_none())
    );
}

#[test]
fn integer_contract_uses_contextual_literals_and_exact_references() {
    for ty in Scalar::ALL_INTEGERS {
        let name = ty.name();
        let max = u128::from(ty.max());
        for base in [2, 8, 10, 16] {
            let maximum = literal(max, base);
            let syntax = parse(&format!(
                "fn main() -> void {{ var x: {name} = {maximum}; x = {maximum}; }}"
            ))
            .unwrap();
            let checked = check_root(&syntax).unwrap();
            assert!(
                checked
                    .bindings
                    .iter()
                    .all(|(_, binding)| binding.ty == value_type(ty))
            );
            assert!(
                checked
                    .expressions
                    .iter()
                    .all(|(_, expression)| expression.ty == value_type(ty))
            );

            let overflow = literal(max + 1, base);
            rejects(
                &format!("exit(0); var x: {name} = {overflow};"),
                &overflow,
                &format!("integer literal out of range for `{name}`"),
            );
        }
    }

    for source in Scalar::ALL_INTEGERS {
        let source_name = source.name();
        for destination in Scalar::ALL_INTEGERS {
            let destination_name = destination.name();
            for target in [
                format!("const target: {destination_name} = source;"),
                format!("var target: {destination_name} = source;"),
                format!("var target: {destination_name} = 0; target = source;"),
            ] {
                for scope in [target.clone(), format!("{{ exit(0); {target} }}")] {
                    let body = format!("const source: {source_name} = 1; {scope}");
                    if source == destination {
                        let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
                        check_root(&syntax).unwrap();
                    } else {
                        rejects(
                            &body,
                            "source",
                            &format!(
                                "cannot implicitly convert `{source_name}` to `{destination_name}`"
                            ),
                        );
                    }
                }
            }
        }
        let body = format!("const source: {source_name} = 1; exit(source);");
        if source == Scalar::Int {
            let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
            check_root(&syntax).unwrap();
        } else {
            rejects(
                &body,
                "source",
                &format!("cannot implicitly convert `{source_name}` to `int`"),
            );
        }
    }
}

#[test]
fn conversions_evaluate_constants_and_preserve_runtime_operands() {
    let syntax = parse(
        "fn main() -> void {
                const literal: u8 = 42;
                const reduced = u8.truncate(340282366920938463463374607431768211498);
                const signed = i8.truncate(255);
                const widened = u64(literal);
                var runtime: u64 = 42;
                const checked = u8(runtime);
                const truncated = u8.truncate(runtime);
                const later = u16(checked);
            }",
    )
    .unwrap();
    let checked = check_root(&syntax).unwrap();
    let constants: Vec<_> = checked
        .bindings
        .iter()
        .map(|(_, binding)| binding.constant.clone())
        .collect();
    assert_eq!(
        constants,
        [
            folded(42),
            folded(42),
            folded(-1),
            folded(42),
            None,
            None,
            None,
            None
        ]
    );
    assert_eq!(
        checked
            .expressions
            .iter()
            .filter(|(_, expression)| {
                matches!(expression.value, ExpressionValue::Conversion { .. })
            })
            .count(),
        3
    );

    rejects(
        "const value: u64 = 18446744073709551615; const narrowed = u8(value);",
        "u8(value)",
        "constant conversion to `u8` would trap",
    );
    rejects(
        "const narrowed = u8(256);",
        "256",
        "integer literal out of range for `u8`",
    );
    rejects(
        "exit(0); const value: u64 = 256; const narrowed = u8(value);",
        "u8(value)",
        "constant conversion to `u8` would trap",
    );
}

#[test]
fn truncating_constants_keep_at_least_256_bits_before_reduction() {
    for literal in [
        format!("0x{}", "F".repeat(64)),
        format!("0b{}", "1".repeat(256)),
        format!("0x1{}FF", "0".repeat(128)),
    ] {
        let syntax = parse(&format!(
            "fn main() -> void {{ const result = int(u8.truncate({literal})); }}"
        ))
        .unwrap();
        let checked = check_root(&syntax).unwrap();
        assert_eq!(
            checked.bindings.iter().next().unwrap().1.constant,
            folded(255)
        );
        rejects(
            &format!("exit(0); const result = u8.truncate(u64({literal}));"),
            &literal,
            "integer literal out of range for `u64`",
        );
    }
}

#[test]
fn constant_classification_follows_copies_and_shadowing() {
    let syntax = parse(
        "fn main() -> void {
                const original: u64 = 255;
                const copy = original;
                { var original = copy;
                  const saved = original;
                  const converted = u8(saved); }
                const folded = i8.truncate(copy);
                const extended = i64(folded);
                const wrapped = u64.truncate(extended);
            }",
    )
    .unwrap();
    let checked = check_root(&syntax).unwrap();
    assert_eq!(
        checked
            .bindings
            .iter()
            .map(|(_, binding)| binding.constant.clone())
            .collect::<Vec<_>>(),
        [
            folded(255),
            folded(255),
            None,
            None,
            None,
            folded(-1),
            folded(-1),
            Some(Constant::Integer(BigInt::from(u64::MAX)))
        ],
    );
    for body in [
        "const original: u64 = 256; const copy = original; const bad = u8(copy);",
        "const original: u64 = 256; { var original: u64 = 1; } const bad = u8(original);",
        "const negative = i8.truncate(255); const bad = u64(negative);",
    ] {
        let syntax = parse(&format!("fn main() -> void {{ exit(0); {body} }}")).unwrap();
        assert!(
            check_root(&syntax)
                .unwrap_err()
                .message
                .contains("would trap")
        );
    }
}

#[test]
fn native_integer_widths_follow_the_host_and_model_both_specified_widths() {
    assert_eq!(Scalar::Int.width(), usize::BITS);
    assert_eq!(Scalar::Uint.width(), usize::BITS);
    for pointer_width in [32, 64] {
        assert_eq!(Scalar::Int.width_on(pointer_width), pointer_width);
        assert_eq!(Scalar::Uint.width_on(pointer_width), pointer_width);
        assert_eq!(Scalar::I32.width_on(pointer_width), 32);
        assert_eq!(Scalar::U64.width_on(pointer_width), 64);
    }
}

#[test]
fn floating_literals_keep_the_exact_value_they_spell() {
    // A binding takes a format, so a literal's exact value is observed
    // through a comparison, which never chooses one.
    for spelling in [
        "1.0 == 1",
        ".5 * 2 == 1",
        "2. == 2",
        "0.1 * 10 == 1",
        "1e3 == 1000",
        "1E+2 == 100",
        "2.5e-2 * 40 == 1",
        "0.000 == 0",
        "123.456 * 1000 == 123456",
        "6.02e-23 * 1e23 == 6.02",
    ] {
        let body = format!("const same = {spelling};");
        assert_eq!(
            checked_bindings(&format!("fn main() -> void {{ {body} }}"))[0],
            (value_type(Scalar::Bool), folded(1)),
            "{spelling}"
        );
    }
}

#[test]
fn an_untyped_floating_constant_takes_its_format_from_context() {
    for (body, ty, constant) in [
        ("const defaulted = 1e0;", Scalar::F64, binary64(1.0)),
        ("const half: f32 = .5;", Scalar::F32, binary32(0.5)),
        ("const rounded: f32 = 0.1;", Scalar::F32, binary32(0.1)),
        ("const wider: f64 = 0.1;", Scalar::F64, binary64(0.1)),
        ("const scaled = 2.5e-2;", Scalar::F64, binary64(0.025)),
        ("const zero: f64 = 0.0;", Scalar::F64, binary64(0.0)),
        ("const whole: f32 = 3;", Scalar::F32, binary32(3.0)),
        ("const negative: f32 = -0.5;", Scalar::F32, binary32(-0.5)),
        // An untyped constant is an exact value, which has no signed zero.
        ("const zero: f32 = -0.0;", Scalar::F32, binary32(0.0)),
        // The subnormal values and the underflow below them.
        ("const small: f32 = 1e-45;", Scalar::F32, binary32(1e-45)),
        ("const under: f32 = 1e-46;", Scalar::F32, binary32(0.0)),
        ("const least: f64 = 5e-324;", Scalar::F64, binary64(5e-324)),
        // The largest finite value of each format.
        (
            "const most: f32 = 340282346638528859811704183484516925440.0;",
            Scalar::F32,
            binary32(f32::MAX),
        ),
        (
            "const big: f64 = 1.7976931348623157e308;",
            Scalar::F64,
            binary64(f64::MAX),
        ),
    ] {
        assert_eq!(
            checked_bindings(&format!("fn main() -> void {{ {body} }}"))[0],
            (value_type(ty), constant),
            "{body}"
        );
    }
}

#[test]
fn rounding_to_a_format_happens_once_and_breaks_ties_to_even() {
    for (body, constant) in [
        // Halfway between two `f32` values, so the tie goes to the even one.
        ("const tie: f32 = 16777217.0;", binary32(16777217.0)),
        ("const above: f32 = 16777219.0;", binary32(16777219.0)),
        // Rounding through `f64` first would land on that tie and then round
        // down. Rounding once from the exact value rounds up.
        (
            "const once: f32 = 16777217.0000000001;",
            binary32(16777217.0000000001),
        ),
        // The exact sum is 0.3, which rounds to a different `f64` value than
        // the sum of the two rounded operands gives.
        ("const sum: f64 = 0.1 + 0.2;", binary64(0.3)),
        ("const third: f64 = 1.0 / 3.0;", binary64(1.0 / 3.0)),
    ] {
        assert_eq!(
            checked_bindings(&format!("fn main() -> void {{ {body} }}"))[0].1,
            constant,
            "{body}"
        );
    }
    // A concrete operation rounds its own result, so the same sum differs.
    assert_eq!(
        checked_bindings("fn main() -> void { const sum = f64(0.1) + f64(0.2); }")
            .last()
            .unwrap()
            .1,
        binary64(0.1 + 0.2),
    );
    assert_eq!(
        checked_bindings("fn main() -> void { const exact = 0.1 + 0.2 == 0.3; }")[0].1,
        folded(1),
    );
    assert_eq!(
        checked_bindings("fn main() -> void { const rounded = f64(0.1) + f64(0.2) == f64(0.3); }")
            .last()
            .unwrap()
            .1,
        folded(0),
    );
}

#[test]
fn untyped_floating_arithmetic_stays_exact_through_every_operator() {
    for (expression, ty, constant) in [
        ("3.0 / 2.0", Scalar::F64, binary64(1.5)),
        ("1 + 0.5", Scalar::F64, binary64(1.5)),
        ("0.5 + 1", Scalar::F64, binary64(1.5)),
        ("2.5 - 1", Scalar::F64, binary64(1.5)),
        ("0.75 * 2", Scalar::F64, binary64(1.5)),
        ("3 / 2.0", Scalar::F64, binary64(1.5)),
        ("-0.25", Scalar::F64, binary64(-0.25)),
        ("-(1.0 + 0.5)", Scalar::F64, binary64(-1.5)),
        ("(1.5 - 0.5) * 4", Scalar::F32, binary32(4.0)),
        ("1e300 * 1e300 / 1e300", Scalar::F64, binary64(1e300)),
    ] {
        let annotation = if ty == Scalar::F32 { ": f32" } else { "" };
        let body = format!("const value{annotation} = {expression};");
        assert_eq!(
            checked_bindings(&format!("fn main() -> void {{ {body} }}"))[0],
            (value_type(ty), constant),
            "{expression}"
        );
    }
}

#[test]
fn floating_constant_operations_reject_what_they_cannot_name() {
    for (body, offending, message) in [
        (
            "const bad: f64 = 1.0 / 0.0;",
            "/",
            "constant `/` divisor is zero",
        ),
        (
            "const bad = 0.0 / 0.0;",
            "/",
            "constant `/` divisor is zero",
        ),
        (
            "const bad = f64(1.0) / f64(0.0);",
            "/",
            "constant `/` divisor is zero",
        ),
        (
            "const bad = -f64(1.0) / (f64(0.0) - f64(0.0));",
            "/",
            "constant `/` divisor is zero",
        ),
        (
            "const bad = f64(1e308) * f64(10.0);",
            "*",
            "constant `*` on `f64` would overflow",
        ),
        (
            "const bad = f32(3e38) + f32(3e38);",
            "+",
            "constant `+` on `f32` would overflow",
        ),
        (
            "const bad = f32(1e38) * 10.0;",
            "*",
            "constant `*` on `f32` would overflow",
        ),
        (
            "const bad: f32 = 1e39;",
            "1e39",
            "floating-point literal out of range for `f32`",
        ),
        (
            "const bad: f64 = 1e309;",
            "1e309",
            "floating-point literal out of range for `f64`",
        ),
        (
            "const bad = 1e600000;",
            "1e600000",
            "floating-point literal exceeds compiler resource limit",
        ),
        (
            "const bad = 1.5e-600000;",
            "1.5e-600000",
            "floating-point literal exceeds compiler resource limit",
        ),
        (
            "const bad: f32 = 1e38 * 10.0;",
            "1e38 * 10.0",
            "floating-point value out of range for `f32`",
        ),
        // The checks do not depend on the operation being reached.
        (
            "exit(0); const bad: f64 = 1.0 / 0.0;",
            "/",
            "constant `/` divisor is zero",
        ),
        (
            "if false { const bad = f64(1e308) * f64(10.0); }",
            "*",
            "constant `*` on `f64` would overflow",
        ),
    ] {
        rejects(body, offending, message);
    }
    // Only an operation the compiler evaluates in full is rejected. A
    // runtime operand makes division by zero and overflow defined results.
    accepts("var x = f64(1.0); const defined = x / f64(0.0);");
    accepts("var x: f32 = 1e38; x /= f32(1e-38);");
    accepts("var x: f32 = 1e38; x = x * f32(10.0);");
}

#[test]
fn a_constant_floating_binding_is_copied_into_the_expressions_that_read_it() {
    let bindings = checked_bindings(
        "fn main() -> void {
            const half: f32 = 0.5;
            const doubled = half + half;
            const compared = doubled == 1.0;
        }",
    );
    assert_eq!(bindings[0], (value_type(Scalar::F32), binary32(0.5)));
    assert_eq!(bindings[1], (value_type(Scalar::F32), binary32(1.0)));
    assert_eq!(bindings[2], (value_type(Scalar::Bool), folded(1)));
    // A `var` is not a constant expression, so nothing folds through it.
    let bindings =
        checked_bindings("fn main() -> void { var half: f32 = 0.5; const doubled = half + half; }");
    assert_eq!(bindings[1], (value_type(Scalar::F32), None));
}

#[test]
fn checked_conversions_between_numbers_preserve_their_value_exactly() {
    for (body, ty, constant) in [
        ("const rounded = f32(0.1);", Scalar::F32, binary32(0.1)),
        ("const whole = f64(2);", Scalar::F64, binary64(2.0)),
        ("const negative = f32(-3);", Scalar::F32, binary32(-3.0)),
        ("const widened = f64(f32(0.5));", Scalar::F64, binary64(0.5)),
        (
            "const narrowed = f32(f64(0.5));",
            Scalar::F32,
            binary32(0.5),
        ),
        ("const back = i32(f64(2.0));", Scalar::I32, folded(2)),
        ("const signed = i8(f32(-128.0));", Scalar::I8, folded(-128)),
        ("const zero = int(f64(0.0));", Scalar::Int, folded(0)),
        (
            "const large = u64(f64(1e18));",
            Scalar::U64,
            folded(1_000_000_000_000_000_000),
        ),
    ] {
        assert_eq!(
            checked_bindings(&format!("fn main() -> void {{ {body} }}"))
                .last()
                .unwrap()
                .clone(),
            (value_type(ty), constant),
            "{body}"
        );
    }
    for (body, offending, message) in [
        (
            "const x: f64 = 0.1; const bad = f32(x);",
            "f32(x)",
            "constant conversion to `f32` would trap",
        ),
        (
            "const bad = i32(f64(1.5));",
            "i32(f64(1.5))",
            "constant conversion to `i32` would trap",
        ),
        (
            "const bad = i32(f64(1e20));",
            "i32(f64(1e20))",
            "constant conversion to `i32` would trap",
        ),
        (
            "const bad = u8(f32(-1.0));",
            "u8(f32(-1.0))",
            "constant conversion to `u8` would trap",
        ),
        (
            "const bad = f32(16777217);",
            "16777217",
            "integer literal not representable in `f32`",
        ),
        (
            "var bad: f32 = 16777217;",
            "16777217",
            "integer literal not representable in `f32`",
        ),
        (
            "const value = 16777216 + 1; var bad: f32 = value;",
            "value",
            "cannot implicitly convert `int` to `f32`",
        ),
        (
            "var bad: f64 = 9007199254740993;",
            "9007199254740993",
            "integer literal not representable in `f64`",
        ),
        (
            "const bad = f32(1e300);",
            "1e300",
            "floating-point literal out of range for `f32`",
        ),
    ] {
        rejects(body, offending, message);
    }
}
