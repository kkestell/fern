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
            StatementKind::Binding { initializer, .. } => {
                checked.expressions[initializer].constant.clone()
            }
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
        StatementKind::Binding { initializer, .. } => initializer,
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
