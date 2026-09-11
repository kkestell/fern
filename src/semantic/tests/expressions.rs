use super::*;

#[test]
fn an_array_literal_takes_its_type_from_context() {
    for (source, expected) in [
        (
            "fn main() -> void { var a: [3]int = [1, 2, 3]; }",
            array_type(3, value_type(Scalar::Int)),
        ),
        (
            "fn main() -> void { var a: [2]u8 = [0, 255]; }",
            array_type(2, value_type(Scalar::U8)),
        ),
        (
            "fn main() -> void { var a: [2]bool = [true, false]; }",
            array_type(2, value_type(Scalar::Bool)),
        ),
        (
            "fn main() -> void { var a: [2][3]int = [[1, 2, 3], [4, 5, 6]]; }",
            array_type(2, array_type(3, value_type(Scalar::Int))),
        ),
    ] {
        assert_eq!(checked_bindings(source)[0].0, expected, "{source}");
    }
    // A parameter, a result, and an assignment target give a literal its
    // type the same way an annotation does.
    accepts_source("fn take(a: [2]u8) -> void {} fn main() -> void { take([0, 255]); }");
    accepts_source("fn make() -> [2]u8 { return [0, 255]; } fn main() -> void {}");
    accepts_source("fn main() -> void { var a: [2]u8 = [0, 0]; a = [1, 255]; }");
    // Context reaches the literal itself and no further, so a grouped
    // literal is checked with none.
    rejects_root(
        "fn main() -> void { var a: [2]u8 = «([0, 255])»; }",
        "cannot implicitly convert `[2]int` to `[2]u8`",
    );
    rejects_root(
        "fn main() -> void { var a: int = «[1, 2]»; }",
        "cannot implicitly convert an array literal to `int`",
    );
    rejects_root(
        "fn main() -> void { var a: [2]u8 = [0, «256»]; }",
        "integer literal out of range for `u8`",
    );
}

#[test]
fn a_literal_without_context_takes_one_common_element_type() {
    for (source, expected) in [
        (
            "fn main() -> void { var a = [1, 2, 3]; }",
            array_type(3, value_type(Scalar::Int)),
        ),
        (
            "fn main() -> void { var a = [true, false]; }",
            array_type(2, value_type(Scalar::Bool)),
        ),
        (
            "fn main() -> void { var x: u8 = 1; var a = [1, x]; }",
            array_type(2, value_type(Scalar::U8)),
        ),
        (
            "fn main() -> void { var x: u8 = 1; var a = [x, 1]; }",
            array_type(2, value_type(Scalar::U8)),
        ),
        (
            "fn main() -> void { var a = [[1, 2], [3, 4]]; }",
            array_type(2, array_type(2, value_type(Scalar::Int))),
        ),
    ] {
        assert_eq!(
            checked_bindings(source).last().unwrap().0,
            expected,
            "{source}"
        );
    }
    rejects_root(
        "fn main() -> void { var x: u8 = 1; var y: int = 1; var a = [x, «y»]; }",
        "cannot implicitly convert `int` to `u8`",
    );
}

#[test]
fn an_array_literal_has_one_element_per_array_element() {
    rejects_root(
        "fn main() -> void { var a: [3]int = «[1, 2]»; }",
        "expected 3 elements for `[3]int`, found 2",
    );
    rejects_root(
        "fn main() -> void { var a: [2]int = «[1, 2, 3]»; }",
        "expected 2 elements for `[2]int`, found 3",
    );
    rejects_root(
        "fn take(a: [3]int) -> void {} fn main() -> void { take(«[1, 2]»); }",
        "expected 3 elements for `[3]int`, found 2",
    );
    rejects_root(
        "fn make() -> [2]int { return «[1, 2, 3]»; } fn main() -> void {}",
        "expected 2 elements for `[2]int`, found 3",
    );
}

#[test]
fn a_fill_repeats_the_last_element_across_the_remaining_elements() {
    for (source, expected) in [
        ("const a: [2]int = [7...]; fn main() -> void {}", vec![7, 7]),
        (
            "const a: [8]int = [1, 2, 3, 0...]; fn main() -> void {}",
            vec![1, 2, 3, 0, 0, 0, 0, 0],
        ),
        (
            "const a: [3]int = [1, 2, 3...]; fn main() -> void {}",
            vec![1, 2, 3],
        ),
    ] {
        assert_eq!(
            checked_bindings(source)[0].1,
            folded_array(&expected),
            "{source}"
        );
    }
    // A nested fill takes its length from the element type.
    let row = Constant::Array(vec![Constant::Integer(big(1)); 3]);
    assert_eq!(
        checked_bindings("const a: [2][3]u8 = [[1...]...]; fn main() -> void {}")[0].1,
        Some(Constant::Array(vec![row.clone(), row]))
    );
    rejects_root(
        "fn main() -> void { var a: [2]int = «[1, 2, 3...]»; }",
        "expected at most 2 elements for `[2]int`, found 3",
    );
    rejects_root(
        "fn main() -> void { var bad = [0«...»]; }",
        "a fill requires a length from context",
    );
}

#[test]
fn a_module_level_declaration_holds_a_constant_array() {
    let source = "var counts: [3]int = [1, 2, 3];
const limits: [2]u8 = [0, 255];
fn main() -> void {}";
    assert_eq!(
        checked_bindings(source),
        [
            (array_type(3, value_type(Scalar::Int)), None),
            (
                array_type(2, value_type(Scalar::U8)),
                folded_array(&[0, 255])
            ),
        ]
    );
    // A module-level `var` records no constant on its binding, so the
    // folded array is read back from the initializer instead.
    let syntax = parse(source).unwrap();
    let checked = check_root(&syntax).unwrap();
    let initializers: Vec<_> = checked
        .module_bindings
        .iter()
        .map(|&statement| match syntax.statements[statement].kind {
            StatementKind::Binding {
                initializer: Some(initializer),
                ..
            } => checked.expressions[initializer].constant.clone(),
            _ => unreachable!("module bindings are binding statements"),
        })
        .collect();
    assert_eq!(
        initializers,
        [folded_array(&[1, 2, 3]), folded_array(&[0, 255])]
    );
    rejects_root(
        "var seed = 1; var a: [2]int = «[seed, 2]»; fn main() -> void {}",
        "module-level initializer must be a constant expression",
    );
}

#[test]
fn a_whole_array_value_requires_an_identical_type() {
    let bindings = checked_bindings("fn main() -> void { var a: [2]int = [1, 2]; var b = a; }");
    assert_eq!(bindings[0].0, bindings[1].0);
    accepts_source(
        "fn take(a: [2]int) -> [2]int { return a; }
             fn main() -> void { var a: [2]int = [1, 2]; var b: [2]int = [3, 4]; a = take(b); }",
    );
    rejects_root(
        "fn main() -> void { var a: [3]int = [1, 2, 3]; var b: [2]int = [1, 2]; a = «b»; }",
        "cannot implicitly convert `[2]int` to `[3]int`",
    );
    rejects_root(
        "fn take(a: [3]int) -> void {}
fn main() -> void { var b: [2]int = [1, 2]; take(«b»); }",
        "cannot implicitly convert `[2]int` to `[3]int`",
    );
    rejects_root(
        "fn make() -> [3]int { var b: [2]int = [1, 2]; return «b»; }
fn main() -> void {}",
        "cannot implicitly convert `[2]int` to `[3]int`",
    );
    rejects_root(
        "fn main() -> void { var a: [2]int = [1, 2]; var b = a «+» 1; }",
        "`+` requires numeric operands",
    );
}

#[test]
fn indexing_an_array_yields_its_element_type() {
    for (source, expected) in [
        (
            "fn main() -> void { var a: [3]int = [1, 2, 3]; var e = a[0]; }",
            value_type(Scalar::Int),
        ),
        (
            "fn main() -> void { var g: [2][3]u8 = [[1...]...]; var e = g[1]; }",
            array_type(3, value_type(Scalar::U8)),
        ),
        (
            "fn main() -> void { var g: [2][3]u8 = [[1...]...]; var e = g[1][2]; }",
            value_type(Scalar::U8),
        ),
    ] {
        assert_eq!(
            checked_bindings(source).last().unwrap().0,
            expected,
            "{source}"
        );
    }
    accepts_source("fn main() -> void { var a: [3]int = [1, 2, 3]; var i = 2; exit(a[i]); }");
    rejects_root(
        "fn main() -> void { var x = 1; var e = «x»[0]; }",
        "cannot index `int`",
    );
    rejects_root(
        "fn main() -> void { var a: [3]int = [1, 2, 3]; var i: u8 = 1; var e = a[«i»]; }",
        "cannot implicitly convert `u8` to `int`",
    );
    rejects_root(
        "fn main() -> void { var a: [3]int = [1, 2, 3]; var e = a[«true»]; }",
        "cannot implicitly convert `bool` to `int`",
    );
}

#[test]
fn a_constant_index_must_be_in_range_and_never_folds() {
    accepts_source("fn main() -> void { const a: [3]int = [1, 2, 3]; exit(a[2]); }");
    rejects_root(
        "fn main() -> void { const a: [3]int = [1, 2, 3]; exit(a[«3»]); }",
        "index 3 is out of range for `[3]int`",
    );
    rejects_root(
        "fn main() -> void { const a: [3]int = [1, 2, 3]; exit(a[«-1»]); }",
        "index -1 is out of range for `[3]int`",
    );
    // `a[i]` is never a constant expression, even when the array and the
    // index both are.
    rejects_root(
        "const a: [3]int = [1, 2, 3]; const first = «a[0]»; fn main() -> void {}",
        "module-level initializer must be a constant expression",
    );
}

#[test]
fn an_element_of_a_mutable_array_may_be_assigned() {
    accepts_source("fn main() -> void { var a: [3]int = [1, 2, 3]; a[0] = 9; a[1] += 1; }");
    accepts_source(
        "fn main() -> void { var g: [2][3]int = [[1...]...]; g[0][1] = 9; g[1] = [4, 5, 6]; }",
    );
    rejects_root(
        "fn main() -> void { const a: [3]int = [1, 2, 3]; «a»[0] = 9; }",
        "cannot assign to immutable binding `a`",
    );
    rejects_root(
        "fn main() -> void { var a: [3]int = [1, 2, 3]; a[0] = «true»; }",
        "cannot implicitly convert `bool` to `int`",
    );
    rejects_root(
        "fn main() -> void { var a: [3]int = [1, 2, 3]; «a»[0][0] = 9; }",
        "cannot index `int`",
    );
    rejects_root(
        "fn main() -> void { var a: [3]int = [1, 2, 3]; a[«3»] = 9; }",
        "index 3 is out of range for `[3]int`",
    );
    rejects_root(
        "fn main() -> void { var g: [2][3]int = [[1...]...]; g[0] «+=» 1; }",
        "`+=` requires numeric operands",
    );
}

#[test]
fn len_reads_the_length_from_the_operands_type() {
    for (source, expected) in [
        (
            "fn main() -> void { const a: [3]int = [1, 2, 3]; const n = len(a); }",
            3,
        ),
        (
            "fn main() -> void { var a: [4]u8 = [0...]; const n = len(a); }",
            4,
        ),
        (
            "fn main() -> void { var g: [2][3]int = [[1...]...]; const n = len(g[0]); }",
            3,
        ),
    ] {
        assert_eq!(
            checked_bindings(source).last().unwrap(),
            &(value_type(Scalar::Int), folded(expected)),
            "{source}"
        );
    }
    // A folded length is usable wherever a constant expression is.
    assert_eq!(
        checked_bindings(
            "const a: [3]int = [1, 2, 3];
                 fn main() -> void { var b: [len(a)]u8 = [0...]; }"
        )[1]
        .0,
        array_type(3, value_type(Scalar::U8))
    );
    rejects_root(
        "fn main() -> void { var x = 1; const n = len(«x»); }",
        "`len` requires an array operand, found `int`",
    );
    // Reaching the operand's type through a call means the length does not
    // fold, because the call is still evaluated.
    rejects_root(
        "fn make() -> [3]int { return [1, 2, 3]; }
const n = len(«make()»);
fn main() -> void {}",
        "module-level initializer must be a constant expression",
    );
}

#[test]
fn arrays_compare_for_equality_only() {
    accepts_source(
        "fn main() -> void {
                 var a: [3]int = [1, 2, 3];
                 var b: [3]int = [1, 2, 3];
                 if a == b { exit(1); }
                 if a != b { exit(2); }
             }",
    );
    for (source, expected) in [
        (
            "const a: [3]int = [1, 2, 3]; const same = a == [1, 2, 3]; fn main() -> void {}",
            folded(1),
        ),
        (
            "const a: [3]int = [1, 2, 3]; const same = a == [1, 2, 4]; fn main() -> void {}",
            folded(0),
        ),
        (
            "const a: [3]int = [1, 2, 3]; const same = a != [1, 2, 3]; fn main() -> void {}",
            folded(0),
        ),
    ] {
        assert_eq!(checked_bindings(source)[1].1, expected, "{source}");
    }
    rejects_root(
        "fn main() -> void { var a: [3]int = [1, 2, 3]; var b: [3]int = [1, 2, 3];
             if a «<» b { exit(1); } }",
        "only `==` and `!=` are defined on `[3]int`",
    );
    rejects_root(
        "fn main() -> void { var a: [3]int = [1, 2, 3]; var b: [2]int = [1, 2];
             if a «==» b { exit(1); } }",
        "comparison operands have different types `[3]int` and `[2]int`",
    );
    rejects_root(
        "fn main() -> void { var a: [3]int = [1, 2, 3]; var x = 1;
             if a «==» x { exit(1); } }",
        "comparison operands have different types `[3]int` and `int`",
    );
}

#[test]
fn boolean_bindings_assignments_and_constants_are_typed() {
    let syntax = parse(
        "const module_copy = module_ready;
             const module_ready: bool = true;
             fn main() -> void {
                 var ready: bool = false;
                 const copied = module_copy;
                 const negated = !copied;
                 ready = negated;
             }",
    )
    .unwrap();
    let checked = check_root(&syntax).unwrap();
    assert!(
        checked
            .bindings
            .iter()
            .all(|(_, binding)| binding.ty == value_type(Scalar::Bool))
    );
    assert_eq!(
        checked
            .bindings
            .iter()
            .map(|(_, binding)| binding.constant.clone())
            .collect::<Vec<_>>(),
        [folded(1), folded(1), None, folded(1), folded(0)]
    );
    assert!(
            checked
                .expressions
                .iter()
                .all(|(_, expression)| expression.ty == value_type(Scalar::Bool)
                    && !expression.untyped)
        );
}

#[test]
fn comparisons_follow_operand_types_and_fold_constants() {
    let syntax = parse(
        "fn main() -> void {
                const less = 1 < 2;
                const equal = true == false;
                const ordered = false < true;
                const chained = (1 < 2) == true;
                var value: u8 = 1;
                const runtime = value >= 0;
            }",
    )
    .unwrap();
    let checked = check_root(&syntax).unwrap();
    assert_eq!(
        checked
            .bindings
            .iter()
            .map(|(_, binding)| (binding.ty.clone(), binding.constant.clone()))
            .collect::<Vec<_>>(),
        [
            (value_type(Scalar::Bool), folded(1)),
            (value_type(Scalar::Bool), folded(0)),
            (value_type(Scalar::Bool), folded(1)),
            (value_type(Scalar::Bool), folded(1)),
            (value_type(Scalar::U8), None),
            (value_type(Scalar::Bool), None),
        ]
    );

    for (operator, expected) in [
        ("==", false),
        ("!=", true),
        ("<", true),
        ("<=", true),
        (">", false),
        (">=", false),
    ] {
        let syntax = parse(&format!(
            "fn main() -> void {{ const result = 1 {operator} 2; }}"
        ))
        .unwrap();
        let checked = check_root(&syntax).unwrap();
        assert_eq!(
            checked.bindings.iter().next().unwrap().1.constant,
            Some(Constant::Integer(BigInt::from(expected)))
        );
    }

    accepts("var value: u8 = 1; const result = value < 255;");
    rejects(
        "var value: u8 = 1; const result = value < 256;",
        "256",
        "integer literal out of range for `u8`",
    );
    rejects(
        "var left: u8 = 1; var right: u16 = 1; const result = left == right;",
        "==",
        "comparison operands have different types `u8` and `u16`",
    );
    rejects(
        "const result = 1 == true;",
        "==",
        "comparison operands have different types `int` and `bool`",
    );
}

#[test]
fn logical_expressions_fold_constants_and_keep_runtime_short_circuiting() {
    let syntax = parse(
        "fn main() -> void {
                const conjunction = true && false;
                const disjunction = false || true;
                const negated = !false;
                var runtime = true;
                const guarded = false && runtime;
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
        [folded(0), folded(1), folded(1), None, None]
    );
    let guarded = match syntax.statements[syntax.functions[checked.main].body[4]].kind {
        StatementKind::Binding {
            initializer: Some(initializer),
            ..
        } => initializer,
        _ => unreachable!(),
    };
    assert!(matches!(
        checked.expressions[guarded].value,
        ExpressionValue::Logical {
            operator: LogicalOperator::And,
            ..
        }
    ));
    assert_eq!(checked.expressions[guarded].constant, None);

    rejects(
        "const invalid = false && 1 / 0 == 0;",
        "/",
        "constant `/` divisor is zero",
    );
    rejects(
        "const invalid = true || u8(255) + 1 == 0;",
        "+",
        "constant `+` on `u8` would overflow",
    );
}

#[test]
fn boolean_operators_and_contexts_reject_integer_mixing() {
    for (body, offending, message) in [
        (
            "const value: bool = 1;",
            "1",
            "cannot implicitly convert `int` to `bool`",
        ),
        (
            "const value: int = true;",
            "true",
            "cannot implicitly convert `bool` to `int`",
        ),
        (
            "const value = true + false;",
            "+",
            "`+` requires numeric operands",
        ),
        (
            "const value = !1;",
            "1",
            "logical operand has type `int`, expected `bool`",
        ),
        (
            "const value = true && 1;",
            "1",
            "logical operand has type `int`, expected `bool`",
        ),
        (
            "const value = u8(true);",
            "u8(true)",
            "cannot convert `bool` to `u8`",
        ),
        (
            "exit(true);",
            "true",
            "cannot implicitly convert `bool` to `int`",
        ),
    ] {
        rejects(body, offending, message);
    }
}

#[test]
fn integer_expression_types_follow_operand_rules() {
    let body = "var a: u8 = 1;
             var b: u8 = 2;
             const add = a + 2;
             const reverse = 2 + a;
             const shift = a << u64(3);
             const exact: u16 = 1 + 2;
             const negative: i8 = -128;
             const complemented = ^a;
             const wrapped = a +% 1;
             const wrapped_negative = -%a;
             exit(0);
             const after = a & ^b;";
    let text = format!("fn main() -> void {{ {body} }}");
    let syntax = parse(&text).unwrap();
    let checked = check_root(&syntax).unwrap();
    let types: Vec<_> = checked
        .bindings
        .iter()
        .map(|(_, binding)| binding.ty.clone())
        .collect();
    assert_eq!(
        types,
        [
            Scalar::U8,
            Scalar::U8,
            Scalar::U8,
            Scalar::U8,
            Scalar::U8,
            Scalar::U16,
            Scalar::I8,
            Scalar::U8,
            Scalar::U8,
            Scalar::U8,
            Scalar::U8,
        ]
        .map(value_type)
    );

    for left in Scalar::ALL_INTEGERS {
        let left_name = left.name();
        for right in Scalar::ALL_INTEGERS {
            let right_name = right.name();
            let body = format!(
                "var left: {left_name} = 1; var right: {right_name} = 1; const result = left + right;"
            );
            if left == right {
                accepts(&body);
            } else {
                rejects(
                    &body,
                    "+",
                    &format!(
                        "binary operands have different types `{left_name}` and `{right_name}`"
                    ),
                );
            }

            accepts(&format!(
                "var left: {left_name} = 1; var count: {right_name} = 1; const result = left << count;"
            ));
        }
    }

    for body in [
        "var x: u8 = 1; const y = x + 256;",
        "var x: u8 = 1; const y = 256 + x;",
    ] {
        rejects(body, "256", "integer literal out of range for `u8`");
    }
    for (body, offending, message) in [
        (
            "const x = 1 +% 2;",
            "+%",
            "wrapping arithmetic requires a typed operand",
        ),
        (
            "const x = -%1;",
            "-%",
            "wrapping negation requires a typed operand",
        ),
        (
            "const x = -u8(1);",
            "-",
            "unary `-` is not permitted on `u8`",
        ),
    ] {
        rejects(body, offending, message);
    }
    accepts("var x: u8 = 1; { var x: u16 = 2; const inner = x + 1; } x = x + 1; exit(0);");
    rejects(
        "exit(0); const after = 1 + missing;",
        "missing",
        "unknown binding `missing`",
    );
}

#[test]
fn contextual_types_reach_nested_runtime_integer_expressions() {
    let text = "fn main() -> void {
            var count: uint = 1;
            const arithmetic: u64 = (1 << count) + 1;
            const negated: i64 = -(1 << count);
            const complemented: u16 = ^(1 << count);
        }";
    let syntax = parse(text).unwrap();
    let checked = check_root(&syntax).unwrap();
    assert_eq!(
        checked
            .bindings
            .iter()
            .map(|(_, binding)| binding.ty.clone())
            .collect::<Vec<_>>(),
        [Scalar::Uint, Scalar::U64, Scalar::I64, Scalar::U16].map(value_type)
    );
    assert!(
        checked
            .expressions
            .iter()
            .all(|(_, expression)| !expression.untyped)
    );

    rejects(
        "var count: uint = 1; const invalid: u8 = -(1 << count);",
        "-",
        "unary `-` is not permitted on `u8`",
    );

    let too_large = BigInt::from(Scalar::Int.max()) + 1u8;
    rejects(
        &format!("var count: uint = 1; const invalid = u8.truncate({too_large} << count);"),
        &too_large.to_string(),
        "integer literal out of range for `int`",
    );
}

#[test]
fn nonconstant_untyped_shift_counts_are_concretized_and_range_checked() {
    accepts("var value: u8 = 1; var n = 1; const shifted = value << ((1 << 2) << n);");

    for body in [
        "var value: u8 = 1; var n = 1; const shifted = value << ((1 << 200) << n);",
        "var value: u8 = 1; var n = 1; const shifted = value << ((1 << 63) << n);",
        "exit(0); var value: u8 = 1; var n = 1; const shifted = value << ((1 << 200) << n);",
    ] {
        let text = format!("fn main() -> void {{ {body} }}");
        let syntax = parse(&text).unwrap();
        let error = check_root(&syntax).unwrap_err();
        assert_eq!(error.message, "integer value out of range for `int`");
        assert!(text[error.span].contains("1 <<"));
    }
}

#[test]
fn floating_expression_types_follow_the_operand_rules() {
    for (body, ty, constant) in [
        ("const value = 1.0 + 2.0;", Scalar::F64, binary64(3.0)),
        (
            "const a: f32 = 1.0; const b: f32 = 2.0; const value = a * b;",
            Scalar::F32,
            binary32(2.0),
        ),
        (
            "const a: f32 = 1.5; const value = a - 0.5;",
            Scalar::F32,
            binary32(1.0),
        ),
        (
            "const a: f64 = 1.5; const value = 4 / a;",
            Scalar::F64,
            binary64(4.0 / 1.5),
        ),
        ("const value = -f32(0.5);", Scalar::F32, binary32(-0.5)),
        ("var a = f64(1.0); const value = -a;", Scalar::F64, None),
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
            "const bad = f32(1.0) + f64(2.0);",
            "+",
            "binary operands have different types `f32` and `f64`",
        ),
        (
            "const a: f32 = 1.0; var b: int = 2; const bad = a * b;",
            "*",
            "binary operands have different types `f32` and `int`",
        ),
        (
            "var i: int = 1; const bad = i + 1.0;",
            "1.0",
            "cannot implicitly convert `f64` to `int`",
        ),
        (
            "var i: int = 1; const bad = 1.0 + i;",
            "1.0",
            "cannot implicitly convert `f64` to `int`",
        ),
        (
            "var n = 1; const bad = (1 << n) + 0.5;",
            "(1 << n)",
            "cannot implicitly convert `int` to `f64`",
        ),
        (
            "const bad = 1.0 + true;",
            "+",
            "`+` requires numeric operands",
        ),
        (
            "exit(1.0);",
            "1.0",
            "cannot implicitly convert `f64` to `int`",
        ),
    ] {
        rejects(body, offending, message);
    }
}

#[test]
fn the_integer_only_operators_reject_floating_operands() {
    for operator in ["%", "*%", "+%", "-%", "&", "|", "^", "<<", ">>"] {
        rejects(
            &format!("const bad = 1.0 {operator} 2.0;"),
            operator,
            &format!("integer `{operator}` requires integer operands"),
        );
        rejects(
            &format!("const bad = f64(1.0) {operator} f64(2.0);"),
            operator,
            &format!("integer `{operator}` requires integer operands"),
        );
        rejects(
            &format!("var x: f32 = 1.0; x {operator}= 2.0;"),
            &format!("{operator}="),
            &format!("integer `{operator}=` requires integer operands"),
        );
    }
    for (body, offending) in [
        ("const bad = ^1.0;", "^"),
        ("const bad = -%f64(1.0);", "-%"),
        ("var w: [2]f64 = [0.5...]; const bad = ^w[0];", "^"),
    ] {
        rejects(
            body,
            offending,
            "integer unary operator requires an integer operand",
        );
    }
    rejects(
        "const bad = -true;",
        "-",
        "unary `-` requires a numeric operand",
    );
    rejects(
        "var a: [2]f64 = [1.0...]; const bad = -a;",
        "-",
        "unary `-` requires a numeric operand",
    );
    rejects(
        "const bad = u8.truncate(1.0);",
        "u8.truncate(1.0)",
        "cannot convert `f64` to `u8`",
    );
}

#[test]
fn floating_comparisons_compare_by_value() {
    for (body, expected) in [
        ("const value = 1.5 == 1.5;", 1),
        ("const value = 1.5 != 1.5;", 0),
        ("const value = 1 < 1.5;", 1),
        ("const value = 2.0 >= 2;", 1),
        ("const value = f32(1.0) < f32(2.0);", 1),
        ("const value = f64(1.0) <= f64(1.0);", 1),
        ("const value = f64(3.0) > f64(4.0);", 0),
        ("const value = f64(1.0) == 1;", 1),
        // A positive and a negative zero hold different bits and compare
        // equal, which no ordering comparison separates either.
        ("const value = -f64(0.0) == f64(0.0);", 1),
        ("const value = -f32(0.0) < f32(0.0);", 0),
        ("const value = -f32(0.0) >= f32(0.0);", 1),
    ] {
        assert_eq!(
            checked_bindings(&format!("fn main() -> void {{ {body} }}"))
                .last()
                .unwrap()
                .clone(),
            (value_type(Scalar::Bool), folded(expected)),
            "{body}"
        );
    }
    for (body, offending, message) in [
        (
            "const bad = f32(1.0) == f64(1.0);",
            "==",
            "comparison operands have different types `f32` and `f64`",
        ),
        (
            "const bad = f64(1.0) == true;",
            "true",
            "cannot implicitly convert `bool` to `f64`",
        ),
        (
            "var i: int = 1; const bad = i < 1.5;",
            "1.5",
            "cannot implicitly convert `f64` to `int`",
        ),
    ] {
        rejects(body, offending, message);
    }
}

#[test]
fn floating_arrays_hold_and_compare_their_elements() {
    let bindings = checked_bindings(
        "fn main() -> void {
            const signed: [2]f32 = [-f32(0.0), 1.0];
            const unsigned: [2]f32 = [0.0, 1.0];
            const same = signed == unsigned;
            const whole: [2]f32 = [1, 2];
            const filled: [3]f64 = [0.5...];
        }",
    );
    assert_eq!(
        bindings[0],
        (
            array_type(2, value_type(Scalar::F32)),
            binary32_array(&[-0.0, 1.0])
        )
    );
    assert_eq!(
        bindings[1],
        (
            array_type(2, value_type(Scalar::F32)),
            binary32_array(&[0.0, 1.0])
        )
    );
    assert_ne!(bindings[0].1, bindings[1].1);
    assert_eq!(bindings[2], (value_type(Scalar::Bool), folded(1)));
    assert_eq!(bindings[3].1, binary32_array(&[1.0, 2.0]));
    assert_eq!(
        bindings[4],
        (
            array_type(3, value_type(Scalar::F64)),
            Some(Constant::Array(vec![
                Constant::Float(Float::Binary64(
                    0.5f64.to_bits()
                ));
                3
            ]))
        )
    );
    accepts_source(
        "fn main() -> void {
            var weights: [2]f64 = [0.5, 1.5];
            weights[0] = 2.5;
            weights[1] += 1.0;
            for weight in weights { exit(0); }
            exit(len(weights));
        }",
    );
    rejects(
        "const bad: [2]f32 = [0.5, 16777217];",
        "16777217",
        "integer literal not representable in `f32`",
    );
}

#[test]
fn floating_values_pass_through_calls_and_returns() {
    accepts_source(
        "fn scale(factor: f32, count: int) -> f32 {
            var total: f32 = 0.0;
            var left = count;
            for left > 0 { total += factor; left = left - 1; }
            return total;
        }
        fn main() -> void { const scaled = scale(0.5, 2); exit(0); }",
    );
    accepts_source(
        "fn half() -> f64 { return 0.5; }
        fn main() -> void { const value = half() + 1.0; exit(0); }",
    );
    rejects_source(
        "fn scale(factor: f32) -> void {}
        fn main() -> void { scale(16777217); }",
        "16777217",
        "integer literal not representable in `f32`",
    );
    rejects_source(
        "fn scale(factor: f32) -> void {}
        fn main() -> void { var value: f64 = 1.0; scale(value); }",
        "value",
        "cannot implicitly convert `f64` to `f32`",
    );
    rejects_source(
        "fn count() -> int { return 1.5; }
        fn main() -> void {}",
        "1.5",
        "cannot implicitly convert `f64` to `int`",
    );
}

/// One checked struct literal: its written initializers as `(field position,
/// source text)` in source order, then the positions of the fields a `...`
/// filled.
type CheckedLiteral<'a> = (Vec<(usize, &'a str)>, Vec<usize>);

/// Every struct literal of a checked program, in the order they were checked.
fn checked_struct_literals(source: &str) -> Vec<CheckedLiteral<'_>> {
    let syntax = parse(source).unwrap();
    let checked = check_root(&syntax).unwrap();
    checked
        .expressions
        .iter()
        .filter_map(|(_, expression)| match &expression.value {
            ExpressionValue::Struct {
                initializers,
                filled,
                ..
            } => Some((
                initializers
                    .iter()
                    .map(|&(ordinal, value)| {
                        (ordinal, &source[syntax.expressions[value].span.clone()])
                    })
                    .collect(),
                filled.iter().map(|&(ordinal, _)| ordinal).collect(),
            )),
            _ => None,
        })
        .collect()
}

#[test]
fn a_struct_literal_takes_the_type_it_names() {
    assert_eq!(
        checked_bindings(
            "type Point struct { x: int, y: u8 }
             const p = Point { x = 1, y = 2 };
             fn main() -> void {}"
        )[0],
        (
            struct_type(0, "Point"),
            Some(Constant::Struct(vec![
                Constant::Integer(big(1)),
                Constant::Integer(big(2)),
            ]))
        )
    );
    // A literal is checked against its own type, so a destination of another
    // struct type is a mismatch rather than a differently checked literal.
    rejects_source(
        "type Point struct { x: int }
         type Line struct { x: int }
         fn main() -> void { var l: Line = Point { x = 1 }; }",
        "Point { x = 1 }",
        "cannot implicitly convert `Point` to `Line`",
    );
    rejects_source(
        "type Point struct { x: int }
         fn main() -> void { var n: int = Point { x = 1 }; }",
        "Point { x = 1 }",
        "cannot implicitly convert `Point` to `int`",
    );
}

#[test]
fn a_struct_literal_keeps_its_fields_in_source_order() {
    // The written fields stay in the order they are evaluated in, each
    // paired with the position of the field it fills.
    assert_eq!(
        checked_struct_literals(
            "type Point struct { x: int, y: int }
             const p = Point { y = 2, x = 1 };
             fn main() -> void {}"
        ),
        [(vec![(1, "2"), (0, "1")], vec![])]
    );
    assert_eq!(
        checked_bindings(
            "type Point struct { x: int, y: int }
             const p = Point { y = 2, x = 1 };
             fn main() -> void {}"
        )[0]
        .1,
        Some(Constant::Struct(vec![
            Constant::Integer(big(1)),
            Constant::Integer(big(2)),
        ]))
    );
}

#[test]
fn a_fill_gives_every_unlisted_field_its_recursive_zero_value() {
    assert_eq!(
        checked_bindings(
            "type Inner struct { flag: bool, ratio: f32 }
             type Outer struct { first: int, inner: Inner, row: [2]u8 }
             const o = Outer { first = 7, ... };
             fn main() -> void {}"
        )[0],
        (
            struct_type(1, "Outer"),
            Some(Constant::Struct(vec![
                Constant::Integer(big(7)),
                Constant::Struct(vec![
                    Constant::Integer(big(0)),
                    Constant::Float(Float::Binary32(0)),
                ]),
                Constant::Array(vec![Constant::Integer(big(0)), Constant::Integer(big(0))]),
            ]))
        )
    );
    // `...` may stand alone, and it may follow a complete field list.
    assert_eq!(
        checked_struct_literals(
            "type Point struct { x: int, y: int }
             const zero = Point { ... };
             const both = Point { x = 1, y = 2, ... };
             fn main() -> void {}"
        ),
        [(vec![], vec![0, 1]), (vec![(0, "1"), (1, "2")], vec![])]
    );
}

#[test]
fn aggregate_fills_stay_within_the_expanded_constant_limit() {
    rejects_source(
        "type Huge struct { values: [1000001]int }
         const value = Huge { ... };
         fn main() -> void {}",
        "Huge",
        "aggregate initializer exceeds compiler limit of 1000000 values",
    );
    rejects_root(
        "fn main() -> void { var values: [1000001]int = «[0...]»; }",
        "aggregate initializer exceeds compiler limit of 1000000 values",
    );
    // The bound counts every scalar one initializer expands, so nested fills
    // and sibling omitted fields are counted together rather than one at a
    // time.
    rejects_root(
        "fn main() -> void { var x: [4][4][300000]u8 = «[[[0...]...]...]»; }",
        "aggregate initializer exceeds compiler limit of 1000000 values",
    );
    rejects_source(
        "type Wide struct { a: [900000]u8, b: [900000]u8, c: [900000]u8, d: [900000]u8 }
         const value = Wide { ... };
         fn main() -> void {}",
        "Wide",
        "aggregate initializer exceeds compiler limit of 1000000 values",
    );
    accepts_source(
        "fn main() -> void { var x: [4][4][60000]u8 = [[[0...]...]...]; exit(int(x[0][0][0])); }",
    );
}

#[test]
fn a_struct_literal_lists_each_field_of_its_type_once() {
    let point = "type Point struct { x: int, y: u8 }\n";
    rejects_source(
        &format!("{point}const p = Point {{ x = 1, z = 2 }}; fn main() -> void {{}}"),
        "z",
        "struct `Point` has no field `z`",
    );
    rejects_source(
        &format!("{point}const p = Point {{ x = 1, x = 2, y = 3 }}; fn main() -> void {{}}"),
        "x",
        "duplicate field `x` in `Point` literal",
    );
    rejects_source(
        &format!("{point}const p = Point {{ x = 1 }}; fn main() -> void {{}}"),
        "Point { x = 1 }",
        "`Point` literal is missing field `y`",
    );
    // Each initializer is checked against its own field's type.
    rejects_source(
        &format!("{point}const p = Point {{ x = 1, y = 256 }}; fn main() -> void {{}}"),
        "256",
        "integer literal out of range for `u8`",
    );
    rejects_source(
        &format!("{point}const p = Point {{ x = true, y = 2 }}; fn main() -> void {{}}"),
        "true",
        "cannot implicitly convert `bool` to `int`",
    );
    rejects_source(
        "type Row struct { cells: [2]int }
         const r = Row { cells = [1, 2, 3] };
         fn main() -> void {}",
        "[1, 2, 3]",
        "expected 2 elements for `[2]int`, found 3",
    );
}

#[test]
fn a_field_selection_reads_the_field_s_type() {
    let source = "type Point struct { x: int, y: u8 }
         type Line struct { start: Point, cells: [2]Point }
         fn main() -> void {
             const l = Line { start = Point { x = 1, y = 2 }, cells = [Point { x = 3, y = 4 }...] };
             const a = l.start.x;
             const b = l.start.y;
             const c = l.cells[1].y;
         }";
    let bindings = checked_bindings(source);
    let types: Vec<_> = bindings.iter().skip(1).map(|(ty, _)| ty.clone()).collect();
    assert_eq!(
        types,
        [
            value_type(Scalar::Int),
            value_type(Scalar::U8),
            value_type(Scalar::U8),
        ]
    );
    // A selection is not a constant expression, so it folds to nothing even
    // when its operand is constant.
    assert!(bindings[1].1.is_none());
    // An array of structs and a struct of arrays both select through.
    accepts_source(
        "type Point struct { x: int }
         fn main() -> void {
             var g: [2]Point = [Point { x = 1 }...];
             exit(g[0].x);
         }",
    );
    rejects_source(
        "type Point struct { x: int }
         fn main() -> void { const p = Point { x = 1 }; exit(p.z); }",
        "z",
        "struct `Point` has no field `z`",
    );
    for source in [
        "fn main() -> void { const n = 1; exit(n.x); }",
        "fn main() -> void { var a: [2]int = [1, 2]; exit(a.x); }",
    ] {
        let operand = source.rfind("exit(").unwrap() + "exit(".len();
        let syntax = parse(source).unwrap();
        let error = check_root(&syntax).unwrap_err();
        assert_eq!(error.span, operand..operand + 1, "{source}");
        assert!(
            error.message.starts_with("cannot select a field of `"),
            "{}",
            error.message
        );
    }
}

#[test]
fn structs_compare_only_for_equality_and_only_with_their_own_type() {
    accepts_source(
        "type Point struct { x: int, y: [2]u8 }
         fn main() -> void {
             var p = Point { x = 1, y = [2, 3] };
             const q = Point { ... };
             if p == q { exit(1); }
             if p != q { exit(2); }
         }",
    );
    // Constant structs compare field by field, through nested aggregates.
    assert_eq!(
        checked_bindings(
            "type Inner struct { row: [2]int }
             type Outer struct { inner: Inner, ratio: f64 }
             const same = Outer { inner = Inner { row = [1, 2] }, ratio = 0.5 }
                 == Outer { inner = Inner { row = [1, 2] }, ratio = 0.5 };
             const other = Outer { inner = Inner { row = [1, 2] }, ratio = 0.5 }
                 != Outer { inner = Inner { row = [1, 3] }, ratio = 0.5 };
             fn main() -> void {}"
        )
        .iter()
        .map(|(_, constant)| constant.clone())
        .collect::<Vec<_>>(),
        [folded(1), folded(1)]
    );
    rejects_source(
        "type Point struct { x: int }
         type Line struct { x: int }
         fn main() -> void {
             const p = Point { x = 1 };
             const l = Line { x = 1 };
             if p == l { exit(1); }
         }",
        "==",
        "comparison operands have different types `Point` and `Line`",
    );
    rejects_source(
        "type Point struct { x: int }
         fn main() -> void {
             const p = Point { x = 1 };
             if p < p { exit(1); }
         }",
        "<",
        "only `==` and `!=` are defined on `Point`",
    );
    // Arithmetic and logical operators are not defined on a struct.
    rejects_source(
        "type Point struct { x: int }
         fn main() -> void { const p = Point { x = 1 }; exit(p + 1); }",
        "+",
        "`+` requires numeric operands",
    );
}

#[test]
fn a_struct_value_passes_through_bindings_calls_returns_and_iteration() {
    accepts_source(
        "type Point struct { x: int, y: int }
         type Board struct { cells: [2]Point }
         fn shift(p: Point) -> Point { return Point { x = p.x + 1, y = p.y }; }
         const start = Board { cells = [Point { x = 1, y = 2 }, Point { ... }] };
         fn main() -> void {
             var board = start;
             board.cells[0] = shift(board.cells[0]);
             for cell, i in board.cells {
                 if cell == start.cells[i] { continue; }
                 exit(cell.x + cell.y);
             }
             exit(0);
         }",
    );
}
