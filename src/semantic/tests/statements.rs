use super::*;

#[test]
fn for_in_binds_an_element_and_an_optional_index() {
    for (source, value, index) in [
        (
            "fn main() -> void { var a: [2]u8 = [0...]; for v in a { exit(int(v)); } }",
            value_type(Scalar::U8),
            None,
        ),
        (
            "fn main() -> void { var g: [2][3]int = [[1...]...]; for row, i in g { exit(i); } }",
            array_type(3, value_type(Scalar::Int)),
            Some(value_type(Scalar::Int)),
        ),
    ] {
        let syntax = parse(source).unwrap();
        let checked = check_root(&syntax).unwrap();
        let (_, bindings) = checked.iterations.iter().next().unwrap();
        assert_eq!(checked.bindings[bindings.value].ty, value, "{source}");
        assert!(!checked.bindings[bindings.value].mutable, "{source}");
        assert_eq!(
            bindings
                .index
                .map(|binding| checked.bindings[binding].ty.clone()),
            index,
            "{source}"
        );
    }
    accepts_source(
        "fn main() -> void { var g: [2][3]int = [[1...]...];
             for row in g { for v in row { exit(v); } } }",
    );
    // The body may shadow the bindings, and neither outlives the loop.
    accepts_source(
        "fn main() -> void { var a: [2]int = [1, 2]; for v in a { const v = 9; exit(v); } }",
    );
    rejects_root(
        "fn main() -> void { var a: [2]int = [1, 2]; for v in a {} exit(«v»); }",
        "unknown binding `v`",
    );
    rejects_root(
        "fn main() -> void { var a: [2]int = [1, 2]; for v in a { «v» = 9; } }",
        "cannot assign to immutable binding `v`",
    );
    rejects_root(
        "fn main() -> void { var a: [2]int = [1, 2]; for v, «v» in a {} }",
        "a `for` loop's value and index bindings must have different names",
    );
    rejects_root(
        "fn main() -> void { var x = 1; for v in «x» {} }",
        "`for … in` requires an array, found `int`",
    );
}

#[test]
fn assignment_errors_use_target_or_value_spans_even_after_nested_exit() {
    for (body, offending, message) in [
        (
            "const x = 1; x = 2;",
            "x",
            "cannot assign to immutable binding `x`",
        ),
        (
            "var x = 1; const x = 2; x = 3;",
            "x",
            "cannot assign to immutable binding `x`",
        ),
        (
            "var x = 1; { const x = 2; x = 3; }",
            "x",
            "cannot assign to immutable binding `x`",
        ),
        (
            "const x = 1; { var x = 2; x = 3; } x = 4;",
            "x",
            "cannot assign to immutable binding `x`",
        ),
        ("x = 2;", "x", "unknown binding `x`"),
        ("{ var x = 1; } x = 2;", "x", "unknown binding `x`"),
        ("{ const x = 1; } exit(x);", "x", "unknown binding `x`"),
        ("{ var x = x; }", "x", "unknown binding `x`"),
        (
            "var x = 1; { x = missing; }",
            "missing",
            "unknown binding `missing`",
        ),
        (
            "var x: u8 = 1; { x = 256; }",
            "256",
            "integer literal out of range for `u8`",
        ),
    ] {
        rejects(body, offending, message);
        rejects(&format!("{{ exit(0); }} {body}"), offending, message);
        rejects(&format!("{{ exit(0); {body} }}"), offending, message);
    }
    for body in [
        "const x = 1; var x = x; x = 2;",
        "const x = 1; { var x = x; x = 2; }",
        "var x = 1; { const x = x; } x = 2;",
    ] {
        let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
        check_root(&syntax).unwrap();
    }
}

#[test]
fn returns_check_against_the_declared_result() {
    let text = "fn nothing() -> void { return; }
                    fn early(flag: bool) -> void {
                        if flag {
                            return;
                        }
                        exit(0);
                    }
                    fn falls_through() -> void {}
                    fn narrow() -> u8 { return 200; }
                    fn wide() -> i64 { return 1 + 2; }
                    fn ready() -> bool { return 1 < 2; }
                    fn branching(flag: bool) -> int {
                        if flag {
                            return 1;
                        } else {
                            return 2;
                        }
                    }
                    fn main() -> void {}";
    let syntax = parse(text).unwrap();
    let checked = check_root(&syntax).unwrap();
    let returned = syntax
        .statements
        .iter()
        .filter_map(|(_, statement)| match &statement.kind {
            StatementKind::Return { value: Some(value) } => Some((
                &text[syntax.expressions[*value].span.clone()],
                checked.expressions[*value].ty.clone(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        returned,
        [
            ("200", value_type(Scalar::U8)),
            ("1 + 2", value_type(Scalar::I64)),
            ("1 < 2", value_type(Scalar::Bool)),
            ("1", value_type(Scalar::Int)),
            ("2", value_type(Scalar::Int)),
        ]
    );

    rejects_source(
        "fn main() -> void {} fn nothing() -> void { return 1; }",
        "1",
        "cannot return a value from a `void` function",
    );
    rejects_source(
        "fn main() -> void {} fn total() -> int { return; }",
        "return;",
        "`return` must supply a value of type `int`",
    );
    rejects_source(
        "fn main() -> void {} fn total() -> int { return true; }",
        "true",
        "cannot implicitly convert `bool` to `int`",
    );
    rejects_source(
        "fn main() -> void {} fn narrow() -> u8 { return 256; }",
        "256",
        "integer literal out of range for `u8`",
    );

    // Statements after a `return` are still checked.
    rejects(
        "return; exit(missing);",
        "missing",
        "unknown binding `missing`",
    );
}

#[test]
fn value_returning_functions_must_not_reach_the_end_of_their_body() {
    for body in [
        "if flag { return 1; } else { return 2; }",
        "if flag { return 1; } else if flag { return 2; } else { return 3; }",
        "{ return 1; }",
        "exit(0);",
        "for { }",
        "for { for { break; } }",
        "for :outer { for :inner { break :inner; } }",
        "for { return 1; break; }",
        "if flag { return 1; } else { for { } }",
    ] {
        accepts_source(&format!(
            "fn main() -> void {{}} fn total(flag: bool) -> int {{ {body} }}"
        ));
    }

    for body in [
        "",
        "if flag { return 1; }",
        "if flag { return 1; } else if flag { return 2; }",
        "for flag { return 1; }",
        "for var i = 0; i < 1; i = i + 1 { return 1; }",
        "for { break; }",
        "for { if flag { break; } }",
        "for :outer { for { break :outer; } }",
        "for { { break; } }",
    ] {
        let text = format!("fn main() -> void {{}} fn total(flag: bool) -> int {{ {body} }}");
        rejects_source(
            &text,
            "total",
            "function `total` can reach the end of its body without returning a value",
        );
    }
}

#[test]
fn function_signatures_are_collected_before_call_checking() {
    let text = "fn caller(value: int, flag: bool,) -> int {
                        callee(value, flag);
                        const nested: int = callee(callee(value, flag), flag);
                        return nested;
                     }
                     fn callee(value: int, flag: bool) -> int { return value; }
                     fn recursive(value: int) -> void { recursive(value); }
                     fn mutual_left(value: int) -> int { mutual_right(value); return value; }
                     fn mutual_right(value: int) -> void { mutual_left(value); }
                     fn main() -> void {
                         caller(1, true);
                         callee(2, false);
                         recursive(3);
                         mutual_left(4);
                     }";
    let syntax = parse(text).unwrap();
    let checked = check_root(&syntax).unwrap();

    assert_eq!(checked.function_names.len(), 6);
    assert_eq!(checked.functions.iter().count(), 6);
    assert_eq!(checked.calls.iter().count(), 8);
    assert!(
        checked
            .expressions
            .iter()
            .any(|(_, expression)| matches!(expression.value, ExpressionValue::Call { .. }))
    );
    for (_, signature) in checked.functions.iter() {
        assert!(
            signature
                .parameters
                .iter()
                .all(|parameter| !checked.bindings[*parameter].mutable)
        );
    }
}

#[test]
fn call_arguments_use_parameter_types_and_source_order() {
    let syntax = parse(
        "fn typed(value: u8, flag: bool) -> void {
                 const copy: u8 = value;
                 const ready: bool = flag;
             }
             fn main() -> void {
                 typed(1, true);
                 typed(1 + 2, false);
             }",
    )
    .unwrap();
    let checked = check_root(&syntax).unwrap();
    assert_eq!(
        checked
            .bindings
            .iter()
            .map(|(_, binding)| (binding.ty.clone(), binding.mutable))
            .collect::<Vec<_>>(),
        [
            (value_type(Scalar::U8), false),
            (value_type(Scalar::Bool), false),
            (value_type(Scalar::U8), false),
            (value_type(Scalar::Bool), false),
        ]
    );

    rejects_source(
        "fn typed(value: u8, flag: bool) -> void {} fn main() -> void { typed(256, missing); }",
        "256",
        "integer literal out of range for `u8`",
    );
}

#[test]
fn calls_respect_shadowing_context_and_result_kind() {
    rejects_source(
        "fn target() -> void {} fn main() -> void { var target = 0; target(); }",
        "target",
        "cannot call non-function binding `target`",
    );
    rejects_source(
        "fn target(target: int) -> void { target(); } fn main() -> void {}",
        "target",
        "cannot call non-function binding `target`",
    );
    rejects_source(
        "fn target() -> void {} fn main() -> void { const value = target(); }",
        "target",
        "void function `target` cannot be used as a value",
    );
    rejects_source(
        "fn target(value: int) -> int { return value; } fn main() -> void { target(); }",
        "target",
        "function `target` expects 1 argument, found 0",
    );
    rejects_source(
        "fn main() -> void { missing(); }",
        "missing",
        "unknown function `missing`",
    );
    rejects_source(
        "fn target(first: int, second: bool) -> void {} fn main() -> void { target(1); }",
        "target",
        "function `target` expects 2 arguments, found 1",
    );
    rejects_source(
        "fn target(value: int) -> void {} fn main() -> void { target(true); }",
        "true",
        "cannot implicitly convert `bool` to `int`",
    );
    rejects_source(
        "fn target() -> void {} fn main(value: int) -> void {}",
        "value",
        "`main` must not have parameters",
    );
    rejects_source(
        "fn target() -> void {} fn main() -> int {}",
        "int",
        "`main` must return `void`",
    );
    rejects_source(
        "fn target() -> void {} fn main() -> void { const value = target; }",
        "target",
        "unknown binding `target`",
    );
}

#[test]
fn parameters_are_shadowed_by_local_bindings_and_restored_after_their_scope() {
    let syntax = parse(
        "fn typed(value: u8) -> u8 {
                 { var value: bool = true; }
                 const copy: u8 = value;
                 return copy;
             }
             fn main() -> void {}",
    )
    .unwrap();
    let checked = check_root(&syntax).unwrap();
    let typed = syntax
        .functions
        .iter()
        .find(|(_, function)| !function.parameters.is_empty())
        .map(|(id, _)| id)
        .unwrap();
    let parameter = checked.functions[typed].parameters[0];
    assert_eq!(
        checked
            .bindings
            .iter()
            .map(|(_, binding)| (binding.ty.clone(), binding.mutable))
            .collect::<Vec<_>>(),
        [
            (value_type(Scalar::U8), false),
            (value_type(Scalar::Bool), true),
            (value_type(Scalar::U8), false),
        ]
    );
    let initializer = match &syntax.statements[syntax.functions[typed].body[1]].kind {
        StatementKind::Binding {
            initializer: Some(initializer),
            ..
        } => *initializer,
        _ => unreachable!("the shadowing scope ends before the copy"),
    };
    assert_eq!(
        checked.expressions[initializer].value,
        ExpressionValue::Reference(parameter)
    );
}

#[test]
fn parameters_are_immutable_and_duplicate_names_are_rejected() {
    rejects_source(
        "fn target(value: int, value: bool) -> void {} fn main() -> void {}",
        "value",
        "duplicate parameter name `value`",
    );
    rejects_source(
        "fn target(value: int) -> void { value = 1; } fn main() -> void {}",
        "value",
        "cannot assign to immutable binding `value`",
    );
    rejects_source(
        "fn target(value: int) -> void { value += 1; } fn main() -> void {}",
        "value",
        "cannot assign to immutable binding `value`",
    );
}

#[test]
fn calls_are_not_constant_expressions() {
    // Signatures resolve after module-level initializers, so a call is
    // reported where it is written rather than where the initializer ends.
    for (source, marked) in [
        (
            "fn value() -> int { return 1; } const result = value(); fn main() -> void {}",
            "value()",
        ),
        (
            "fn value() -> int { return 1; } const result = 1 + value(); fn main() -> void {}",
            "value()",
        ),
    ] {
        rejects_source(
            source,
            marked,
            "module-level initializer must be a constant expression",
        );
    }
}

#[test]
fn structured_control_flow_checks_conditions_and_nested_scopes() {
    for body in ["if 1 {}", "for 1 {}"] {
        rejects(body, "1", "cannot implicitly convert `int` to `bool`");
    }

    accepts(
        "var outer = 0;
             if true { var branch = outer; }
             else if false { var branch = outer; }
             else { var branch = outer; }
             for {}
             for false {}
             for var i = outer; i < 3; i += 1 {
                 var i = i;
                 if i == 2 { continue; }
             }
             for outer = 0; outer < 1; outer = outer + 1 {}
             exit(outer);",
    );

    for body in [
        "if true { var hidden = 1; } exit(hidden);",
        "if true {} else { var hidden = 1; } exit(hidden);",
        "for { var hidden = 1; break; } exit(hidden);",
        "for var hidden = 0; hidden < 1; hidden += 1 {} exit(hidden);",
    ] {
        rejects(body, "hidden", "unknown binding `hidden`");
    }

    rejects(
        "for const iterator = 0; iterator < 1; iterator = 1 {}",
        "iterator",
        "cannot assign to immutable binding `iterator`",
    );
}

#[test]
fn loop_control_resolves_enclosing_labels() {
    accepts(
        "var outer = 0;
             for :outer {
                 for :inner {
                     continue;
                     continue :outer;
                     break :inner;
                 }
             }
             for :same { break; }
             for :same { break :same; }",
    );

    for (body, offending, message) in [
        ("break;", "break", "`break` is not inside a loop"),
        (
            "continue :missing;",
            "continue",
            "`continue` is not inside a loop",
        ),
        (
            "for :outer { break :missing; }",
            "missing",
            "unknown enclosing loop label `missing`",
        ),
        (
            "for :same { for :same {} }",
            "same",
            "duplicate enclosing loop label `same`",
        ),
    ] {
        rejects(body, offending, message);
    }
}

#[test]
fn compound_assignments_follow_binary_and_assignment_rules() {
    accepts(
        "var value: u8 = 1;
             value += 1;
             value +%= 255;
             value <<= u16(2);
             for var i: u8 = 0; i < 2; i += 1 {}",
    );
    for (body, offending, message) in [
        (
            "const value = 1; value += 1;",
            "value",
            "cannot assign to immutable binding `value`",
        ),
        (
            "var value = true; value += true;",
            "+=",
            "`+=` requires numeric operands",
        ),
        (
            "var value: u8 = 1; value += u16(1);",
            "+=",
            "binary operands have different types `u8` and `u16`",
        ),
        (
            "var value = 1; value /= 0;",
            "/=",
            "constant `/=` divisor is zero",
        ),
        (
            "var value = 1; value <<= -1;",
            "<<=",
            "constant `<<=` shift count is negative",
        ),
    ] {
        rejects(body, offending, message);
    }
}

#[test]
fn names_require_a_preceding_binding_even_after_exit() {
    for body in [
        "const x = x;",
        "exit(x); const x = 1;",
        "var y = x;",
        "exit(0); exit(x);",
        "const y = 1; exit(0); var x = x;",
    ] {
        // In the forward-reference case the later declaration has the same name.
        let text = format!("/* 🌿 */ fn main() -> void {{ {body} }}");
        let syntax = parse(&text).unwrap();
        let error = check_root(&syntax).unwrap_err();
        let start = if body.starts_with("exit(x)") {
            text.find("exit(x)").unwrap() + 5
        } else {
            text.rfind('x').unwrap()
        };
        assert_eq!(error.span, start..start + 1);
        assert_eq!(error.message, "unknown binding `x`");
    }
}

#[test]
fn floating_bindings_and_assignments_follow_their_format() {
    accepts_source(
        "const ratio = 1.5;
        var scale: f64 = 0.5;
        fn main() -> void {
            var value: f32 = 1.0;
            value = 2.0;
            value = 1;
            value += 0.5;
            value -= 1;
            value *= 2.0;
            value /= 4.0;
            scale = scale * ratio;
            exit(0);
        }",
    );
    assert_eq!(
        checked_bindings("const ratio = 1.5; fn main() -> void {}")[0],
        (value_type(Scalar::F64), binary64(1.5))
    );
    for (body, offending, message) in [
        (
            "var value: f32 = 1.0; value = f64(1.0);",
            "f64(1.0)",
            "cannot implicitly convert `f64` to `f32`",
        ),
        (
            "var value: f32 = 1.0; value = 16777217;",
            "16777217",
            "integer literal not representable in `f32`",
        ),
        (
            "var value: int = 1; value = 1.5;",
            "1.5",
            "cannot implicitly convert `f64` to `int`",
        ),
        (
            "var value: f32 = 1.0; value += true;",
            "+=",
            "`+=` requires numeric operands",
        ),
        (
            "var value: f32 = 1.0; value += f64(1.0);",
            "+=",
            "binary operands have different types `f32` and `f64`",
        ),
        (
            "var value: bool = true; value = 1.0;",
            "1.0",
            "cannot implicitly convert `f64` to `bool`",
        ),
    ] {
        rejects(body, offending, message);
    }
}

/// The resolved steps of every assignment target of a checked program, in the
/// order the statements were checked.
fn checked_target_steps(source: &str) -> Vec<Vec<String>> {
    let syntax = parse(source).unwrap();
    let checked = check_root(&syntax).unwrap();
    checked
        .assignments
        .iter()
        .map(|(_, target)| {
            fn steps(location: &CheckedLocation, found: &mut Vec<String>) {
                match &location.kind {
                    CheckedLocationKind::Binding(_) | CheckedLocationKind::Dereference { .. } => {}
                    CheckedLocationKind::Index { operand, .. } => {
                        steps(operand, found);
                        found.push("index".to_owned());
                    }
                    CheckedLocationKind::Field {
                        operand, ordinal, ..
                    } => {
                        steps(operand, found);
                        found.push(format!("field {ordinal}"));
                    }
                }
            }
            let mut found = Vec::new();
            steps(&target.location, &mut found);
            found
        })
        .collect()
}

#[test]
fn a_field_of_a_mutable_struct_may_be_assigned() {
    accepts_source(
        "type Point struct { x: int, y: u8 }
         fn main() -> void {
             var p = Point { x = 1, y = 2 };
             p.x = 5;
             p.y += 1;
             p = Point { x = 6, y = 7 };
         }",
    );
    // A field of a `const` binding is rejected with the binding itself.
    rejects_source(
        "type Point struct { x: int }
         fn main() -> void { const p = Point { x = 1 }; p.x = 5; }",
        "p",
        "cannot assign to immutable binding `p`",
    );
    rejects_source(
        "type Point struct { x: int }
         fn main() -> void { var p = Point { x = 1 }; p.x = true; }",
        "true",
        "cannot implicitly convert `bool` to `int`",
    );
    rejects_source(
        "type Point struct { x: int }
         fn main() -> void { var p = Point { x = 1 }; p.z = 5; }",
        "z",
        "struct `Point` has no field `z`",
    );
    rejects_source(
        "type Point struct { x: int }
         fn main() -> void { var p = Point { x = 1 }; p += 1; }",
        "+=",
        "`+=` requires numeric operands",
    );
    rejects_source(
        "fn main() -> void { var n = 1; n.x = 5; }",
        "n",
        "cannot select a field of `int`",
    );
}

#[test]
fn an_assignment_target_records_its_index_and_field_steps_in_order() {
    assert_eq!(
        checked_target_steps(
            "type Point struct { x: int, y: int }
             type Cell struct { point: Point }
             type Board struct { rows: [2]Cell, count: int }
             fn main() -> void {
                 var grid: [2]Cell = [Cell { point = Point { ... } }...];
                 var board = Board { rows = grid, count = 0 };
                 const i = 1;
                 grid[i].point.y = 3;
                 board.rows[i].point.x = 4;
                 board.count = 5;
             }"
        ),
        [
            vec![
                "index".to_owned(),
                "field 0".to_owned(),
                "field 1".to_owned()
            ],
            vec![
                "field 0".to_owned(),
                "index".to_owned(),
                "field 0".to_owned(),
                "field 0".to_owned()
            ],
            vec!["field 1".to_owned()],
        ]
    );
    // Every step is checked, so an index out of range or a field of a
    // non-struct is rejected part way along a chain.
    rejects_source(
        "type Cell struct { point: int }
         fn main() -> void {
             var grid: [2]Cell = [Cell { point = 0 }...];
             grid[2].point = 1;
         }",
        "2",
        "index 2 is out of range for `[2]Cell`",
    );
    rejects_source(
        "type Cell struct { point: int }
         fn main() -> void {
             var grid: [2]Cell = [Cell { point = 0 }...];
             grid[0].point.x = 1;
         }",
        "grid",
        "cannot select a field of `int`",
    );
}

#[test]
fn a_declaration_without_an_initializer_takes_its_type_s_zero_value() {
    let source = "type Pair struct { a: int, b: f64 }
         var counter: int;
         const limit: u8;
         fn main() -> void {
             var row: [2]int;
             var pair: Pair;
             exit(counter + int(limit) + row[0] + pair.a);
         }";
    let syntax = parse(source).unwrap();
    let checked = check_root(&syntax).unwrap();
    let zeroes = checked
        .bindings
        .iter()
        .map(|(_, binding)| (binding.ty.clone(), binding.constant.clone()))
        .collect::<Vec<_>>();
    assert_eq!(
        zeroes,
        vec![
            // A module-level `var` keeps its zero out of `Binding::constant`,
            // which would fold it into its use sites.
            (value_type(Scalar::Int), None),
            (value_type(Scalar::U8), folded(0)),
            (array_type(2, value_type(Scalar::Int)), None),
            (struct_type(0, "Pair"), None),
        ]
    );
    // Every declaration written without an initializer records its zero for
    // lowering, whether or not the binding also folds.
    assert_eq!(checked.zero_declarations.len(), 4);
    let values = checked.zero_declarations.values().collect::<Vec<_>>();
    assert!(values.contains(&&Constant::Integer(big(0))));
    assert!(values.contains(&&Constant::Array(vec![
        Constant::Integer(big(0)),
        Constant::Integer(big(0))
    ])));
    assert!(values.contains(&&Constant::Struct(vec![
        Constant::Integer(big(0)),
        Constant::Float(Float::Binary64(0)),
    ])));
}

#[test]
fn a_declaration_without_an_initializer_is_still_a_const_or_a_typed_binding() {
    rejects(
        "const x: int; x = 1;",
        "x",
        "cannot assign to immutable binding `x`",
    );
    // `[_]` has no initializer to take a length from.
    rejects_source(
        "fn main() -> void { var x: [_]int; }",
        "[_]int",
        "`[_]` requires an array-literal initializer",
    );
    accepts("var x: int; x = 1; exit(x);");
}

#[test]
fn indirect_assignment_observes_pointer_mutability() {
    accepts("var value = 1; const p = &value; *p = 2; *p += 1;");
    accepts_source(
        "type Point struct { x: int } fn main() -> void { var point = Point { x = 1 }; const p = &point; p.x = 2; }",
    );
    rejects(
        "var p: *const int = null; *p = 1;",
        "*p",
        "cannot assign to an immutable location",
    );
    rejects(
        "var value = 1; const p = &value; var q: *const int = p; *q = 2;",
        "*q",
        "cannot assign to an immutable location",
    );
    accepts_source(
        "type Point struct { x: int }
         fn point(pointer: *Point) -> *Point { return pointer; }
         fn values(pointer: *[2]int) -> *[2]int { return pointer; }
         fn main() -> void {
             var p: Point;
             var a: [2]int;
             point(&p).x = 1;
             point(&p).x += 1;
             values(&a)[0] = 2;
             values(&a)[0] += 1;
         }",
    );
}

#[test]
fn an_assignment_target_must_be_a_location() {
    // A statement starts an assignment target at a name, a `*`, or a `(`, so
    // those are the heads a non-location target can reach checking through.
    for (body, offending) in [
        ("var x = 1; int(x) = 2;", "int(x)"),
        ("var x = 1; i64.truncate(x) = 2;", "i64.truncate(x)"),
        ("var x = 1; (x + 1) = 2;", "x + 1"),
        ("var a = [1, 2]; (len(a)) = 2;", "len(a)"),
    ] {
        rejects(
            body,
            offending,
            "cannot assign to an expression that is not a location",
        );
    }
    // A grouped location is still a location.
    accepts_source("fn main() -> void { var x = 1; (x) = 2; }");
    accepts_source(
        "type Point struct { x: int }
         fn main() -> void { var p = Point { x = 1 }; (p).x = 2; }",
    );
}
