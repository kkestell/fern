use super::*;

#[test]
fn mixed_body_snapshot() {
    let source = "fn main() -> void { const exit_code = 0x2A; var copy: int = exit_code; exit(copy,); const after = missing; } fn helper() -> void { exit(000,); }";
    insta::assert_snapshot!(projected(source));
}

#[test]
fn interleaved_top_level_bindings_snapshot() {
    let source = "const start: int = 40; fn main() -> void { var local = start; exit(local); } var counter = start + 2; fn helper() -> void {} const last = counter;";
    insta::assert_snapshot!(projected(source));
}

#[test]
fn function_signatures_calls_and_returns_snapshot() {
    let source = "fn mark(digit: int, flag: bool,) -> int { return digit; } fn nothing() -> void { return; } fn main() -> void { mark(1, true,); nothing(); const value = mark(mark(2, false), true) + mark(3, true); if value > 0 && true { return; } }";
    insta::assert_snapshot!(projected(source));
}

#[test]
fn malformed_function_signatures_calls_and_returns_report_the_offending_token() {
    for (marked, message) in [
        (
            "fn f(«,» value: int) -> void {}",
            "expected a parameter name",
        ),
        (
            "fn f(value «int») -> void {}",
            "expected `:` after parameter name",
        ),
        ("fn f(value: «)» -> void {}", "expected a type"),
        ("fn f(value: «void» ) -> void {}", "expected a type"),
        (
            "fn f(value: int «other»: bool) -> void {}",
            "expected `)` after parameter list",
        ),
        (
            "fn f(value: int «->» void {})",
            "expected `)` after parameter list",
        ),
        ("fn f(value: int) -> «{»", "expected a type"),
        (
            "fn f(value: int) -> int { return «+»; }",
            "expected an expression",
        ),
        ("fn f() -> void { target(«,»); }", "expected an expression"),
        (
            "fn f() -> void { target(1 «2»); }",
            "expected `,` or `)` after call argument",
        ),
        (
            "fn f() -> void { target(1«;» }",
            "expected `,` or `)` after call argument",
        ),
        ("fn f() -> void { target(1) «}»", "expected `;`"),
        ("fn f() -> void { return 1 «}»", "expected `;`"),
    ] {
        let prefix = "/* 🌿 */ ";
        let start = marked.find('«').unwrap();
        let end = marked.find('»').unwrap() - '«'.len_utf8();
        let source = format!("{prefix}{}", marked.replace(['«', '»'], ""));
        let error = parse(&source).unwrap_err();
        assert_eq!(error.message, message, "{marked}");
        assert_eq!(
            error.span,
            prefix.len() + start..prefix.len() + end,
            "{marked}"
        );
    }
}

#[test]
fn modules_imports_and_qualified_names_snapshot() {
    let source = "\
use fmt;
use network::http;
use fs::{flag, mode,};
pub const limit: int = 4;
pub var total = 0;
const private = 1;
fn helper() -> int { return private; }
pub fn main() -> void {
    http::serve(fmt::width(limit) + helper());
    http::total = limit;
    http::total += 2;
    const value = http::total;
    exit(value);
}
";
    insta::assert_snapshot!(projected(source));
}

#[test]
fn module_files_parse_into_one_syntax_with_file_local_spans_snapshot() {
    let sources = SourceMap::from_named_texts(&[
        ("first.fern", "use fmt;\nconst base = 1;\n"),
        ("second.fern", "pub fn helper() -> int { return base; }\n"),
    ]);
    let syntax = crate::frontend::parser::parse(&sources).unwrap();
    insta::assert_snapshot!(project(&sources, &syntax));

    // The second file's spans start past the first file's text and its
    // one-byte gap, so a span identifies the file it points into.
    let base = sources.files()[1].base;
    assert_eq!(base, sources.files()[0].text.len() + 1);
    let helper = syntax.functions.iter().next().unwrap().1;
    assert!(helper.name_span.start >= base);
    assert_eq!(sources.index_at(helper.name_span.start).0, 1);
    assert_eq!(sources.index_at(base - 1).0, 0);
}

#[test]
fn malformed_modules_imports_and_qualified_names_report_the_offending_token() {
    for (marked, message) in [
        (
            "«pub» use fmt;",
            "`pub` is not permitted on a `use` declaration",
        ),
        (
            "fn main() -> void {} «use» fmt;",
            "`use` declarations must precede the first declaration",
        ),
        (
            "fn main() -> void { «pub» var x = 1; }",
            "`pub` is not permitted on a local declaration",
        ),
        ("use «::»fmt;", "expected a module name after `use`"),
        ("use fmt «as» f;", "expected `;` after import path"),
        ("use fmt«»", "expected `;` after import path"),
        ("use fmt::«;»", "expected a name after `::`"),
        (
            "use fmt::«const»;",
            "reserved word cannot be used as an identifier",
        ),
        ("use fs::{«}»;", "expected an imported name"),
        ("use fs::{flag,«,»};", "expected an imported name"),
        (
            "use fs::{flag«::»mode};",
            "expected `}` after imported names",
        ),
        ("fn a«::»b() -> void {}", "expected `(`"),
        (
            "fn f(a«::»b: int) -> void {}",
            "expected `:` after parameter name",
        ),
        (
            "fn main() -> void { var «a»::b = 1; }",
            "a declaration without an initializer requires a type annotation",
        ),
        (
            "fn main() -> void { for :a«::»b {} }",
            "expected an expression",
        ),
        ("fn main() -> void { const x = a::b«::»c; }", "expected `;`"),
        (
            "fn main() -> void { a::«=» 1; }",
            "expected a name after `::`",
        ),
        (
            "fn main() -> void { a::«(»1); }",
            "expected a name after `::`",
        ),
    ] {
        let prefix = "/* 🌿 */ ";
        let start = marked.find('«').unwrap();
        let end = marked.find('»').unwrap() - '«'.len_utf8();
        let source = format!("{prefix}{}", marked.replace(['«', '»'], ""));
        let error = parse(&source).unwrap_err();
        assert_eq!(error.message, message, "{marked}");
        assert_eq!(
            error.span,
            prefix.len() + start..prefix.len() + end,
            "{marked}"
        );
    }
}

#[test]
fn statements_are_rejected_at_top_level() {
    for marked in [
        "fn main() -> void {} «exit»(0);",
        "fn main() -> void {} «value» = 1;",
        "fn main() -> void {} «{» exit(0); }",
        "fn main() -> void {} «42»;",
        "fn main() -> void {} «;»",
    ] {
        let start = marked.find('«').unwrap();
        let end = marked.find('»').unwrap() - '«'.len_utf8();
        let source = marked.replace(['«', '»'], "");
        let error = parse(&source).unwrap_err();
        assert_eq!(error.message, "expected a top-level declaration");
        assert_eq!(error.span, start..end, "{marked}");
    }
}

#[test]
fn a_declaration_without_an_initializer_parses_from_its_annotation() {
    for source in [
        "var x: int; fn main() -> void {}",
        "const x: int; fn main() -> void {}",
        "fn main() -> void { var x: [2]int; const y: int; }",
    ] {
        let syntax = parse(source).unwrap();
        assert!(
            syntax.statements.iter().all(|(_, statement)| matches!(
                statement.kind,
                StatementKind::Binding {
                    annotation: Some(_),
                    initializer: None,
                    ..
                }
            )),
            "{source}"
        );
    }
}

#[test]
fn malformed_top_level_bindings_report_the_offending_token() {
    for (source, message, span) in [
        ("var = 1;", "expected a binding name after `var`", 4..5),
        (
            "const value int = 1;",
            "a declaration without an initializer requires a type annotation",
            6..11,
        ),
        ("var value: void = 1;", "expected a type", 11..15),
        ("const value = ;", "expected an expression", 14..15),
        ("var value = 1", "expected `;`", 13..13),
    ] {
        let error = parse(source).unwrap_err();
        assert_eq!(error.message, message, "{source}");
        assert_eq!(error.span, span, "{source}");
    }
}

#[test]
fn assignment_and_blocks_snapshot() {
    let source = "fn main() -> void { var x = 1; x = x; {} { const copy = x; { x = 42; } exit(copy); } exit(x); }";
    insta::assert_snapshot!(projected(source));
}

#[test]
fn integer_annotations_snapshot() {
    let mut source = String::from("/* 🌿 */ fn main() -> void {\n");
    for name in Scalar::ALL_INTEGERS.map(Scalar::name) {
        source.push_str(&format!("var value: {name} = 0x2A;\n"));
        source.push_str(&format!("{{ const copy: /* type */ {name} = value; }}\n"));
    }
    source.push('}');
    insta::assert_snapshot!(projected(&source));
}

#[test]
fn integer_operator_precedence_and_grouping_snapshot() {
    let source = "fn main() -> void { const high = 1 * 2 / 3 % 4 *% 5 << 6 >> 7 & 8; const low = 9 + 10 - 11 +% 12 -% 13 | 14 ^ 15; const shift = 1 + 2 << 3; const add_or = 1 | 2 + 3; const and_not = 1 & ^2; const unary = -^-%u8(1); const grouping = (1 + 2) * (3 - 4); const and_negative = 7 & -2; }";
    insta::assert_snapshot!(projected(source));
}

#[test]
fn floating_literals_snapshot() {
    let source = "fn main() -> void { const whole = 1.0; const fraction = .5; const trailing = 2.; const exponent = 1e3; const scaled = 6.02e-23; const upper = 1E+10; const negated = -1.5; const sum = 1.5 + .5 * 2.; const grouped = (1.0 + 2.5) / 0.5; const compared = 1.0 < 2.0; const mixed = 1 + 0.5; var annotated: int = 1.0; }";
    insta::assert_snapshot!(projected(source));
}

#[test]
fn floating_annotations_and_conversions_snapshot() {
    let source = "fn scale(factor: f32, weights: [2]f64) -> f64 { var narrow: f32 = factor; var matrix: [2][3]f32 = weights; const widened = f64(factor); const narrowed = f32(weights[0]); return f64(1); }";
    insta::assert_snapshot!(projected(source));
}

#[test]
fn a_floating_type_has_no_truncating_conversion() {
    for name in ["f32", "f64"] {
        let prefix = "/* 🌿 */ fn main() -> void { const x = ";
        let source = format!("{prefix}{name}.truncate(1); }}");
        let error = parse(&source).unwrap_err();
        assert_eq!(
            error.message,
            format!("`{name}` has no truncating conversion")
        );
        assert_eq!(error.span, prefix.len()..prefix.len() + name.len());
    }
}

#[test]
fn boolean_expressions_and_control_flow_snapshot() {
    let source = "fn main() -> void { var ready: bool = true; const stopped = false; const result = 1 + 2 < 4 && !stopped || ready == false; if ready { exit(1); } else if stopped { exit(2); } else {} for { break; } for ready { continue; } for :rows var i: int = 0; i < 4; i += 1 { if i >= 2 { break :rows; } } for cursor = 0; cursor != 2; cursor = cursor + 1 { continue; } }";
    insta::assert_snapshot!(projected(source));
}

#[test]
fn every_compound_assignment_operator_parses() {
    let operators = [
        "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "+%=", "-%=", "*%=",
    ];
    let body = operators
        .iter()
        .map(|operator| format!("value {operator} 1;"))
        .collect::<Vec<_>>()
        .join(" ");
    let syntax = parse(&format!("fn main() -> void {{ var value = 0; {body} }}")).unwrap();
    assert_eq!(syntax.statements.len(), operators.len() + 1);
    for &statement in &syntax.functions.iter().next().unwrap().1.body[1..] {
        assert!(matches!(
            syntax.statements[statement].kind,
            StatementKind::CompoundAssignment { .. }
        ));
    }
}

#[test]
fn every_comparison_operator_parses() {
    for operator in ["==", "!=", "<", "<=", ">", ">="] {
        let source = format!("fn main() -> void {{ const result = left {operator} right; }}");
        let syntax = parse(&source).unwrap();
        let statement = syntax.functions.iter().next().unwrap().1.body[0];
        let StatementKind::Binding {
            initializer: Some(initializer),
            ..
        } = syntax.statements[statement].kind
        else {
            panic!("expected binding")
        };
        assert!(matches!(
            syntax.expressions[initializer].kind,
            ExpressionKind::Comparison { .. }
        ));
    }
}

#[test]
fn malformed_control_flow_reports_the_offending_token() {
    for (marked, message) in [
        ("if «{»}", "expected an expression"),
        ("if true «;»", "expected `{`"),
        ("if true {} else «exit»(0);", "expected `{`"),
        ("for «;»", "expected an expression"),
        (
            "for var i = 0 «i» < 2; i = i + 1 {}",
            "expected `;` after for initializer",
        ),
        ("for var i = 0; «;» i = i + 1 {}", "expected an expression"),
        (
            "for var i = 0; i < 2; «{»}",
            "expected an assignment after second `;`",
        ),
        (
            "for var i = 0; i < 2; «i» + 1 {}",
            "expected an assignment after second `;`",
        ),
        ("for :«{»}", "expected a loop label after `:`"),
        ("break :«;»", "expected a label after `:`"),
        ("continue«}»", "expected `;`"),
        (
            "for i «+=» 1; true; i = i + 1 {}",
            "for initializer does not permit compound assignment",
        ),
    ] {
        let prefix = "fn main() -> void { ";
        let start = marked.find('«').unwrap();
        let end = marked.find('»').unwrap() - '«'.len_utf8();
        let source = format!("{prefix}{}}}", marked.replace(['«', '»'], ""));
        let error = parse(&source).unwrap_err();
        assert_eq!(error.message, message, "{marked}");
        assert_eq!(
            error.span,
            prefix.len() + start..prefix.len() + end,
            "{marked}"
        );
    }
}

#[test]
fn boolean_and_control_flow_nesting_obeys_the_source_limit() {
    let accepted_logical = format!(
        "fn main() -> void {{ const value = {}true; }}",
        "true && ".repeat(127)
    );
    parse(&accepted_logical).unwrap();
    let rejected_logical = format!(
        "fn main() -> void {{ const value = {}true; }}",
        "true && ".repeat(128)
    );
    assert_eq!(
        parse(&rejected_logical).unwrap_err().message,
        "source nesting exceeds compiler limit of 128"
    );

    let chain = |count: usize| {
        format!(
            "fn main() -> void {{ {}if true {{}} }}",
            "if true {} else ".repeat(count - 1)
        )
    };
    parse(&chain(127)).unwrap();
    assert_eq!(
        parse(&chain(128)).unwrap_err().message,
        "source nesting exceeds compiler limit of 128"
    );
}

#[test]
fn malformed_annotations_report_the_offending_token() {
    // An ordinary identifier names a declared type, so only a reserved word
    // that is not a type name and a non-name token are rejected here.
    for spelling in ["void", "42", "=", ";", ":", ""] {
        let prefix = "/* 🌿 */ fn main() -> void { const x: ";
        let source = format!("{prefix}{spelling}");
        let error = parse(&source).unwrap_err();
        assert_eq!(error.message, "expected a type", "{spelling}");
        assert_eq!(error.span, prefix.len()..source.len(), "{spelling}");
    }
}

#[test]
fn malformed_statements_report_the_offending_token() {
    // The marker surrounds the expected diagnostic span, including EOF.
    for marked in [
        "var «=» 1;",
        "const «;»",
        "var x = «;»",
        "var x: «=» 1;",
        "var x: «void» = 1;",
        "var «x» 1;",
        "var x: int «1»;",
        "const x = 1 «}»",
        "exit «0»;",
        "exit(«)»;",
        "exit(«,»);",
        "exit(0, «1»);",
        "exit(0 «1»);",
        "exit(0,«,»);",
        "exit(0«;»",
        "exit(0) «}»",
        "x «1»;",
        "x = «;»",
        "x = 1 «}»",
        "x = 1«»",
        "{ x = 1; «»",
        "{ {} } «»",
        "{}«;»",
        "«1» = 2;",
        "«=» 2;",
        "«)»",
        "x.«=» 1;",
        "const x = 1 + «;»",
        "exit(-«)»);",
        "var x = (1 + 2«;»",
        "var x = 1«»",
        "exit(0,«»",
        "var x:«»",
    ] {
        let prefix = "/* 🌿 */ fn main() -> void { ";
        let start = marked.find('«').unwrap();
        let end = marked.find('»').unwrap() - '«'.len_utf8();
        let source = format!("{prefix}{}", marked.replace(['«', '»'], ""));
        assert_eq!(
            parse(&source).unwrap_err().span,
            prefix.len() + start..prefix.len() + end,
            "{marked}"
        );
    }
}

#[test]
fn parsing_does_not_check_names_or_reachability() {
    let syntax = parse("fn exit_code() -> void { var const_value: int = unknown; exit(const_value); var after = missing; } fn exit_code() -> void {}").unwrap();
    assert_eq!(syntax.functions.len(), 2);
    assert_eq!(syntax.statements.len(), 3);
}

#[test]
fn malformed_expressions_and_calls_have_specific_diagnostics() {
    for (body, message) in [
        ("var = 1;", "expected a binding name after `var`"),
        ("const = 1;", "expected a binding name after `const`"),
        ("var x = ;", "expected an expression"),
        ("exit();", "exit requires one argument"),
        ("exit(1..5);", "expected a field name after `.`"),
        ("exit(1, 2);", "exit takes one argument"),
        ("var x = int();", "expected an expression"),
        ("var x = 1 + ;", "expected an expression"),
        ("var x = 1 + / 2;", "expected an expression"),
        ("var x = ();", "expected an expression"),
        ("var x = (1 + 2;", "expected `)` after grouped expression"),
        // Two decimal points make two literals, not one candidate, and a
        // lone point between them selects a field.
        ("var x = 1.5.5;", "expected `;`"),
        ("var x = 1..5;", "expected a field name after `.`"),
    ] {
        let error = parse(&format!("fn main() -> void {{ {body} }}")).unwrap_err();
        assert_eq!(error.message, message, "{body}");
    }
}

#[test]
fn nesting_limit_protects_parsing_checking_and_lowering() {
    for (blocks, conversions) in [(127, 0), (0, 127), (63, 64)] {
        let text = format!(
            "fn main() -> void {{ var x = 42; {}exit({}x{});{} }}",
            "{".repeat(blocks),
            "int(".repeat(conversions),
            ")".repeat(conversions),
            "}".repeat(blocks),
        );
        let syntax = parse(&text).unwrap();
        let checked = crate::semantic::namespaces::check_root(&syntax).unwrap();
        crate::ir::lower::lower(checked).verify().unwrap();
    }
    for (blocks, conversions) in [(128, 0), (0, 128), (64, 64), (100_000, 0), (0, 100_000)] {
        let text = format!(
            "fn main() -> void {{ {}exit({}42{});{} }}",
            "{".repeat(blocks),
            "int(".repeat(conversions),
            ")".repeat(conversions),
            "}".repeat(blocks),
        );
        let error = parse(&text).unwrap_err();
        assert_eq!(
            error.message,
            "source nesting exceeds compiler limit of 128"
        );
        assert!(["{", "int"].contains(&&text[error.span]));
    }

    let grouped = format!(
        "fn main() -> void {{ exit({}42{}); }}",
        "(".repeat(127),
        ")".repeat(127),
    );
    let syntax = parse(&grouped).unwrap();
    crate::ir::lower::lower(crate::semantic::namespaces::check_root(&syntax).unwrap())
        .verify()
        .unwrap();
    let binary = format!("fn main() -> void {{ exit({}1); }}", "1 + ".repeat(127));
    let syntax = parse(&binary).unwrap();
    crate::ir::lower::lower(crate::semantic::namespaces::check_root(&syntax).unwrap())
        .verify()
        .unwrap();
    for source in [
        format!(
            "fn main() -> void {{ exit({}42{}); }}",
            "(".repeat(128),
            ")".repeat(128),
        ),
        format!("fn main() -> void {{ exit({}42); }}", "-".repeat(128)),
        format!("fn main() -> void {{ exit({}1); }}", "1 + ".repeat(128)),
    ] {
        assert_eq!(
            parse(&source).unwrap_err().message,
            "source nesting exceeds compiler limit of 128"
        );
    }
}

#[test]
fn array_types_literals_indexing_and_iteration_snapshot() {
    let source = "\
fn rows(grid: [2][3]int) -> [2]int {
    var out: [2]int = [0...];
    for v, i in grid {
        out[i] = v[0];
    }
    return out;
}
fn main() -> void {
    var a: [3]int = [1, 2, 3,];
    const inferred: [_]int = [0...];
    const seeded: [4]int = [1, 2, 3, 0...];
    const board: [2][3]int = [[1...]...];
    var grid: [2][3]int = [[1, 2, 3], [4, 5, 6]];
    const r = 0;
    const c = 1;
    const cell = grid[r][c];
    const first = rows(grid)[0];
    const negated = -a[0];
    const length = len(a);
    const trailing = len(a,);
    grid[r][c] = 7;
    a[r] += 1;
    for :outer t in a {
        continue :outer;
    }
    for t, i in a {
        break;
    }
}
";
    insta::assert_snapshot!(projected(source));
}

#[test]
fn malformed_array_syntax_reports_the_offending_token() {
    for (marked, message) in [
        ("const x = [«]»;", "expected an array element"),
        ("const x = [1,«,»2];", "expected an expression"),
        (
            "const x = [1 «2»];",
            "expected `,` or `]` after array element",
        ),
        ("const x = [«...»];", "expected an expression"),
        ("var x: [3] «=» 1;", "expected a type"),
        ("var x: [3]«void» = 1;", "expected a type"),
        (
            "var x: [_ «+» 1]int = 1;",
            "expected `]` after array length",
        ),
        ("const x = a[«]»;", "expected an expression"),
        ("a[0 «=» 1;", "expected `]` after index"),
        (
            "const x = «len»;",
            "reserved word cannot be used as an identifier",
        ),
        ("const x = len(«)»;", "len requires one argument"),
        ("const x = len(a, «b»);", "len takes one argument"),
        (
            "for v, «in» a {}",
            "reserved word cannot be used as an identifier",
        ),
        ("for v «i» in a {}", "expected `{`"),
        ("for v, i «a» in b {}", "expected `in`"),
        (
            "for «in» a {}",
            "reserved word cannot be used as an identifier",
        ),
        ("for v in «{»}", "expected an expression"),
    ] {
        let prefix = "/* 🌿 */ fn main() -> void { ";
        let start = marked.find('«').unwrap();
        let end = marked.find('»').unwrap() - '«'.len_utf8();
        let source = format!("{prefix}{}", marked.replace(['«', '»'], ""));
        let error = parse(&source).unwrap_err();
        assert_eq!(error.message, message, "{marked}");
        assert_eq!(
            error.span,
            prefix.len() + start..prefix.len() + end,
            "{marked}"
        );
    }
}

#[test]
fn array_nesting_obeys_the_source_limit() {
    // A function body already holds one level, so 127 more are accepted.
    let indexes = |count: usize| {
        format!(
            "fn main() -> void {{ const x = a{}; }}",
            "[0]".repeat(count)
        )
    };
    let literals = |count: usize| {
        format!(
            "fn main() -> void {{ const x = {}1{}; }}",
            "[".repeat(count),
            "]".repeat(count),
        )
    };
    let annotations = |count: usize| {
        format!(
            "fn main() -> void {{ var x: {}int = 1; }}",
            "[1]".repeat(count)
        )
    };
    for build in [indexes, literals, annotations] {
        parse(&build(127)).unwrap();
        assert_eq!(
            parse(&build(128)).unwrap_err().message,
            "source nesting exceeds compiler limit of 128"
        );
    }
}

#[test]
fn struct_declarations_literals_and_selection_snapshot() {
    let source = "\
use geom;
type Point struct {
    x: int,
    y: int,
}
pub type Shape struct {
    origin: Point,
    corners: [2]Point,
    label: geom::Label,
    scale: f64,
}
fn make(p: Point) -> Point {
    return p;
}
fn main() -> void {
    var origin = Point { x = 1, y = 2 };
    const zeroed = Point { ... };
    const partial = Point { x = 3, ... };
    const nested = Shape {
        origin = Point { x = 4, y = 5 },
        corners = [Point { ... }...],
        label = geom::Label { text = 6 },
        scale = 0.5,
    };
    const value = make(Point { x = 7, y = 8 }).x;
    const deep = nested.corners[0].y;
    var grid: [2]Shape = [nested...];
    grid[0].origin.x = 9;
    grid[1].corners[1].y += 10;
    origin.x = origin.y;
}
";
    insta::assert_snapshot!(projected(source));
}

#[test]
fn a_struct_literal_in_a_condition_is_parenthesized() {
    for body in [
        "if ready {}",
        "for ready {}",
        "for v in a {}",
        "for var i = 0; i < limit; i = i + step {}",
        "if (Point { x = 1 }) == p {}",
        "if (Point { x = 1 } == p) {}",
        "for (Point { x = 1 }) == p {}",
        "for v in (Point { x = 1 }) {}",
        "for var p = Point { x = 1 }; ready; p = p {}",
        "for var i = 0; i < limit; i = (Point { x = 1 }) {}",
    ] {
        parse(&format!("fn main() -> void {{ {body} }}"))
            .unwrap_or_else(|error| panic!("{body}: {error:?}"));
    }

    // A direct literal's brace reads as the statement body's brace, so the
    // header ends at it and the literal's fields become statements.
    for body in [
        "if Point { x = 1 } == p {}",
        "for Point { x = 1 } == p {}",
        "for v in Point { x = 1 } {}",
        "for var i = 0; i < limit; i = Point { x = 1 } {}",
    ] {
        let error = parse(&format!("fn main() -> void {{ {body} }}")).unwrap_err();
        assert_eq!(error.message, "expected `;`", "{body}");
    }
}

#[test]
fn malformed_struct_declarations_report_the_offending_token() {
    for (marked, message) in [
        (
            "type «struct» { x: int }",
            "reserved word cannot be used as an identifier",
        ),
        (
            "type Point «int»;",
            "named types other than structs are not yet implemented",
        ),
        (
            "type Point «{» x: int }",
            "named types other than structs are not yet implemented",
        ),
        (
            "type Point struct «(» x: int )",
            "expected `{` after `struct`",
        ),
        ("type Point struct {«}»", "expected a field declaration"),
        ("type Point struct { «=» }", "expected a field name"),
        (
            "type Point struct { x«,» y: int }",
            "expected `:` after field name",
        ),
        ("type Point struct { x: «}» }", "expected a type"),
        ("type Point struct { x: int, «,» }", "expected a field name"),
        (
            "type Point struct { x: int«»",
            "expected `}` after struct fields",
        ),
        (
            "type Point struct { x: int }«;»",
            "expected a top-level declaration",
        ),
    ] {
        let prefix = "/* 🌿 */ ";
        let start = marked.find('«').unwrap();
        let end = marked.find('»').unwrap() - '«'.len_utf8();
        let source = format!("{prefix}{}", marked.replace(['«', '»'], ""));
        let error = parse(&source).unwrap_err();
        assert_eq!(error.message, message, "{marked}");
        assert_eq!(
            error.span,
            prefix.len() + start..prefix.len() + end,
            "{marked}"
        );
    }
}

#[test]
fn malformed_struct_literals_and_selections_report_the_offending_token() {
    for (marked, message) in [
        ("const p = Point {«}»;", "expected a field initializer"),
        ("const p = Point { «=» 1 };", "expected a field name"),
        (
            "const p = Point { «const» = 1 };",
            "reserved word cannot be used as an identifier",
        ),
        (
            "const p = Point { x «1» };",
            "expected `=` after field name",
        ),
        ("const p = Point { x = «}» };", "expected an expression"),
        (
            "const p = Point { x = 1 «2» };",
            "expected `,` or `}` after field initializer",
        ),
        (
            "const p = Point { ...«,» x = 1 };",
            "expected `}` after field initializers",
        ),
        (
            "const p = Point { x = 1, ...«,» };",
            "expected `}` after field initializers",
        ),
        (
            "const p = Point { x = 1«»",
            "expected `,` or `}` after field initializer",
        ),
        ("const v = p.«=»;", "expected a field name after `.`"),
        (
            "const v = p.«len»;",
            "reserved word cannot be used as an identifier",
        ),
        ("p[0].«[»1] = 2;", "expected a field name after `.`"),
    ] {
        let prefix = "/* 🌿 */ fn main() -> void { ";
        let start = marked.find('«').unwrap();
        let end = marked.find('»').unwrap() - '«'.len_utf8();
        let source = format!("{prefix}{}", marked.replace(['«', '»'], ""));
        let error = parse(&source).unwrap_err();
        assert_eq!(error.message, message, "{marked}");
        assert_eq!(
            error.span,
            prefix.len() + start..prefix.len() + end,
            "{marked}"
        );
    }
}

#[test]
fn struct_nesting_obeys_the_source_limit() {
    // A function body already holds one level, so 127 more are accepted.
    let literals = |count: usize| {
        format!(
            "fn main() -> void {{ const x = {}1{}; }}",
            "Point { f = ".repeat(count),
            " }".repeat(count),
        )
    };
    let selections =
        |count: usize| format!("fn main() -> void {{ const x = a{}; }}", ".f".repeat(count));
    let mixed = |count: usize| {
        format!(
            "fn main() -> void {{ const x = a{}; }}",
            ".f[0]".repeat(count / 2)
        )
    };
    for build in [literals, selections, mixed] {
        parse(&build(127)).unwrap();
        assert_eq!(
            parse(&build(128)).unwrap_err().message,
            "source nesting exceeds compiler limit of 128"
        );
    }
}

#[test]
fn pointer_types_address_of_and_dereference_snapshot() {
    let source = "\
type Node struct {
    value: int,
    next: *Node,
    cells: *[3]int,
    slots: [3]*int,
}
fn head(list: *const Node, table: *const *Node) -> *int {
    return null;
}
fn main() -> void {
    var x = 7;
    var mask = 3;
    const p: *int = &x;
    const frozen: *const int = null;
    var pp: **Node = null;
    var node: Node = Node { ... };
    const element = &node.slots[0];
    const reached = *node.next.value;
    const anded = x & mask;
    const sum = *p + 1;
    const same = p == null;
    head(null, null);
    pp = null;
    *p = 42;
    **pp = 1;
    (*pp).value = 2;
    *node.slots[0] = 3;
    *head(null, null) = 4;
    *p += 5;
    for *p = 0; x < 1; *p += 1 {}
    for *p {}
}
";
    insta::assert_snapshot!(projected(source));
}

#[test]
fn a_pointer_returning_call_may_head_an_implicit_assignment() {
    parse(
        "type Point struct { x: int }
         fn point(pointer: *Point) -> *Point { return pointer; }
         fn values(pointer: *[2]int) -> *[2]int { return pointer; }
         fn main() -> void {
             var p: Point;
             var a: [2]int;
             point(&p).x = 1;
             values(&a)[0] += 1;
         }",
    )
    .unwrap();
}

#[test]
fn malformed_pointer_syntax_reports_the_offending_token() {
    for (marked, message) in [
        ("var x: *«;»", "expected a type"),
        ("var x: *const «;»", "expected a type"),
        ("var x: **«1»;", "expected a type"),
        ("var x: *«void»;", "expected a type"),
        ("const v = &«;»", "expected an expression"),
        ("const v = *«;»", "expected an expression"),
        ("const v = &«»", "expected an expression"),
        (
            "var «null» = 1;",
            "reserved word cannot be used as an identifier",
        ),
        (
            "const «null» = 1;",
            "reserved word cannot be used as an identifier",
        ),
        ("* «=» 1;", "expected an expression"),
        ("*p «1»;", "expected `=`"),
    ] {
        let prefix = "/* 🌿 */ fn main() -> void { ";
        let start = marked.find('«').unwrap();
        let end = marked.find('»').unwrap() - '«'.len_utf8();
        let source = format!("{prefix}{}", marked.replace(['«', '»'], ""));
        let error = parse(&source).unwrap_err();
        assert_eq!(error.message, message, "{marked}");
        assert_eq!(
            error.span,
            prefix.len() + start..prefix.len() + end,
            "{marked}"
        );
    }
}

#[test]
fn pointer_nesting_obeys_the_source_limit() {
    // A function body already holds one level, so 127 more are accepted.
    let annotation =
        |count: usize| format!("fn main() -> void {{ var x: {}int; }}", "*".repeat(count));
    let dereferences =
        |count: usize| format!("fn main() -> void {{ exit({}42); }}", "*".repeat(count));
    for build in [annotation, dereferences] {
        parse(&build(127)).unwrap();
        assert_eq!(
            parse(&build(128)).unwrap_err().message,
            "source nesting exceeds compiler limit of 128"
        );
    }
}
