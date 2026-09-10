use super::*;

#[test]
fn public_declarations_and_qualified_names_are_recorded_on_their_nodes() {
    let syntax = parse("pub fn f() -> void {} var private = 0; pub const shared = 1;").unwrap();
    let public: Vec<_> = syntax
        .files
        .iter()
        .flat_map(|file| &file.items)
        .map(|item| match item {
            TopLevelItem::Function { public, .. }
            | TopLevelItem::Binding { public, .. }
            | TopLevelItem::Struct { public, .. } => *public,
        })
        .collect();
    assert_eq!(public, [true, false, true]);

    let syntax = parse("fn main() -> void { const x = plain; const y = a::b; }").unwrap();
    let qualifiers: Vec<_> = syntax
        .expressions
        .iter()
        .filter_map(|(_, expression)| match &expression.kind {
            ExpressionKind::Reference(name) => Some(name.qualifier.is_some()),
            _ => None,
        })
        .collect();
    assert_eq!(qualifiers, [false, true]);
}

#[test]
fn nested_checked_and_truncating_conversions_preserve_their_forms() {
    let source = "fn main() -> void { exit(u8.truncate(i16(u64(42)))); }";
    let syntax = parse(source).unwrap();
    let statement = syntax.functions.iter().next().unwrap().1.body[0];
    let StatementKind::Exit { argument } = syntax.statements[statement].kind else {
        panic!("expected exit")
    };
    let ExpressionKind::Conversion {
        destination,
        truncating,
        operand,
        ..
    } = &syntax.expressions[argument].kind
    else {
        panic!("expected truncating conversion")
    };
    assert_eq!(*destination, Scalar::U8);
    assert!(*truncating);
    let ExpressionKind::Conversion {
        destination,
        truncating,
        operand,
        ..
    } = &syntax.expressions[*operand].kind
    else {
        panic!("expected checked conversion")
    };
    assert_eq!(*destination, Scalar::I16);
    assert!(!*truncating);
    let ExpressionKind::Conversion {
        destination,
        truncating,
        ..
    } = &syntax.expressions[*operand].kind
    else {
        panic!("expected nested checked conversion")
    };
    assert_eq!(*destination, Scalar::U64);
    assert!(!*truncating);

    for (body, message) in [
        ("var value = u8(1;", "expected `)`"),
        ("var value = u8.truncate(1;", "expected `)`"),
        ("var value = u8.(1);", "expected `truncate`"),
        ("var value = u8.truncate 1;", "expected `(`"),
    ] {
        let text = format!("fn main() -> void {{ {body} }}");
        assert_eq!(parse(&text).unwrap_err().message, message, "{body}");
    }
}

#[test]
fn syntax_snapshot() {
    let syntax = parse("fn main() -> void {} ").unwrap();
    let functions: Vec<_> = syntax
        .functions
        .iter()
        .map(|(_, f)| {
            (
                syntax.names.resolve(&f.name),
                f.name_span.clone(),
                f.span.clone(),
            )
        })
        .collect();
    insta::assert_debug_snapshot!(functions, @r#"
        [
            (
                "main",
                3..7,
                0..20,
            ),
        ]
        "#);
}
