use super::*;
use crate::frontend::lexer::{MAX_INTEGER_LITERAL_DIGITS, is_floating};

#[test]
fn comments_may_touch_integer_operators() {
    let source = "fn main() -> void { const x = ^/*a*/1/*b*/+%/*c*/2<<// d\n3; }";
    let syntax = parse(source).unwrap();
    assert_eq!(syntax.expressions.len(), 6);
}

#[test]
fn unsuffixed_integer_spellings_survive_parsing() {
    for digits in [
        "0",
        "00042",
        "42",
        "0xabcdefABCDEF",
        "0b001010",
        "0o00752",
        "9999999999999999999999999999999999999999999999999999999999",
    ] {
        let source = format!("fn main() -> void {{ exit({digits}); }}");
        let syntax = parse(&source).unwrap();
        let (_, expression) = syntax.expressions.iter().next().unwrap();
        let ExpressionKind::Integer(actual) = &expression.kind else {
            panic!("expected integer")
        };
        assert_eq!(actual, digits);
        assert_eq!(&source[expression.span.clone()], digits);
    }
}

#[test]
fn floating_spellings_survive_parsing() {
    for spelling in [
        "1.0", ".5", "2.", "1e3", "6.02e-23", "0.0", ".0", "9.", "1E3", ".5e+2", "1.e3", "0e0",
        "1E+10", "00.50",
    ] {
        let source = format!("fn main() -> void {{ exit({spelling}); }}");
        let syntax = parse(&source).unwrap();
        let (_, expression) = syntax.expressions.iter().next().unwrap();
        let ExpressionKind::Floating(actual) = &expression.kind else {
            panic!("expected a floating-point literal for {spelling}")
        };
        assert_eq!(actual, spelling);
        assert_eq!(&source[expression.span.clone()], spelling);
    }
}

#[test]
fn a_decimal_point_or_exponent_marker_makes_a_candidate_floating() {
    for spelling in ["1.0", ".5", "2.", "1e3", "6.02e-23", "1E3", "0x1.8"] {
        assert!(is_floating(spelling), "{spelling}");
    }
    // A plain integer spelling is not a floating-point candidate, and neither
    // is a hexadecimal `e` digit or a suffix that merely contains one.
    for spelling in [
        "42",
        "0x2A",
        "0xabcdefABCDEF",
        "0b101010",
        "0o52",
        "42size",
        "0x1p3",
    ] {
        assert!(!is_floating(spelling), "{spelling}");
    }
}

#[test]
fn malformed_floating_literals_cover_the_whole_candidate() {
    for spelling in [
        // Exponents
        "1e", "1E", "1e+", "1e-", "1.0e", ".5e", "1.0e+", "1.0E-", "1e2e3", ".5e2e",
        // Suffixes
        "1.0f", "1.0f32", ".5f64", "1.0p3", // Digit separators
        "1.2_3", "1_0.5", "1.0_", // Hexadecimal, binary, and octal forms
        "0x1.8p3", "0x.8p3", "0b1.1", "0o1.7",
    ] {
        let prefix = "/* é */ fn main() -> void { const x = ";
        let source = format!("{prefix}{spelling}; }}");
        let error = parse(&source).unwrap_err();
        assert_eq!(
            error.span,
            prefix.len()..prefix.len() + spelling.len(),
            "{spelling}"
        );
        assert_eq!(
            error.message, "malformed floating-point literal",
            "{spelling}"
        );
    }
}

#[test]
fn a_fill_marker_keeps_its_element_an_integer() {
    // `0.` is a floating-point literal, so the fill marker's first `.` must
    // not join the element before it.
    // A `.` before another `.` is never a decimal point, so a trailing-dot
    // element needs a space before the marker.
    for (element, expected) in [("0", "0"), ("0.0", "0.0"), ("2. ", "2.")] {
        let source = format!("fn main() -> void {{ var a: [3]int = [{element}...]; }}");
        let syntax = parse(&source).unwrap();
        let (elements, fill) = syntax
            .expressions
            .iter()
            .find_map(|(_, expression)| match &expression.kind {
                ExpressionKind::ArrayLiteral { elements, fill } => Some((elements, fill)),
                _ => None,
            })
            .expect("the initializer is an array literal");
        assert!(fill.is_some(), "{element}");
        let spelling = match &syntax.expressions[elements[0]].kind {
            ExpressionKind::Integer(spelling) | ExpressionKind::Floating(spelling) => spelling,
            other => panic!("expected a numeric literal for {element}, got {other:?}"),
        };
        assert_eq!(spelling, expected, "{element}");
    }
    parse("fn main() -> void { var a: [3][2]int = [[0...]...]; }").unwrap();
}

#[test]
fn integer_literal_digit_limit_is_checked_before_semantic_parsing() {
    let accepted = "9".repeat(MAX_INTEGER_LITERAL_DIGITS);
    parse(&format!(
        "fn main() -> void {{ const value = {accepted}; }}"
    ))
    .unwrap();

    let rejected = "9".repeat(MAX_INTEGER_LITERAL_DIGITS + 1);
    let text = format!("fn main() -> void {{ const value = {rejected}; }}");
    let error = parse(&text).unwrap_err();
    assert_eq!(error.span.len(), rejected.len());
    assert_eq!(
        error.message,
        "integer literal exceeds compiler limit of 4096 digits"
    );
}

#[test]
fn malformed_integers_cover_the_whole_candidate() {
    for spelling in [
        "0x",
        "0b",
        "0o",
        "0xi",
        "0bu8",
        "0oz",
        "0b2",
        "0b102",
        "0o8",
        "0o178",
        "0xG",
        "0x12G",
        "1_000",
        "0x_ff",
        "42foo",
        "42i",
        "42u",
        "42z",
        "42i8",
        "42i16",
        "42i32",
        "42i64",
        "42u8",
        "42u16",
        "42u32",
        "42u64",
        "42int",
        "42uint",
        "42size",
        "42uintptr",
        "42i128",
        "42u7",
        "0XFF",
        "0B10",
        "0O12",
        "0x1int",
    ] {
        let prefix = "/* é */ fn main() -> void { const x = ";
        let source = format!("{prefix}{spelling}; }}");
        let error = parse(&source).unwrap_err();
        assert_eq!(
            error.span,
            prefix.len()..prefix.len() + spelling.len(),
            "{spelling}"
        );
        assert_eq!(error.message, "malformed integer literal", "{spelling}");
    }
}

#[test]
fn reserved_names_agree_in_all_name_positions() {
    for name in [
        "const", "var", "true", "false", "null", "fn", "void", "exit", "i8", "i16", "i32", "i64",
        "u8", "u16", "u32", "u64", "int", "uint", "f32", "f64", "bool", "if", "else", "for",
        "break", "continue", "return", "in", "len", "pub", "use", "type", "struct",
    ] {
        for (prefix, suffix) in [
            ("fn ", "() -> void {}"),
            ("fn main() -> void { var ", " = 0; }"),
            ("fn main() -> void { const ", " = 0; }"),
        ] {
            let error = parse(&format!("{prefix}{name}{suffix}")).unwrap_err();
            assert_eq!(error.span, prefix.len()..prefix.len() + name.len());
            assert!(error.message.contains("reserved word"));
        }
    }

    for name in [
        "const", "var", "fn", "void", "exit", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
        "int", "uint", "f32", "f64", "bool", "if", "else", "for", "break", "continue", "return",
        "in", "len", "pub", "use", "type", "struct",
    ] {
        for (prefix, suffix) in [
            ("fn main() -> void { const x = ", "; }"),
            ("fn main() -> void { exit(", "); }"),
        ] {
            let error = parse(&format!("{prefix}{name}{suffix}")).unwrap_err();
            assert_eq!(error.span, prefix.len()..prefix.len() + name.len());
            assert!(error.message.contains("reserved word"));
        }
    }

    for name in ["alloc", "free", "rune", "str", "uintptr", "size"] {
        parse(&format!("fn {name}() -> void {{}}"))
            .unwrap_or_else(|error| panic!("{name} should not be reserved: {error:?}"));
    }
}

#[test]
fn bodies_ignore_whitespace_and_comments() {
    let source = "fn main() -> void { const x: int = 1; var y = x; exit(y,); }";
    for separator in [
        " ",
        "\t",
        "\n",
        "\r",
        "\u{b}",
        "\u{c}",
        "/* 🌿 /* nested */ é */",
        "// é\n",
        "// é\r",
    ] {
        let syntax = parse(&source.replace(' ', separator)).unwrap();
        assert_eq!(syntax.statements.len(), 3);
        assert_eq!(syntax.expressions.len(), 3);
    }
    for comment in ["/*", "/* é", "/* outer /* inner */"] {
        let prefix = "fn main() -> void { exit(0); ";
        let source = format!("{prefix}{comment}");
        let error = parse(&source).unwrap_err();
        assert_eq!(error.span, prefix.len()..source.len());
        assert_eq!(error.message, "unterminated block comment");
    }
}

#[test]
fn errors_use_byte_spans() {
    for (text, span) in [
        ("/* é */ @", 9..10),
        ("fn main() -> void {", 19..19),
        ("fn main() -> void {} fn", 23..23),
        ("fn main() -> void {} extra", 21..26),
        ("/* é */ fn main() -> void {} 💥", 30..34),
        ("/* outer /* inner */", 0..20),
    ] {
        assert_eq!(parse(text).unwrap_err().span, span, "{text}");
    }
}
