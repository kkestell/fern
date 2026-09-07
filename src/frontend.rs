use crate::diagnostic::Diagnostic;
use la_arena::{Arena, Idx};
use lasso::{Rodeo, Spur};
use logos::Logos;
use std::ops::Range;

#[derive(Logos, Debug, PartialEq)]
#[logos(skip r"[ \t\n\r\x0B\x0C]+")]
enum Token {
    #[token("fn")]
    Fn,
    #[token("void")]
    Void,
    #[token("const")]
    Const,
    #[token("int")]
    Int,
    #[token("exit")]
    Exit,
    #[regex("[0-9][a-zA-Z0-9_]*")]
    Integer,
    #[token(":")]
    Colon,
    #[token("=")]
    Equals,
    #[token(";")]
    Semicolon,
    #[token(",")]
    Comma,
    #[regex("[a-zA-Z_][a-zA-Z0-9_]*")]
    Name,
    #[token("(")]
    LeftParen,
    #[token(")")]
    RightParen,
    #[token("->")]
    Arrow,
    #[token("{")]
    LeftBrace,
    #[token("}")]
    RightBrace,
    #[regex(r"//[^\r\n]*", logos::skip, allow_greedy = true)]
    LineComment,
    #[token("/*", block_comment)]
    BlockComment,
}

fn block_comment(lexer: &mut logos::Lexer<'_, Token>) -> Result<logos::Skip, ()> {
    let bytes = lexer.remainder().as_bytes();
    let mut depth = 1;
    let mut i = 0;
    while i + 1 < bytes.len() {
        match &bytes[i..i + 2] {
            b"/*" => {
                depth += 1;
                i += 2;
            }
            b"*/" => {
                depth -= 1;
                i += 2;
                if depth == 0 {
                    lexer.bump(i);
                    return Ok(logos::Skip);
                }
            }
            _ => i += 1,
        }
    }
    lexer.bump(bytes.len());
    Err(())
}

#[derive(Debug)]
pub(crate) struct Function {
    pub name: Spur,
    pub name_span: Range<usize>,
    pub span: Range<usize>,
    pub body: Vec<Idx<Statement>>,
}

#[derive(Debug)]
pub(crate) struct Statement {
    pub kind: StatementKind,
    #[cfg_attr(not(test), expect(dead_code, reason = "preserved in syntax snapshots"))]
    pub span: Range<usize>,
}

#[derive(Debug)]
pub(crate) enum StatementKind {
    Binding {
        #[cfg_attr(not(test), expect(dead_code, reason = "preserved in syntax snapshots"))]
        mutable: bool,
        name: Spur,
        #[cfg_attr(not(test), expect(dead_code, reason = "preserved in syntax snapshots"))]
        name_span: Range<usize>,
        int_annotation: Option<Range<usize>>,
        initializer: Idx<Expression>,
    },
    Exit {
        argument: Idx<Expression>,
    },
}

#[derive(Debug)]
pub(crate) struct Expression {
    pub kind: ExpressionKind,
    pub span: Range<usize>,
}

#[derive(Debug)]
pub(crate) enum ExpressionKind {
    Integer(String),
    Reference(Spur),
}

#[derive(Debug, Default)]
pub(crate) struct Syntax {
    pub names: Rodeo,
    pub functions: Arena<Function>,
    pub statements: Arena<Statement>,
    pub expressions: Arena<Expression>,
}

fn reserved(name: &str) -> bool {
    matches!(
        name,
        "alloc"
            | "free"
            | "const"
            | "fn"
            | "void"
            | "exit"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "int"
            | "uint"
            | "f32"
            | "f64"
            | "bool"
            | "rune"
            | "str"
            | "uintptr"
            | "size"
    )
}

pub(crate) fn integer_parts(spelling: &str) -> (u32, &str, &str) {
    let (digits, base) = if let Some(digits) = spelling.strip_prefix("0x") {
        (digits, 16)
    } else if let Some(digits) = spelling.strip_prefix("0b") {
        (digits, 2)
    } else if let Some(digits) = spelling.strip_prefix("0o") {
        (digits, 8)
    } else {
        (spelling, 10)
    };
    let end = digits
        .bytes()
        .take_while(|b| char::from(*b).is_digit(base))
        .count();
    (base, &digits[..end], &digits[end..])
}

fn valid_integer(spelling: &str) -> bool {
    let (_, digits, suffix) = integer_parts(spelling);
    !digits.is_empty()
        && matches!(
            suffix,
            "" | "i" | "u" | "z" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64"
        )
}

struct Parser<'a> {
    lexer: logos::Lexer<'a, Token>,
    current: Option<Token>,
    span: Range<usize>,
    syntax: Syntax,
}

pub(crate) fn parse(text: &str) -> Result<Syntax, Diagnostic> {
    let mut parser = Parser {
        lexer: Token::lexer(text),
        current: None,
        span: 0..0,
        syntax: Syntax::default(),
    };
    parser.advance()?;
    while parser.current.is_some() {
        parser.function()?;
    }
    Ok(parser.syntax)
}

impl Parser<'_> {
    fn advance(&mut self) -> Result<(), Diagnostic> {
        let token = self.lexer.next();
        self.span = if token.is_some() {
            self.lexer.span()
        } else {
            self.lexer.source().len()..self.lexer.source().len()
        };
        self.current = match token {
            Some(Ok(Token::Integer)) if !valid_integer(self.lexer.slice()) => {
                return Err(self.error("malformed integer literal"));
            }
            Some(Ok(token)) => Some(token),
            Some(Err(())) => {
                return Err(self.error(if self.lexer.slice().starts_with("/*") {
                    "unterminated block comment"
                } else {
                    "invalid token"
                }));
            }
            None => None,
        };
        Ok(())
    }

    fn error(&self, message: &str) -> Diagnostic {
        Diagnostic::new(self.span.clone(), message)
    }

    fn expect(&mut self, token: Token, message: &str) -> Result<Range<usize>, Diagnostic> {
        if self.current != Some(token) {
            return Err(self.error(message));
        }
        let span = self.span.clone();
        self.advance()?;
        Ok(span)
    }

    fn name(&mut self) -> Result<(Spur, Range<usize>), Diagnostic> {
        if self.current.is_some() && reserved(self.lexer.slice()) {
            return Err(self.error("reserved word cannot be used as an identifier"));
        }
        if self.current != Some(Token::Name) {
            return Err(self.error("expected a name"));
        }
        let name = self.syntax.names.get_or_intern(self.lexer.slice());
        let span = self.span.clone();
        self.advance()?;
        Ok((name, span))
    }

    fn function(&mut self) -> Result<(), Diagnostic> {
        let start = self.expect(Token::Fn, "expected `fn`")?.start;
        let (name, name_span) = self.name()?;
        self.expect(Token::LeftParen, "expected `(`")?;
        self.expect(
            Token::RightParen,
            "expected `)`; parameters are not supported",
        )?;
        self.expect(Token::Arrow, "expected `->`")?;
        self.expect(Token::Void, "expected `void`")?;
        self.expect(Token::LeftBrace, "expected `{`")?;
        let mut body = Vec::new();
        while self.current != Some(Token::RightBrace) {
            if self.current.is_none() {
                return Err(self.error("expected `}`"));
            }
            body.push(self.statement()?);
        }
        let end = self.expect(Token::RightBrace, "expected `}`")?.end;
        self.syntax.functions.alloc(Function {
            name,
            name_span,
            span: start..end,
            body,
        });
        Ok(())
    }

    fn statement(&mut self) -> Result<Idx<Statement>, Diagnostic> {
        let start = self.span.start;
        let kind = match self.current {
            Some(Token::Const) | Some(Token::Name)
                if self.lexer.slice() == "const" || self.lexer.slice() == "var" =>
            {
                let mutable = self.lexer.slice() == "var";
                self.advance()?;
                let (name, name_span) = self.name()?;
                let int_annotation = if self.current == Some(Token::Colon) {
                    self.advance()?;
                    Some(self.expect(Token::Int, "expected `int`")?)
                } else {
                    None
                };
                self.expect(Token::Equals, "expected `=`")?;
                let initializer = self.expression()?;
                StatementKind::Binding {
                    mutable,
                    name,
                    name_span,
                    int_annotation,
                    initializer,
                }
            }
            Some(Token::Exit) => {
                self.advance()?;
                self.expect(Token::LeftParen, "expected `(`")?;
                let argument = self.expression()?;
                if self.current == Some(Token::Comma) {
                    self.advance()?;
                }
                self.expect(Token::RightParen, "expected `)`; exit takes one argument")?;
                StatementKind::Exit { argument }
            }
            _ => return Err(self.error("expected a declaration or `exit`")),
        };
        let end = self.expect(Token::Semicolon, "expected `;`")?.end;
        Ok(self.syntax.statements.alloc(Statement {
            kind,
            span: start..end,
        }))
    }

    fn expression(&mut self) -> Result<Idx<Expression>, Diagnostic> {
        let span = self.span.clone();
        let kind = if self.current == Some(Token::Integer) {
            let spelling = self.lexer.slice().to_owned();
            self.advance()?;
            ExpressionKind::Integer(spelling)
        } else {
            let (name, _) = self.name()?;
            ExpressionKind::Reference(name)
        };
        Ok(self.syntax.expressions.alloc(Expression { kind, span }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(syntax: &Syntax) -> String {
        use std::fmt::Write;
        let mut output = String::new();
        for (_, function) in syntax.functions.iter() {
            writeln!(
                output,
                "fn {} name={:?} span={:?}",
                syntax.names.resolve(&function.name),
                function.name_span,
                function.span
            )
            .unwrap();
            for &id in &function.body {
                let statement = &syntax.statements[id];
                let expression = match &statement.kind {
                    StatementKind::Binding {
                        mutable,
                        name,
                        name_span,
                        int_annotation,
                        initializer,
                    } => {
                        write!(
                            output,
                            "  {} {} name={name_span:?} int={int_annotation:?}",
                            if *mutable { "var" } else { "const" },
                            syntax.names.resolve(name)
                        )
                        .unwrap();
                        *initializer
                    }
                    StatementKind::Exit { argument } => {
                        write!(output, "  exit").unwrap();
                        *argument
                    }
                };
                writeln!(output, " span={:?}", statement.span).unwrap();
                let expression = &syntax.expressions[expression];
                match &expression.kind {
                    ExpressionKind::Integer(spelling) => {
                        write!(output, "    integer {spelling}").unwrap()
                    }
                    ExpressionKind::Reference(name) => {
                        write!(output, "    reference {}", syntax.names.resolve(name)).unwrap()
                    }
                }
                writeln!(output, " span={:?}", expression.span).unwrap();
            }
        }
        output
    }

    #[test]
    fn mixed_body_snapshot() {
        let source = "fn main() -> void { const exit_code = 0x2Ai; var copy: int = exit_code; exit(copy,); const after = missing; } fn helper() -> void { exit(000,); }";
        insta::assert_snapshot!(project(&parse(source).unwrap()));
    }

    #[test]
    fn integer_spellings_survive_parsing() {
        for digits in [
            "0",
            "00042",
            "42",
            "0xabcdefABCDEF",
            "0b001010",
            "0o00752",
            "9999999999999999999999999999999999999999999999999999999999",
        ] {
            for suffix in [
                "", "i", "u", "z", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
            ] {
                let spelling = format!("{digits}{suffix}");
                let source = format!("fn main() -> void {{ exit({spelling}); }}");
                let syntax = parse(&source).unwrap();
                let (_, expression) = syntax.expressions.iter().next().unwrap();
                let ExpressionKind::Integer(actual) = &expression.kind else {
                    panic!("expected integer")
                };
                assert_eq!(actual, &spelling);
                assert_eq!(&source[expression.span.clone()], spelling);
            }
        }
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
    fn malformed_statements_report_the_offending_token() {
        // The marker surrounds the expected diagnostic span, including EOF.
        for marked in [
            "var «=» 1;",
            "const «;»",
            "var x = «;»",
            "var x: «=» 1;",
            "var x: «u8» = 1;",
            "var x «1»;",
            "var x: int «;»",
            "const x = 1 «}»",
            "exit «0»;",
            "exit(«)»;",
            "exit(«,»);",
            "exit(0, «1»);",
            "exit(0 «1»);",
            "exit(0,«,»);",
            "exit(0«;»",
            "exit(0) «}»",
            "«x» = 1;",
            "«{»}",
            "const x = 1 «+» 2;",
            "exit(«-»1);",
            "var x = «(»1);",
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
    fn reserved_names_agree_in_all_name_positions() {
        for name in [
            "alloc", "free", "const", "fn", "void", "exit", "i8", "i16", "i32", "i64", "u8", "u16",
            "u32", "u64", "int", "uint", "f32", "f64", "bool", "rune", "str", "uintptr", "size",
        ] {
            for (prefix, suffix) in [
                ("fn ", "() -> void {}"),
                ("fn main() -> void { var ", " = 0; }"),
                ("fn main() -> void { const ", " = 0; }"),
                ("fn main() -> void { const x = ", "; }"),
                ("fn main() -> void { exit(", "); }"),
            ] {
                let error = parse(&format!("{prefix}{name}{suffix}")).unwrap_err();
                assert_eq!(error.span, prefix.len()..prefix.len() + name.len());
                assert!(error.message.contains("reserved word"));
            }
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
    fn parsing_does_not_check_names_or_reachability() {
        let syntax = parse("fn exit_code() -> void { var const_value: int = unknown; exit(const_value); var after = missing; } fn exit_code() -> void {}").unwrap();
        assert_eq!(syntax.functions.len(), 2);
        assert_eq!(syntax.statements.len(), 3);
        // `var` has a declaration role but is not reserved by the specification.
        parse("fn var() -> void { const var = 1; exit(var); }").unwrap();
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
}
