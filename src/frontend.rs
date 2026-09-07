use crate::diagnostic::Diagnostic;
use la_arena::Arena;
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
}

#[derive(Debug, Default)]
pub(crate) struct Syntax {
    pub names: Rodeo,
    pub functions: Arena<Function>,
}

pub(crate) fn parse(text: &str) -> Result<Syntax, Diagnostic> {
    let mut lexer = Token::lexer(text);
    let mut syntax = Syntax::default();
    while let Some(token) = lexer.next() {
        let start = lexer.span().start;
        require(&lexer, token, Token::Fn, "expected `fn`")?;
        let name_span = expect(&mut lexer, Token::Name, "expected a function name")?;
        let name = &text[name_span.clone()];
        if matches!(
            name,
            "alloc"
                | "free"
                | "const"
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
        ) {
            return Err(Diagnostic::new(
                name_span,
                "reserved word cannot name a function",
            ));
        }
        expect(&mut lexer, Token::LeftParen, "expected `(`")?;
        expect(
            &mut lexer,
            Token::RightParen,
            "expected `)`; parameters are not supported",
        )?;
        expect(&mut lexer, Token::Arrow, "expected `->`")?;
        expect(&mut lexer, Token::Void, "expected `void`")?;
        expect(&mut lexer, Token::LeftBrace, "expected `{`")?;
        let end = expect(
            &mut lexer,
            Token::RightBrace,
            "expected `}`; only empty bodies are supported",
        )?
        .end;
        syntax.functions.alloc(Function {
            name: syntax.names.get_or_intern(name),
            name_span,
            span: start..end,
        });
    }
    Ok(syntax)
}

fn expect(
    lexer: &mut logos::Lexer<'_, Token>,
    expected: Token,
    message: &str,
) -> Result<Range<usize>, Diagnostic> {
    let token = lexer
        .next()
        .ok_or_else(|| Diagnostic::new(lexer.source().len()..lexer.source().len(), message))?;
    require(lexer, token, expected, message)?;
    Ok(lexer.span())
}

fn require(
    lexer: &logos::Lexer<'_, Token>,
    token: Result<Token, ()>,
    expected: Token,
    message: &str,
) -> Result<(), Diagnostic> {
    match token {
        Ok(token) if token == expected => Ok(()),
        Ok(_) => Err(Diagnostic::new(lexer.span(), message)),
        Err(()) => Err(Diagnostic::new(
            lexer.span(),
            if lexer.slice().starts_with("/*") {
                "unterminated block comment"
            } else {
                "invalid token"
            },
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
