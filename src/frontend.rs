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
    #[token("var")]
    Var,
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
    #[token(".")]
    Dot,
    #[token("+%")]
    WrappingPlus,
    #[token("-%")]
    WrappingMinus,
    #[token("*%")]
    WrappingStar,
    #[token("<<")]
    ShiftLeft,
    #[token(">>")]
    ShiftRight,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("&")]
    Ampersand,
    #[token("^")]
    Caret,
    #[token("|")]
    Pipe,
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
        mutable: bool,
        name: Spur,
        #[cfg_attr(not(test), expect(dead_code, reason = "preserved in syntax snapshots"))]
        name_span: Range<usize>,
        annotation: Option<TypeAnnotation>,
        initializer: Idx<Expression>,
    },
    Assignment {
        name: Spur,
        name_span: Range<usize>,
        value: Idx<Expression>,
    },
    Block {
        body: Vec<Idx<Statement>>,
    },
    Exit {
        argument: Idx<Expression>,
    },
}

#[derive(Debug)]
pub(crate) struct TypeAnnotation {
    pub name: String,
    pub span: Range<usize>,
}

#[derive(Debug)]
pub(crate) struct Expression {
    pub kind: ExpressionKind,
    pub span: Range<usize>,
    depth: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnaryOperator {
    Negate,
    WrappingNegate,
    Complement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryOperator {
    Multiply,
    Divide,
    Remainder,
    WrappingMultiply,
    Add,
    Subtract,
    WrappingAdd,
    WrappingSubtract,
    ShiftLeft,
    ShiftRight,
    And,
    Xor,
    Or,
}

impl BinaryOperator {
    pub(crate) fn spelling(self) -> &'static str {
        match self {
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::Remainder => "%",
            Self::WrappingMultiply => "*%",
            Self::Add => "+",
            Self::Subtract => "-",
            Self::WrappingAdd => "+%",
            Self::WrappingSubtract => "-%",
            Self::ShiftLeft => "<<",
            Self::ShiftRight => ">>",
            Self::And => "&",
            Self::Xor => "^",
            Self::Or => "|",
        }
    }
}

#[derive(Debug)]
pub(crate) enum ExpressionKind {
    Integer(String),
    Reference(Spur),
    Grouping {
        expression: Idx<Expression>,
    },
    Unary {
        operator: UnaryOperator,
        operator_span: Range<usize>,
        operand: Idx<Expression>,
    },
    Binary {
        operator: BinaryOperator,
        operator_span: Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
    },
    Conversion {
        destination: TypeAnnotation,
        truncating: bool,
        operand: Idx<Expression>,
    },
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
        "const" | "var" | "true" | "false" | "fn" | "void" | "exit" | "bool"
    ) || integer_type(name)
}

const INTEGER_TYPE_NAMES: [&str; 10] = [
    "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "int", "uint",
];

fn integer_type(name: &str) -> bool {
    INTEGER_TYPE_NAMES.contains(&name)
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

pub(crate) const MAX_INTEGER_LITERAL_DIGITS: usize = 4_096;

fn valid_integer(spelling: &str) -> Result<(), &'static str> {
    let (_, digits, suffix) = integer_parts(spelling);
    if digits.is_empty() || !suffix.is_empty() {
        return Err("malformed integer literal");
    }
    if digits.len() > MAX_INTEGER_LITERAL_DIGITS {
        return Err("integer literal exceeds compiler limit of 4096 digits");
    }
    Ok(())
}

struct Parser<'a> {
    lexer: logos::Lexer<'a, Token>,
    current: Option<Token>,
    span: Range<usize>,
    syntax: Syntax,
    nesting: usize,
}

// Bound recursive parsing, checking, and lowering before entering another level.
const MAX_NESTING: usize = 128;

pub(crate) fn parse(text: &str) -> Result<Syntax, Diagnostic> {
    let mut parser = Parser {
        lexer: Token::lexer(text),
        current: None,
        span: 0..0,
        syntax: Syntax::default(),
        nesting: 0,
    };
    parser.advance()?;
    while parser.current.is_some() {
        parser.function()?;
    }
    Ok(parser.syntax)
}

impl Parser<'_> {
    fn enter_nesting(&mut self) -> Result<(), Diagnostic> {
        if self.nesting == MAX_NESTING {
            return Err(self.error("source nesting exceeds compiler limit of 128"));
        }
        self.nesting += 1;
        Ok(())
    }

    fn advance(&mut self) -> Result<(), Diagnostic> {
        let token = self.lexer.next();
        self.span = if token.is_some() {
            self.lexer.span()
        } else {
            self.lexer.source().len()..self.lexer.source().len()
        };
        self.current = match token {
            Some(Ok(Token::Integer)) => {
                if let Err(message) = valid_integer(self.lexer.slice()) {
                    return Err(self.error(message));
                }
                Some(Token::Integer)
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

    fn name(&mut self, expected: &str) -> Result<(Spur, Range<usize>), Diagnostic> {
        if self.current.is_some() && reserved(self.lexer.slice()) {
            return Err(self.error("reserved word cannot be used as an identifier"));
        }
        if self.current != Some(Token::Name) {
            return Err(self.error(expected));
        }
        let name = self.syntax.names.get_or_intern(self.lexer.slice());
        let span = self.span.clone();
        self.advance()?;
        Ok((name, span))
    }

    fn function(&mut self) -> Result<(), Diagnostic> {
        let start = self.expect(Token::Fn, "expected `fn`")?.start;
        let (name, name_span) = self.name("expected a function name")?;
        self.expect(Token::LeftParen, "expected `(`")?;
        self.expect(
            Token::RightParen,
            "expected `)`; parameters are not supported",
        )?;
        self.expect(Token::Arrow, "expected `->`")?;
        self.expect(Token::Void, "expected `void`")?;
        let (body, end) = self.body()?;
        self.syntax.functions.alloc(Function {
            name,
            name_span,
            span: start..end,
            body,
        });
        Ok(())
    }

    fn body(&mut self) -> Result<(Vec<Idx<Statement>>, usize), Diagnostic> {
        self.enter_nesting()?;
        self.expect(Token::LeftBrace, "expected `{`")?;
        let mut body = Vec::new();
        while self.current != Some(Token::RightBrace) {
            if self.current.is_none() {
                return Err(self.error("expected `}`"));
            }
            body.push(self.statement()?);
        }
        let end = self.expect(Token::RightBrace, "expected `}`")?.end;
        self.nesting -= 1;
        Ok((body, end))
    }

    fn statement(&mut self) -> Result<Idx<Statement>, Diagnostic> {
        let start = self.span.start;
        if self.current == Some(Token::LeftBrace) {
            let (body, end) = self.body()?;
            return Ok(self.syntax.statements.alloc(Statement {
                kind: StatementKind::Block { body },
                span: start..end,
            }));
        }
        let kind = match self.current {
            Some(Token::Const) | Some(Token::Var) => {
                let mutable = self.current == Some(Token::Var);
                self.advance()?;
                let (name, name_span) = self.name(if mutable {
                    "expected a binding name after `var`"
                } else {
                    "expected a binding name after `const`"
                })?;
                let annotation = if self.current == Some(Token::Colon) {
                    self.advance()?;
                    Some(self.integer_annotation()?)
                } else {
                    None
                };
                self.expect(Token::Equals, "expected `=`")?;
                let initializer = self.expression()?;
                StatementKind::Binding {
                    mutable,
                    name,
                    name_span,
                    annotation,
                    initializer,
                }
            }
            Some(Token::Name) => {
                let (name, name_span) = self.name("expected an assignment target")?;
                self.expect(Token::Equals, "expected `=`")?;
                let value = self.expression()?;
                StatementKind::Assignment {
                    name,
                    name_span,
                    value,
                }
            }
            Some(Token::Exit) => {
                self.advance()?;
                self.expect(Token::LeftParen, "expected `(`")?;
                if self.current == Some(Token::RightParen) {
                    return Err(self.error("exit requires one argument"));
                }
                let argument = self.expression()?;
                if self.current == Some(Token::Comma) {
                    self.advance()?;
                    if self.current != Some(Token::RightParen) {
                        return Err(self.error("exit takes one argument"));
                    }
                }
                self.expect(Token::RightParen, "expected `)` after exit argument")?;
                StatementKind::Exit { argument }
            }
            _ => return Err(self.error("expected a declaration, assignment, block, or `exit`")),
        };
        let end = self.expect(Token::Semicolon, "expected `;`")?.end;
        Ok(self.syntax.statements.alloc(Statement {
            kind,
            span: start..end,
        }))
    }

    fn integer_annotation(&mut self) -> Result<TypeAnnotation, Diagnostic> {
        if self.current != Some(Token::Name) || !integer_type(self.lexer.slice()) {
            return Err(self.error("expected an integer type"));
        }
        let annotation = TypeAnnotation {
            name: self.lexer.slice().to_owned(),
            span: self.span.clone(),
        };
        self.advance()?;
        Ok(annotation)
    }

    fn expression(&mut self) -> Result<Idx<Expression>, Diagnostic> {
        self.binary_expression(1)
    }

    fn binary_expression(&mut self, minimum_precedence: u8) -> Result<Idx<Expression>, Diagnostic> {
        let mut left = self.unary_expression()?;
        while let Some((operator, precedence)) = self.binary_operator() {
            if precedence < minimum_precedence {
                break;
            }
            let operator_span = self.span.clone();
            self.advance()?;
            let right = self.binary_expression(precedence + 1)?;
            let depth = self.syntax.expressions[left]
                .depth
                .max(self.syntax.expressions[right].depth)
                + 1;
            if self.nesting + depth > MAX_NESTING {
                return Err(Diagnostic::new(
                    operator_span,
                    "source nesting exceeds compiler limit of 128",
                ));
            }
            let span =
                self.syntax.expressions[left].span.start..self.syntax.expressions[right].span.end;
            left = self.syntax.expressions.alloc(Expression {
                kind: ExpressionKind::Binary {
                    operator,
                    operator_span,
                    left,
                    right,
                },
                span,
                depth,
            });
        }
        Ok(left)
    }

    fn binary_operator(&self) -> Option<(BinaryOperator, u8)> {
        Some(match self.current.as_ref()? {
            Token::Star => (BinaryOperator::Multiply, 2),
            Token::Slash => (BinaryOperator::Divide, 2),
            Token::Percent => (BinaryOperator::Remainder, 2),
            Token::WrappingStar => (BinaryOperator::WrappingMultiply, 2),
            Token::ShiftLeft => (BinaryOperator::ShiftLeft, 2),
            Token::ShiftRight => (BinaryOperator::ShiftRight, 2),
            Token::Ampersand => (BinaryOperator::And, 2),
            Token::Plus => (BinaryOperator::Add, 1),
            Token::Minus => (BinaryOperator::Subtract, 1),
            Token::WrappingPlus => (BinaryOperator::WrappingAdd, 1),
            Token::WrappingMinus => (BinaryOperator::WrappingSubtract, 1),
            Token::Caret => (BinaryOperator::Xor, 1),
            Token::Pipe => (BinaryOperator::Or, 1),
            _ => return None,
        })
    }

    fn unary_expression(&mut self) -> Result<Idx<Expression>, Diagnostic> {
        let operator = match self.current {
            Some(Token::Minus) => UnaryOperator::Negate,
            Some(Token::WrappingMinus) => UnaryOperator::WrappingNegate,
            Some(Token::Caret) => UnaryOperator::Complement,
            _ => return self.primary_expression(),
        };
        self.enter_nesting()?;
        let operator_span = self.span.clone();
        self.advance()?;
        let operand = self.unary_expression()?;
        self.nesting -= 1;
        Ok(self.syntax.expressions.alloc(Expression {
            span: operator_span.start..self.syntax.expressions[operand].span.end,
            depth: self.syntax.expressions[operand].depth + 1,
            kind: ExpressionKind::Unary {
                operator,
                operator_span,
                operand,
            },
        }))
    }

    fn primary_expression(&mut self) -> Result<Idx<Expression>, Diagnostic> {
        let span = self.span.clone();
        let kind = if self.current == Some(Token::Integer) {
            let spelling = self.lexer.slice().to_owned();
            self.advance()?;
            ExpressionKind::Integer(spelling)
        } else if self.current == Some(Token::Name) && integer_type(self.lexer.slice()) {
            self.enter_nesting()?;
            let destination = self.integer_annotation()?;
            let truncating = if self.current == Some(Token::Dot) {
                self.advance()?;
                if self.current != Some(Token::Name) || self.lexer.slice() != "truncate" {
                    return Err(self.error("expected `truncate`"));
                }
                self.advance()?;
                true
            } else {
                false
            };
            if !truncating && self.current != Some(Token::LeftParen) {
                return Err(Diagnostic::new(
                    destination.span.clone(),
                    "reserved word cannot be used as an identifier",
                ));
            }
            self.expect(Token::LeftParen, "expected `(`")?;
            let operand = self.expression()?;
            let end = self.expect(Token::RightParen, "expected `)`")?.end;
            self.nesting -= 1;
            return Ok(self.syntax.expressions.alloc(Expression {
                kind: ExpressionKind::Conversion {
                    destination,
                    truncating,
                    operand,
                },
                span: span.start..end,
                depth: self.syntax.expressions[operand].depth + 1,
            }));
        } else if self.current == Some(Token::LeftParen) {
            self.enter_nesting()?;
            self.advance()?;
            let expression = self.expression()?;
            let end = self
                .expect(Token::RightParen, "expected `)` after grouped expression")?
                .end;
            self.nesting -= 1;
            return Ok(self.syntax.expressions.alloc(Expression {
                kind: ExpressionKind::Grouping { expression },
                span: span.start..end,
                depth: self.syntax.expressions[expression].depth + 1,
            }));
        } else {
            let (name, _) = self.name("expected an expression")?;
            ExpressionKind::Reference(name)
        };
        Ok(self.syntax.expressions.alloc(Expression {
            kind,
            span,
            depth: 0,
        }))
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
            project_body(syntax, &function.body, 1, &mut output);
        }
        output
    }

    fn project_body(syntax: &Syntax, body: &[Idx<Statement>], depth: usize, output: &mut String) {
        use std::fmt::Write;
        let indent = "  ".repeat(depth);
        for &id in body {
            let statement = &syntax.statements[id];
            let expression = match &statement.kind {
                StatementKind::Binding {
                    mutable,
                    name,
                    name_span,
                    annotation,
                    initializer,
                } => {
                    write!(
                        output,
                        "{indent}{} {} name={name_span:?} annotation={annotation:?}",
                        if *mutable { "var" } else { "const" },
                        syntax.names.resolve(name)
                    )
                    .unwrap();
                    *initializer
                }
                StatementKind::Assignment {
                    name,
                    name_span,
                    value,
                } => {
                    write!(
                        output,
                        "{indent}assign {} name={name_span:?}",
                        syntax.names.resolve(name)
                    )
                    .unwrap();
                    *value
                }
                StatementKind::Block { body } => {
                    writeln!(output, "{indent}block span={:?}", statement.span).unwrap();
                    project_body(syntax, body, depth + 1, output);
                    continue;
                }
                StatementKind::Exit { argument } => {
                    write!(output, "{indent}exit").unwrap();
                    *argument
                }
            };
            writeln!(output, " span={:?}", statement.span).unwrap();
            project_expression(syntax, expression, depth + 1, output);
        }
    }

    fn project_expression(syntax: &Syntax, id: Idx<Expression>, depth: usize, output: &mut String) {
        use std::fmt::Write;
        let indent = "  ".repeat(depth);
        let expression = &syntax.expressions[id];
        match &expression.kind {
            ExpressionKind::Integer(spelling) => writeln!(
                output,
                "{indent}integer {spelling} span={:?}",
                expression.span
            )
            .unwrap(),
            ExpressionKind::Reference(name) => writeln!(
                output,
                "{indent}reference {} span={:?}",
                syntax.names.resolve(name),
                expression.span
            )
            .unwrap(),
            ExpressionKind::Grouping { expression: inner } => {
                writeln!(output, "{indent}group span={:?}", expression.span).unwrap();
                project_expression(syntax, *inner, depth + 1, output);
            }
            ExpressionKind::Unary {
                operator,
                operator_span,
                operand,
            } => {
                writeln!(
                    output,
                    "{indent}unary {operator:?} operator={operator_span:?} span={:?}",
                    expression.span
                )
                .unwrap();
                project_expression(syntax, *operand, depth + 1, output);
            }
            ExpressionKind::Binary {
                operator,
                operator_span,
                left,
                right,
            } => {
                writeln!(
                    output,
                    "{indent}binary {operator:?} operator={operator_span:?} span={:?}",
                    expression.span
                )
                .unwrap();
                project_expression(syntax, *left, depth + 1, output);
                project_expression(syntax, *right, depth + 1, output);
            }
            ExpressionKind::Conversion {
                destination,
                truncating,
                operand,
            } => {
                writeln!(
                    output,
                    "{indent}{} conversion {} span={:?}",
                    if *truncating { "truncating" } else { "checked" },
                    destination.name,
                    expression.span
                )
                .unwrap();
                project_expression(syntax, *operand, depth + 1, output);
            }
        }
    }

    #[test]
    fn mixed_body_snapshot() {
        let source = "fn main() -> void { const exit_code = 0x2A; var copy: int = exit_code; exit(copy,); const after = missing; } fn helper() -> void { exit(000,); }";
        insta::assert_snapshot!(project(&parse(source).unwrap()));
    }

    #[test]
    fn assignment_and_blocks_snapshot() {
        let source = "fn main() -> void { var x = 1; x = x; {} { const copy = x; { x = 42; } exit(copy); } exit(x); }";
        insta::assert_snapshot!(project(&parse(source).unwrap()));
    }

    #[test]
    fn integer_annotations_snapshot() {
        let mut source = String::from("/* 🌿 */ fn main() -> void {\n");
        for name in INTEGER_TYPE_NAMES {
            source.push_str(&format!("var value: {name} = 0x2A;\n"));
            source.push_str(&format!("{{ const copy: /* type */ {name} = value; }}\n"));
        }
        source.push('}');
        insta::assert_snapshot!(project(&parse(&source).unwrap()));
    }

    #[test]
    fn integer_operator_precedence_and_grouping_snapshot() {
        let source = "fn main() -> void { const high = 1 * 2 / 3 % 4 *% 5 << 6 >> 7 & 8; const low = 9 + 10 - 11 +% 12 -% 13 | 14 ^ 15; const shift = 1 + 2 << 3; const add_or = 1 | 2 + 3; const and_not = 1 & ^2; const unary = -^-%u8(1); const grouping = (1 + 2) * (3 - 4); const and_negative = 7 & -2; }";
        insta::assert_snapshot!(project(&parse(source).unwrap()));
    }

    #[test]
    fn comments_may_touch_integer_operators() {
        let source = "fn main() -> void { const x = ^/*a*/1/*b*/+%/*c*/2<<// d\n3; }";
        let syntax = parse(source).unwrap();
        assert_eq!(syntax.expressions.len(), 6);
    }

    #[test]
    fn removed_operator_spellings_are_separate_tokens() {
        let tokens = Token::lexer("&+ &- &* &^")
            .map(|token| token.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            tokens,
            [
                Token::Ampersand,
                Token::Plus,
                Token::Ampersand,
                Token::Minus,
                Token::Ampersand,
                Token::Star,
                Token::Ampersand,
                Token::Caret,
            ]
        );
    }

    #[test]
    fn malformed_annotations_report_the_offending_token() {
        for spelling in [
            "size",
            "uintptr",
            "bool",
            "void",
            "f32",
            "i128",
            "u7",
            "int_value",
            "i",
            "u",
            "42",
            "=",
            ";",
            ":",
            "",
        ] {
            let prefix = "/* 🌿 */ fn main() -> void { const x: ";
            let source = format!("{prefix}{spelling}");
            let error = parse(&source).unwrap_err();
            assert_eq!(error.message, "expected an integer type", "{spelling}");
            assert_eq!(error.span, prefix.len()..source.len(), "{spelling}");
        }
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
        } = &syntax.expressions[argument].kind
        else {
            panic!("expected truncating conversion")
        };
        assert_eq!(destination.name, "u8");
        assert!(*truncating);
        let ExpressionKind::Conversion {
            destination,
            truncating,
            operand,
        } = &syntax.expressions[*operand].kind
        else {
            panic!("expected checked conversion")
        };
        assert_eq!(destination.name, "i16");
        assert!(!*truncating);
        let ExpressionKind::Conversion {
            destination,
            truncating,
            ..
        } = &syntax.expressions[*operand].kind
        else {
            panic!("expected nested checked conversion")
        };
        assert_eq!(destination.name, "u64");
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
    fn malformed_statements_report_the_offending_token() {
        // The marker surrounds the expected diagnostic span, including EOF.
        for marked in [
            "var «=» 1;",
            "const «;»",
            "var x = «;»",
            "var x: «=» 1;",
            "var x: «size» = 1;",
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
            "x «.»field = 1;",
            "x «[»0] = 1;",
            "const x = 1 + «;»",
            "const x = 1 &«+» 2;",
            "const x = 1 &«*» 2;",
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
    fn reserved_names_agree_in_all_name_positions() {
        for name in [
            "const", "var", "true", "false", "fn", "void", "exit", "i8", "i16", "i32", "i64", "u8",
            "u16", "u32", "u64", "int", "uint", "bool",
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

        for name in [
            "alloc", "free", "f32", "f64", "rune", "str", "uintptr", "size",
        ] {
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
            ("exit(1.5);", "expected `)` after exit argument"),
            ("exit(1, 2);", "exit takes one argument"),
            ("var x = int();", "expected an expression"),
            ("var x = 1 + ;", "expected an expression"),
            ("var x = 1 + * 2;", "expected an expression"),
            ("var x = ();", "expected an expression"),
            ("var x = (1 + 2;", "expected `)` after grouped expression"),
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
            let checked = crate::semantic::check(&syntax).unwrap();
            crate::ir::lower(checked).verify().unwrap();
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
        crate::ir::lower(crate::semantic::check(&syntax).unwrap())
            .verify()
            .unwrap();
        let binary = format!("fn main() -> void {{ exit({}1); }}", "1 + ".repeat(127));
        let syntax = parse(&binary).unwrap();
        crate::ir::lower(crate::semantic::check(&syntax).unwrap())
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
