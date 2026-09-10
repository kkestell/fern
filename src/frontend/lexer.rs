//! Token recognition and lexical validation.

use logos::Logos;

#[derive(Logos, Debug, Clone, Copy, PartialEq)]
#[logos(skip r"[ \t\n\r\x0B\x0C]+")]
pub(super) enum Token {
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
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("for")]
    For,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("return")]
    Return,
    #[token("in")]
    In,
    #[token("len")]
    Len,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[regex("[0-9][a-zA-Z0-9_]*")]
    Integer,
    #[token("pub")]
    Pub,
    #[token("use")]
    Use,
    #[token("::")]
    ColonColon,
    #[token(":")]
    Colon,
    #[token("=")]
    Equals,
    #[token("==")]
    EqualsEquals,
    #[token("!=")]
    NotEquals,
    #[token("<=")]
    LessEqual,
    #[token(">=")]
    GreaterEqual,
    #[token("<")]
    Less,
    #[token(">")]
    Greater,
    #[token(";")]
    Semicolon,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("+%")]
    WrappingPlus,
    #[token("+=")]
    PlusEquals,
    #[token("+%=")]
    WrappingPlusEquals,
    #[token("-%")]
    WrappingMinus,
    #[token("-=")]
    MinusEquals,
    #[token("-%=")]
    WrappingMinusEquals,
    #[token("*%")]
    WrappingStar,
    #[token("*=")]
    StarEquals,
    #[token("*%=")]
    WrappingStarEquals,
    #[token("<<")]
    ShiftLeft,
    #[token("<<=")]
    ShiftLeftEquals,
    #[token(">>")]
    ShiftRight,
    #[token(">>=")]
    ShiftRightEquals,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("/=")]
    SlashEquals,
    #[token("%")]
    Percent,
    #[token("%=")]
    PercentEquals,
    #[token("&")]
    Ampersand,
    #[token("&&")]
    LogicalAnd,
    #[token("&=")]
    AmpersandEquals,
    #[token("^")]
    Caret,
    #[token("^=")]
    CaretEquals,
    #[token("|")]
    Pipe,
    #[token("||")]
    LogicalOr,
    #[token("|=")]
    PipeEquals,
    #[token("!")]
    Bang,
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
    #[token("[")]
    LeftBracket,
    #[token("]")]
    RightBracket,
    #[token("...")]
    Ellipsis,
    #[regex(r"//[^\r\n]*", logos::skip, allow_greedy = true)]
    LineComment,
    #[token("/*", block_comment)]
    BlockComment,
}

fn block_comment(lexer: &mut logos::Lexer<'_, Token>) -> Result<logos::Skip, ()> {
    let bytes = lexer.remainder().as_bytes();
    let mut depth = 1;
    let mut index = 0;
    while index + 1 < bytes.len() {
        match &bytes[index..index + 2] {
            b"/*" => {
                depth += 1;
                index += 2;
            }
            b"*/" => {
                depth -= 1;
                index += 2;
                if depth == 0 {
                    lexer.bump(index);
                    return Ok(logos::Skip);
                }
            }
            _ => index += 1,
        }
    }
    lexer.bump(bytes.len());
    Err(())
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
        .take_while(|byte| char::from(*byte).is_digit(base))
        .count();
    (base, &digits[..end], &digits[end..])
}

pub(super) const MAX_INTEGER_LITERAL_DIGITS: usize = 4_096;

pub(super) fn valid_integer(spelling: &str) -> Result<(), String> {
    let (_, digits, suffix) = integer_parts(spelling);
    if digits.is_empty() || !suffix.is_empty() {
        return Err("malformed integer literal".to_owned());
    }
    if digits.len() > MAX_INTEGER_LITERAL_DIGITS {
        return Err(format!(
            "integer literal exceeds compiler limit of {MAX_INTEGER_LITERAL_DIGITS} digits"
        ));
    }
    Ok(())
}
