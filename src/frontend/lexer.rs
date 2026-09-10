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
    /// An integer or floating-point literal candidate, matched loosely so a
    /// malformed spelling stays one token. `is_floating` says which literal
    /// a candidate spells and `valid_number` says whether it is one.
    #[regex("[0-9][a-zA-Z0-9_]*", number)]
    #[regex(r"\.[0-9][a-zA-Z0-9_]*", number)]
    Number,
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

/// Extends a numeric candidate over the decimal point and exponent sign its
/// regex cannot reach: a point separates two runs of word characters and a
/// sign may only follow the `e` or `E` a run ended on.
///
/// A point that begins an `...` fill marker is left alone, so `[0...]` lexes
/// as an integer and an ellipsis rather than as `0.` and two dots.
fn number(lexer: &mut logos::Lexer<'_, Token>) {
    let remainder = lexer.remainder();
    if remainder.starts_with('.') && !remainder[1..].starts_with('.') {
        lexer.bump(1);
        bump_word(lexer);
    }
    while lexer.slice().ends_with(['e', 'E']) && lexer.remainder().starts_with(['+', '-']) {
        lexer.bump(1);
        bump_word(lexer);
    }
}

fn bump_word(lexer: &mut logos::Lexer<'_, Token>) {
    let length = lexer
        .remainder()
        .bytes()
        .take_while(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        .count();
    lexer.bump(length);
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

/// Whether a numeric candidate spells a floating-point literal rather than an
/// integer one. An exponent marker follows the decimal digit run directly, so
/// neither a hexadecimal `e` digit nor a suffix containing one reads as one.
pub(super) fn is_floating(spelling: &str) -> bool {
    let (base, _, suffix) = integer_parts(spelling);
    spelling.contains('.') || (base == 10 && suffix.starts_with(['e', 'E']))
}

/// Rejects a malformed candidate before parsing reads it, so a loose literal
/// regex cannot admit a spelling the language forbids.
pub(super) fn valid_number(spelling: &str) -> Result<(), String> {
    if is_floating(spelling) {
        valid_floating(spelling)
    } else {
        valid_integer(spelling)
    }
}

fn valid_integer(spelling: &str) -> Result<(), String> {
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

fn decimal_digits(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit())
}

fn valid_floating(spelling: &str) -> Result<(), String> {
    let malformed = || Err("malformed floating-point literal".to_owned());
    let (mantissa, exponent) = match spelling.find(['e', 'E']) {
        Some(marker) => (&spelling[..marker], Some(&spelling[marker + 1..])),
        None => (spelling, None),
    };
    let mantissa_digits = match mantissa.split_once('.') {
        // A decimal point may omit the integer part or the fractional part,
        // but not both.
        Some((integer, fraction)) => match (integer, fraction) {
            ("", part) | (part, "") => decimal_digits(part),
            _ => decimal_digits(integer) && decimal_digits(fraction),
        },
        // Without a decimal point, the exponent is what separates a
        // floating-point literal from an integer one.
        None => exponent.is_some() && decimal_digits(mantissa),
    };
    if !mantissa_digits {
        return malformed();
    }
    if let Some(exponent) = exponent {
        let digits = exponent.strip_prefix(['+', '-']).unwrap_or(exponent);
        if !decimal_digits(digits) {
            return malformed();
        }
    }
    Ok(())
}
