//! Token recognition and lexical validation.

use logos::Logos;

#[derive(Logos, Debug, Clone, Copy, PartialEq)]
#[logos(skip r"[ \t\n\r\x0B\x0C]+")]
pub(super) enum Token {
    #[token("extern")]
    Extern,
    #[token("sizeof")]
    Sizeof,
    #[token("fn")]
    Fn,
    #[token("void")]
    Void,
    #[token("const")]
    Const,
    #[token("var")]
    Var,
    #[token("type")]
    Type,
    #[token("struct")]
    Struct,
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
    #[token("len")]
    Len,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("null")]
    Null,
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
    #[token("...")]
    Ellipsis,
    #[token("+=")]
    PlusEquals,
    #[token("-=")]
    MinusEquals,
    #[token("*=")]
    StarEquals,
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
    #[regex(r"//[^\r\n]*", logos::skip, allow_greedy = true)]
    LineComment,
    #[token("/*", block_comment)]
    BlockComment,
    #[token("\"", quoted::<b'"'>)]
    String,
    #[token("'", quoted::<b'\''>)]
    Character,
}

fn number(lexer: &mut logos::Lexer<'_, Token>) {
    let remainder = lexer.remainder();
    if remainder.starts_with('.') {
        lexer.bump(1);
        bump_word(lexer);
    }
    // Only an exponent marker takes a sign; a hexadecimal `e` digit before
    // `+` or `-` ends the literal.
    while is_floating(lexer.slice())
        && lexer.slice().ends_with(['e', 'E'])
        && lexer.remainder().starts_with(['+', '-'])
    {
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

/// Scans a string or character literal to its closing delimiter, skipping the
/// byte after each backslash. A line break or the end of input comes first in
/// an unterminated literal.
fn quoted<const DELIMITER: u8>(lexer: &mut logos::Lexer<'_, Token>) -> Result<(), ()> {
    let bytes = lexer.remainder().as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'\n' | b'\r' => break,
            b'\\' if index + 1 < bytes.len() && !matches!(bytes[index + 1], b'\n' | b'\r') => {
                index += 2;
            }
            byte if byte == DELIMITER => {
                lexer.bump(index + 1);
                return Ok(());
            }
            _ => index += 1,
        }
    }
    lexer.bump(index);
    Err(())
}

/// Decodes the text between a string or character literal's delimiters into
/// the bytes it denotes.
pub(super) fn literal_bytes(body: &str) -> Result<Vec<u8>, String> {
    let bytes = body.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'\\' {
            decoded.push(bytes[index]);
            index += 1;
            continue;
        }
        let escape = body[index + 1..]
            .chars()
            .next()
            .expect("the lexer keeps a backslash's next character in the literal");
        let byte = match escape {
            'n' => b'\n',
            'r' => b'\r',
            't' => b'\t',
            '0' => 0,
            '\\' => b'\\',
            '"' => b'"',
            '\'' => b'\'',
            'x' => {
                let digits = bytes[index + 2..]
                    .iter()
                    .take(2)
                    .take_while(|byte| byte.is_ascii_hexdigit())
                    .count();
                let digits = &body[index + 2..index + 2 + digits];
                if digits.len() != 2 {
                    return Err(format!("invalid escape sequence `\\x{digits}`"));
                }
                decoded.push(u8::from_str_radix(digits, 16).expect("hexadecimal digits"));
                index += 4;
                continue;
            }
            other => return Err(format!("invalid escape sequence `\\{other}`")),
        };
        decoded.push(byte);
        index += 2;
    }
    Ok(decoded)
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
