//! Floating-point literal, operation, comparison, and conversion emission.

use crate::{
    ir::model::{Operand, Value},
    types::{BinaryOperator, ComparisonOperator, Float, Scalar},
};

use std::fmt::Write;

use super::qbe::*;

/// The QBE spelling of a floating-point value. QBE parses a hexadecimal
/// significand exactly, so the emitted literal is the value that rounded to
/// this format and keeps its signed zero.
pub(super) fn literal(value: Float) -> String {
    format!("{}_{}", qbe_type(value.ty()), hexadecimal(value))
}

/// The parts of a value's interchange format: its sign, its unbiased
/// exponent, and its stored significand.
fn hexadecimal(value: Float) -> String {
    let (significand_bits, exponent_bits): (u32, u32) = match value {
        Float::Binary32(_) => (23, 8),
        Float::Binary64(_) => (52, 11),
    };
    let bits = value.bits();
    let sign = if bits >> (significand_bits + exponent_bits) == 1 {
        "-"
    } else {
        ""
    };
    let exponent = (bits >> significand_bits) & ((1u64 << exponent_bits) - 1);
    let significand = bits & ((1u64 << significand_bits) - 1);
    if exponent == (1u64 << exponent_bits) - 1 {
        // QBE accepts IEEE special values in floating literal syntax.
        return format!("{sign}{}", if significand == 0 { "inf" } else { "nan" });
    }
    // A hexadecimal digit spells four bits, so a binary32 significand needs
    // one padding bit and a binary64 significand needs none.
    let digits = significand_bits.div_ceil(4);
    let fraction = fraction(significand << (digits * 4 - significand_bits), digits);
    let bias = i64::from((1u32 << (exponent_bits - 1)) - 1);
    if exponent == 0 {
        // A zero has no significant digit at all. A subnormal has no leading
        // one, and its exponent is the smallest a normal value has rather
        // than the zero it encodes.
        let exponent = if significand == 0 { 0 } else { 1 - bias };
        return format!("{sign}0x0{fraction}p{exponent:+}");
    }
    format!("{sign}0x1{fraction}p{:+}", exponent as i64 - bias)
}

/// The hexadecimal fraction of a significand, empty when every digit is zero.
/// Dropping trailing zeroes does not change the value.
fn fraction(significand: u64, digits: u32) -> String {
    let digits = format!("{significand:0width$x}", width = digits as usize);
    let trimmed = digits.trim_end_matches('0');
    if trimmed.is_empty() {
        String::new()
    } else {
        format!(".{trimmed}")
    }
}

pub(super) fn emit_unary(text: &mut String, id: usize, ty: Scalar, source: Operand) {
    let class = qbe_type(ty);
    writeln!(text, "    %v{id} ={class} neg {}", operand(source)).unwrap();
}

pub(super) fn emit_binary(
    text: &mut String,
    id: usize,
    ty: Scalar,
    operator: BinaryOperator,
    left: Operand,
    right: Operand,
) {
    let instruction = match operator {
        BinaryOperator::Add => "add",
        BinaryOperator::Subtract => "sub",
        BinaryOperator::Multiply => "mul",
        BinaryOperator::Divide => "div",
        _ => {
            unreachable!("only the four arithmetic operators are defined on floating-point values")
        }
    };
    writeln!(
        text,
        "    %v{id} ={} {instruction} {}, {}",
        qbe_type(ty),
        operand(left),
        operand(right)
    )
    .unwrap();
}

/// QBE's floating-point comparisons are the IEEE 754 ones, so a NaN operand
/// makes every comparison but `!=` false and a signed zero compares equal to
/// the other zero.
pub(super) fn emit_comparison(
    text: &mut String,
    id: usize,
    operator: ComparisonOperator,
    left: Operand,
    right: Operand,
    ty: Scalar,
) {
    writeln!(
        text,
        "    %v{id} =w {}{} {}, {}",
        relation(operator),
        qbe_type(ty),
        operand(left),
        operand(right)
    )
    .unwrap();
}

fn relation(operator: ComparisonOperator) -> &'static str {
    match operator {
        ComparisonOperator::Equal => "ceq",
        ComparisonOperator::NotEqual => "cne",
        ComparisonOperator::Less => "clt",
        ComparisonOperator::LessEqual => "cle",
        ComparisonOperator::Greater => "cgt",
        ComparisonOperator::GreaterEqual => "cge",
    }
}

/// Emits the ordinary IEEE format or numeric conversion, without precision checks.
pub(super) fn emit_conversion(
    emitter: &mut Emitter<'_>,
    id: usize,
    _value: &Value,
    source: Operand,
    source_ty: Scalar,
    destination: Scalar,
) {
    let instruction = if source_ty == destination {
        "copy"
    } else if source_ty.is_floating() && destination.is_floating() {
        if destination == Scalar::F32 {
            "truncd"
        } else {
            "exts"
        }
    } else if destination.is_floating() {
        integer_to_floating(source_ty)
    } else {
        floating_to_integer(source_ty, destination)
    };
    let class = qbe_type(destination);
    if destination.is_integer() && destination.width() < 32 {
        writeln!(
            emitter.text,
            "    %conversion{id}_integer =w {instruction} {}",
            operand(source)
        )
        .unwrap();
        let normalized = match (destination.width(), destination.signed()) {
            (8, true) => "extsb",
            (8, false) => "extub",
            (16, true) => "extsh",
            _ => "extuh",
        };
        writeln!(
            emitter.text,
            "    %v{id} =w {normalized} %conversion{id}_integer"
        )
        .unwrap();
    } else {
        writeln!(
            emitter.text,
            "    %v{id} ={class} {instruction} {}",
            operand(source)
        )
        .unwrap();
    }
}

fn integer_to_floating(source: Scalar) -> &'static str {
    match (source.width() == 64, source.signed()) {
        (false, true) => "swtof",
        (false, false) => "uwtof",
        (true, true) => "sltof",
        (true, false) => "ultof",
    }
}

fn floating_to_integer(source: Scalar, destination: Scalar) -> &'static str {
    match (source, destination.signed()) {
        (Scalar::F32, true) => "stosi",
        (Scalar::F32, false) => "stoui",
        (_, true) => "dtosi",
        (_, false) => "dtoui",
    }
}
