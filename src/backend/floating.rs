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
        // Arithmetic produces these at run time. A Fern constant expression
        // whose result is not finite is a compile-time error, so no literal
        // reaches here.
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

/// Negation flips a sign bit and cannot trap, so it needs no overflow check
/// and no normalization.
pub(super) fn emit_unary(text: &mut String, id: usize, ty: Scalar, source: Operand) {
    let class = qbe_type(ty);
    writeln!(text, "    %v{id} ={class} neg {}", operand(source)).unwrap();
}

/// The four arithmetic operators round their IEEE 754 result to the operand
/// format, which QBE's operations already do, and none of them traps.
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

/// Emits a conversion with a floating-point type on one side or both.
///
/// Every check runs before the cast it guards, so no QBE conversion ever reads
/// a value outside its destination's range.
pub(super) fn emit_conversion(
    emitter: &mut Emitter<'_>,
    id: usize,
    value: &Value,
    source: Operand,
    source_ty: Scalar,
    destination: Scalar,
) {
    if destination.is_floating() && source_ty.is_floating() {
        emit_format_conversion(emitter, id, value, source, source_ty, destination)
    } else if destination.is_floating() {
        emit_integer_to_floating(emitter, id, value, source, source_ty, destination)
    } else {
        emit_floating_to_integer(emitter, id, value, source, source_ty, destination)
    }
}

/// Converts between the two floating-point formats. Widening is exact.
/// Narrowing keeps the value only when reconstructing it returns the original,
/// which a NaN never does and every other preserved value does.
fn emit_format_conversion(
    emitter: &mut Emitter<'_>,
    id: usize,
    value: &Value,
    source: Operand,
    source_ty: Scalar,
    destination: Scalar,
) {
    let class = qbe_type(destination);
    let source = operand(source);
    if source_ty == destination {
        writeln!(emitter.text, "    %v{id} ={class} copy {source}").unwrap();
        return;
    }
    if source_ty.all_values_fit(destination) {
        writeln!(emitter.text, "    %v{id} ={class} exts {source}").unwrap();
        return;
    }
    let wide = qbe_type(source_ty);
    writeln!(emitter.text, "    %v{id} ={class} truncd {source}").unwrap();
    writeln!(emitter.text, "    %conversion{id}_wide ={wide} exts %v{id}").unwrap();
    writeln!(
        emitter.text,
        "    %conversion{id}_same =w ceq{wide} %conversion{id}_wide, {source}"
    )
    .unwrap();
    writeln!(
        emitter.text,
        "    %conversion{id}_nan =w cuo{wide} {source}, {source}"
    )
    .unwrap();
    writeln!(
        emitter.text,
        "    %conversion{id}_kept =w or %conversion{id}_same, %conversion{id}_nan"
    )
    .unwrap();
    writeln!(
        emitter.text,
        "    %conversion{id}_lost =w ceqw %conversion{id}_kept, 0"
    )
    .unwrap();
    emit_trap(emitter, id, value, "conversion", (source_ty, destination));
}

/// Converts an integer to a floating-point value, keeping it only when
/// converting the result back returns the original integer. A result outside
/// the source type's range rounded away from every value of that type, so it
/// fails the check rather than being cast back.
fn emit_integer_to_floating(
    emitter: &mut Emitter<'_>,
    id: usize,
    value: &Value,
    source: Operand,
    source_ty: Scalar,
    destination: Scalar,
) {
    let class = qbe_type(destination);
    let source = operand(source);
    writeln!(
        emitter.text,
        "    %v{id} ={class} {} {source}",
        integer_to_floating(source_ty)
    )
    .unwrap();
    if source_ty.all_values_fit(destination) {
        return;
    }
    emit_range_check(emitter, id, &format!("%v{id}"), destination, source_ty);
    emit_trap(emitter, id, value, "range", (source_ty, destination));
    writeln!(
        emitter.text,
        "    %conversion{id}_back ={} {} %v{id}",
        qbe_type(source_ty),
        floating_to_integer(destination, source_ty)
    )
    .unwrap();
    writeln!(
        emitter.text,
        "    %conversion{id}_lost =w cne{} %conversion{id}_back, {source}",
        qbe_type(source_ty)
    )
    .unwrap();
    emit_trap(emitter, id, value, "conversion", (source_ty, destination));
}

/// Converts a floating-point value to an integer. The range check rejects a
/// NaN, an infinity, and every value the destination cannot hold, so the cast
/// that follows it is defined; converting the result back rejects a value with
/// a fractional part.
fn emit_floating_to_integer(
    emitter: &mut Emitter<'_>,
    id: usize,
    value: &Value,
    source: Operand,
    source_ty: Scalar,
    destination: Scalar,
) {
    let source = operand(source);
    emit_range_check(emitter, id, &source, source_ty, destination);
    emit_trap(emitter, id, value, "range", (source_ty, destination));
    writeln!(
        emitter.text,
        "    %v{id} ={} {} {source}",
        qbe_type(destination),
        floating_to_integer(source_ty, destination)
    )
    .unwrap();
    writeln!(
        emitter.text,
        "    %conversion{id}_back ={} {} %v{id}",
        qbe_type(source_ty),
        integer_to_floating(destination)
    )
    .unwrap();
    writeln!(
        emitter.text,
        "    %conversion{id}_lost =w cne{} %conversion{id}_back, {source}",
        qbe_type(source_ty)
    )
    .unwrap();
    emit_trap(emitter, id, value, "conversion", (source_ty, destination));
}

/// Rejects a floating-point value that no value of `integer` is, leaving
/// `%conversion{id}_lost` set when the cast to it would be undefined. The
/// bounds are powers of two, so both are exact in either format and one
/// ordered comparison against each also rejects a NaN.
fn emit_range_check(
    emitter: &mut Emitter<'_>,
    id: usize,
    source: &str,
    format: Scalar,
    integer: Scalar,
) {
    let class = qbe_type(format);
    let (minimum, bound) = bounds(format, integer);
    writeln!(
        emitter.text,
        "    %conversion{id}_low =w cge{class} {source}, {}",
        literal(minimum)
    )
    .unwrap();
    writeln!(
        emitter.text,
        "    %conversion{id}_high =w clt{class} {source}, {}",
        literal(bound)
    )
    .unwrap();
    writeln!(
        emitter.text,
        "    %conversion{id}_within =w and %conversion{id}_low, %conversion{id}_high"
    )
    .unwrap();
    writeln!(
        emitter.text,
        "    %conversion{id}_lost =w ceqw %conversion{id}_within, 0"
    )
    .unwrap();
}

/// The smallest value of `integer` and the first power of two above its
/// largest, in `format`. A type's largest value is one below a power of two
/// and may round when a format cannot hold it, so the upper bound is that
/// power of two and the comparison against it is strict.
fn bounds(format: Scalar, integer: Scalar) -> (Float, Float) {
    let minimum = if integer.signed() {
        -(2f64.powi(integer.width() as i32 - 1))
    } else {
        0.0
    };
    let bound = 2f64.powi(integer.width() as i32 - i32::from(integer.signed()));
    match format {
        Scalar::F32 => (
            Float::Binary32((minimum as f32).to_bits()),
            Float::Binary32((bound as f32).to_bits()),
        ),
        _ => (
            Float::Binary64(minimum.to_bits()),
            Float::Binary64(bound.to_bits()),
        ),
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

/// Reports a conversion that lost its value, which every check leading here
/// leaves in `%conversion{id}_lost`.
fn emit_trap(
    emitter: &mut Emitter<'_>,
    id: usize,
    value: &Value,
    cause: &str,
    (source, destination): (Scalar, Scalar),
) {
    let message = conversion_message(emitter, value, source, destination);
    emit_conditional_trap(
        &mut emitter.text,
        &mut emitter.data,
        emitter.function,
        id,
        cause,
        &format!("%conversion{id}_lost"),
        message,
    );
}
