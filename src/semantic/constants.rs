//! Constant folding, numeric range operations, shifts, conversions, and comparisons.

use crate::frontend::lexer::integer_parts;
use crate::{
    diagnostic::Diagnostic,
    frontend::syntax::{Expression, ExpressionKind, Syntax},
    types::{BinaryOperator, ComparisonOperator, Float, Scalar, UnaryOperator},
};
use la_arena::Idx;
use num_bigint::{BigInt, BigUint};
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};

use super::model::*;
/// The value an array literal folds to, which is one constant per element with
/// the fill repeating the last element. It folds only when every element does.
pub(super) fn fold_elements(elements: &[CheckedExpression], length: u64) -> Option<Constant> {
    let mut values = elements
        .iter()
        .map(|element| element.constant.clone())
        .collect::<Option<Vec<_>>>()?;
    let last = values
        .last()
        .expect("an array literal has at least one element")
        .clone();
    values.resize(
        usize::try_from(length).expect("an array fits in the host's address space"),
        last,
    );
    Some(Constant::Array(values))
}

/// The scalar type an integer operator requires of its operand, or `None` when
/// the operand is a floating-point value, a `bool`, or an array.
pub(super) fn integer_operand(checked: &CheckedExpression) -> Option<Scalar> {
    checked.ty.scalar().filter(|scalar| scalar.is_integer())
}

/// The scalar type an arithmetic operator requires of its operand, or `None`
/// when the operand is a `bool` or an array.
pub(super) fn numeric_operand(checked: &CheckedExpression) -> Option<Scalar> {
    checked.ty.scalar().filter(|scalar| scalar.is_numeric())
}

/// The scalar types a binary operator requires of its operands. An integer
/// operator rejects a floating-point operand, and every operator rejects a
/// `bool` or an array.
pub(super) fn binary_operand_types(
    operator: BinaryOperator,
    operator_span: &std::ops::Range<usize>,
    spelling: &str,
    operands: (&CheckedExpression, &CheckedExpression),
) -> Result<(Scalar, Scalar), Diagnostic> {
    let (left, right) = operands;
    if operator.defined_on_floating() {
        if let (Some(left), Some(right)) = (numeric_operand(left), numeric_operand(right)) {
            return Ok((left, right));
        }
        return Err(Diagnostic::new(
            operator_span.clone(),
            format!("`{spelling}` requires numeric operands"),
        ));
    }
    if let (Some(left), Some(right)) = (integer_operand(left), integer_operand(right)) {
        return Ok((left, right));
    }
    Err(Diagnostic::new(
        operator_span.clone(),
        format!("integer `{spelling}` requires integer operands"),
    ))
}

/// Whether an untyped expression may take `destination` from context. An
/// untyped integer also reaches a floating-point type, but only as the exact
/// value of a constant.
pub(super) fn contextualizable(checked: &CheckedExpression, destination: Scalar) -> bool {
    let ty = checked
        .ty
        .scalar()
        .expect("an untyped expression has a scalar type");
    if ty == destination {
        return true;
    }
    if ty.is_integer() {
        return destination.is_integer()
            || (destination.is_floating() && checked.constant.is_some());
    }
    ty.is_floating() && destination.is_floating()
}

/// Rewrites an untyped integer constant as the exact floating-point value it
/// names, reporting whether it had one to rewrite.
pub(super) fn to_untyped_floating(checked: &mut CheckedExpression) -> bool {
    let Some(value) = checked.integer() else {
        return false;
    };
    checked.constant = Some(BigRational::from(value.clone()).into());
    checked.ty = Scalar::F64.into();
    true
}

pub(super) fn evaluate_unary(
    operator: UnaryOperator,
    operator_span: std::ops::Range<usize>,
    ty: Scalar,
    operand: &CheckedExpression,
) -> Result<Option<Constant>, Diagnostic> {
    if ty.is_floating() {
        debug_assert!(operator == UnaryOperator::Negate);
        return Ok(negate_floating(operand));
    }
    let Some(value) = operand.integer() else {
        return Ok(None);
    };
    let result = match operator {
        UnaryOperator::Negate => -value,
        UnaryOperator::WrappingNegate => truncate_integer(&-value, ty),
        UnaryOperator::Complement if operand.untyped => !value,
        UnaryOperator::Complement => truncate_integer(&!value, ty),
    };
    if operand.untyped && result.bits() > MAX_UNTYPED_INTEGER_BITS {
        return Err(Diagnostic::new(
            operator_span,
            "constant expression exceeds compiler resource limit",
        ));
    }
    if operator == UnaryOperator::Negate && !operand.untyped && !integer_fits(&result, ty) {
        return Err(Diagnostic::new(
            operator_span,
            format!("constant unary `-` on `{ty}` would trap"),
        ));
    }
    Ok(Some(result.into()))
}

/// Negates a floating-point constant. Negation is exact in both forms, and
/// flipping the sign bit gives a concrete zero the other zero.
fn negate_floating(operand: &CheckedExpression) -> Option<Constant> {
    if let Some(value) = operand.rational() {
        return Some((-value).into());
    }
    operand.float().map(|value| negate_float(value).into())
}

const MAX_UNTYPED_INTEGER_BITS: u64 = 2_000_000;

/// The largest shift count the compiler folds. An untyped shift has no width
/// to wrap against, so a large count is a resource limit rather than a result.
const MAX_CONSTANT_SHIFT: usize = 1_000_000;

/// Folds a shift of two constants. `right` is not negative: a negative shift
/// count is rejected before folding.
fn evaluate_shift(
    operator: BinaryOperator,
    operator_span: &std::ops::Range<usize>,
    ty: Scalar,
    untyped: bool,
    left: &BigInt,
    right: &BigInt,
) -> Result<BigInt, Diagnostic> {
    debug_assert!(right >= &BigInt::from(0u8));
    if untyped {
        return shift_untyped(operator, operator_span, left, right);
    }
    Ok(shift_typed(operator, ty, left, right))
}

/// Shifts an untyped constant, which has no width to shift bits out of.
fn shift_untyped(
    operator: BinaryOperator,
    operator_span: &std::ops::Range<usize>,
    left: &BigInt,
    right: &BigInt,
) -> Result<BigInt, Diagnostic> {
    let count = right.to_usize();
    if operator == BinaryOperator::ShiftRight && count.is_none() {
        return Ok(sign_fill(left));
    }
    let Some(count) = count else {
        return Err(Diagnostic::new(
            operator_span.clone(),
            "constant shift exceeds compiler resource limit",
        ));
    };
    if operator == BinaryOperator::ShiftLeft && shift_exceeds_limit(count, left) {
        return Err(Diagnostic::new(
            operator_span.clone(),
            "constant expression exceeds compiler resource limit",
        ));
    }
    Ok(if operator == BinaryOperator::ShiftLeft {
        left << count
    } else {
        left >> count
    })
}

/// Whether shifting `left` left by `count` would need more bits than the
/// compiler folds.
fn shift_exceeds_limit(count: usize, left: &BigInt) -> bool {
    count > MAX_CONSTANT_SHIFT
        || u64::try_from(count)
            .unwrap_or(u64::MAX)
            .saturating_add(left.bits())
            > MAX_UNTYPED_INTEGER_BITS
}

/// Shifts a typed constant. A count at or past the type's width shifts every
/// bit out.
fn shift_typed(operator: BinaryOperator, ty: Scalar, left: &BigInt, right: &BigInt) -> BigInt {
    if right >= &BigInt::from(ty.width()) {
        return if operator == BinaryOperator::ShiftRight && ty.signed() {
            sign_fill(left)
        } else {
            BigInt::from(0u8)
        };
    }
    let count = right
        .to_usize()
        .expect("count below every Fern integer width fits usize");
    if operator == BinaryOperator::ShiftLeft {
        truncate_integer(&(left << count), ty)
    } else {
        left >> count
    }
}

/// Folds a binary operation on two constants, reporting the operations the
/// program is not allowed to perform even when it never runs them.
pub(super) fn evaluate_binary(
    operator: BinaryOperator,
    operator_span: std::ops::Range<usize>,
    spelling: &str,
    ty: Scalar,
    untyped: bool,
    operands: (&CheckedExpression, &CheckedExpression),
) -> Result<Option<Constant>, Diagnostic> {
    if ty.is_floating() {
        return evaluate_floating_binary(operator, operator_span, spelling, ty, untyped, operands);
    }
    let (left, right) = operands;
    let right_constant = right.integer();
    reject_constant_divisor(operator, &operator_span, spelling, right_constant)?;
    let (Some(left), Some(right)) = (left.integer(), right_constant) else {
        return Ok(None);
    };
    let result = fold_binary(operator, &operator_span, ty, untyped, left, right)?;
    check_constant_range(operator, &operator_span, ty, untyped, result)
        .map(|result| Some(result.into()))
}

/// Folds an operation on two floating-point constants. An untyped pair is
/// exact, so only a zero divisor can fail it. A concrete pair rounds to its
/// format, where a result that is not finite is an overflow the program may
/// not name.
fn evaluate_floating_binary(
    operator: BinaryOperator,
    operator_span: std::ops::Range<usize>,
    spelling: &str,
    ty: Scalar,
    untyped: bool,
    operands: (&CheckedExpression, &CheckedExpression),
) -> Result<Option<Constant>, Diagnostic> {
    let (left, right) = operands;
    // Dividing a runtime value by zero is defined, so only a division the
    // compiler evaluates in full is rejected.
    if operator == BinaryOperator::Divide && left.constant.is_some() && is_zero(right) {
        return Err(Diagnostic::new(
            operator_span,
            format!("constant `{spelling}` divisor is zero"),
        ));
    }
    if untyped {
        let (Some(left), Some(right)) = (left.rational(), right.rational()) else {
            return Ok(None);
        };
        let result = fold_rational(operator, left, right);
        limit_rational(&operator_span, &result)?;
        return Ok(Some(result.into()));
    }
    let (Some(left), Some(right)) = (left.float(), right.float()) else {
        return Ok(None);
    };
    let Some(result) = fold_float(operator, left, right) else {
        return Err(Diagnostic::new(
            operator_span,
            format!(
                "constant `{}` on `{ty}` would overflow",
                operator.spelling()
            ),
        ));
    };
    Ok(Some(result.into()))
}

/// Whether an operand is a floating-point constant zero, of either sign.
fn is_zero(operand: &CheckedExpression) -> bool {
    operand.rational().is_some_and(BigRational::is_zero)
        || operand
            .float()
            .is_some_and(|value| float_value(value).is_zero())
}

/// Rejects a right operand the operator cannot accept, whether or not the left
/// operand is constant.
fn reject_constant_divisor(
    operator: BinaryOperator,
    operator_span: &std::ops::Range<usize>,
    spelling: &str,
    right: Option<&BigInt>,
) -> Result<(), Diagnostic> {
    if matches!(operator, BinaryOperator::Divide | BinaryOperator::Remainder)
        && right == Some(&BigInt::from(0u8))
    {
        return Err(Diagnostic::new(
            operator_span.clone(),
            format!("constant `{spelling}` divisor is zero"),
        ));
    }
    if matches!(
        operator,
        BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight
    ) && right.is_some_and(|count| count < &BigInt::from(0u8))
    {
        let message = if spelling.ends_with('=') {
            format!("constant `{spelling}` shift count is negative")
        } else {
            "constant shift count is negative".to_owned()
        };
        return Err(Diagnostic::new(operator_span.clone(), message));
    }
    Ok(())
}

/// Applies the operator to two constants. The result is not range-checked
/// here, except where the operation itself would exceed what the compiler
/// folds.
fn fold_binary(
    operator: BinaryOperator,
    operator_span: &std::ops::Range<usize>,
    ty: Scalar,
    untyped: bool,
    left: &BigInt,
    right: &BigInt,
) -> Result<BigInt, Diagnostic> {
    Ok(match operator {
        BinaryOperator::Multiply => {
            if untyped
                && left.bits().saturating_add(right.bits()).saturating_sub(1)
                    > MAX_UNTYPED_INTEGER_BITS
            {
                return Err(Diagnostic::new(
                    operator_span.clone(),
                    "constant expression exceeds compiler resource limit",
                ));
            }
            left * right
        }
        BinaryOperator::Divide => {
            reject_division_trap("/", operator_span, ty, untyped, left, right)?;
            left / right
        }
        BinaryOperator::Remainder => {
            reject_division_trap("%", operator_span, ty, untyped, left, right)?;
            left % right
        }
        BinaryOperator::WrappingMultiply => truncate_integer(&(left * right), ty),
        BinaryOperator::Add => left + right,
        BinaryOperator::Subtract => left - right,
        BinaryOperator::WrappingAdd => truncate_integer(&(left + right), ty),
        BinaryOperator::WrappingSubtract => truncate_integer(&(left - right), ty),
        BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => {
            evaluate_shift(operator, operator_span, ty, untyped, left, right)?
        }
        BinaryOperator::And => left & right,
        BinaryOperator::Xor => left ^ right,
        BinaryOperator::Or => left | right,
    })
}

/// Dividing the most negative value of a type by `-1` traps, so a program that
/// spells it out is rejected at check time.
fn reject_division_trap(
    spelling: &str,
    operator_span: &std::ops::Range<usize>,
    ty: Scalar,
    untyped: bool,
    left: &BigInt,
    right: &BigInt,
) -> Result<(), Diagnostic> {
    if !untyped && is_minimum(left, ty) && right == &BigInt::from(-1) {
        return Err(Diagnostic::new(
            operator_span.clone(),
            format!("constant `{spelling}` on `{}` would trap", ty.name()),
        ));
    }
    Ok(())
}

/// Checks a folded result against the type it must fit, or against the
/// compiler's limit on how large an untyped constant may grow.
fn check_constant_range(
    operator: BinaryOperator,
    operator_span: &std::ops::Range<usize>,
    ty: Scalar,
    untyped: bool,
    result: BigInt,
) -> Result<BigInt, Diagnostic> {
    let checked_arithmetic = matches!(
        operator,
        BinaryOperator::Multiply | BinaryOperator::Add | BinaryOperator::Subtract
    );
    if checked_arithmetic && !untyped && !integer_fits(&result, ty) {
        return Err(Diagnostic::new(
            operator_span.clone(),
            format!(
                "constant `{}` on `{}` would overflow",
                operator.spelling(),
                ty.name()
            ),
        ));
    }
    if untyped && result.bits() > MAX_UNTYPED_INTEGER_BITS {
        return Err(Diagnostic::new(
            operator_span.clone(),
            "constant expression exceeds compiler resource limit",
        ));
    }
    Ok(result)
}

/// What shifting every bit out of a value leaves behind: its sign.
pub(super) fn sign_fill(left: &BigInt) -> BigInt {
    if left < &BigInt::from(0u8) {
        BigInt::from(-1)
    } else {
        BigInt::from(0u8)
    }
}

pub(super) fn concretize_value(
    syntax: &Syntax,
    id: Idx<Expression>,
    checked: &mut CheckedExpression,
    destination: Scalar,
) -> Result<Option<bool>, Diagnostic> {
    if !checked.untyped {
        return Ok(None);
    }
    if let Some(value) = &checked.constant {
        let contextualized = contextualize_constant(value, destination)
            .ok_or_else(|| not_representable(syntax, id, value, destination))?;
        checked.constant = Some(contextualized);
    }
    let constant = checked.constant.is_some();
    checked.ty = destination.into();
    checked.untyped = false;
    Ok(Some(constant))
}

/// The value an untyped constant takes on in a `destination` context. An
/// untyped integer must be representable there exactly, while an untyped
/// floating-point constant rounds once to the nearest value of the
/// destination format.
fn contextualize_constant(value: &Constant, destination: Scalar) -> Option<Constant> {
    match value {
        Constant::Integer(value) if destination.is_floating() => {
            representable_exactly(&BigRational::from(value.clone()), destination)
        }
        Constant::Integer(value) => {
            integer_fits(value, destination).then(|| Constant::Integer(value.clone()))
        }
        Constant::Rational(value) => round_to_float(value, destination).map(Constant::from),
        Constant::Null
        | Constant::EmptySlice
        | Constant::Float(_)
        | Constant::Array(_)
        | Constant::Struct(_) => {
            unreachable!("an untyped constant is an integer, an exact value, or a boolean")
        }
    }
}

/// Converts a constant operand at check time. A checked conversion that would
/// trap is an error rather than a trap at run time.
pub(super) fn convert_constant(
    operand: &CheckedExpression,
    destination: Scalar,
    truncating: bool,
    span: &std::ops::Range<usize>,
) -> Result<Option<Constant>, Diagnostic> {
    let Some(value) = &operand.constant else {
        return Ok(None);
    };
    if truncating {
        let value = value
            .integer()
            .expect("a truncated constant has an integer type");
        return Ok(Some(truncate_integer(value, destination).into()));
    }
    converted(value, destination).map(Some).ok_or_else(|| {
        Diagnostic::new(
            span.clone(),
            format!("constant conversion to `{destination}` would trap"),
        )
    })
}

/// The value a checked conversion produces, or `None` for the trap it would
/// take at run time. A checked conversion preserves its value exactly.
fn converted(value: &Constant, destination: Scalar) -> Option<Constant> {
    match value {
        Constant::Integer(value) if destination.is_floating() => {
            representable_exactly(&BigRational::from(value.clone()), destination)
        }
        Constant::Integer(value) => {
            integer_fits(value, destination).then(|| Constant::Integer(value.clone()))
        }
        Constant::Float(value) if destination.is_floating() => {
            representable_exactly(&float_value(*value), destination)
        }
        Constant::Float(value) => {
            let exact = float_value(*value);
            let whole = exact.is_integer().then(|| exact.to_integer())?;
            integer_fits(&whole, destination).then_some(Constant::Integer(whole))
        }
        Constant::Null
        | Constant::EmptySlice
        | Constant::Rational(_)
        | Constant::Array(_)
        | Constant::Struct(_) => {
            unreachable!("a converted constant has a concrete numeric type")
        }
    }
}

/// The value rounded to `destination`, when the rounding loses nothing.
fn representable_exactly(value: &BigRational, destination: Scalar) -> Option<Constant> {
    let rounded = round_to_float(value, destination)?;
    (float_value(rounded) == *value).then_some(rounded.into())
}

/// The largest power of ten a floating-point literal may name. Ten to this
/// power stays inside the bit limit an untyped constant is folded within.
const MAX_LITERAL_EXPONENT: u64 = 500_000;

/// The checked form of a floating-point literal. A literal starts untyped, so
/// its exact value is kept until the expression's format is known.
pub(super) fn floating_literal(
    spelling: &str,
    span: &std::ops::Range<usize>,
) -> Result<CheckedExpression, Diagnostic> {
    let (mantissa, exponent) = match spelling.find(['e', 'E']) {
        Some(marker) => (&spelling[..marker], &spelling[marker + 1..]),
        None => (spelling, ""),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = BigUint::parse_bytes(format!("{whole}{fraction}").as_bytes(), 10)
        .expect("frontend validated floating-point digits");
    let fraction_digits =
        i64::try_from(fraction.len()).expect("a source file holds fewer digits than i64::MAX");
    let scale = literal_exponent(exponent).saturating_sub(fraction_digits);
    if scale.unsigned_abs() > MAX_LITERAL_EXPONENT {
        return Err(Diagnostic::new(
            span.clone(),
            "floating-point literal exceeds compiler resource limit",
        ));
    }
    let power = BigInt::from(10u8).pow(
        u32::try_from(scale.unsigned_abs()).expect("a power below the literal limit fits u32"),
    );
    let digits = BigInt::from(digits);
    let value = if scale >= 0 {
        BigRational::from(digits * power)
    } else {
        BigRational::new(digits, power)
    };
    limit_rational(span, &value)?;
    Ok(CheckedExpression {
        ty: Scalar::F64.into(),
        untyped: true,
        value: ExpressionValue::Floating,
        constant: Some(value.into()),
    })
}

/// The power of ten a literal's exponent names. A spelling that overflows is
/// saturated, because it is far outside the limit it is checked against.
fn literal_exponent(spelling: &str) -> i64 {
    if spelling.is_empty() {
        return 0;
    }
    spelling.parse().unwrap_or(if spelling.starts_with('-') {
        i64::MIN
    } else {
        i64::MAX
    })
}

/// Rejects an exact value that has grown past what the compiler folds.
fn limit_rational(span: &std::ops::Range<usize>, value: &BigRational) -> Result<(), Diagnostic> {
    if value.numer().bits() > MAX_UNTYPED_INTEGER_BITS
        || value.denom().bits() > MAX_UNTYPED_INTEGER_BITS
    {
        return Err(Diagnostic::new(
            span.clone(),
            "constant expression exceeds compiler resource limit",
        ));
    }
    Ok(())
}

/// The checked form of an integer literal. A literal starts untyped, so its
/// value is kept exactly until the expression's type is known.
pub(super) fn integer_literal(spelling: &str) -> CheckedExpression {
    let (base, digits, suffix) = integer_parts(spelling);
    debug_assert!(suffix.is_empty(), "frontend rejects literal suffixes");
    let value =
        BigUint::parse_bytes(digits.as_bytes(), base).expect("frontend validated integer digits");
    CheckedExpression {
        ty: Scalar::Int.into(),
        untyped: true,
        value: ExpressionValue::Integer,
        constant: Some(BigInt::from(value).into()),
    }
}

/// Compares two constant operands of one type. Equality reaches every value
/// type, so an array folds elementwise; ordering reaches the numbers, which is
/// all checking lets through.
pub(super) fn compare_constants(
    operator: ComparisonOperator,
    left: &Constant,
    right: &Constant,
) -> BigInt {
    BigInt::from(match operator {
        ComparisonOperator::Equal => constants_equal(left, right),
        ComparisonOperator::NotEqual => !constants_equal(left, right),
        ComparisonOperator::Less => compare_numbers(left, right).is_lt(),
        ComparisonOperator::LessEqual => compare_numbers(left, right).is_le(),
        ComparisonOperator::Greater => compare_numbers(left, right).is_gt(),
        ComparisonOperator::GreaterEqual => compare_numbers(left, right).is_ge(),
    })
}

/// Whether two constants of one type hold the same value. Floating-point
/// values compare by value rather than by their bits, so a positive and a
/// negative zero are equal, and an array or struct compares componentwise.
fn constants_equal(left: &Constant, right: &Constant) -> bool {
    match (left, right) {
        (Constant::Float(left), Constant::Float(right)) => {
            float_value(*left) == float_value(*right)
        }
        (Constant::Array(left), Constant::Array(right))
        | (Constant::Struct(left), Constant::Struct(right)) => {
            left.len() == right.len()
                && std::iter::zip(left, right).all(|(left, right)| constants_equal(left, right))
        }
        _ => left == right,
    }
}

/// Orders two numeric constants of one type.
fn compare_numbers(left: &Constant, right: &Constant) -> std::cmp::Ordering {
    match (left, right) {
        (Constant::Float(left), Constant::Float(right)) => {
            float_value(*left).cmp(&float_value(*right))
        }
        (Constant::Rational(left), Constant::Rational(right)) => left.cmp(right),
        _ => {
            let message = "ordering compares numeric constants of one type";
            left.integer()
                .expect(message)
                .cmp(right.integer().expect(message))
        }
    }
}

pub(super) fn require_boolean(
    syntax: &Syntax,
    id: Idx<Expression>,
    checked: &CheckedExpression,
) -> Result<(), Diagnostic> {
    if checked.ty == Scalar::Bool.into() {
        Ok(())
    } else {
        Err(Diagnostic::new(
            syntax.expressions[id].span.clone(),
            format!("logical operand has type `{}`, expected `bool`", checked.ty),
        ))
    }
}

pub(super) fn constant_boolean(value: &BigInt) -> bool {
    debug_assert!(value == &BigInt::from(0u8) || value == &BigInt::from(1u8));
    value == &BigInt::from(1u8)
}

/// Reports an untyped constant the destination type cannot hold.
fn not_representable(
    syntax: &Syntax,
    id: Idx<Expression>,
    value: &Constant,
    destination: Scalar,
) -> Diagnostic {
    let literal = matches!(
        syntax.expressions[id].kind,
        ExpressionKind::Integer(_) | ExpressionKind::Floating(_)
    );
    let noun = if literal { "literal" } else { "value" };
    let message = match value {
        Constant::Integer(_) if destination.is_floating() => {
            format!("integer {noun} not representable in `{destination}`")
        }
        Constant::Integer(_) => format!("integer {noun} out of range for `{destination}`"),
        _ => format!("floating-point {noun} out of range for `{destination}`"),
    };
    Diagnostic::new(syntax.expressions[id].span.clone(), message)
}

fn is_minimum(value: &BigInt, ty: Scalar) -> bool {
    value == &BigInt::from(ty.min())
}

pub(super) fn integer_fits(value: &BigInt, ty: Scalar) -> bool {
    value >= &BigInt::from(ty.min()) && value <= &BigInt::from(ty.max())
}

fn integer_from_bits(bits: BigInt, ty: Scalar) -> BigInt {
    if ty.signed() && bits >= (BigInt::from(1u8) << (ty.width() - 1)) {
        bits - (BigInt::from(1u8) << ty.width())
    } else {
        bits
    }
}

pub(super) fn truncate_integer(value: &BigInt, ty: Scalar) -> BigInt {
    let modulus = BigInt::from(1u8) << ty.width();
    let bits = ((value % &modulus) + &modulus) % &modulus;
    integer_from_bits(bits, ty)
}

/// The IEEE 754 interchange format a Fern floating-point type stores its
/// values in. Every rule below follows from its two widths, so `f32` and
/// `f64` share one implementation of rounding, encoding, and decoding.
#[derive(Clone, Copy)]
struct Format {
    ty: Scalar,
    /// Significand bits, counting the leading bit a normal value implies.
    significand: u32,
}

impl Format {
    fn of(ty: Scalar) -> Self {
        let significand = match ty {
            Scalar::F32 => 24,
            Scalar::F64 => 53,
            _ => unreachable!("`{ty}` is not a floating-point type"),
        };
        Self { ty, significand }
    }

    /// What a stored exponent adds to the exponent it encodes.
    fn bias(self) -> i64 {
        (1i64 << (self.ty.width() - self.significand - 1)) - 1
    }

    /// The exponent of the least significant bit of a subnormal value, which
    /// is the smallest exponent any value of the format has.
    fn min_exponent(self) -> i64 {
        2 - self.bias() - i64::from(self.significand)
    }

    /// The largest exponent a full significand may be scaled by while the
    /// value it names stays finite.
    fn max_exponent(self) -> i64 {
        self.bias() - i64::from(self.significand) + 1
    }

    /// The leading significand bit, which a normal value implies rather than
    /// stores.
    fn implicit_bit(self) -> u64 {
        1 << (self.significand - 1)
    }

    fn exponent_mask(self) -> u64 {
        (1 << (self.ty.width() - self.significand)) - 1
    }

    /// The format value `significand` times two to the `exponent`, which the
    /// rounding above has already fitted to the format.
    fn encode(self, negative: bool, significand: &BigInt, exponent: i64) -> Float {
        debug_assert!((self.min_exponent()..=self.max_exponent()).contains(&exponent));
        let significand = significand
            .to_u64()
            .expect("a rounded significand fits its format");
        debug_assert!(significand < self.implicit_bit() << 1);
        let sign = u64::from(negative) << (self.ty.width() - 1);
        let bits = if significand < self.implicit_bit() {
            // Zero and the subnormal values store their whole significand
            // under a stored exponent of zero.
            debug_assert!(exponent == self.min_exponent());
            sign | significand
        } else {
            let stored = exponent + i64::from(self.significand) - 1 + self.bias();
            let stored = u64::try_from(stored).expect("a finite value has a stored exponent");
            sign | (stored << (self.significand - 1)) | (significand - self.implicit_bit())
        };
        match self.ty {
            Scalar::F32 => {
                Float::Binary32(u32::try_from(bits).expect("a binary32 value has 32 bits"))
            }
            _ => Float::Binary64(bits),
        }
    }

    /// The exact value a format value names.
    fn decode(self, value: Float) -> BigRational {
        let bits = value.bits();
        let stored = i64::try_from((bits >> (self.significand - 1)) & self.exponent_mask())
            .expect("a stored exponent is narrower than i64");
        debug_assert!(
            stored != i64::try_from(self.exponent_mask()).expect("the mask fits i64"),
            "a Fern floating-point constant is finite"
        );
        let significand = bits & (self.implicit_bit() - 1);
        let (significand, exponent) = if stored == 0 {
            (significand, self.min_exponent())
        } else {
            (
                significand | self.implicit_bit(),
                stored - self.bias() - i64::from(self.significand) + 1,
            )
        };
        let magnitude = scale_by_two(&BigInt::from(significand), exponent);
        if bits >> (self.ty.width() - 1) == 1 {
            -magnitude
        } else {
            magnitude
        }
    }
}

/// The exact value a concrete floating-point constant names.
pub(super) fn float_value(value: Float) -> BigRational {
    Format::of(value.ty()).decode(value)
}

/// Rounds an exact value to `ty` with round-to-nearest, ties-to-even. `None`
/// when it rounds to an infinity, which no Fern constant may hold.
pub(super) fn round_to_float(value: &BigRational, ty: Scalar) -> Option<Float> {
    let format = Format::of(ty);
    let (significand, exponent) = round_significand(&value.abs(), format);
    (exponent <= format.max_exponent())
        .then(|| format.encode(value.is_negative(), &significand, exponent))
}

/// The significand and exponent of the `format` value nearest to a magnitude,
/// with a tie going to the even significand. The exponent may exceed the
/// format's largest, which is how the caller sees an overflow.
fn round_significand(magnitude: &BigRational, format: Format) -> (BigInt, i64) {
    if magnitude.is_zero() {
        return (BigInt::from(0u8), format.min_exponent());
    }
    let leading = floor_log2(magnitude.numer(), magnitude.denom());
    let mut exponent = (leading - i64::from(format.significand) + 1).max(format.min_exponent());
    let mut significand = round_to_nearest_even(magnitude, exponent);
    // Rounding up can carry into one bit more than the significand holds,
    // which is exactly one halving away from fitting again.
    if significand.bits() > u64::from(format.significand) {
        significand >>= 1;
        exponent += 1;
    }
    (significand, exponent)
}

/// The exponent of the highest set bit of a positive quotient.
fn floor_log2(numerator: &BigInt, denominator: &BigInt) -> i64 {
    let estimate = bit_length(numerator) - bit_length(denominator);
    // The two bit lengths bracket the quotient within one power of two.
    if at_least_power_of_two(numerator, denominator, estimate) {
        estimate
    } else {
        estimate - 1
    }
}

/// Whether a positive quotient is at least two to the `exponent`.
fn at_least_power_of_two(numerator: &BigInt, denominator: &BigInt, exponent: i64) -> bool {
    let shift = shift_width(exponent);
    if exponent >= 0 {
        numerator >= &(denominator << shift)
    } else {
        &(numerator << shift) >= denominator
    }
}

/// A magnitude divided by two to the `exponent`, rounded to the nearest
/// integer with a tie going to the even one.
fn round_to_nearest_even(magnitude: &BigRational, exponent: i64) -> BigInt {
    let shift = shift_width(exponent);
    let (numerator, denominator) = if exponent >= 0 {
        (magnitude.numer().clone(), magnitude.denom() << shift)
    } else {
        (magnitude.numer() << shift, magnitude.denom().clone())
    };
    let quotient = &numerator / &denominator;
    let doubled = (numerator - &quotient * &denominator) * 2u8;
    match doubled.cmp(&denominator) {
        std::cmp::Ordering::Less => quotient,
        std::cmp::Ordering::Greater => quotient + 1u8,
        std::cmp::Ordering::Equal if quotient.bit(0) => quotient + 1u8,
        std::cmp::Ordering::Equal => quotient,
    }
}

/// An integer times two to the `exponent`, as an exact value.
fn scale_by_two(value: &BigInt, exponent: i64) -> BigRational {
    let shift = shift_width(exponent);
    if exponent >= 0 {
        BigRational::from(value << shift)
    } else {
        BigRational::new(value.clone(), BigInt::from(1u8) << shift)
    }
}

fn bit_length(value: &BigInt) -> i64 {
    i64::try_from(value.bits()).expect("a folded constant is narrower than i64::MAX bits")
}

/// The shift an exponent names, which the limit on a folded constant keeps
/// small enough to shift by.
fn shift_width(exponent: i64) -> u32 {
    u32::try_from(exponent.unsigned_abs()).expect("a folded constant has a small exponent")
}

/// Applies an arithmetic operator to two values of one floating-point format.
fn fold_float(operator: BinaryOperator, left: Float, right: Float) -> Option<Float> {
    match (left, right) {
        (Float::Binary32(left), Float::Binary32(right)) => {
            let result = apply(operator, f32::from_bits(left), f32::from_bits(right));
            result
                .is_finite()
                .then(|| Float::Binary32(result.to_bits()))
        }
        (Float::Binary64(left), Float::Binary64(right)) => {
            let result = apply(operator, f64::from_bits(left), f64::from_bits(right));
            result
                .is_finite()
                .then(|| Float::Binary64(result.to_bits()))
        }
        _ => unreachable!("binary operands share one type"),
    }
}

/// Applies an arithmetic operator to two exact values.
fn fold_rational(operator: BinaryOperator, left: &BigRational, right: &BigRational) -> BigRational {
    apply(operator, left.clone(), right.clone())
}

/// The one arithmetic rule the exact and the rounded values share. Only the
/// operators floating-point checking permits reach it.
fn apply<T: num_traits::Num>(operator: BinaryOperator, left: T, right: T) -> T {
    match operator {
        BinaryOperator::Add => left + right,
        BinaryOperator::Subtract => left - right,
        BinaryOperator::Multiply => left * right,
        BinaryOperator::Divide => left / right,
        _ => unreachable!("only arithmetic is defined on floating-point values"),
    }
}

/// Negates a concrete value by flipping its sign bit, which is exact for
/// every value, including a zero.
fn negate_float(value: Float) -> Float {
    match value {
        Float::Binary32(bits) => Float::Binary32(bits ^ (1 << 31)),
        Float::Binary64(bits) => Float::Binary64(bits ^ (1 << 63)),
    }
}
