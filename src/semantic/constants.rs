//! Constant folding, integer range operations, shifts, conversions, and comparisons.

use crate::frontend::lexer::integer_parts;
use crate::{
    diagnostic::Diagnostic,
    frontend::syntax::{Expression, ExpressionKind, Syntax},
    types::{BinaryOperator, ComparisonOperator, Scalar, UnaryOperator},
};
use la_arena::Idx;
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;

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
/// the operand is a `bool` or an array.
pub(super) fn integer_operand(checked: &CheckedExpression) -> Option<Scalar> {
    checked.ty.scalar().filter(|scalar| scalar.is_integer())
}

pub(super) fn evaluate_unary(
    operator: UnaryOperator,
    operator_span: std::ops::Range<usize>,
    ty: Scalar,
    operand: &CheckedExpression,
) -> Result<Option<BigInt>, Diagnostic> {
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
    Ok(Some(result))
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
) -> Result<Option<BigInt>, Diagnostic> {
    let (left, right) = operands;
    let right_constant = right.integer();
    reject_constant_divisor(operator, &operator_span, spelling, right_constant)?;
    let (Some(left), Some(right)) = (left.integer(), right_constant) else {
        return Ok(None);
    };
    let result = fold_binary(operator, &operator_span, ty, untyped, left, right)?;
    check_constant_range(operator, &operator_span, ty, untyped, result).map(Some)
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
    let ty = checked
        .ty
        .scalar()
        .expect("an untyped expression has a scalar type");
    if ty.is_integer()
        && checked
            .integer()
            .is_some_and(|value| !integer_fits(value, destination))
    {
        return Err(out_of_range(syntax, id, destination));
    }
    let constant = checked.constant.is_some();
    checked.ty = destination.into();
    checked.untyped = false;
    Ok(Some(constant))
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

/// Compares two constant operands. Equality reaches every value type, so an
/// array folds elementwise; ordering reaches only integers, which is all
/// checking lets through.
pub(super) fn compare_constants(
    operator: ComparisonOperator,
    left: &Constant,
    right: &Constant,
) -> BigInt {
    let ordering = || {
        let message = "ordering compares integer constants";
        left.integer()
            .expect(message)
            .cmp(right.integer().expect(message))
    };
    BigInt::from(match operator {
        ComparisonOperator::Equal => left == right,
        ComparisonOperator::NotEqual => left != right,
        ComparisonOperator::Less => ordering().is_lt(),
        ComparisonOperator::LessEqual => ordering().is_le(),
        ComparisonOperator::Greater => ordering().is_gt(),
        ComparisonOperator::GreaterEqual => ordering().is_ge(),
    })
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

fn out_of_range(syntax: &Syntax, id: Idx<Expression>, destination: Scalar) -> Diagnostic {
    let literal = matches!(syntax.expressions[id].kind, ExpressionKind::Integer(_));
    Diagnostic::new(
        syntax.expressions[id].span.clone(),
        format!(
            "integer {} out of range for `{}`",
            if literal { "literal" } else { "value" },
            destination
        ),
    )
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
