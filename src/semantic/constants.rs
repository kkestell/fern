//! Literal typing and constant evaluation with concrete machine semantics.
use super::model::*;
use crate::{
    diagnostic::Diagnostic,
    frontend::{
        lexer::integer_parts,
        syntax::{Expression, ExpressionKind, Syntax},
    },
    types::{BinaryOperator, ComparisonOperator, Float, Scalar, UnaryOperator},
};
use la_arena::Idx;
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};

pub(super) fn fold_elements(elements: &[CheckedExpression], _length: u64) -> Option<Constant> {
    Some(Constant::Array(
        elements
            .iter()
            .map(|e| e.constant.clone())
            .collect::<Option<Vec<_>>>()?,
    ))
}
pub(super) fn integer_operand(e: &CheckedExpression) -> Option<Scalar> {
    e.ty.scalar().filter(|s| s.is_integer())
}
pub(super) fn numeric_operand(e: &CheckedExpression) -> Option<Scalar> {
    e.ty.scalar().filter(|s| s.is_numeric())
}
pub(super) fn binary_operand_types(
    op: BinaryOperator,
    span: &std::ops::Range<usize>,
    spelling: &str,
    (left, right): (&CheckedExpression, &CheckedExpression),
) -> Result<(Scalar, Scalar), Diagnostic> {
    let operand = if op.defined_on_floating() {
        numeric_operand
    } else {
        integer_operand
    };
    match (operand(left), operand(right)) {
        (Some(a), Some(b)) => Ok((a, b)),
        _ => Err(Diagnostic::new(
            span.clone(),
            format!(
                "`{spelling}` requires {} operands",
                if op.defined_on_floating() {
                    "numeric"
                } else {
                    "integer"
                }
            ),
        )),
    }
}
pub(super) fn contextualizable(e: &CheckedExpression, to: Scalar) -> bool {
    let from = e.ty.scalar().unwrap();
    from == to || (from.is_integer() && to.is_numeric()) || (from.is_floating() && to.is_floating())
}
pub(super) fn evaluate_unary(
    op: UnaryOperator,
    _span: std::ops::Range<usize>,
    ty: Scalar,
    e: &CheckedExpression,
) -> Result<Option<Constant>, Diagnostic> {
    Ok(match &e.constant {
        Some(Constant::Integer(n)) => {
            let result = match op {
                UnaryOperator::Negate => -n,
                UnaryOperator::Complement => !n,
            };
            Some(truncate_integer(&result, ty).into())
        }
        Some(Constant::Float(Float::Binary32(n))) => Some(Float::Binary32(n ^ (1 << 31)).into()),
        Some(Constant::Float(Float::Binary64(n))) => Some(Float::Binary64(n ^ (1 << 63)).into()),
        _ => None,
    })
}
pub(super) fn evaluate_binary(
    op: BinaryOperator,
    span: std::ops::Range<usize>,
    _spelling: &str,
    ty: Scalar,
    _untyped: bool,
    (left, right): (&CheckedExpression, &CheckedExpression),
) -> Result<Option<Constant>, Diagnostic> {
    let (Some(a), Some(b)) = (&left.constant, &right.constant) else {
        return Ok(None);
    };
    if let (Constant::Integer(a), Constant::Integer(b)) = (a, b) {
        if matches!(op, BinaryOperator::Divide | BinaryOperator::Remainder)
            && (b.is_zero()
                || (ty.signed() && a == &BigInt::from(ty.min()) && b == &BigInt::from(-1)))
        {
            return Err(Diagnostic::new(
                span,
                "undefined constant integer division or remainder",
            ));
        }
        if op.is_shift() && b.to_u32().is_none_or(|n| n >= ty.width()) {
            return Err(Diagnostic::new(span, "undefined constant shift count"));
        }
        let result = match op {
            BinaryOperator::Add => a + b,
            BinaryOperator::Subtract => a - b,
            BinaryOperator::Multiply => a * b,
            BinaryOperator::Divide => a / b,
            BinaryOperator::Remainder => a % b,
            BinaryOperator::And => a & b,
            BinaryOperator::Or => a | b,
            BinaryOperator::Xor => a ^ b,
            BinaryOperator::ShiftLeft => a << b.to_u32().unwrap(),
            BinaryOperator::ShiftRight => a >> b.to_u32().unwrap(),
        };
        return Ok(Some(truncate_integer(&result, ty).into()));
    }
    macro_rules! floating {
        ($a:expr,$b:expr,$f:ty,$v:ident) => {{
            let a = <$f>::from_bits(*$a);
            let b = <$f>::from_bits(*$b);
            let n = match op {
                BinaryOperator::Add => a + b,
                BinaryOperator::Subtract => a - b,
                BinaryOperator::Multiply => a * b,
                BinaryOperator::Divide => a / b,
                _ => unreachable!(),
            };
            Float::$v(n.to_bits()).into()
        }};
    }
    Ok(Some(match (a, b) {
        (Constant::Float(Float::Binary32(a)), Constant::Float(Float::Binary32(b))) => {
            floating!(a, b, f32, Binary32)
        }
        (Constant::Float(Float::Binary64(a)), Constant::Float(Float::Binary64(b))) => {
            floating!(a, b, f64, Binary64)
        }
        _ => unreachable!("numeric operands share a concrete type"),
    }))
}
pub(super) fn integer_fits(n: &BigInt, ty: Scalar) -> bool {
    n >= &BigInt::from(ty.min()) && n <= &BigInt::from(ty.max())
}
pub(super) fn truncate_integer(n: &BigInt, ty: Scalar) -> BigInt {
    let modulus = BigInt::from(1u8) << ty.width();
    let bits = ((n % &modulus) + &modulus) % &modulus;
    if ty.signed() && bits >= (BigInt::from(1u8) << (ty.width() - 1)) {
        bits - modulus
    } else {
        bits
    }
}
pub(super) fn concretize_value(
    syntax: &Syntax,
    id: Idx<Expression>,
    e: &mut CheckedExpression,
    to: Scalar,
) -> Result<Option<bool>, Diagnostic> {
    if !e.untyped {
        return Ok(None);
    }
    let span = syntax.expressions[id].span.clone();
    if let Some(value) = e.constant.clone() {
        e.constant = Some(match value {
            Constant::Integer(n) if to.is_floating() => integer_to_float(&n, to),
            Constant::Integer(n) => {
                if !integer_fits(&n, to) {
                    return Err(Diagnostic::new(
                        span,
                        format!("integer literal out of range for `{to}`"),
                    ));
                }
                n.into()
            }
            Constant::Float(n) => {
                if let ExpressionKind::Floating(text) = &syntax.expressions[id].kind {
                    parse_float(text, to, &span)?
                } else {
                    float_to_float(n, to)
                }
            }
            _ => unreachable!("only scalar literals need context"),
        });
    }
    e.ty = to.into();
    e.untyped = false;
    Ok(Some(e.constant.is_some()))
}
fn integer_to_float(n: &BigInt, to: Scalar) -> Constant {
    if to == Scalar::F32 {
        Float::Binary32(
            n.to_f32()
                .unwrap_or(if n < &BigInt::from(0) {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                })
                .to_bits(),
        )
        .into()
    } else {
        Float::Binary64(
            n.to_f64()
                .unwrap_or(if n < &BigInt::from(0) {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                })
                .to_bits(),
        )
        .into()
    }
}
fn float_to_float(n: Float, to: Scalar) -> Constant {
    match (n, to) {
        (Float::Binary32(n), Scalar::F32) => Float::Binary32(n),
        (Float::Binary64(n), Scalar::F64) => Float::Binary64(n),
        (Float::Binary32(n), _) => Float::Binary64(f64::from(f32::from_bits(n)).to_bits()),
        (Float::Binary64(n), _) => Float::Binary32((f64::from_bits(n) as f32).to_bits()),
    }
    .into()
}
fn as_f64(n: Float) -> f64 {
    match n {
        Float::Binary32(n) => f64::from(f32::from_bits(n)),
        Float::Binary64(n) => f64::from_bits(n),
    }
}
pub(super) fn convert_constant(
    e: &CheckedExpression,
    to: Scalar,
    span: &std::ops::Range<usize>,
) -> Result<Option<Constant>, Diagnostic> {
    let Some(n) = &e.constant else {
        return Ok(None);
    };
    Ok(Some(match n {
        Constant::Integer(n) if to.is_integer() => truncate_integer(n, to).into(),
        Constant::Integer(n) => integer_to_float(n, to),
        Constant::Float(n) if to.is_floating() => float_to_float(*n, to),
        Constant::Float(n) => {
            let n = as_f64(*n).trunc();
            let low = if to.signed() {
                -2f64.powi(to.width() as i32 - 1)
            } else {
                0.0
            };
            let high = 2f64.powi(to.width() as i32 - i32::from(to.signed()));
            if !n.is_finite() || n < low || n >= high {
                return Err(Diagnostic::new(
                    span.clone(),
                    "undefined constant float-to-integer conversion",
                ));
            }
            let n = if to.signed() {
                BigInt::from(n as i64)
            } else {
                BigInt::from(n as u64)
            };
            n.into()
        }
        _ => unreachable!(),
    }))
}
fn parse_float(
    text: &str,
    ty: Scalar,
    span: &std::ops::Range<usize>,
) -> Result<Constant, Diagnostic> {
    match ty {
        Scalar::F32 => text
            .parse::<f32>()
            .map(|n| Float::Binary32(n.to_bits()).into())
            .map_err(|_| Diagnostic::new(span.clone(), "invalid floating-point literal")),
        _ => text
            .parse::<f64>()
            .map(|n| Float::Binary64(n.to_bits()).into())
            .map_err(|_| Diagnostic::new(span.clone(), "invalid floating-point literal")),
    }
}
pub(super) fn floating_literal(
    text: &str,
    span: &std::ops::Range<usize>,
) -> Result<CheckedExpression, Diagnostic> {
    Ok(CheckedExpression {
        ty: Scalar::F64.into(),
        untyped: true,
        value: ExpressionValue::Floating,
        constant: Some(parse_float(text, Scalar::F64, span)?),
    })
}
pub(super) fn integer_literal(text: &str) -> CheckedExpression {
    let (base, digits, _) = integer_parts(text);
    CheckedExpression {
        ty: Scalar::Int.into(),
        untyped: true,
        value: ExpressionValue::Integer,
        constant: Some(BigInt::parse_bytes(digits.as_bytes(), base).unwrap().into()),
    }
}
pub(super) fn compare_constants(op: ComparisonOperator, a: &Constant, b: &Constant) -> BigInt {
    use ComparisonOperator::*;
    let order = match (a, b) {
        (Constant::Integer(a), Constant::Integer(b)) => Some(a.cmp(b)),
        (Constant::Float(a), Constant::Float(b)) => as_f64(*a).partial_cmp(&as_f64(*b)),
        _ => None,
    };
    let equal = match (a, b) {
        (Constant::Array(a), Constant::Array(b)) | (Constant::Struct(a), Constant::Struct(b)) => a
            .iter()
            .zip(b)
            .all(|(a, b)| constant_boolean(&compare_constants(Equal, a, b))),
        (Constant::Float(a), Constant::Float(b)) => as_f64(*a) == as_f64(*b),
        _ => a == b,
    };
    BigInt::from(match op {
        Equal => equal,
        NotEqual => !equal,
        Less => order.is_some_and(|o| o.is_lt()),
        LessEqual => order.is_some_and(|o| o.is_le()),
        Greater => order.is_some_and(|o| o.is_gt()),
        GreaterEqual => order.is_some_and(|o| o.is_ge()),
    })
}
pub(super) fn require_boolean(
    syntax: &Syntax,
    id: Idx<Expression>,
    e: &CheckedExpression,
) -> Result<(), Diagnostic> {
    if e.ty == Scalar::Bool.into() {
        Ok(())
    } else {
        Err(Diagnostic::new(
            syntax.expressions[id].span.clone(),
            format!("logical operand has type `{}`, expected `bool`", e.ty),
        ))
    }
}
pub(super) fn constant_boolean(n: &BigInt) -> bool {
    !n.is_zero()
}
