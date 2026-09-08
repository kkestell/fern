use crate::{
    diagnostic::Diagnostic,
    frontend::{
        BinaryOperator, Expression, ExpressionKind, Function, Statement, StatementKind, Syntax,
        UnaryOperator, integer_parts,
    },
};
use la_arena::{Arena, ArenaMap, Idx};
use lasso::Spur;
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Type {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Int,
    Uint,
}

impl Type {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 10] = [
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::Int,
        Self::Uint,
    ];

    fn named(name: &str) -> Option<Self> {
        Some(match name {
            "i8" => Self::I8,
            "i16" => Self::I16,
            "i32" => Self::I32,
            "i64" => Self::I64,
            "u8" => Self::U8,
            "u16" => Self::U16,
            "u32" => Self::U32,
            "u64" => Self::U64,
            "int" => Self::Int,
            "uint" => Self::Uint,
            _ => return None,
        })
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::Int => "int",
            Self::Uint => "uint",
        }
    }

    pub(crate) fn width(self) -> u32 {
        self.width_on(usize::BITS)
    }

    pub(crate) fn width_on(self, pointer_width: u32) -> u32 {
        match self {
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 => 16,
            Self::I32 | Self::U32 => 32,
            Self::I64 | Self::U64 => 64,
            Self::Int | Self::Uint => pointer_width,
        }
    }

    pub(crate) fn signed(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::Int
        )
    }

    pub(crate) fn max(self) -> u64 {
        u64::MAX >> (64 - self.width() + u32::from(self.signed()))
    }

    pub(crate) fn min(self) -> i128 {
        if self.signed() {
            -(1i128 << (self.width() - 1))
        } else {
            0
        }
    }
}

#[derive(Debug)]
pub(crate) struct Binding {
    pub ty: Type,
    pub mutable: bool,
    pub constant: Option<BigInt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExpressionValue {
    Integer,
    Reference(Idx<Binding>),
    Grouping {
        expression: Idx<Expression>,
    },
    Conversion {
        operand: Idx<Expression>,
        truncating: bool,
    },
    Unary {
        operator: UnaryOperator,
        operator_span: std::ops::Range<usize>,
        operand: Idx<Expression>,
    },
    Binary {
        operator: BinaryOperator,
        operator_span: std::ops::Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
    },
}

#[derive(Debug)]
pub(crate) struct CheckedExpression {
    pub ty: Type,
    pub untyped: bool,
    pub value: ExpressionValue,
    pub constant: Option<BigInt>,
}

#[derive(Debug)]
pub(crate) struct CheckedEntry<'a> {
    pub syntax: &'a Syntax,
    pub main: Idx<Function>,
    pub expressions: ArenaMap<Idx<Expression>, CheckedExpression>,
    pub declarations: ArenaMap<Idx<Statement>, Idx<Binding>>,
    pub bindings: Arena<Binding>,
    pub assignments: ArenaMap<Idx<Statement>, Idx<Binding>>,
}

pub(crate) fn check(syntax: &Syntax) -> Result<CheckedEntry<'_>, Diagnostic> {
    let mut main = None;
    for (id, function) in syntax.functions.iter() {
        if syntax.names.resolve(&function.name) != "main" {
            return Err(Diagnostic::new(
                function.span.clone(),
                "only the `main` function is supported",
            ));
        }
        if main.replace(id).is_some() {
            return Err(Diagnostic::new(
                function.name_span.clone(),
                "duplicate `main` function",
            ));
        }
    }
    let main = main.ok_or_else(|| Diagnostic::new(0..0, "missing `main` function"))?;
    let mut checked = CheckedEntry {
        syntax,
        main,
        expressions: ArenaMap::default(),
        declarations: ArenaMap::default(),
        bindings: Arena::default(),
        assignments: ArenaMap::default(),
    };
    checked.check_body(&syntax.functions[main].body, &mut Vec::new())?;
    Ok(checked)
}

impl CheckedEntry<'_> {
    fn check_body(
        &mut self,
        body: &[Idx<Statement>],
        scopes: &mut Vec<HashMap<Spur, Idx<Binding>>>,
    ) -> Result<(), Diagnostic> {
        scopes.push(HashMap::new());
        for &statement in body {
            match &self.syntax.statements[statement].kind {
                StatementKind::Binding {
                    name,
                    mutable,
                    annotation,
                    initializer,
                    ..
                } => {
                    let destination = annotation
                        .as_ref()
                        .map(|a| Type::named(&a.name).expect("frontend validates integer types"));
                    let expression = self.check_expression(*initializer, scopes, destination)?;
                    let ty = expression.ty;
                    let constant = if *mutable {
                        None
                    } else {
                        expression.constant.clone()
                    };
                    self.expressions.insert(*initializer, expression);
                    let binding = self.bindings.alloc(Binding {
                        ty,
                        mutable: *mutable,
                        constant,
                    });
                    self.declarations.insert(statement, binding);
                    scopes.last_mut().unwrap().insert(*name, binding);
                }
                StatementKind::Assignment {
                    name,
                    name_span,
                    value,
                } => {
                    let binding = self.resolve(*name, name_span.clone(), scopes)?;
                    if !self.bindings[binding].mutable {
                        return Err(Diagnostic::new(
                            name_span.clone(),
                            format!(
                                "cannot assign to immutable binding `{}`",
                                self.syntax.names.resolve(name)
                            ),
                        ));
                    }
                    let expression =
                        self.check_expression(*value, scopes, Some(self.bindings[binding].ty))?;
                    self.expressions.insert(*value, expression);
                    self.assignments.insert(statement, binding);
                }
                StatementKind::Block { body } => self.check_body(body, scopes)?,
                StatementKind::Exit { argument } => {
                    let expression = self.check_expression(*argument, scopes, Some(Type::Int))?;
                    self.expressions.insert(*argument, expression);
                }
            }
        }
        scopes.pop();
        Ok(())
    }

    fn resolve(
        &self,
        name: Spur,
        span: std::ops::Range<usize>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<Idx<Binding>, Diagnostic> {
        scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(&name).copied())
            .ok_or_else(|| {
                Diagnostic::new(
                    span,
                    format!("unknown binding `{}`", self.syntax.names.resolve(&name)),
                )
            })
    }

    fn check_expression(
        &mut self,
        id: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
        destination: Option<Type>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let mut checked = self.infer_expression(id, scopes)?;
        if checked.untyped {
            self.concretize(id, &mut checked, destination.unwrap_or(Type::Int))?;
        } else if let Some(destination) = destination
            && checked.ty != destination
        {
            return Err(Diagnostic::new(
                self.syntax.expressions[id].span.clone(),
                format!(
                    "cannot implicitly convert `{}` to `{}`",
                    checked.ty.name(),
                    destination.name()
                ),
            ));
        }
        Ok(checked)
    }

    fn infer_expression(
        &mut self,
        id: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let expression = &self.syntax.expressions[id];
        let error = |message| Diagnostic::new(expression.span.clone(), message);
        let checked = match &expression.kind {
            ExpressionKind::Integer(spelling) => {
                let (base, digits, suffix) = integer_parts(spelling);
                debug_assert!(suffix.is_empty(), "frontend rejects literal suffixes");
                let value = BigUint::parse_bytes(digits.as_bytes(), base)
                    .expect("frontend validated integer digits");
                let value = BigInt::from(value);
                CheckedExpression {
                    ty: Type::Int,
                    untyped: true,
                    value: ExpressionValue::Integer,
                    constant: Some(value),
                }
            }
            ExpressionKind::Reference(name) => {
                let binding = self.resolve(*name, expression.span.clone(), scopes)?;
                CheckedExpression {
                    ty: self.bindings[binding].ty,
                    untyped: false,
                    value: ExpressionValue::Reference(binding),
                    constant: self.bindings[binding].constant.clone(),
                }
            }
            ExpressionKind::Grouping { expression: inner } => {
                let mut checked = self.infer_expression(*inner, scopes)?;
                let result = CheckedExpression {
                    ty: checked.ty,
                    untyped: checked.untyped,
                    value: ExpressionValue::Grouping { expression: *inner },
                    constant: checked.constant.clone(),
                };
                if result.constant.is_some() {
                    checked.constant = None;
                }
                self.expressions.insert(*inner, checked);
                result
            }
            ExpressionKind::Unary {
                operator,
                operator_span,
                operand,
            } => {
                let operand_id = *operand;
                let mut checked_operand = self.infer_expression(operand_id, scopes)?;
                if *operator == UnaryOperator::Negate
                    && !checked_operand.untyped
                    && !checked_operand.ty.signed()
                {
                    return Err(Diagnostic::new(
                        operator_span.clone(),
                        format!(
                            "unary `-` is not permitted on `{}`",
                            checked_operand.ty.name()
                        ),
                    ));
                }
                if *operator == UnaryOperator::WrappingNegate && checked_operand.untyped {
                    return Err(Diagnostic::new(
                        operator_span.clone(),
                        "wrapping negation requires a typed operand",
                    ));
                }
                let constant =
                    self.evaluate_unary(*operator, operator_span.clone(), &checked_operand)?;
                let result = CheckedExpression {
                    ty: checked_operand.ty,
                    untyped: checked_operand.untyped,
                    value: ExpressionValue::Unary {
                        operator: *operator,
                        operator_span: operator_span.clone(),
                        operand: operand_id,
                    },
                    constant,
                };
                if result.constant.is_some() {
                    checked_operand.constant = None;
                }
                self.expressions.insert(operand_id, checked_operand);
                result
            }
            ExpressionKind::Binary {
                operator,
                operator_span,
                left,
                right,
            } => {
                let left_id = *left;
                let right_id = *right;
                let mut checked_left = self.infer_expression(left_id, scopes)?;
                let mut checked_right = self.infer_expression(right_id, scopes)?;
                let shift = matches!(
                    operator,
                    BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight
                );
                let wrapping = matches!(
                    operator,
                    BinaryOperator::WrappingAdd
                        | BinaryOperator::WrappingSubtract
                        | BinaryOperator::WrappingMultiply
                );
                if wrapping && checked_left.untyped && checked_right.untyped {
                    return Err(Diagnostic::new(
                        operator_span.clone(),
                        "wrapping arithmetic requires a typed operand",
                    ));
                }
                if !shift {
                    match (checked_left.untyped, checked_right.untyped) {
                        (false, false) if checked_left.ty != checked_right.ty => {
                            return Err(Diagnostic::new(
                                operator_span.clone(),
                                format!(
                                    "binary operands have different types `{}` and `{}`",
                                    checked_left.ty.name(),
                                    checked_right.ty.name()
                                ),
                            ));
                        }
                        (false, true) => {
                            self.concretize(right_id, &mut checked_right, checked_left.ty)?;
                        }
                        (true, false) => {
                            self.concretize(left_id, &mut checked_left, checked_right.ty)?;
                        }
                        _ => {}
                    }
                } else if (checked_left.constant.is_none() || checked_right.constant.is_none())
                    && checked_right.untyped
                {
                    self.concretize(right_id, &mut checked_right, Type::Int)?;
                }
                let (ty, untyped) = if shift {
                    (checked_left.ty, checked_left.untyped)
                } else if checked_left.untyped {
                    (checked_right.ty, checked_right.untyped)
                } else {
                    (checked_left.ty, false)
                };
                let constant = self.evaluate_binary(
                    *operator,
                    operator_span.clone(),
                    ty,
                    untyped,
                    &checked_left,
                    &checked_right,
                )?;
                let result = CheckedExpression {
                    ty,
                    untyped,
                    value: ExpressionValue::Binary {
                        operator: *operator,
                        operator_span: operator_span.clone(),
                        left: left_id,
                        right: right_id,
                    },
                    constant,
                };
                if result.constant.is_some() {
                    checked_left.constant = None;
                    checked_right.constant = None;
                }
                self.expressions.insert(left_id, checked_left);
                self.expressions.insert(right_id, checked_right);
                result
            }
            ExpressionKind::Conversion {
                destination: annotation,
                truncating,
                operand,
            } => {
                let destination =
                    Type::named(&annotation.name).expect("frontend validates integer types");
                let operand_id = *operand;
                let mut checked_operand = self.infer_expression(operand_id, scopes)?;
                if checked_operand.untyped && (!*truncating || checked_operand.constant.is_none()) {
                    let operand_type = if *truncating { Type::Int } else { destination };
                    self.concretize(operand_id, &mut checked_operand, operand_type)?;
                }
                let constant = if let Some(value) = checked_operand.constant.as_ref() {
                    Some(if *truncating {
                        truncate_integer(value, destination)
                    } else if integer_fits(value, destination) {
                        value.clone()
                    } else {
                        return Err(error(format!(
                            "constant conversion to `{}` would trap",
                            destination.name()
                        )));
                    })
                } else {
                    None
                };
                let value = if constant.is_some() {
                    ExpressionValue::Integer
                } else {
                    ExpressionValue::Conversion {
                        operand: operand_id,
                        truncating: *truncating,
                    }
                };
                if constant.is_some() {
                    checked_operand.constant = None;
                }
                self.expressions.insert(operand_id, checked_operand);
                CheckedExpression {
                    ty: destination,
                    untyped: false,
                    value,
                    constant,
                }
            }
        };
        Ok(checked)
    }

    fn concretize(
        &mut self,
        id: Idx<Expression>,
        checked: &mut CheckedExpression,
        destination: Type,
    ) -> Result<(), Diagnostic> {
        debug_assert!(checked.untyped);
        let constant = concretize_value(self.syntax, id, checked, destination)?
            .expect("caller provides an untyped expression");
        self.concretize_children(id, destination, constant)?;
        Ok(())
    }

    fn concretize_stored(
        &mut self,
        id: Idx<Expression>,
        destination: Type,
    ) -> Result<(), Diagnostic> {
        let Some(constant) =
            concretize_value(self.syntax, id, &mut self.expressions[id], destination)?
        else {
            return Ok(());
        };
        self.concretize_children(id, destination, constant)?;
        Ok(())
    }

    fn concretize_children(
        &mut self,
        id: Idx<Expression>,
        destination: Type,
        constant: bool,
    ) -> Result<(), Diagnostic> {
        let (first, second) = match &self.syntax.expressions[id].kind {
            ExpressionKind::Grouping { expression } => (Some(*expression), None),
            ExpressionKind::Unary {
                operator,
                operator_span,
                operand,
            } if !constant => {
                if *operator == UnaryOperator::Negate && !destination.signed() {
                    return Err(Diagnostic::new(
                        operator_span.clone(),
                        format!("unary `-` is not permitted on `{}`", destination.name()),
                    ));
                }
                (Some(*operand), None)
            }
            ExpressionKind::Binary {
                operator: BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight,
                left,
                ..
            } if !constant => (Some(*left), None),
            ExpressionKind::Binary { left, right, .. } if !constant => (Some(*left), Some(*right)),
            _ => (None, None),
        };
        for child in first.into_iter().chain(second) {
            self.concretize_stored(child, destination)?;
        }
        Ok(())
    }

    fn evaluate_unary(
        &self,
        operator: UnaryOperator,
        operator_span: std::ops::Range<usize>,
        operand: &CheckedExpression,
    ) -> Result<Option<BigInt>, Diagnostic> {
        let Some(value) = operand.constant.as_ref() else {
            return Ok(None);
        };
        let result = match operator {
            UnaryOperator::Negate => -value,
            UnaryOperator::WrappingNegate => truncate_integer(&-value, operand.ty),
            UnaryOperator::Complement if operand.untyped => !value,
            UnaryOperator::Complement => truncate_integer(&!value, operand.ty),
        };
        if operand.untyped && result.bits() > MAX_UNTYPED_INTEGER_BITS {
            return Err(Diagnostic::new(
                operator_span,
                "constant expression exceeds compiler resource limit",
            ));
        }
        if operator == UnaryOperator::Negate
            && !operand.untyped
            && !integer_fits(&result, operand.ty)
        {
            return Err(Diagnostic::new(
                operator_span,
                format!("constant unary `-` on `{}` would trap", operand.ty.name()),
            ));
        }
        Ok(Some(result))
    }

    fn evaluate_binary(
        &self,
        operator: BinaryOperator,
        operator_span: std::ops::Range<usize>,
        ty: Type,
        untyped: bool,
        left: &CheckedExpression,
        right: &CheckedExpression,
    ) -> Result<Option<BigInt>, Diagnostic> {
        let right_constant = right.constant.as_ref();
        if matches!(operator, BinaryOperator::Divide | BinaryOperator::Remainder)
            && right_constant == Some(&BigInt::from(0u8))
        {
            return Err(Diagnostic::new(
                operator_span,
                format!("constant `{}` divisor is zero", operator.spelling()),
            ));
        }
        if matches!(
            operator,
            BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight
        ) && right_constant.is_some_and(|count| count < &BigInt::from(0u8))
        {
            return Err(Diagnostic::new(
                operator_span,
                "constant shift count is negative",
            ));
        }

        let (Some(left), Some(right)) = (left.constant.as_ref(), right_constant) else {
            return Ok(None);
        };
        if untyped
            && operator == BinaryOperator::Multiply
            && left.bits().saturating_add(right.bits()).saturating_sub(1) > MAX_UNTYPED_INTEGER_BITS
        {
            return Err(Diagnostic::new(
                operator_span,
                "constant expression exceeds compiler resource limit",
            ));
        }
        let result = match operator {
            BinaryOperator::Multiply => left * right,
            BinaryOperator::Divide => {
                if !untyped && is_minimum(left, ty) && right == &BigInt::from(-1) {
                    return Err(Diagnostic::new(
                        operator_span,
                        format!("constant `/` on `{}` would trap", ty.name()),
                    ));
                }
                left / right
            }
            BinaryOperator::Remainder => {
                if !untyped && is_minimum(left, ty) && right == &BigInt::from(-1) {
                    return Err(Diagnostic::new(
                        operator_span,
                        format!("constant `%` on `{}` would trap", ty.name()),
                    ));
                }
                left % right
            }
            BinaryOperator::WrappingMultiply => truncate_integer(&(left * right), ty),
            BinaryOperator::Add => left + right,
            BinaryOperator::Subtract => left - right,
            BinaryOperator::WrappingAdd => truncate_integer(&(left + right), ty),
            BinaryOperator::WrappingSubtract => truncate_integer(&(left - right), ty),
            BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => {
                return self.evaluate_shift(operator, operator_span, ty, untyped, left, right);
            }
            BinaryOperator::And => left & right,
            BinaryOperator::AndNot => left & !right,
            BinaryOperator::Xor => left ^ right,
            BinaryOperator::Or => left | right,
        };
        let checked_arithmetic = matches!(
            operator,
            BinaryOperator::Multiply | BinaryOperator::Add | BinaryOperator::Subtract
        );
        if checked_arithmetic && !untyped && !integer_fits(&result, ty) {
            return Err(Diagnostic::new(
                operator_span,
                format!(
                    "constant `{}` on `{}` would overflow",
                    operator.spelling(),
                    ty.name()
                ),
            ));
        }
        if untyped && result.bits() > MAX_UNTYPED_INTEGER_BITS {
            return Err(Diagnostic::new(
                operator_span,
                "constant expression exceeds compiler resource limit",
            ));
        }
        Ok(Some(result))
    }

    fn evaluate_shift(
        &self,
        operator: BinaryOperator,
        operator_span: std::ops::Range<usize>,
        ty: Type,
        untyped: bool,
        left: &BigInt,
        right: &BigInt,
    ) -> Result<Option<BigInt>, Diagnostic> {
        debug_assert!(right >= &BigInt::from(0u8));
        if untyped {
            const MAX_CONSTANT_SHIFT: usize = 1_000_000;
            let count = right.to_usize();
            if operator == BinaryOperator::ShiftRight && count.is_none() {
                return Ok(Some(if left < &BigInt::from(0u8) {
                    BigInt::from(-1)
                } else {
                    BigInt::from(0u8)
                }));
            }
            let Some(count) = count else {
                return Err(Diagnostic::new(
                    operator_span,
                    "constant shift exceeds compiler resource limit",
                ));
            };
            if operator == BinaryOperator::ShiftLeft
                && (count > MAX_CONSTANT_SHIFT
                    || count
                        .try_into()
                        .unwrap_or(u64::MAX)
                        .saturating_add(left.bits())
                        > MAX_UNTYPED_INTEGER_BITS)
            {
                return Err(Diagnostic::new(
                    operator_span,
                    "constant expression exceeds compiler resource limit",
                ));
            }
            return Ok(Some(if operator == BinaryOperator::ShiftLeft {
                left << count
            } else {
                left >> count
            }));
        }

        if right >= &BigInt::from(ty.width()) {
            return Ok(Some(
                if operator == BinaryOperator::ShiftRight
                    && ty.signed()
                    && left < &BigInt::from(0u8)
                {
                    BigInt::from(-1)
                } else {
                    BigInt::from(0u8)
                },
            ));
        }
        let count = right
            .to_usize()
            .expect("count below every Fern integer width fits usize");
        Ok(Some(if operator == BinaryOperator::ShiftLeft {
            truncate_integer(&(left << count), ty)
        } else {
            left >> count
        }))
    }
}

const MAX_UNTYPED_INTEGER_BITS: u64 = 2_000_000;

fn concretize_value(
    syntax: &Syntax,
    id: Idx<Expression>,
    checked: &mut CheckedExpression,
    destination: Type,
) -> Result<Option<bool>, Diagnostic> {
    if !checked.untyped {
        return Ok(None);
    }
    if checked
        .constant
        .as_ref()
        .is_some_and(|value| !integer_fits(value, destination))
    {
        return Err(out_of_range(syntax, id, destination));
    }
    let constant = checked.constant.is_some();
    checked.ty = destination;
    checked.untyped = false;
    Ok(Some(constant))
}

fn out_of_range(syntax: &Syntax, id: Idx<Expression>, destination: Type) -> Diagnostic {
    let literal = matches!(syntax.expressions[id].kind, ExpressionKind::Integer(_));
    Diagnostic::new(
        syntax.expressions[id].span.clone(),
        format!(
            "integer {} out of range for `{}`",
            if literal { "literal" } else { "value" },
            destination.name()
        ),
    )
}

fn is_minimum(value: &BigInt, ty: Type) -> bool {
    value == &BigInt::from(ty.min())
}

fn integer_fits(value: &BigInt, ty: Type) -> bool {
    value >= &BigInt::from(ty.min()) && value <= &BigInt::from(ty.max())
}

fn integer_from_bits(bits: BigInt, ty: Type) -> BigInt {
    if ty.signed() && bits >= (BigInt::from(1u8) << (ty.width() - 1)) {
        bits - (BigInt::from(1u8) << ty.width())
    } else {
        bits
    }
}

fn truncate_integer(value: &BigInt, ty: Type) -> BigInt {
    let modulus = BigInt::from(1u8) << ty.width();
    let bits = ((value % &modulus) + &modulus) % &modulus;
    integer_from_bits(bits, ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::parse;

    fn literal(value: u128, base: u32) -> String {
        match base {
            2 => format!("0b{value:b}"),
            8 => format!("0o{value:o}"),
            10 => value.to_string(),
            16 => format!("0x{value:X}"),
            _ => unreachable!(),
        }
    }

    fn big(value: i128) -> BigInt {
        BigInt::from(value)
    }

    #[test]
    fn bindings_have_concrete_types_and_distinct_identities() {
        let text =
            "fn main() -> void { const x = 1; var x: int = x; const x = x; exit(x); exit(0); }";
        let syntax = parse(text).unwrap();
        let checked = check(&syntax).unwrap();
        assert!(std::ptr::eq(checked.syntax, &syntax));
        let statements = &syntax.functions[checked.main].body;
        let ids: Vec<_> = statements[..3]
            .iter()
            .map(|s| checked.declarations[*s])
            .collect();
        assert_ne!(ids[0], ids[1]);
        assert_ne!(ids[1], ids[2]);
        assert_ne!(ids[0], ids[2]);
        for id in &ids {
            assert_eq!(checked.bindings[*id].ty, Type::Int);
        }
        let facts: Vec<_> = syntax
            .expressions
            .iter()
            .map(|(id, _)| &checked.expressions[id])
            .collect();
        assert_eq!(facts.len(), 5);
        assert!(facts.iter().all(|fact| fact.ty == Type::Int));
        assert_eq!(facts[0].value, ExpressionValue::Integer);
        assert_eq!(facts[1].value, ExpressionValue::Reference(ids[0]));
        assert_eq!(facts[2].value, ExpressionValue::Reference(ids[1]));
        assert_eq!(facts[3].value, ExpressionValue::Reference(ids[2]));
        assert_eq!(facts[4].value, ExpressionValue::Integer);
    }

    #[test]
    fn nested_scopes_resolve_binding_identity_and_mutability() {
        let syntax = parse("fn main() -> void { var x = 1; { x = 2; const x = x; { var x = x; x = x; } exit(x); } x = x; const x = x; exit(x); }").unwrap();
        let checked = check(&syntax).unwrap();
        let ids: Vec<_> = checked.bindings.iter().map(|(id, _)| id).collect();
        assert_eq!(ids.len(), 4);
        let mutable: Vec<_> = checked.bindings.iter().map(|(_, b)| b.mutable).collect();
        assert_eq!(mutable, [true, false, true, false]);
        let targets: Vec<_> = checked.assignments.iter().map(|(_, id)| *id).collect();
        assert_eq!(targets, [ids[0], ids[2], ids[0]]);
        let references: Vec<_> = checked
            .expressions
            .iter()
            .filter_map(|(_, expression)| match &expression.value {
                ExpressionValue::Reference(id) => Some(*id),
                ExpressionValue::Integer
                | ExpressionValue::Conversion { .. }
                | ExpressionValue::Grouping { .. }
                | ExpressionValue::Unary { .. }
                | ExpressionValue::Binary { .. } => None,
            })
            .collect();
        assert_eq!(
            references,
            [ids[0], ids[1], ids[2], ids[1], ids[0], ids[0], ids[3]]
        );
    }

    #[test]
    fn assignment_errors_use_target_or_value_spans_even_after_nested_exit() {
        for (body, offending, message) in [
            (
                "const x = 1; x = 2;",
                "x",
                "cannot assign to immutable binding `x`",
            ),
            (
                "var x = 1; const x = 2; x = 3;",
                "x",
                "cannot assign to immutable binding `x`",
            ),
            (
                "var x = 1; { const x = 2; x = 3; }",
                "x",
                "cannot assign to immutable binding `x`",
            ),
            (
                "const x = 1; { var x = 2; x = 3; } x = 4;",
                "x",
                "cannot assign to immutable binding `x`",
            ),
            ("x = 2;", "x", "unknown binding `x`"),
            ("{ var x = 1; } x = 2;", "x", "unknown binding `x`"),
            ("{ const x = 1; } exit(x);", "x", "unknown binding `x`"),
            ("{ var x = x; }", "x", "unknown binding `x`"),
            (
                "var x = 1; { x = missing; }",
                "missing",
                "unknown binding `missing`",
            ),
            (
                "var x: u8 = 1; { x = 256; }",
                "256",
                "integer literal out of range for `u8`",
            ),
        ] {
            rejects(body, offending, message);
            rejects(&format!("{{ exit(0); }} {body}"), offending, message);
            rejects(&format!("{{ exit(0); {body} }}"), offending, message);
        }
        for body in [
            "const x = 1; var x = x; x = 2;",
            "const x = 1; { var x = x; x = 2; }",
            "var x = 1; { const x = x; } x = 2;",
        ] {
            let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
            check(&syntax).unwrap();
        }
    }

    fn rejects(body: &str, offending: &str, message: &str) {
        let text = format!("/* 🌿 */ fn main() -> void {{ {body} }}");
        let syntax = parse(&text).unwrap();
        let error = check(&syntax).unwrap_err();
        let start = text.rfind(offending).unwrap();
        assert_eq!(error.span, start..start + offending.len(), "{body}");
        assert_eq!(error.message, message, "{body}");
    }

    fn accepts(body: &str) {
        let text = format!("fn main() -> void {{ {body} }}");
        let syntax = parse(&text).unwrap();
        check(&syntax).unwrap();
    }

    #[test]
    fn integer_expression_types_follow_operand_rules() {
        let body = "var a: u8 = 1;
             var b: u8 = 2;
             const add = a + 2;
             const reverse = 2 + a;
             const shift = a << u64(3);
             const exact: u16 = 1 + 2;
             const negative: i8 = -128;
             const complemented = ^a;
             const wrapped = a &+ 1;
             const wrapped_negative = &-a;
             exit(0);
             const after = a &^ b;";
        let text = format!("fn main() -> void {{ {body} }}");
        let syntax = parse(&text).unwrap();
        let checked = check(&syntax).unwrap();
        let types: Vec<_> = checked
            .bindings
            .iter()
            .map(|(_, binding)| binding.ty)
            .collect();
        assert_eq!(
            types,
            [
                Type::U8,
                Type::U8,
                Type::U8,
                Type::U8,
                Type::U8,
                Type::U16,
                Type::I8,
                Type::U8,
                Type::U8,
                Type::U8,
                Type::U8,
            ]
        );

        for left in Type::ALL {
            let left_name = left.name();
            for right in Type::ALL {
                let right_name = right.name();
                let body = format!(
                    "var left: {left_name} = 1; var right: {right_name} = 1; const result = left + right;"
                );
                if left == right {
                    accepts(&body);
                } else {
                    rejects(
                        &body,
                        "+",
                        &format!(
                            "binary operands have different types `{left_name}` and `{right_name}`"
                        ),
                    );
                }

                accepts(&format!(
                    "var left: {left_name} = 1; var count: {right_name} = 1; const result = left << count;"
                ));
            }
        }

        for body in [
            "var x: u8 = 1; const y = x + 256;",
            "var x: u8 = 1; const y = 256 + x;",
        ] {
            rejects(body, "256", "integer literal out of range for `u8`");
        }
        for (body, offending, message) in [
            (
                "const x = 1 &+ 2;",
                "&+",
                "wrapping arithmetic requires a typed operand",
            ),
            (
                "const x = &-1;",
                "&-",
                "wrapping negation requires a typed operand",
            ),
            (
                "const x = -u8(1);",
                "-",
                "unary `-` is not permitted on `u8`",
            ),
        ] {
            rejects(body, offending, message);
        }
        accepts("var x: u8 = 1; { var x: u16 = 2; const inner = x + 1; } x = x + 1; exit(0);");
        rejects(
            "exit(0); const after = 1 + missing;",
            "missing",
            "unknown binding `missing`",
        );
    }

    #[test]
    fn signed_minima_and_expression_constants_keep_their_contracts() {
        for (name, minimum) in [
            ("i8", 1u128 << 7),
            ("i16", 1u128 << 15),
            ("i32", 1u128 << 31),
            ("i64", 1u128 << 63),
            ("int", 1u128 << (usize::BITS - 1)),
        ] {
            accepts(&format!(
                "const direct: {name} = -{minimum}; const converted = {name}(-{minimum});"
            ));
            let invalid = minimum + 1;
            rejects(
                &format!("const value: {name} = -{invalid};"),
                &format!("-{invalid}"),
                &format!("integer value out of range for `{name}`"),
            );
        }

        let body = "const exact = 1 + 2;
             const copy = exact;
             var runtime = 1;
             const mixed = runtime + 2;
             { const exact = runtime; const shadowed = exact + 1; }
             exit(0);
             const after = copy ^ 1;";
        let text = format!("fn main() -> void {{ {body} }}");
        let syntax = parse(&text).unwrap();
        let checked = check(&syntax).unwrap();
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.constant.is_some())
                .collect::<Vec<_>>(),
            [true, true, false, false, false, false, true]
        );
    }

    #[test]
    fn constant_evaluation_preserves_exact_and_typed_operations() {
        let text = "fn main() -> void {
            const exact: u8 = (250 + 10) / 2;
            const ordinary = 9 * 5 - 3;
            const quotient = -7 / 3;
            const remainder = -7 % 3;
            const complement = ^0;
            const cleared = 15 &^ 3;
            const bits = (12 & 10) ^ 3 | 16;
            const large: i64 = 1 << 40;
            const signed_shift = -8 >> 2;
            const wrapped_add = u8(250) &+ 10;
            const wrapped_subtract = u8(1) &- 2;
            const wrapped_multiply = u8(200) &* 2;
            const wrapped_negate = &-u8(1);
            const high: u8 = 128;
            const discarded = high << 1;
            const negative: i8 = -1;
            const sign_fill = negative >> 8;
            const huge = u8.truncate((1 << 255) + 42);
            const distant_bit = u8.truncate(((1 << 1000000) * 2) >> 1000001);
            const copy = exact;
            const combined = copy + u8(1);
            var runtime = 2;
            const saved = runtime;
            const not_constant = saved + 1;
        }";
        let syntax = parse(text).unwrap();
        let checked = check(&syntax).unwrap();
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.constant.clone())
                .collect::<Vec<_>>(),
            [
                Some(big(130)),
                Some(big(42)),
                Some(big(-2)),
                Some(big(-1)),
                Some(big(-1)),
                Some(big(12)),
                Some(big(27)),
                Some(BigInt::from(1u8) << 40),
                Some(big(-2)),
                Some(big(4)),
                Some(big(255)),
                Some(big(144)),
                Some(big(255)),
                Some(big(128)),
                Some(big(0)),
                Some(big(-1)),
                Some(big(-1)),
                Some(big(42)),
                Some(big(1)),
                Some(big(130)),
                Some(big(131)),
                None,
                None,
                None,
            ]
        );
    }

    #[test]
    fn constant_failures_are_diagnosed_before_runtime_lowering() {
        for (body, offending, message) in [
            (
                "const x = u8(255) + 1;",
                "+",
                "constant `+` on `u8` would overflow",
            ),
            (
                "const x = u8(0) - 1;",
                "-",
                "constant `-` on `u8` would overflow",
            ),
            (
                "const x = u8(128) * 2;",
                "*",
                "constant `*` on `u8` would overflow",
            ),
            (
                "const minimum: i8 = -128; const x = -minimum;",
                "-",
                "constant unary `-` on `i8` would trap",
            ),
            (
                "const x = i8(-128) / -1;",
                "/",
                "constant `/` on `i8` would trap",
            ),
            (
                "const x = i8(-128) % -1;",
                "%",
                "constant `%` on `i8` would trap",
            ),
            (
                "var x: u8 = 1; const y = x / 0;",
                "/",
                "constant `/` divisor is zero",
            ),
            (
                "var x: u8 = 1; const y = x % 0;",
                "%",
                "constant `%` divisor is zero",
            ),
            (
                "var x: u8 = 1; const y = x << -1;",
                "<<",
                "constant shift count is negative",
            ),
            (
                "const x: u8 = 250 + 10;",
                "250 + 10",
                "integer value out of range for `u8`",
            ),
            (
                "const x = u8(250 + 10);",
                "250 + 10",
                "integer value out of range for `u8`",
            ),
        ] {
            rejects(body, offending, message);
            rejects(&format!("exit(0); {body}"), offending, message);
        }
        rejects(
            "var runtime: u8 = 1; const outer = runtime + (u8(255) + 1);",
            "+",
            "constant `+` on `u8` would overflow",
        );
    }

    #[test]
    fn constant_classification_does_not_depend_on_binding_mutability() {
        let text = "fn main() -> void {
            const immutable = 1 + 2;
            var mutable = 1 + 2;
            const immutable_copy = immutable;
            const mutable_copy = mutable;
        }";
        let syntax = parse(text).unwrap();
        let checked = check(&syntax).unwrap();
        let expression_constants: Vec<_> = syntax.functions[checked.main]
            .body
            .iter()
            .map(|statement| match syntax.statements[*statement].kind {
                StatementKind::Binding { initializer, .. } => {
                    checked.expressions[initializer].constant.clone()
                }
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(
            expression_constants,
            [Some(big(3)), Some(big(3)), Some(big(3)), None]
        );
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.constant.clone())
                .collect::<Vec<_>>(),
            [Some(big(3)), None, Some(big(3)), None]
        );
    }

    #[test]
    fn revised_precedence_wrapping_and_shift_boundaries_are_preserved() {
        let text = "fn main() -> void {
            const precedence_left = 1 + 2 << 1;
            const precedence_right = 1 << 2 + 1;
            const same_level = 15 &^ 3 & 6;
            const bit_levels = 1 | 2 ^ 3 & 4;
            const wrapped_left = u8(250) &+ 10;
            const wrapped_right = 250 &+ u8(10);
            const high: u8 = 128;
            const discarded = high << 1;
            var runtime_high: u8 = 128;
            const runtime_discarded = runtime_high << 1;
            const overshift = u8(1) << 999999999999999999999999999999999999;
            const negative: i8 = -1;
            const sign_fill = negative >> 8;
            const exact: u16 = 1 << 8;
            const reduced: u8 = 256 >> u8(8);
        }";
        let syntax = parse(text).unwrap();
        let checked = check(&syntax).unwrap();
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.constant.clone())
                .collect::<Vec<_>>(),
            [
                Some(big(6)),
                Some(big(8)),
                Some(big(4)),
                Some(big(3)),
                Some(big(4)),
                Some(big(4)),
                Some(big(128)),
                Some(big(0)),
                None,
                None,
                Some(big(0)),
                Some(big(-1)),
                Some(big(-1)),
                Some(big(256)),
                Some(big(1)),
            ]
        );

        for (body, offending, message) in [
            (
                "const invalid: u8 = 250 &+ 10;",
                "&+",
                "wrapping arithmetic requires a typed operand",
            ),
            (
                "const invalid = &-1;",
                "&-",
                "wrapping negation requires a typed operand",
            ),
            (
                "const typed = u8(1) &+ 256;",
                "256",
                "integer literal out of range for `u8`",
            ),
            (
                "const invalid: u8 = 1 << 8;",
                "1 << 8",
                "integer value out of range for `u8`",
            ),
        ] {
            rejects(body, offending, message);
            rejects(&format!("exit(0); {body}"), offending, message);
        }
    }

    #[test]
    fn contextual_types_reach_nested_runtime_integer_expressions() {
        let text = "fn main() -> void {
            var count: uint = 1;
            const arithmetic: u64 = (1 << count) + 1;
            const negated: i64 = -(1 << count);
            const complemented: u16 = ^(1 << count);
        }";
        let syntax = parse(text).unwrap();
        let checked = check(&syntax).unwrap();
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.ty)
                .collect::<Vec<_>>(),
            [Type::Uint, Type::U64, Type::I64, Type::U16]
        );
        assert!(
            checked
                .expressions
                .iter()
                .all(|(_, expression)| !expression.untyped)
        );

        rejects(
            "var count: uint = 1; const invalid: u8 = -(1 << count);",
            "-",
            "unary `-` is not permitted on `u8`",
        );

        let too_large = BigInt::from(Type::Int.max()) + 1u8;
        rejects(
            &format!("var count: uint = 1; const invalid = u8.truncate({too_large} << count);"),
            &too_large.to_string(),
            "integer literal out of range for `int`",
        );
    }

    #[test]
    fn nonconstant_untyped_shift_counts_are_concretized_and_range_checked() {
        accepts("var value: u8 = 1; var n = 1; const shifted = value << ((1 << 2) << n);");

        for body in [
            "var value: u8 = 1; var n = 1; const shifted = value << ((1 << 200) << n);",
            "var value: u8 = 1; var n = 1; const shifted = value << ((1 << 63) << n);",
            "exit(0); var value: u8 = 1; var n = 1; const shifted = value << ((1 << 200) << n);",
        ] {
            let text = format!("fn main() -> void {{ {body} }}");
            let syntax = parse(&text).unwrap();
            let error = check(&syntax).unwrap_err();
            assert_eq!(error.message, "integer value out of range for `int`");
            assert!(text[error.span].contains("1 <<"));
        }
    }

    #[test]
    fn untyped_constant_folding_is_bounded_and_discards_child_values() {
        for body in [
            "const value = (1 << 1000000) * (1 << 1000000);",
            "exit(0); const value = (1 << 1000000) * (1 << 1000000);",
        ] {
            rejects(
                body,
                "*",
                "constant expression exceeds compiler resource limit",
            );
        }

        let syntax = parse("fn main() -> void { const value = (1 + 2) * (3 + 4); }").unwrap();
        let checked = check(&syntax).unwrap();
        let root = match syntax.statements[syntax.functions[checked.main].body[0]].kind {
            StatementKind::Binding { initializer, .. } => initializer,
            _ => unreachable!(),
        };
        assert_eq!(checked.expressions[root].constant, Some(big(21)));
        assert!(
            checked
                .expressions
                .iter()
                .all(|(id, expression)| id == root || expression.constant.is_none())
        );
    }

    #[test]
    fn names_require_a_preceding_binding_even_after_exit() {
        for body in [
            "const x = x;",
            "exit(x); const x = 1;",
            "var y = x;",
            "exit(0); exit(x);",
            "const y = 1; exit(0); var x = x;",
        ] {
            // In the forward-reference case the later declaration has the same name.
            let text = format!("/* 🌿 */ fn main() -> void {{ {body} }}");
            let syntax = parse(&text).unwrap();
            let error = check(&syntax).unwrap_err();
            let start = if body.starts_with("exit(x)") {
                text.find("exit(x)").unwrap() + 5
            } else {
                text.rfind('x').unwrap()
            };
            assert_eq!(error.span, start..start + 1);
            assert_eq!(error.message, "unknown binding `x`");
        }
    }

    #[test]
    fn integer_contract_uses_contextual_literals_and_exact_references() {
        for ty in Type::ALL {
            let name = ty.name();
            let max = u128::from(ty.max());
            for base in [2, 8, 10, 16] {
                let maximum = literal(max, base);
                let syntax = parse(&format!(
                    "fn main() -> void {{ var x: {name} = {maximum}; x = {maximum}; }}"
                ))
                .unwrap();
                let checked = check(&syntax).unwrap();
                assert!(checked.bindings.iter().all(|(_, binding)| binding.ty == ty));
                assert!(
                    checked
                        .expressions
                        .iter()
                        .all(|(_, expression)| expression.ty == ty)
                );

                let overflow = literal(max + 1, base);
                rejects(
                    &format!("exit(0); var x: {name} = {overflow};"),
                    &overflow,
                    &format!("integer literal out of range for `{name}`"),
                );
            }
        }

        for source in Type::ALL {
            let source_name = source.name();
            for destination in Type::ALL {
                let destination_name = destination.name();
                for target in [
                    format!("const target: {destination_name} = source;"),
                    format!("var target: {destination_name} = source;"),
                    format!("var target: {destination_name} = 0; target = source;"),
                ] {
                    for scope in [target.clone(), format!("{{ exit(0); {target} }}")] {
                        let body = format!("const source: {source_name} = 1; {scope}");
                        if source == destination {
                            let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
                            check(&syntax).unwrap();
                        } else {
                            rejects(
                                &body,
                                "source",
                                &format!(
                                    "cannot implicitly convert `{source_name}` to `{destination_name}`"
                                ),
                            );
                        }
                    }
                }
            }
            let body = format!("const source: {source_name} = 1; exit(source);");
            if source == Type::Int {
                let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
                check(&syntax).unwrap();
            } else {
                rejects(
                    &body,
                    "source",
                    &format!("cannot implicitly convert `{source_name}` to `int`"),
                );
            }
        }
    }

    #[test]
    fn conversions_evaluate_constants_and_preserve_runtime_operands() {
        let syntax = parse(
            "fn main() -> void {
                const literal: u8 = 42;
                const reduced = u8.truncate(340282366920938463463374607431768211498);
                const signed = i8.truncate(255);
                const widened = u64(literal);
                var runtime: u64 = 42;
                const checked = u8(runtime);
                const truncated = u8.truncate(runtime);
                const later = u16(checked);
            }",
        )
        .unwrap();
        let checked = check(&syntax).unwrap();
        let constants: Vec<_> = checked
            .bindings
            .iter()
            .map(|(_, binding)| binding.constant.clone())
            .collect();
        assert_eq!(
            constants,
            [
                Some(big(42)),
                Some(big(42)),
                Some(big(-1)),
                Some(big(42)),
                None,
                None,
                None,
                None
            ]
        );
        assert_eq!(
            checked
                .expressions
                .iter()
                .filter(|(_, expression)| {
                    matches!(expression.value, ExpressionValue::Conversion { .. })
                })
                .count(),
            3
        );

        rejects(
            "const value: u64 = 18446744073709551615; const narrowed = u8(value);",
            "u8(value)",
            "constant conversion to `u8` would trap",
        );
        rejects(
            "const narrowed = u8(256);",
            "256",
            "integer literal out of range for `u8`",
        );
        rejects(
            "exit(0); const value: u64 = 256; const narrowed = u8(value);",
            "u8(value)",
            "constant conversion to `u8` would trap",
        );
    }

    #[test]
    fn truncating_constants_keep_at_least_256_bits_before_reduction() {
        for literal in [
            format!("0x{}", "F".repeat(64)),
            format!("0b{}", "1".repeat(256)),
            format!("0x1{}FF", "0".repeat(128)),
        ] {
            let syntax = parse(&format!(
                "fn main() -> void {{ const result = int(u8.truncate({literal})); }}"
            ))
            .unwrap();
            let checked = check(&syntax).unwrap();
            assert_eq!(
                checked.bindings.iter().next().unwrap().1.constant,
                Some(big(255))
            );
            rejects(
                &format!("exit(0); const result = u8.truncate(u64({literal}));"),
                &literal,
                "integer literal out of range for `u64`",
            );
        }
    }

    #[test]
    fn constant_classification_follows_copies_and_shadowing() {
        let syntax = parse(
            "fn main() -> void {
                const original: u64 = 255;
                const copy = original;
                { var original = copy;
                  const saved = original;
                  const converted = u8(saved); }
                const folded = i8.truncate(copy);
                const extended = i64(folded);
                const wrapped = u64.truncate(extended);
            }",
        )
        .unwrap();
        let checked = check(&syntax).unwrap();
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.constant.clone())
                .collect::<Vec<_>>(),
            [
                Some(big(255)),
                Some(big(255)),
                None,
                None,
                None,
                Some(big(-1)),
                Some(big(-1)),
                Some(BigInt::from(u64::MAX))
            ],
        );
        for body in [
            "const original: u64 = 256; const copy = original; const bad = u8(copy);",
            "const original: u64 = 256; { var original: u64 = 1; } const bad = u8(original);",
            "const negative = i8.truncate(255); const bad = u64(negative);",
        ] {
            let syntax = parse(&format!("fn main() -> void {{ exit(0); {body} }}")).unwrap();
            assert!(check(&syntax).unwrap_err().message.contains("would trap"));
        }
    }

    #[test]
    fn native_integer_widths_follow_the_host_and_model_both_specified_widths() {
        assert_eq!(Type::Int.width(), usize::BITS);
        assert_eq!(Type::Uint.width(), usize::BITS);
        for pointer_width in [32, 64] {
            assert_eq!(Type::Int.width_on(pointer_width), pointer_width);
            assert_eq!(Type::Uint.width_on(pointer_width), pointer_width);
            assert_eq!(Type::I32.width_on(pointer_width), 32);
            assert_eq!(Type::U64.width_on(pointer_width), 64);
        }
    }

    #[test]
    fn entry_checks_keep_their_spans_and_precede_body_checks() {
        for (text, span, message) in [
            ("/* 🌿 */", 0..0, "missing `main` function"),
            (
                "fn main() -> void {} fn main() -> void {}",
                24..28,
                "duplicate `main` function",
            ),
            (
                "fn other() -> void {}",
                0..21,
                "only the `main` function is supported",
            ),
        ] {
            let syntax = parse(text).unwrap();
            let error = check(&syntax).unwrap_err();
            assert_eq!(error.span, span);
            assert_eq!(error.message, message);
        }
        let syntax = parse("fn main() -> void { exit(x); } fn other() -> void {}").unwrap();
        assert_eq!(
            check(&syntax).unwrap_err().message,
            "only the `main` function is supported"
        );
    }
}
