//! Expression inference, contextualization, indexing, lengths, and array literals.

use crate::{
    diagnostic::Diagnostic,
    frontend::syntax::{Expression, ExpressionKind, find_call},
    types::{BinaryOperator, ComparisonOperator, LogicalOperator, Scalar, Type, UnaryOperator},
};
use la_arena::Idx;
use lasso::Spur;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use std::collections::HashMap;

use super::{annotations::check_element_count, constants::*, model::*};

impl CheckedProgram<'_> {
    pub(super) fn check_expression(
        &mut self,
        id: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
        destination: Option<Type>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let syntax = self.syntax;
        // An array literal is the one expression whose type comes from its
        // destination rather than from concretizing an inferred type.
        if matches!(
            syntax.expressions[id].kind,
            ExpressionKind::ArrayLiteral { .. }
        ) {
            return self.check_array_literal(id, scopes, destination);
        }
        let mut checked = self.infer_expression(id, scopes)?;
        let mismatch = |checked: &CheckedExpression, destination: &Type| {
            Diagnostic::new(
                syntax.expressions[id].span.clone(),
                format!(
                    "cannot implicitly convert `{}` to `{destination}`",
                    checked.ty
                ),
            )
        };
        if checked.untyped {
            let destination = destination.unwrap_or_else(|| checked.ty.clone());
            let Some(scalar) = destination.scalar() else {
                return Err(mismatch(&checked, &destination));
            };
            self.concretize(id, &mut checked, scalar)?;
        } else if let Some(destination) = destination
            && checked.ty != destination
        {
            return Err(mismatch(&checked, &destination));
        }
        Ok(checked)
    }

    pub(super) fn infer_expression(
        &mut self,
        id: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let expression = &self.syntax.expressions[id];
        match &expression.kind {
            ExpressionKind::Integer(spelling) => Ok(integer_literal(spelling)),
            ExpressionKind::Boolean(value) => Ok(CheckedExpression {
                ty: Scalar::Bool.into(),
                untyped: true,
                value: ExpressionValue::Boolean,
                constant: Some(BigInt::from(*value).into()),
            }),
            ExpressionKind::Reference(name) => {
                let binding = self.resolve(name, scopes)?;
                Ok(CheckedExpression {
                    ty: self.bindings[binding].ty.clone(),
                    untyped: false,
                    value: ExpressionValue::Reference(binding),
                    constant: self.bindings[binding].constant.clone(),
                })
            }
            ExpressionKind::Grouping { expression } => self.infer_grouping(*expression, scopes),
            ExpressionKind::Unary {
                operator,
                operator_span,
                operand,
            } => self.infer_unary(*operator, operator_span, *operand, scopes),
            ExpressionKind::Binary {
                operator,
                operator_span,
                left,
                right,
            } => self.infer_binary(*operator, operator_span, *left, *right, scopes),
            ExpressionKind::Comparison {
                operator,
                operator_span,
                left,
                right,
            } => self.infer_comparison(*operator, operator_span, *left, *right, scopes),
            ExpressionKind::Logical {
                operator,
                operator_span,
                left,
                right,
            } => self.infer_logical(*operator, operator_span, *left, *right, scopes),
            ExpressionKind::LogicalNot {
                operator_span,
                operand,
            } => self.infer_logical_not(operator_span, *operand, scopes),
            ExpressionKind::Conversion {
                destination,
                truncating,
                operand,
                ..
            } => self.infer_conversion(
                *destination,
                *truncating,
                *operand,
                &expression.span,
                scopes,
            ),
            ExpressionKind::ArrayLiteral { .. } => self.check_array_literal(id, scopes, None),
            ExpressionKind::Index { operand, index, .. } => {
                self.infer_index(*operand, *index, scopes)
            }
            ExpressionKind::Length { operand } => self.infer_length(*operand, scopes),
            ExpressionKind::Call(call) => {
                let (function, result) = self.check_call(call, scopes, true)?;
                Ok(CheckedExpression {
                    ty: result.expect("value-context call has a value result"),
                    untyped: false,
                    value: ExpressionValue::Call { function },
                    constant: None,
                })
            }
        }
    }

    /// Checks an array literal. Every element is checked against the array's
    /// element type, which the destination supplies when there is one and the
    /// elements themselves supply when there is not.
    fn check_array_literal(
        &mut self,
        id: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
        destination: Option<Type>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let span = self.syntax.expressions[id].span.clone();
        let ExpressionKind::ArrayLiteral { elements, fill } = &self.syntax.expressions[id].kind
        else {
            unreachable!("an array literal is checked from its own syntax")
        };
        let (elements, fill) = (elements.clone(), fill.clone());
        let (length, element_type) =
            self.array_literal_type(&span, scopes, destination, &elements, fill.as_ref())?;
        let ty = Type::Array {
            length,
            element: Box::new(element_type.clone()),
        };
        check_element_count(span, &ty, length, elements.len(), fill.is_some())?;
        let mut checked_elements = Vec::with_capacity(elements.len());
        for &element in &elements {
            checked_elements.push(self.check_expression(
                element,
                scopes,
                Some(element_type.clone()),
            )?);
        }
        let constant = fold_elements(&checked_elements, length);
        let folded = constant.is_some();
        for (&element, checked) in elements.iter().zip(checked_elements) {
            self.record_operand(element, checked, folded);
        }
        Ok(CheckedExpression {
            ty,
            untyped: false,
            value: ExpressionValue::Array {
                elements,
                fill: fill.is_some(),
            },
            constant,
        })
    }

    /// The length and element type an array literal is checked against.
    fn array_literal_type(
        &mut self,
        span: &std::ops::Range<usize>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
        destination: Option<Type>,
        elements: &[Idx<Expression>],
        fill: Option<&std::ops::Range<usize>>,
    ) -> Result<(u64, Type), Diagnostic> {
        match destination {
            Some(Type::Array { length, element }) => Ok((length, *element)),
            Some(destination) => Err(Diagnostic::new(
                span.clone(),
                format!("cannot implicitly convert an array literal to `{destination}`"),
            )),
            None => {
                if let Some(fill) = fill {
                    return Err(Diagnostic::new(
                        fill.clone(),
                        "a fill requires a length from context",
                    ));
                }
                let length = u64::try_from(elements.len())
                    .expect("a source file holds fewer elements than u64::MAX");
                Ok((length, self.common_element_type(elements, scopes)?))
            }
        }
    }

    /// The one type the elements of a literal with no context must share: the
    /// type of the first typed element, or the untyped default when every
    /// element is an untyped constant. The checked elements are discarded, so
    /// every element goes through the one checking path afterwards.
    fn common_element_type(
        &mut self,
        elements: &[Idx<Expression>],
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<Type, Diagnostic> {
        let mut default = None;
        for &element in elements {
            let checked = self.infer_expression(element, scopes)?;
            if !checked.untyped {
                return Ok(checked.ty);
            }
            default.get_or_insert(checked.ty);
        }
        Ok(default.expect("an array literal has at least one element"))
    }

    /// Checks one `[ … ]` step and gives the element type it reaches. The
    /// operand must be an array, and the index must be an `int` that is in
    /// range whenever it is constant. Index expressions and assignment targets
    /// share this, so both spell one rule.
    pub(super) fn check_index_step(
        &mut self,
        operand: &Type,
        operand_span: &std::ops::Range<usize>,
        index: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<Type, Diagnostic> {
        let Type::Array { length, element } = operand else {
            return Err(Diagnostic::new(
                operand_span.clone(),
                format!("cannot index `{operand}`"),
            ));
        };
        let checked = self.check_expression(index, scopes, Some(Scalar::Int.into()))?;
        if let Some(value) = checked.integer()
            && value.to_u64().is_none_or(|value| value >= *length)
        {
            return Err(Diagnostic::new(
                self.syntax.expressions[index].span.clone(),
                format!("index {value} is out of range for `{operand}`"),
            ));
        }
        self.expressions.insert(index, checked);
        Ok((**element).clone())
    }

    fn infer_index(
        &mut self,
        operand: Idx<Expression>,
        index: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        let span = self.syntax.expressions[operand].span.clone();
        let element = self.check_index_step(&checked_operand.ty, &span, index, scopes)?;
        // `a[i]` is never a constant expression, so both sub-expressions keep
        // their own constants and are still evaluated.
        self.record_operand(operand, checked_operand, false);
        Ok(CheckedExpression {
            ty: element,
            untyped: false,
            value: ExpressionValue::Index { operand, index },
            constant: None,
        })
    }

    fn infer_length(
        &mut self,
        operand: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        let Type::Array { length, .. } = &checked_operand.ty else {
            return Err(Diagnostic::new(
                self.syntax.expressions[operand].span.clone(),
                format!(
                    "`len` requires an array operand, found `{}`",
                    checked_operand.ty
                ),
            ));
        };
        // The length comes from the operand's type, so it folds unless
        // reaching the type needs a call. The operand is evaluated either way.
        let constant = find_call(self.syntax, operand)
            .is_none()
            .then(|| Constant::Integer(BigInt::from(*length)));
        self.record_operand(operand, checked_operand, false);
        Ok(CheckedExpression {
            ty: Scalar::Int.into(),
            untyped: false,
            value: ExpressionValue::Length { operand },
            constant,
        })
    }

    /// Records a checked operand. A folded parent owns the constant value, so
    /// the operand it was folded from no longer carries one.
    fn record_operand(
        &mut self,
        id: Idx<Expression>,
        mut operand: CheckedExpression,
        folded: bool,
    ) {
        if folded {
            operand.constant = None;
        }
        self.expressions.insert(id, operand);
    }

    fn infer_grouping(
        &mut self,
        inner: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked = self.infer_expression(inner, scopes)?;
        let result = CheckedExpression {
            ty: checked.ty.clone(),
            untyped: checked.untyped,
            value: ExpressionValue::Grouping { expression: inner },
            constant: checked.constant.clone(),
        };
        self.record_operand(inner, checked, result.constant.is_some());
        Ok(result)
    }

    fn infer_unary(
        &mut self,
        operator: UnaryOperator,
        operator_span: &std::ops::Range<usize>,
        operand: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        let error = |message| Diagnostic::new(operator_span.clone(), message);
        let Some(operand_type) = integer_operand(&checked_operand) else {
            return Err(error(
                "integer unary operator requires an integer operand".to_string(),
            ));
        };
        if operator == UnaryOperator::Negate && !checked_operand.untyped && !operand_type.signed() {
            return Err(error(format!(
                "unary `-` is not permitted on `{operand_type}`"
            )));
        }
        if operator == UnaryOperator::WrappingNegate && checked_operand.untyped {
            return Err(error(
                "wrapping negation requires a typed operand".to_string(),
            ));
        }
        let constant = evaluate_unary(
            operator,
            operator_span.clone(),
            operand_type,
            &checked_operand,
        )?;
        let result = CheckedExpression {
            ty: checked_operand.ty.clone(),
            untyped: checked_operand.untyped,
            value: ExpressionValue::Unary {
                operator,
                operator_span: operator_span.clone(),
                operand,
            },
            constant: constant.map(Constant::Integer),
        };
        self.record_operand(operand, checked_operand, result.constant.is_some());
        Ok(result)
    }

    fn infer_binary(
        &mut self,
        operator: BinaryOperator,
        operator_span: &std::ops::Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_left = self.infer_expression(left, scopes)?;
        let checked_right = self.infer_expression(right, scopes)?;
        let (checked_left, checked_right, ty, untyped, constant) = self.check_integer_binary(
            operator,
            operator_span,
            operator.spelling(),
            CheckedBinaryOperand {
                id: Some(left),
                expression: checked_left,
            },
            CheckedBinaryOperand {
                id: Some(right),
                expression: checked_right,
            },
        )?;
        let result = CheckedExpression {
            ty: ty.into(),
            untyped,
            value: ExpressionValue::Binary {
                operator,
                operator_span: operator_span.clone(),
                left,
                right,
            },
            constant: constant.map(Constant::Integer),
        };
        let folded = result.constant.is_some();
        self.record_operand(left, checked_left.expression, folded);
        self.record_operand(right, checked_right.expression, folded);
        Ok(result)
    }

    fn infer_comparison(
        &mut self,
        operator: ComparisonOperator,
        operator_span: &std::ops::Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let mut checked_left = self.infer_expression(left, scopes)?;
        let mut checked_right = self.infer_expression(right, scopes)?;
        self.unify_comparison(
            operator,
            operator_span,
            (left, &mut checked_left),
            (right, &mut checked_right),
        )?;
        let constant = match (
            checked_left.constant.as_ref(),
            checked_right.constant.as_ref(),
        ) {
            (Some(left), Some(right)) => Some(compare_constants(operator, left, right)),
            _ => None,
        };
        let result = CheckedExpression {
            ty: Scalar::Bool.into(),
            untyped: constant.is_some(),
            value: ExpressionValue::Comparison {
                operator,
                operator_span: operator_span.clone(),
                left,
                right,
            },
            constant: constant.map(Constant::Integer),
        };
        let folded = result.constant.is_some();
        self.record_operand(left, checked_left, folded);
        self.record_operand(right, checked_right, folded);
        Ok(result)
    }

    /// Gives both comparison operands one type. An untyped operand takes the
    /// type of a typed one; otherwise the two types must already agree. Arrays
    /// are never untyped, so they only have to agree, and only `==` and `!=`
    /// reach them.
    fn unify_comparison(
        &mut self,
        operator: ComparisonOperator,
        operator_span: &std::ops::Range<usize>,
        left: (Idx<Expression>, &mut CheckedExpression),
        right: (Idx<Expression>, &mut CheckedExpression),
    ) -> Result<(), Diagnostic> {
        let (left_id, left) = left;
        let (right_id, right) = right;
        let unified = match (left.ty.scalar(), right.ty.scalar()) {
            (Some(left_type), Some(right_type)) => match (left.untyped, right.untyped) {
                (false, true) => return self.concretize(right_id, right, left_type),
                (true, false) => return self.concretize(left_id, left, right_type),
                _ => left_type == right_type,
            },
            _ => left.ty == right.ty,
        };
        if !unified {
            return Err(Diagnostic::new(
                operator_span.clone(),
                format!(
                    "comparison operands have different types `{}` and `{}`",
                    left.ty, right.ty
                ),
            ));
        }
        if left.ty.scalar().is_none()
            && !matches!(
                operator,
                ComparisonOperator::Equal | ComparisonOperator::NotEqual
            )
        {
            return Err(Diagnostic::new(
                operator_span.clone(),
                format!("only `==` and `!=` are defined on `{}`", left.ty),
            ));
        }
        Ok(())
    }

    fn infer_logical(
        &mut self,
        operator: LogicalOperator,
        operator_span: &std::ops::Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let mut checked_left = self.infer_expression(left, scopes)?;
        let mut checked_right = self.infer_expression(right, scopes)?;
        require_boolean(self.syntax, left, &checked_left)?;
        require_boolean(self.syntax, right, &checked_right)?;
        match (checked_left.untyped, checked_right.untyped) {
            (true, false) => self.concretize(left, &mut checked_left, Scalar::Bool)?,
            (false, true) => self.concretize(right, &mut checked_right, Scalar::Bool)?,
            _ => {}
        }
        let constant = match (checked_left.integer(), checked_right.integer()) {
            (Some(left), Some(right)) => Some(BigInt::from(match operator {
                LogicalOperator::And => constant_boolean(left) && constant_boolean(right),
                LogicalOperator::Or => constant_boolean(left) || constant_boolean(right),
            })),
            _ => None,
        };
        let result = CheckedExpression {
            ty: Scalar::Bool.into(),
            untyped: checked_left.untyped && checked_right.untyped,
            value: ExpressionValue::Logical {
                operator,
                operator_span: operator_span.clone(),
                left,
                right,
            },
            constant: constant.map(Constant::Integer),
        };
        let folded = result.constant.is_some();
        self.record_operand(left, checked_left, folded);
        self.record_operand(right, checked_right, folded);
        Ok(result)
    }

    fn infer_logical_not(
        &mut self,
        operator_span: &std::ops::Range<usize>,
        operand: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        require_boolean(self.syntax, operand, &checked_operand)?;
        let constant = checked_operand
            .integer()
            .map(|value| BigInt::from(!constant_boolean(value)));
        let result = CheckedExpression {
            ty: Scalar::Bool.into(),
            untyped: checked_operand.untyped,
            value: ExpressionValue::LogicalNot {
                operator_span: operator_span.clone(),
                operand,
            },
            constant: constant.map(Constant::Integer),
        };
        self.record_operand(operand, checked_operand, result.constant.is_some());
        Ok(result)
    }

    fn infer_conversion(
        &mut self,
        destination: Scalar,
        truncating: bool,
        operand: Idx<Expression>,
        span: &std::ops::Range<usize>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<CheckedExpression, Diagnostic> {
        let mut checked_operand = self.infer_expression(operand, scopes)?;
        if integer_operand(&checked_operand).is_none() {
            return Err(Diagnostic::new(
                span.clone(),
                format!("cannot convert `{}` to `{destination}`", checked_operand.ty),
            ));
        }
        if checked_operand.untyped && (!truncating || checked_operand.constant.is_none()) {
            let operand_type = if truncating { Scalar::Int } else { destination };
            self.concretize(operand, &mut checked_operand, operand_type)?;
        }
        let constant = self.convert_constant(&checked_operand, destination, truncating, span)?;
        let value = if constant.is_some() {
            ExpressionValue::Integer
        } else {
            ExpressionValue::Conversion {
                operand,
                truncating,
            }
        };
        self.record_operand(operand, checked_operand, constant.is_some());
        Ok(CheckedExpression {
            ty: destination.into(),
            untyped: false,
            value,
            constant: constant.map(Constant::Integer),
        })
    }

    /// Converts a constant operand at check time. A non-truncating conversion
    /// that would trap is an error rather than a trap at run time.
    fn convert_constant(
        &self,
        operand: &CheckedExpression,
        destination: Scalar,
        truncating: bool,
        span: &std::ops::Range<usize>,
    ) -> Result<Option<BigInt>, Diagnostic> {
        let Some(value) = operand.integer() else {
            return Ok(None);
        };
        if truncating {
            return Ok(Some(truncate_integer(value, destination)));
        }
        if !integer_fits(value, destination) {
            return Err(Diagnostic::new(
                span.clone(),
                format!("constant conversion to `{destination}` would trap"),
            ));
        }
        Ok(Some(value.clone()))
    }

    pub(super) fn check_integer_binary(
        &mut self,
        operator: BinaryOperator,
        operator_span: &std::ops::Range<usize>,
        spelling: &str,
        mut left: CheckedBinaryOperand,
        mut right: CheckedBinaryOperand,
    ) -> Result<
        (
            CheckedBinaryOperand,
            CheckedBinaryOperand,
            Scalar,
            bool,
            Option<BigInt>,
        ),
        Diagnostic,
    > {
        let (Some(mut left_type), Some(right_type)) = (
            integer_operand(&left.expression),
            integer_operand(&right.expression),
        ) else {
            return Err(Diagnostic::new(
                operator_span.clone(),
                format!("integer `{spelling}` requires integer operands"),
            ));
        };
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
        if wrapping && left.expression.untyped && right.expression.untyped {
            return Err(Diagnostic::new(
                operator_span.clone(),
                "wrapping arithmetic requires a typed operand",
            ));
        }
        if !shift {
            match (left.expression.untyped, right.expression.untyped) {
                (false, false) if left_type != right_type => {
                    return Err(Diagnostic::new(
                        operator_span.clone(),
                        format!(
                            "binary operands have different types `{left_type}` and `{right_type}`"
                        ),
                    ));
                }
                (false, true) => self.concretize(
                    right
                        .id
                        .expect("an untyped right operand has an expression"),
                    &mut right.expression,
                    left_type,
                )?,
                (true, false) => {
                    self.concretize(
                        left.id.expect("an untyped left operand has an expression"),
                        &mut left.expression,
                        right_type,
                    )?;
                    left_type = right_type;
                }
                _ => {}
            }
        } else if (left.expression.constant.is_none() || right.expression.constant.is_none())
            && right.expression.untyped
        {
            self.concretize(
                right
                    .id
                    .expect("an untyped right operand has an expression"),
                &mut right.expression,
                Scalar::Int,
            )?;
        }
        let (ty, untyped) = if shift {
            (left_type, left.expression.untyped)
        } else if left.expression.untyped {
            (right_type, right.expression.untyped)
        } else {
            (left_type, false)
        };
        let constant = evaluate_binary(
            operator,
            operator_span.clone(),
            spelling,
            ty,
            untyped,
            (&left.expression, &right.expression),
        )?;
        Ok((left, right, ty, untyped, constant))
    }

    fn concretize(
        &mut self,
        id: Idx<Expression>,
        checked: &mut CheckedExpression,
        destination: Scalar,
    ) -> Result<(), Diagnostic> {
        debug_assert!(checked.untyped);
        let ty = checked
            .ty
            .scalar()
            .expect("an untyped expression has a scalar type");
        if ty.is_integer() != destination.is_integer() {
            return Err(Diagnostic::new(
                self.syntax.expressions[id].span.clone(),
                format!("cannot implicitly convert `{ty}` to `{destination}`"),
            ));
        }
        let constant = concretize_value(self.syntax, id, checked, destination)?
            .expect("caller provides an untyped expression");
        self.concretize_children(id, destination, constant)?;
        Ok(())
    }

    fn concretize_stored(
        &mut self,
        id: Idx<Expression>,
        destination: Scalar,
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
        destination: Scalar,
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
                        format!("unary `-` is not permitted on `{destination}`"),
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
}
