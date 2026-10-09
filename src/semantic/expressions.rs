//! Expression inference, contextualization, indexing, lengths, and array literals.

use crate::{
    diagnostic::Diagnostic,
    frontend::syntax::{Expression, ExpressionKind, Syntax, TypeAnnotation},
    types::{BinaryOperator, ComparisonOperator, LogicalOperator, Scalar, Type, UnaryOperator},
};
use la_arena::Idx;
use lasso::Spur;
use num_bigint::BigInt;
use num_traits::ToPrimitive;

use super::{annotations::check_element_count, constants::*, model::*};

/// Reports an expression used where a location is required.
fn not_a_location(span: &std::ops::Range<usize>, use_: LocationUse) -> Diagnostic {
    Diagnostic::new(
        span.clone(),
        format!(
            "cannot {} an expression that is not a location",
            use_.verb()
        ),
    )
}

/// What a location is checked for, which words the diagnostics that differ
/// between an assignment target and an `&` operand.
#[derive(Clone, Copy)]
pub(super) enum LocationUse {
    Assign,
    Address,
}

impl LocationUse {
    pub(super) fn verb(self) -> &'static str {
        match self {
            Self::Assign => "assign to",
            Self::Address => "take the address of",
        }
    }
}

/// Reports a `null` whose context supplies no type.
fn null_without_context(span: &std::ops::Range<usize>) -> Diagnostic {
    Diagnostic::new(span.clone(), "`null` requires a pointer type from context")
}

/// Reports a field name the struct does not declare.
fn no_field(
    syntax: &Syntax,
    ty: &Type,
    name: Spur,
    name_span: &std::ops::Range<usize>,
) -> Diagnostic {
    Diagnostic::new(
        name_span.clone(),
        format!(
            "struct `{ty}` has no field `{}`",
            syntax.names.resolve(&name)
        ),
    )
}

impl CheckedProgram<'_> {
    pub(super) fn check_expression(
        &mut self,
        id: Idx<Expression>,
        scopes: &ScopeStack<'_>,
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
        // `null` has no type of its own, so it takes the destination's.
        if self.is_null(id) {
            return self.check_null(id, destination.as_ref());
        }
        let mut checked = match (
            &syntax.expressions[id].kind,
            destination.as_ref().and_then(Type::scalar),
        ) {
            (
                ExpressionKind::Binary {
                    operator,
                    operator_span,
                    left,
                    right,
                },
                Some(ty),
            ) if ty.is_numeric() => self.infer_binary_context(
                *operator,
                operator_span,
                *left,
                *right,
                scopes,
                Some(ty),
            )?,
            (
                ExpressionKind::Unary {
                    operator,
                    operator_span,
                    operand,
                },
                Some(ty),
            ) if ty.is_numeric() => {
                self.infer_unary_context(*operator, operator_span, *operand, scopes, Some(ty))?
            }
            (ExpressionKind::Grouping { expression }, Some(ty)) if ty.is_numeric() => {
                let inner = self.check_expression(*expression, scopes, Some(ty.into()))?;
                let result = CheckedExpression {
                    ty: inner.ty.clone(),
                    untyped: false,
                    value: ExpressionValue::Grouping {
                        expression: *expression,
                    },
                    constant: inner.constant.clone(),
                };
                self.record_operand(*expression, inner, result.constant.is_some());
                result
            }
            _ => self.infer_expression(id, scopes)?,
        };
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

    /// Whether this expression is `null`, in any grouping.
    fn is_null(&self, id: Idx<Expression>) -> bool {
        match &self.syntax.expressions[id].kind {
            ExpressionKind::Null => true,
            ExpressionKind::Grouping { expression } => self.is_null(*expression),
            _ => false,
        }
    }

    fn check_null(
        &mut self,
        id: Idx<Expression>,
        destination: Option<&Type>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let span = &self.syntax.expressions[id].span;
        match &self.syntax.expressions[id].kind {
            ExpressionKind::Null => match destination {
                Some(ty @ Type::Pointer { .. }) => Ok(CheckedExpression {
                    ty: ty.clone(),
                    untyped: false,
                    value: ExpressionValue::Null,
                    constant: Some(Constant::Null),
                }),
                Some(destination) => Err(Diagnostic::new(
                    span.clone(),
                    format!("cannot implicitly convert `null` to `{destination}`"),
                )),
                None => Err(null_without_context(span)),
            },
            ExpressionKind::Grouping { expression } => {
                let inner = *expression;
                let checked = self.check_null(inner, destination)?;
                let result = CheckedExpression {
                    ty: checked.ty.clone(),
                    untyped: false,
                    value: ExpressionValue::Grouping { expression: inner },
                    constant: checked.constant.clone(),
                };
                self.record_operand(inner, checked, true);
                Ok(result)
            }
            _ => unreachable!("only `null` and its groupings are checked as `null`"),
        }
    }

    fn infer_pointer_conversion(
        &mut self,
        destination: Idx<TypeAnnotation>,
        operand: Idx<Expression>,
        span: &std::ops::Range<usize>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let ty = self.resolve_annotation(destination, scopes)?;
        let checked_operand = if self.is_null(operand) {
            self.check_null(operand, Some(&ty))?
        } else {
            self.infer_expression(operand, scopes)?
        };
        if !matches!(checked_operand.ty, Type::Pointer { .. }) {
            return Err(Diagnostic::new(
                span.clone(),
                "pointer conversion requires a pointer operand",
            ));
        }
        let constant = checked_operand.constant.clone();
        self.record_operand(operand, checked_operand, false);
        Ok(CheckedExpression {
            ty,
            untyped: false,
            value: ExpressionValue::PointerConversion { operand },
            constant,
        })
    }

    pub(super) fn infer_expression(
        &mut self,
        id: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let expression = &self.syntax.expressions[id];
        match &expression.kind {
            ExpressionKind::Integer(spelling) => Ok(integer_literal(spelling)),
            ExpressionKind::Sizeof { annotation } => {
                let ty = self.resolve_annotation(*annotation, scopes)?;
                let size = self.layouts.size(self, &ty).expect("validated layout");
                Ok(CheckedExpression {
                    ty: Scalar::Uint.into(),
                    untyped: false,
                    value: ExpressionValue::Integer,
                    constant: Some(BigInt::from(size).into()),
                })
            }
            ExpressionKind::Floating(spelling) => floating_literal(spelling, &expression.span),
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
                operand,
                ..
            } => self.infer_conversion(*destination, *operand, &expression.span, scopes),
            ExpressionKind::ArrayLiteral { .. } => self.check_array_literal(id, scopes, None),
            ExpressionKind::Index { operand, index, .. } => {
                self.infer_index(*operand, *index, scopes)
            }
            ExpressionKind::Length { operand } => self.infer_length(*operand, scopes),
            ExpressionKind::StructLiteral { .. } => self.check_struct_literal(id, scopes),
            ExpressionKind::Field {
                operand,
                name,
                name_span,
            } => self.infer_field(*operand, *name, name_span, scopes),
            ExpressionKind::AddressOf { operand, .. } => {
                self.infer_address_of(id, *operand, scopes)
            }
            ExpressionKind::Dereference { operand, .. } => self.infer_dereference(*operand, scopes),
            ExpressionKind::Null => Err(null_without_context(&expression.span)),
            ExpressionKind::PointerConversion {
                destination,
                operand,
            } => self.infer_pointer_conversion(*destination, *operand, &expression.span, scopes),
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
        scopes: &ScopeStack<'_>,
        destination: Option<Type>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let span = self.syntax.expressions[id].span.clone();
        let ExpressionKind::ArrayLiteral { elements } = &self.syntax.expressions[id].kind else {
            unreachable!("an array literal is checked from its own syntax")
        };
        let elements = elements.clone();
        let (length, element_type) =
            self.array_literal_type(&span, scopes, destination, &elements)?;
        let ty = Type::Array {
            length,
            element: Box::new(element_type.clone()),
        };
        self.validate_aggregate_layout(&ty, &span)?;
        check_element_count(span.clone(), &ty, length, elements.len())?;
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
            value: ExpressionValue::Array { elements },
            constant,
        })
    }

    /// The length and element type an array literal is checked against.
    fn array_literal_type(
        &mut self,
        span: &std::ops::Range<usize>,
        scopes: &ScopeStack<'_>,
        destination: Option<Type>,
        elements: &[Idx<Expression>],
    ) -> Result<(u64, Type), Diagnostic> {
        match destination {
            Some(Type::Array { length, element }) => Ok((length, *element)),
            Some(destination) => Err(Diagnostic::new(
                span.clone(),
                format!("cannot implicitly convert an array literal to `{destination}`"),
            )),
            None => {
                let length = u64::try_from(elements.len())
                    .expect("a source file holds fewer elements than u64::MAX");
                Ok((length, self.common_element_type(elements, scopes)?))
            }
        }
    }

    /// An inferred array takes its element type from its first element.
    fn common_element_type(
        &mut self,
        elements: &[Idx<Expression>],
        scopes: &ScopeStack<'_>,
    ) -> Result<Type, Diagnostic> {
        Ok(self.check_expression(elements[0], scopes, None)?.ty)
    }

    /// Checks one `[ … ]` step and gives the element type it reaches. The
    /// operand must be an array, or a pointer to one, whose type supplies a
    /// constant range check, and the index must be an `int`. Index
    /// expressions and assignment targets share this, so both spell one rule.
    pub(super) fn check_index_step(
        &mut self,
        operand: &Type,
        operand_span: &std::ops::Range<usize>,
        index: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<Type, Diagnostic> {
        let checked = self.check_expression(index, scopes, Some(Scalar::Int.into()))?;
        let element = match operand {
            Type::Array { length, element } => {
                if let Some(value) = checked.integer()
                    && value.to_u64().is_none_or(|n| n >= *length)
                {
                    return Err(Diagnostic::new(
                        self.syntax.expressions[index].span.clone(),
                        format!("constant index out of range for `{operand}`"),
                    ));
                }
                (**element).clone()
            }
            Type::Pointer { target, .. } => (**target).clone(),
            _ => {
                return Err(Diagnostic::new(
                    operand_span.clone(),
                    format!("cannot index `{operand}`"),
                ));
            }
        };
        self.expressions.insert(index, checked);
        Ok(element)
    }

    /// Checks a struct literal against the type it names itself, so an
    /// unexpected struct type is reported by the destination it does not fit.
    /// Its written fields are checked in source order, which is the order they
    /// are evaluated in.
    fn check_struct_literal(
        &mut self,
        id: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let syntax = self.syntax;
        let span = syntax.expressions[id].span.clone();
        let ExpressionKind::StructLiteral { name, fields } = &syntax.expressions[id].kind else {
            unreachable!("a struct literal is checked from its own syntax")
        };
        let declared = self.resolve_type_name(name, scopes)?;
        self.resolve_struct_fields(declared, &name.span, scopes)?;
        let ty = self.struct_type(declared);
        let count = self.structs[declared.0].fields.len();
        let mut values: Vec<Option<Constant>> = vec![None; count];
        let mut written = vec![false; count];
        let mut initializers = Vec::with_capacity(fields.len());
        let mut checked = Vec::with_capacity(fields.len());
        for field in fields {
            let Some(&ordinal) = self.structs[declared.0].ordinals.get(&field.name) else {
                return Err(no_field(syntax, &ty, field.name, &field.name_span));
            };
            if written[ordinal] {
                return Err(Diagnostic::new(
                    field.name_span.clone(),
                    format!(
                        "duplicate field `{}` in `{ty}` literal",
                        syntax.names.resolve(&field.name)
                    ),
                ));
            }
            let field_type = self.structs[declared.0].fields[ordinal].ty.clone();
            let value = self.check_expression(field.value, scopes, Some(field_type))?;
            written[ordinal] = true;
            values[ordinal] = value.constant.clone();
            initializers.push((ordinal, field.value));
            checked.push(value);
        }
        let omitted = written
            .iter()
            .enumerate()
            .filter_map(|(ordinal, written)| (!written).then_some(ordinal))
            .collect::<Vec<_>>();
        if let Some(&ordinal) = omitted.first() {
            return Err(Diagnostic::new(
                span,
                format!(
                    "`{ty}` literal is missing field `{}`",
                    syntax
                        .names
                        .resolve(&self.structs[declared.0].fields[ordinal].name)
                ),
            ));
        }
        let constant = values
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .map(Constant::Struct);
        let folded = constant.is_some();
        for (&(_, value), checked) in initializers.iter().zip(checked) {
            self.record_operand(value, checked, folded);
        }
        Ok(CheckedExpression {
            ty,
            untyped: false,
            value: ExpressionValue::Struct {
                id: declared,
                initializers,
            },
            constant,
        })
    }

    /// Checks one `.name` step and gives the field's position and type. The
    /// operand is a struct or a pointer to one. Field expressions and
    /// assignment targets share this, so both spell one rule.
    pub(super) fn check_field_step(
        &self,
        operand: &Type,
        operand_span: &std::ops::Range<usize>,
        name: Spur,
        name_span: &std::ops::Range<usize>,
    ) -> Result<(usize, Type), Diagnostic> {
        let reached = operand;
        let Type::Struct(struct_type) = reached else {
            return Err(Diagnostic::new(
                operand_span.clone(),
                format!("cannot select a field of `{operand}`"),
            ));
        };
        let declared = &self.structs[struct_type.id.0];
        let Some(&ordinal) = declared.ordinals.get(&name) else {
            return Err(no_field(self.syntax, reached, name, name_span));
        };
        Ok((ordinal, declared.fields[ordinal].ty.clone()))
    }

    fn infer_field(
        &mut self,
        operand: Idx<Expression>,
        name: Spur,
        name_span: &std::ops::Range<usize>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        let span = self.syntax.expressions[operand].span.clone();
        let (ordinal, ty) = self.check_field_step(&checked_operand.ty, &span, name, name_span)?;
        // `p.x` is not a constant expression, so the operand keeps its own
        // constant and is still evaluated.
        let constant = match &checked_operand.constant {
            Some(Constant::Struct(fields)) => Some(fields[ordinal].clone()),
            _ => None,
        };
        self.record_operand(operand, checked_operand, constant.is_some());
        Ok(CheckedExpression {
            ty,
            untyped: false,
            value: ExpressionValue::Field { operand, ordinal },
            constant,
        })
    }

    fn infer_index(
        &mut self,
        operand: Idx<Expression>,
        index: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        let span = self.syntax.expressions[operand].span.clone();
        let element = self.check_index_step(&checked_operand.ty, &span, index, scopes)?;
        // `a[i]` is never a constant expression, so both sub-expressions keep
        // their own constants and are still evaluated.
        let constant = match (
            &checked_operand.constant,
            self.expressions[index]
                .integer()
                .and_then(ToPrimitive::to_usize),
        ) {
            (Some(Constant::Array(values)), Some(index)) => Some(values[index].clone()),
            _ => None,
        };
        self.record_operand(operand, checked_operand, constant.is_some());
        Ok(CheckedExpression {
            ty: element,
            untyped: false,
            value: ExpressionValue::Index { operand, index },
            constant,
        })
    }

    fn infer_length(
        &mut self,
        operand: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        let Type::Array { length, .. } = checked_operand.ty else {
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
        let constant = checked_operand
            .constant
            .is_some()
            .then(|| Constant::Integer(BigInt::from(length)));
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
        scopes: &ScopeStack<'_>,
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

    /// Checks an expression in a position that requires a stored location: a
    /// binding, the target of a pointer, or an index or field selection from
    /// another location, in any grouping. Index expressions are checked in
    /// source order.
    pub(super) fn check_location(
        &mut self,
        id: Idx<Expression>,
        scopes: &ScopeStack<'_>,
        use_: LocationUse,
    ) -> Result<CheckedLocation, Diagnostic> {
        match &self.syntax.expressions[id].kind {
            ExpressionKind::Reference(name) => {
                let binding = self.resolve(name, scopes)?;
                Ok(CheckedLocation {
                    kind: CheckedLocationKind::Binding(binding),
                    ty: self.bindings[binding].ty.clone(),
                    span: name.span.clone(),
                })
            }
            ExpressionKind::Grouping { expression } => {
                self.check_location(*expression, scopes, use_)
            }
            ExpressionKind::Index { operand, index } => {
                let operand_location = self.check_step_operand(*operand, scopes, use_)?;
                let span = operand_location.span.clone();
                let ty = self.check_index_step(&operand_location.ty, &span, *index, scopes)?;
                Ok(CheckedLocation {
                    kind: CheckedLocationKind::Index {
                        operand: Box::new(operand_location),
                        index: *index,
                    },
                    ty,
                    span,
                })
            }
            ExpressionKind::Field {
                operand,
                name,
                name_span,
            } => {
                let operand_location = self.check_step_operand(*operand, scopes, use_)?;
                let span = operand_location.span.clone();
                let (ordinal, ty) =
                    self.check_field_step(&operand_location.ty, &span, *name, name_span)?;
                Ok(CheckedLocation {
                    kind: CheckedLocationKind::Field {
                        operand: Box::new(operand_location),
                        ordinal,
                    },
                    ty,
                    span,
                })
            }
            ExpressionKind::Dereference { operand, .. } => {
                let operand_location = self.check_pointer_operand(*operand, scopes, use_)?;
                let ty = self.pointer_target(
                    &operand_location.ty,
                    &self.syntax.expressions[*operand].span,
                )?;
                Ok(CheckedLocation {
                    kind: CheckedLocationKind::Dereference {
                        operand: Box::new(operand_location),
                    },
                    ty,
                    span: self.syntax.expressions[id].span.clone(),
                })
            }
            _ => Err(not_a_location(&self.syntax.expressions[id].span, use_)),
        }
    }

    /// Checks the operand of an index or field step, which is a location or
    /// an expression that computes a pointer, such as a call.
    fn check_step_operand(
        &mut self,
        id: Idx<Expression>,
        scopes: &ScopeStack<'_>,
        use_: LocationUse,
    ) -> Result<CheckedLocation, Diagnostic> {
        let location = self.check_pointer_operand(id, scopes, use_)?;
        if matches!(location.kind, CheckedLocationKind::Pointer(_))
            && !matches!(location.ty, Type::Pointer { .. })
        {
            return Err(not_a_location(&location.span, use_));
        }
        Ok(location)
    }

    /// Checks the operand of a step or of `*`, which is a location or an
    /// expression that computes a value, such as a call. A computed value
    /// becomes a `Pointer` location whatever its type, and each caller
    /// rejects one that is not a pointer.
    fn check_pointer_operand(
        &mut self,
        id: Idx<Expression>,
        scopes: &ScopeStack<'_>,
        use_: LocationUse,
    ) -> Result<CheckedLocation, Diagnostic> {
        match &self.syntax.expressions[id].kind {
            ExpressionKind::Reference(_)
            | ExpressionKind::Index { .. }
            | ExpressionKind::Field { .. }
            | ExpressionKind::Dereference { .. } => self.check_location(id, scopes, use_),
            ExpressionKind::Grouping { expression } => {
                self.check_pointer_operand(*expression, scopes, use_)
            }
            _ => {
                let checked = self.infer_expression(id, scopes)?;
                let ty = checked.ty.clone();
                self.expressions.insert(id, checked);
                Ok(CheckedLocation {
                    kind: CheckedLocationKind::Pointer(id),
                    ty,
                    span: self.syntax.expressions[id].span.clone(),
                })
            }
        }
    }

    fn pointer_target(
        &self,
        operand: &Type,
        operand_span: &std::ops::Range<usize>,
    ) -> Result<Type, Diagnostic> {
        match operand {
            Type::Pointer { target } => Ok((**target).clone()),
            _ => Err(Diagnostic::new(
                operand_span.clone(),
                format!("cannot dereference `{operand}`"),
            )),
        }
    }

    /// `&location`, a pointer to a mutable location.
    fn infer_address_of(
        &mut self,
        id: Idx<Expression>,
        operand: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let location = self.mutable_location(operand, scopes, LocationUse::Address)?;
        let ty = Type::Pointer {
            target: Box::new(location.ty.clone()),
        };
        self.addresses.insert(id, location);
        Ok(CheckedExpression {
            ty,
            untyped: false,
            value: ExpressionValue::AddressOf,
            constant: None,
        })
    }

    /// `*p` read as a value, a copy of the pointer's target.
    fn infer_dereference(
        &mut self,
        operand: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        let ty =
            self.pointer_target(&checked_operand.ty, &self.syntax.expressions[operand].span)?;
        self.record_operand(operand, checked_operand, false);
        Ok(CheckedExpression {
            ty,
            untyped: false,
            value: ExpressionValue::Dereference { operand },
            constant: None,
        })
    }

    fn infer_unary(
        &mut self,
        operator: UnaryOperator,
        operator_span: &std::ops::Range<usize>,
        operand: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        self.infer_unary_context(operator, operator_span, operand, scopes, None)
    }

    fn infer_unary_context(
        &mut self,
        operator: UnaryOperator,
        operator_span: &std::ops::Range<usize>,
        operand: Idx<Expression>,
        scopes: &ScopeStack<'_>,
        destination: Option<Scalar>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let hint = self.type_hint(operand, scopes)?.and_then(|ty| ty.scalar());
        let ty = destination
            .or(hint)
            .unwrap_or(if self.has_float_literal(operand) {
                Scalar::F64
            } else {
                Scalar::Int
            });
        // The signed minimum is written as negation of its magnitude, which
        // is one beyond the largest positive value of the destination.
        if operator == UnaryOperator::Negate
            && ty.is_integer()
            && ty.signed()
            && matches!(
                self.syntax.expressions[operand].kind,
                ExpressionKind::Integer(_)
            )
        {
            let checked = self.infer_expression(operand, scopes)?;
            if checked.integer() == Some(&BigInt::from(-ty.min())) {
                self.record_operand(operand, checked, true);
                return Ok(CheckedExpression {
                    ty: ty.into(),
                    untyped: false,
                    value: ExpressionValue::Unary { operator, operand },
                    constant: Some(BigInt::from(ty.min()).into()),
                });
            }
        }
        let checked_operand = self.check_expression(operand, scopes, Some(ty.into()))?;
        let Some(operand_type) = numeric_operand(&checked_operand) else {
            return Err(Diagnostic::new(
                operator_span.clone(),
                "unary operator requires a number",
            ));
        };
        if operator == UnaryOperator::Complement && !operand_type.is_integer() {
            return Err(Diagnostic::new(
                operator_span.clone(),
                "integer unary operator requires an integer operand",
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
            value: ExpressionValue::Unary { operator, operand },
            constant,
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
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        self.infer_binary_context(operator, operator_span, left, right, scopes, None)
    }

    fn type_hint(
        &mut self,
        id: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<Option<Type>, Diagnostic> {
        Ok(match &self.syntax.expressions[id].kind {
            ExpressionKind::Reference(name) => {
                let binding = self.resolve(name, scopes)?;
                Some(self.bindings[binding].ty.clone())
            }
            ExpressionKind::Conversion { destination, .. } => Some((*destination).into()),
            ExpressionKind::Grouping { expression } => self.type_hint(*expression, scopes)?,
            ExpressionKind::Unary { operand, .. } => self.type_hint(*operand, scopes)?,
            ExpressionKind::Binary {
                operator,
                left,
                right,
                ..
            } => {
                let left = self.type_hint(*left, scopes)?;
                let right = self.type_hint(*right, scopes)?;
                if *operator == BinaryOperator::Subtract
                    && matches!(left, Some(Type::Pointer { .. }))
                    && matches!(right, Some(Type::Pointer { .. }))
                {
                    Some(Scalar::Int.into())
                } else {
                    left.or(if operator.is_shift() { None } else { right })
                }
            }
            ExpressionKind::Call(call) => {
                let f = self.resolve_call(call, scopes)?;
                self.functions[f].result.clone()
            }
            ExpressionKind::Integer(_) | ExpressionKind::Floating(_) | ExpressionKind::Null => None,
            _ => Some(self.infer_expression(id, scopes)?.ty),
        })
    }

    fn has_float_literal(&self, id: Idx<Expression>) -> bool {
        match &self.syntax.expressions[id].kind {
            ExpressionKind::Floating(_) => true,
            ExpressionKind::Grouping { expression } => self.has_float_literal(*expression),
            ExpressionKind::Unary { operand, .. } => self.has_float_literal(*operand),
            ExpressionKind::Binary { left, right, .. } => {
                self.has_float_literal(*left) || self.has_float_literal(*right)
            }
            _ => false,
        }
    }

    fn infer_binary_context(
        &mut self,
        operator: BinaryOperator,
        operator_span: &std::ops::Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
        scopes: &ScopeStack<'_>,
        destination: Option<Scalar>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let left_hint = self.type_hint(left, scopes)?;
        let right_hint = self.type_hint(right, scopes)?;
        if let Some(pointer_ty @ Type::Pointer { .. }) = &left_hint
            && matches!(operator, BinaryOperator::Add | BinaryOperator::Subtract)
        {
            let checked_left = self.check_expression(left, scopes, Some(pointer_ty.clone()))?;
            let difference = operator == BinaryOperator::Subtract
                && matches!(right_hint, Some(Type::Pointer { .. }));
            let checked_right = self.check_expression(
                right,
                scopes,
                Some(if difference {
                    pointer_ty.clone()
                } else {
                    Scalar::Int.into()
                }),
            )?;
            self.record_operand(left, checked_left, false);
            self.record_operand(right, checked_right, false);
            return Ok(CheckedExpression {
                ty: if difference {
                    Scalar::Int.into()
                } else {
                    pointer_ty.clone()
                },
                untyped: false,
                value: if difference {
                    ExpressionValue::PointerDifference { left, right }
                } else {
                    ExpressionValue::PointerOffset {
                        pointer: left,
                        offset: right,
                        subtract: operator == BinaryOperator::Subtract,
                    }
                },
                constant: None,
            });
        }
        let ty = destination
            .or(left_hint.and_then(|t| t.scalar()))
            .or(if operator.is_shift() {
                None
            } else {
                right_hint.and_then(|t| t.scalar())
            })
            .unwrap_or(
                if self.has_float_literal(left)
                    || (!operator.is_shift() && self.has_float_literal(right))
                {
                    Scalar::F64
                } else {
                    Scalar::Int
                },
            );
        let checked_left = self.check_expression(left, scopes, Some(ty.into()))?;
        let checked_right = self.check_expression(
            right,
            scopes,
            if operator.is_shift() {
                None
            } else {
                Some(ty.into())
            },
        )?;
        let (checked_left, checked_right, ty, untyped, constant) = self.check_numeric_binary(
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
                left,
                right,
            },
            constant,
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
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let left_hint = self.type_hint(left, scopes)?;
        let right_hint = self.type_hint(right, scopes)?;
        let numeric_context = left_hint
            .as_ref()
            .or(right_hint.as_ref())
            .and_then(Type::scalar)
            .filter(|ty| ty.is_numeric());
        // A `null` operand takes the other operand's type.
        let (mut checked_left, mut checked_right) = match (self.is_null(left), self.is_null(right))
        {
            (true, false) => {
                let checked_right = self.infer_expression(right, scopes)?;
                (
                    self.check_null(left, Some(&checked_right.ty))?,
                    checked_right,
                )
            }
            (false, true) => {
                let checked_left = self.infer_expression(left, scopes)?;
                let checked_right = self.check_null(right, Some(&checked_left.ty))?;
                (checked_left, checked_right)
            }
            _ => (
                self.check_expression(left, scopes, numeric_context.map(Type::Scalar))?,
                self.check_expression(right, scopes, numeric_context.map(Type::Scalar))?,
            ),
        };
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
    /// and structs are never untyped, so they only have to agree, and only
    /// `==` and `!=` reach them.
    fn unify_comparison(
        &mut self,
        operator: ComparisonOperator,
        operator_span: &std::ops::Range<usize>,
        left: (Idx<Expression>, &mut CheckedExpression),
        right: (Idx<Expression>, &mut CheckedExpression),
    ) -> Result<(), Diagnostic> {
        let (left_id, left) = left;
        let (right_id, right) = right;
        if let (Some(a), Some(b)) = (left.ty.scalar(), right.ty.scalar()) {
            let ty = if !left.untyped {
                a
            } else if !right.untyped {
                b
            } else if a.is_floating() || b.is_floating() {
                Scalar::F64
            } else {
                a
            };
            if !operator.is_equality() && !ty.is_numeric() {
                return Err(Diagnostic::new(
                    operator_span.clone(),
                    "ordering requires numeric operands",
                ));
            }
            if left.untyped {
                self.concretize(left_id, left, ty)?;
            }
            if right.untyped {
                self.concretize(right_id, right, ty)?;
            }
        } else if !operator.is_equality() {
            return Err(Diagnostic::new(
                operator_span.clone(),
                format!("only `==` and `!=` are defined on `{}`", left.ty),
            ));
        }
        if left.ty != right.ty {
            return Err(Diagnostic::new(
                operator_span.clone(),
                format!(
                    "comparison operands have different types `{}` and `{}`",
                    left.ty, right.ty
                ),
            ));
        }
        Ok(())
    }

    fn infer_logical(
        &mut self,
        operator: LogicalOperator,
        _operator_span: &std::ops::Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
        scopes: &ScopeStack<'_>,
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
        _operator_span: &std::ops::Range<usize>,
        operand: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        require_boolean(self.syntax, operand, &checked_operand)?;
        let constant = checked_operand
            .integer()
            .map(|value| BigInt::from(!constant_boolean(value)));
        let result = CheckedExpression {
            ty: Scalar::Bool.into(),
            untyped: checked_operand.untyped,
            value: ExpressionValue::LogicalNot { operand },
            constant: constant.map(Constant::Integer),
        };
        self.record_operand(operand, checked_operand, result.constant.is_some());
        Ok(result)
    }

    fn infer_conversion(
        &mut self,
        destination: Scalar,
        operand: Idx<Expression>,
        span: &std::ops::Range<usize>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let mut checked_operand = self.infer_expression(operand, scopes)?;
        // Only integers reinterpret their bits, while every number converts.
        let convertible = numeric_operand(&checked_operand);
        if convertible.is_none() {
            return Err(Diagnostic::new(
                span.clone(),
                format!("cannot convert `{}` to `{destination}`", checked_operand.ty),
            ));
        }
        if checked_operand.untyped {
            let operand_type = checked_operand.ty.scalar().expect("numeric operand");
            self.concretize(operand, &mut checked_operand, operand_type)?;
        }
        let constant = convert_constant(&checked_operand, destination, span)?;
        self.record_operand(operand, checked_operand, constant.is_some());
        Ok(CheckedExpression {
            ty: destination.into(),
            untyped: false,
            value: ExpressionValue::Conversion { operand },
            constant,
        })
    }

    /// Gives a binary operation's operands one numeric type and folds it.
    /// Ordinary and compound assignment reach this, so both spell one rule.
    pub(super) fn check_numeric_binary(
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
            Option<Constant>,
        ),
        Diagnostic,
    > {
        let (mut left_type, right_type) = binary_operand_types(
            operator,
            operator_span,
            spelling,
            (&left.expression, &right.expression),
        )?;
        if left.expression.untyped && !right.expression.untyped && !operator.is_shift() {
            left_type = right_type;
        }
        if left.expression.untyped {
            self.concretize(
                left.id.expect("literal operand"),
                &mut left.expression,
                left_type,
            )?;
        }
        if right.expression.untyped {
            self.concretize(
                right.id.expect("literal operand"),
                &mut right.expression,
                if operator.is_shift() {
                    Scalar::Int
                } else {
                    left_type
                },
            )?;
        }
        if !operator.is_shift() && left.expression.ty != right.expression.ty {
            return Err(Diagnostic::new(
                operator_span.clone(),
                "binary operands have different types",
            ));
        }
        let (ty, untyped) = (left_type, false);
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
        if !contextualizable(checked, destination) {
            return Err(Diagnostic::new(
                self.syntax.expressions[id].span.clone(),
                format!(
                    "cannot implicitly convert `{}` to `{destination}`",
                    checked.ty
                ),
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
        _constant: bool,
    ) -> Result<(), Diagnostic> {
        let (first, second) = match &self.syntax.expressions[id].kind {
            ExpressionKind::Grouping { expression } => (Some(*expression), None),
            _ => (None, None),
        };
        for child in first.into_iter().chain(second) {
            self.concretize_stored(child, destination)?;
        }
        Ok(())
    }
}
