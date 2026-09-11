//! Expression inference, contextualization, indexing, lengths, and array literals.

use crate::{
    diagnostic::Diagnostic,
    frontend::syntax::{Expression, ExpressionKind, Syntax, find_call},
    types::{BinaryOperator, ComparisonOperator, LogicalOperator, Scalar, Type, UnaryOperator},
};
use la_arena::Idx;
use lasso::Spur;
use num_bigint::BigInt;
use num_traits::ToPrimitive;

use super::{annotations::check_element_count, constants::*, model::*};

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

/// Whether an expression is `null` wrapped in any number of groupings.
fn is_null_expression(syntax: &Syntax, id: Idx<Expression>) -> bool {
    match syntax.expressions[id].kind {
        ExpressionKind::Null => true,
        ExpressionKind::Grouping { expression } => is_null_expression(syntax, expression),
        _ => false,
    }
}

#[derive(Clone, Copy)]
enum LocationUse {
    Assignment,
    AddressOf,
}

impl LocationUse {
    fn non_location_message(self) -> &'static str {
        match self {
            Self::Assignment => "cannot assign to an expression that is not a location",
            Self::AddressOf => "cannot take the address of an expression that is not a location",
        }
    }
}

struct LocationStep {
    location: CheckedLocation,
    ty: Type,
    implicit_dereference: Option<std::ops::Range<usize>>,
    mutable: bool,
}

impl CheckedProgram<'_> {
    pub(super) fn check_expression(
        &mut self,
        id: Idx<Expression>,
        scopes: &ScopeStack<'_>,
        destination: Option<Type>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let syntax = self.syntax;
        if matches!(syntax.expressions[id].kind, ExpressionKind::Null) {
            let Some(ty @ Type::Pointer { .. }) = destination else {
                return Err(Diagnostic::new(
                    syntax.expressions[id].span.clone(),
                    "`null` requires a pointer type from context",
                ));
            };
            return Ok(CheckedExpression {
                ty,
                untyped: false,
                value: ExpressionValue::Null,
                constant: Some(Constant::Null),
            });
        }
        if let ExpressionKind::Grouping { expression } = syntax.expressions[id].kind
            && is_null_expression(syntax, expression)
        {
            return self.check_grouping(expression, scopes, destination);
        }
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
            && !checked.ty.value_compatible(&destination)
        {
            return Err(mismatch(&checked, &destination));
        }
        Ok(checked)
    }

    pub(super) fn infer_expression(
        &mut self,
        id: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let expression = &self.syntax.expressions[id];
        match &expression.kind {
            ExpressionKind::Integer(spelling) => Ok(integer_literal(spelling)),
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
            ExpressionKind::Null => Err(Diagnostic::new(
                expression.span.clone(),
                "`null` requires a pointer type from context",
            )),
            ExpressionKind::AddressOf { operand, .. } => self.infer_address_of(*operand, scopes),
            ExpressionKind::Dereference { operand, .. } => self.infer_dereference(*operand, scopes),
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
            ExpressionKind::StructLiteral { .. } => self.check_struct_literal(id, scopes),
            ExpressionKind::Field {
                operand,
                name,
                name_span,
            } => self.infer_field(*operand, *name, name_span, scopes),
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
        self.validate_aggregate_layout(&ty, &span)?;
        check_element_count(span.clone(), &ty, length, elements.len(), fill.is_some())?;
        // A fill expands to one constant per scalar the whole literal holds,
        // so the bound counts the elements an element itself expands to.
        if fill.is_some()
            && self
                .aggregate_value_count(&ty)
                .is_none_or(|count| count > MAX_AGGREGATE_INITIALIZER_VALUES)
        {
            return Err(Diagnostic::new(
                span,
                format!(
                    "aggregate initializer exceeds compiler limit of {MAX_AGGREGATE_INITIALIZER_VALUES} values"
                ),
            ));
        }
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
        scopes: &ScopeStack<'_>,
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
        scopes: &ScopeStack<'_>,
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
        scopes: &ScopeStack<'_>,
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
        let ExpressionKind::StructLiteral { name, fields, fill } = &syntax.expressions[id].kind
        else {
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
        if let Some(&ordinal) = omitted.first()
            && fill.is_none()
        {
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
        // The fill expands every omitted field, so the bound covers them
        // together rather than one field at a time.
        let total = omitted.iter().try_fold(0u64, |count, &ordinal| {
            count.checked_add(
                self.aggregate_value_count(&self.structs[declared.0].fields[ordinal].ty)?,
            )
        });
        if total.is_none_or(|count| count > MAX_AGGREGATE_INITIALIZER_VALUES) {
            return Err(Diagnostic::new(
                name.span.clone(),
                format!(
                    "aggregate initializer exceeds compiler limit of {MAX_AGGREGATE_INITIALIZER_VALUES} values"
                ),
            ));
        }
        let mut filled = Vec::new();
        for ordinal in omitted {
            let zero = self.zero_value_unchecked(&self.structs[declared.0].fields[ordinal].ty);
            values[ordinal] = Some(zero.clone());
            filled.push((ordinal, zero));
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
                filled,
            },
            constant,
        })
    }

    /// Checks one `.name` step and gives the field's position and type. Field
    /// expressions and assignment targets share this, so both spell one rule.
    pub(super) fn check_field_step(
        &self,
        operand: &Type,
        operand_span: &std::ops::Range<usize>,
        name: Spur,
        name_span: &std::ops::Range<usize>,
    ) -> Result<(usize, Type), Diagnostic> {
        let Type::Struct(struct_type) = operand else {
            return Err(Diagnostic::new(
                operand_span.clone(),
                format!("cannot select a field of `{operand}`"),
            ));
        };
        let declared = &self.structs[struct_type.id.0];
        let Some(&ordinal) = declared.ordinals.get(&name) else {
            return Err(no_field(self.syntax, operand, name, name_span));
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
        let (operand_type, implicit_dereference) = match &checked_operand.ty {
            Type::Pointer { target, .. } => (&**target, true),
            _ => (&checked_operand.ty, false),
        };
        let (ordinal, ty) = self.check_field_step(operand_type, &span, name, name_span)?;
        // `p.x` is not a constant expression, so the operand keeps its own
        // constant and is still evaluated.
        self.record_operand(operand, checked_operand, false);
        Ok(CheckedExpression {
            ty,
            untyped: false,
            value: ExpressionValue::Field {
                operand,
                ordinal,
                implicit_dereference,
            },
            constant: None,
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
        let (operand_type, implicit_dereference) = match &checked_operand.ty {
            Type::Pointer { target, .. } => (&**target, true),
            _ => (&checked_operand.ty, false),
        };
        let element = self.check_index_step(operand_type, &span, index, scopes)?;
        // `a[i]` is never a constant expression, so both sub-expressions keep
        // their own constants and are still evaluated.
        self.record_operand(operand, checked_operand, false);
        Ok(CheckedExpression {
            ty: element,
            untyped: false,
            value: ExpressionValue::Index {
                operand,
                index,
                implicit_dereference,
            },
            constant: None,
        })
    }

    fn infer_length(
        &mut self,
        operand: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        let (operand_type, implicit_dereference) = match &checked_operand.ty {
            Type::Pointer { target, .. } => (&**target, true),
            _ => (&checked_operand.ty, false),
        };
        let Type::Array { length, .. } = operand_type else {
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
        let constant = (!implicit_dereference && find_call(self.syntax, operand).is_none())
            .then(|| Constant::Integer(BigInt::from(*length)));
        self.record_operand(operand, checked_operand, false);
        Ok(CheckedExpression {
            ty: Scalar::Int.into(),
            untyped: false,
            value: ExpressionValue::Length {
                operand,
                implicit_dereference,
            },
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

    /// A grouped `null` keeps the pointer context its position supplies. Other
    /// grouping stays inference-only, so array literals do not gain context
    /// through parentheses.
    fn check_grouping(
        &mut self,
        inner: Idx<Expression>,
        scopes: &ScopeStack<'_>,
        destination: Option<Type>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked = self.check_expression(inner, scopes, destination)?;
        let result = CheckedExpression {
            ty: checked.ty.clone(),
            untyped: checked.untyped,
            value: ExpressionValue::Grouping { expression: inner },
            constant: checked.constant.clone(),
        };
        self.record_operand(inner, checked, result.constant.is_some());
        Ok(result)
    }

    fn infer_address_of(
        &mut self,
        operand: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let location =
            self.check_location_with_facts(operand, scopes, true, LocationUse::AddressOf)?;
        Ok(CheckedExpression {
            ty: Type::Pointer {
                constant: !location.mutable,
                target: Box::new(location.ty),
            },
            untyped: false,
            value: ExpressionValue::AddressOf { operand },
            constant: None,
        })
    }

    fn infer_dereference(
        &mut self,
        operand: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        let Type::Pointer {
            constant: _,
            target,
        } = &checked_operand.ty
        else {
            return Err(Diagnostic::new(
                self.syntax.expressions[operand].span.clone(),
                format!("cannot dereference `{}`", checked_operand.ty),
            ));
        };
        let ty = (**target).clone();
        self.record_operand(operand, checked_operand, false);
        Ok(CheckedExpression {
            ty,
            untyped: false,
            value: ExpressionValue::Dereference { operand },
            constant: None,
        })
    }

    /// Checks an expression in a position that requires a stored location.
    /// This follows the source order of every index, field, and dereference
    /// operand while retaining an arbitrary dereference value for later IR.
    pub(super) fn check_location(
        &mut self,
        id: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedLocation, Diagnostic> {
        self.check_location_with_facts(id, scopes, false, LocationUse::Assignment)
    }

    fn check_location_with_facts(
        &mut self,
        id: Idx<Expression>,
        scopes: &ScopeStack<'_>,
        record: bool,
        use_: LocationUse,
    ) -> Result<CheckedLocation, Diagnostic> {
        match &self.syntax.expressions[id].kind {
            ExpressionKind::Reference(name) => {
                let binding = self.resolve(name, scopes)?;
                let location = CheckedLocation {
                    kind: CheckedLocationKind::Binding(binding),
                    ty: self.bindings[binding].ty.clone(),
                    mutable: self.bindings[binding].mutable,
                    span: name.span.clone(),
                };
                if record {
                    self.expressions.insert(
                        id,
                        CheckedExpression {
                            ty: location.ty.clone(),
                            untyped: false,
                            value: ExpressionValue::Reference(binding),
                            constant: self.bindings[binding].constant.clone(),
                        },
                    );
                }
                Ok(location)
            }
            ExpressionKind::Grouping { expression } => {
                let location = self.check_location_with_facts(*expression, scopes, record, use_)?;
                if record {
                    self.expressions.insert(
                        id,
                        CheckedExpression {
                            ty: location.ty.clone(),
                            untyped: false,
                            value: ExpressionValue::Grouping {
                                expression: *expression,
                            },
                            constant: None,
                        },
                    );
                }
                Ok(location)
            }
            ExpressionKind::Index { operand, index, .. } => {
                let LocationStep {
                    location: operand_location,
                    ty: operand_type,
                    implicit_dereference,
                    mutable,
                } = self.location_step_operand(*operand, scopes, record, use_)?;
                let span = operand_location.span.clone();
                let ty = self.check_index_step(&operand_type, &span, *index, scopes)?;
                let implicit = implicit_dereference.is_some();
                let location = CheckedLocation {
                    kind: CheckedLocationKind::Index {
                        operand: Box::new(operand_location),
                        index: *index,
                        implicit_dereference,
                    },
                    ty,
                    mutable,
                    span,
                };
                if record {
                    self.expressions.insert(
                        id,
                        CheckedExpression {
                            ty: location.ty.clone(),
                            untyped: false,
                            value: ExpressionValue::Index {
                                operand: *operand,
                                index: *index,
                                implicit_dereference: implicit,
                            },
                            constant: None,
                        },
                    );
                }
                Ok(location)
            }
            ExpressionKind::Field {
                operand,
                name,
                name_span,
            } => {
                let LocationStep {
                    location: operand_location,
                    ty: operand_type,
                    implicit_dereference,
                    mutable,
                } = self.location_step_operand(*operand, scopes, record, use_)?;
                let span = operand_location.span.clone();
                let (ordinal, ty) =
                    self.check_field_step(&operand_type, &span, *name, name_span)?;
                let implicit = implicit_dereference.is_some();
                let location = CheckedLocation {
                    kind: CheckedLocationKind::Field {
                        operand: Box::new(operand_location),
                        ordinal,
                        implicit_dereference,
                    },
                    ty,
                    mutable,
                    span,
                };
                if record {
                    self.expressions.insert(
                        id,
                        CheckedExpression {
                            ty: location.ty.clone(),
                            untyped: false,
                            value: ExpressionValue::Field {
                                operand: *operand,
                                ordinal,
                                implicit_dereference: implicit,
                            },
                            constant: None,
                        },
                    );
                }
                Ok(location)
            }
            ExpressionKind::Dereference { operand, .. } => {
                let checked = self.infer_dereference(*operand, scopes)?;
                let Type::Pointer { constant, .. } = &self
                    .expressions
                    .get(*operand)
                    .map(|value| &value.ty)
                    .unwrap_or(&checked.ty)
                else {
                    unreachable!("dereference checking requires a pointer operand")
                };
                let mutable = !*constant;
                let ty = checked.ty.clone();
                self.expressions.insert(id, checked);
                Ok(CheckedLocation {
                    kind: CheckedLocationKind::Dereference { operand: *operand },
                    ty,
                    mutable,
                    span: self.syntax.expressions[id].span.clone(),
                })
            }
            _ => Err(Diagnostic::new(
                self.syntax.expressions[id].span.clone(),
                use_.non_location_message(),
            )),
        }
    }

    /// Checks the shared base of an index or field location, dereferencing one
    /// pointer level when needed and retaining the source span for its trap.
    fn location_step_operand(
        &mut self,
        id: Idx<Expression>,
        scopes: &ScopeStack<'_>,
        record: bool,
        use_: LocationUse,
    ) -> Result<LocationStep, Diagnostic> {
        let operand = self.location_operand(id, scopes, record, use_)?;
        if let Type::Pointer { constant, target } = &operand.ty {
            let target = (**target).clone();
            let mutable = !*constant;
            return Ok(LocationStep {
                location: operand,
                ty: target,
                implicit_dereference: Some(self.syntax.expressions[id].span.clone()),
                mutable,
            });
        }
        let ty = operand.ty.clone();
        let mutable = operand.mutable;
        Ok(LocationStep {
            location: operand,
            ty,
            implicit_dereference: None,
            mutable,
        })
    }

    /// Gives field and index locations a pointer result to dereference even
    /// when that result is not itself a location, such as a call result.
    fn location_operand(
        &mut self,
        id: Idx<Expression>,
        scopes: &ScopeStack<'_>,
        record: bool,
        use_: LocationUse,
    ) -> Result<CheckedLocation, Diagnostic> {
        let location = self.check_location_with_facts(id, scopes, record, use_);
        let Err(location_error) = location else {
            return location;
        };
        let checked = match self.infer_expression(id, scopes) {
            Ok(checked) => checked,
            Err(_) => return Err(location_error),
        };
        let Type::Pointer { constant, target } = &checked.ty else {
            return Err(location_error);
        };
        let location = CheckedLocation {
            kind: CheckedLocationKind::Dereference { operand: id },
            ty: (**target).clone(),
            mutable: !*constant,
            span: self.syntax.expressions[id].span.clone(),
        };
        self.expressions.insert(id, checked);
        Ok(location)
    }

    fn infer_unary(
        &mut self,
        operator: UnaryOperator,
        operator_span: &std::ops::Range<usize>,
        operand: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let checked_operand = self.infer_expression(operand, scopes)?;
        let error = |message| Diagnostic::new(operator_span.clone(), message);
        let negation = operator == UnaryOperator::Negate;
        let integer_only = || "integer unary operator requires an integer operand".to_string();
        let Some(operand_type) = numeric_operand(&checked_operand) else {
            return Err(error(if negation {
                "unary `-` requires a numeric operand".to_string()
            } else {
                integer_only()
            }));
        };
        if operand_type.is_floating() {
            if !negation {
                return Err(error(integer_only()));
            }
        } else {
            if negation && !checked_operand.untyped && !operand_type.signed() {
                return Err(error(format!(
                    "unary `-` is not permitted on `{operand_type}`"
                )));
            }
            if operator == UnaryOperator::WrappingNegate && checked_operand.untyped {
                return Err(error(
                    "wrapping negation requires a typed operand".to_string(),
                ));
            }
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
        let checked_left = self.infer_expression(left, scopes)?;
        let checked_right = self.infer_expression(right, scopes)?;
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
                operator_span: operator_span.clone(),
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
        let left_null = is_null_expression(self.syntax, left);
        let right_null = is_null_expression(self.syntax, right);
        let (mut checked_left, mut checked_right) = if left_null {
            // `null` has no evaluation of its own, so the right operand can
            // provide its pointer type without changing runtime order.
            let right = self.infer_expression(right, scopes)?;
            let left = self.check_expression(left, scopes, Some(right.ty.clone()))?;
            (left, right)
        } else {
            let left = self.infer_expression(left, scopes)?;
            if right_null {
                let right = self.check_expression(right, scopes, Some(left.ty.clone()))?;
                (left, right)
            } else {
                (left, self.infer_expression(right, scopes)?)
            }
        };
        self.unify_comparison(
            operator,
            operator_span,
            (left, &mut checked_left),
            (right, &mut checked_right),
        )?;
        let pointer_comparison = matches!(checked_left.ty, Type::Pointer { .. });
        let constant = match (
            checked_left.constant.as_ref(),
            checked_right.constant.as_ref(),
        ) {
            (Some(left), Some(right)) if !pointer_comparison => {
                Some(compare_constants(operator, left, right))
            }
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
        if let (
            Type::Pointer {
                target: left_target,
                ..
            },
            Type::Pointer {
                target: right_target,
                ..
            },
        ) = (&left.ty, &right.ty)
        {
            if left_target != right_target {
                return Err(Diagnostic::new(
                    operator_span.clone(),
                    format!(
                        "comparison operands have different types `{}` and `{}`",
                        left.ty, right.ty
                    ),
                ));
            }
            if !matches!(
                operator,
                ComparisonOperator::Equal | ComparisonOperator::NotEqual
            ) {
                return Err(Diagnostic::new(
                    operator_span.clone(),
                    format!("only `==` and `!=` are defined on `{}`", left.ty),
                ));
            }
            return Ok(());
        }
        let unified = match (left.ty.scalar(), right.ty.scalar()) {
            (Some(left_type), Some(right_type)) => match (left.untyped, right.untyped) {
                (false, true) => return self.concretize(right_id, right, left_type),
                (true, false) => return self.concretize(left_id, left, right_type),
                (true, true) if left_type.is_integer() && right_type.is_floating() => {
                    return self.make_untyped_floating(left_id, left);
                }
                (true, true) if left_type.is_floating() && right_type.is_integer() => {
                    return self.make_untyped_floating(right_id, right);
                }
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
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let mut checked_operand = self.infer_expression(operand, scopes)?;
        // Only integers reinterpret their bits, while every number converts.
        if !truncating
            && destination == Scalar::Uint
            && matches!(checked_operand.ty, Type::Pointer { .. })
        {
            self.record_operand(operand, checked_operand, false);
            return Ok(CheckedExpression {
                ty: Scalar::Uint.into(),
                untyped: false,
                value: ExpressionValue::Conversion {
                    operand,
                    truncating,
                },
                constant: None,
            });
        }
        let convertible = if truncating {
            integer_operand(&checked_operand)
        } else {
            numeric_operand(&checked_operand)
        };
        if convertible.is_none() {
            return Err(Diagnostic::new(
                span.clone(),
                format!("cannot convert `{}` to `{destination}`", checked_operand.ty),
            ));
        }
        if checked_operand.untyped && (!truncating || checked_operand.constant.is_none()) {
            let operand_type = if truncating { Scalar::Int } else { destination };
            self.concretize(operand, &mut checked_operand, operand_type)?;
        }
        let constant = convert_constant(&checked_operand, destination, truncating, span)?;
        let value = match constant {
            Some(Constant::Float(_)) => ExpressionValue::Floating,
            Some(_) => ExpressionValue::Integer,
            None => ExpressionValue::Conversion {
                operand,
                truncating,
            },
        };
        self.record_operand(operand, checked_operand, constant.is_some());
        Ok(CheckedExpression {
            ty: destination.into(),
            untyped: false,
            value,
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
        let (mut left_type, mut right_type) = binary_operand_types(
            operator,
            operator_span,
            spelling,
            (&left.expression, &right.expression),
        )?;
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
        if !operator.is_shift() {
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
                // An untyped integer joins an untyped floating-point constant
                // as the exact value it names.
                (true, true) if left_type.is_integer() && right_type.is_floating() => {
                    self.make_untyped_floating(
                        left.id.expect("an untyped left operand has an expression"),
                        &mut left.expression,
                    )?;
                    left_type = right_type;
                }
                (true, true) if left_type.is_floating() && right_type.is_integer() => {
                    self.make_untyped_floating(
                        right
                            .id
                            .expect("an untyped right operand has an expression"),
                        &mut right.expression,
                    )?;
                    right_type = left_type;
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
        let (ty, untyped) = if operator.is_shift() {
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

    /// Gives an untyped integer constant the exact floating-point value it
    /// names, which is the one form it combines with an untyped
    /// floating-point constant in.
    fn make_untyped_floating(
        &self,
        id: Idx<Expression>,
        checked: &mut CheckedExpression,
    ) -> Result<(), Diagnostic> {
        if to_untyped_floating(checked) {
            return Ok(());
        }
        Err(Diagnostic::new(
            self.syntax.expressions[id].span.clone(),
            format!(
                "cannot implicitly convert `{}` to `{}`",
                checked.ty,
                Scalar::F64
            ),
        ))
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
