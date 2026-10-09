//! Statement, scope, control-flow, call, and assignment checking.

use crate::{
    diagnostic::Diagnostic,
    frontend::syntax::{
        Call, Expression, ForHeader, Function, Statement, StatementKind, Syntax, TypeAnnotation,
    },
    types::{BinaryOperator, Scalar, Type},
};
use la_arena::Idx;
use lasso::Spur;
use std::collections::HashMap;

use super::{expressions::LocationUse, model::*};

impl CheckedProgram<'_> {
    pub(super) fn check_body(
        &mut self,
        body: &[Idx<Statement>],
        result: Option<&Type>,
        scopes: &mut ScopeStack<'_>,
        loops: &mut Vec<()>,
    ) -> Result<(), Diagnostic> {
        scopes.push(HashMap::new());
        for &statement in body {
            self.check_statement(statement, result, scopes, loops)?;
        }
        scopes.pop();
        Ok(())
    }

    fn check_statement(
        &mut self,
        statement: Idx<Statement>,
        result: Option<&Type>,
        scopes: &mut ScopeStack<'_>,
        loops: &mut Vec<()>,
    ) -> Result<(), Diagnostic> {
        match &self.syntax.statements[statement].kind {
            StatementKind::Binding {
                name,
                mutable,
                annotation,
                initializer,
                ..
            } => self.check_binding(
                statement,
                *name,
                *mutable,
                *annotation,
                *initializer,
                scopes,
            ),
            StatementKind::Assignment { target, value } => {
                let target = self.assignment_target(*target, scopes)?;
                let expression = self.check_expression(*value, scopes, Some(target.ty.clone()))?;
                self.expressions.insert(*value, expression);
                self.assignments.insert(statement, target);
                Ok(())
            }
            StatementKind::CompoundAssignment {
                target,
                operator,
                operator_span,
                value,
            } => self.check_compound_assignment(
                statement,
                *target,
                *operator,
                operator_span,
                *value,
                scopes,
            ),
            StatementKind::Block { body } => self.check_body(body, result, scopes, loops),
            StatementKind::If {
                condition,
                then_body,
                else_branch,
            } => {
                self.check_condition(*condition, scopes)?;
                self.check_body(then_body, result, scopes, loops)?;
                if let Some(else_branch) = else_branch {
                    self.check_statement(*else_branch, result, scopes, loops)?;
                }
                Ok(())
            }
            StatementKind::For { .. } => self.check_for(statement, result, scopes, loops),
            StatementKind::Break { .. } => self.check_loop_jump(statement, "break", loops),
            StatementKind::Continue { .. } => self.check_loop_jump(statement, "continue", loops),
            StatementKind::Call { call } => {
                let (function, _) = self.check_call(call, scopes, false)?;
                self.calls.insert(statement, function);
                Ok(())
            }
            StatementKind::Return { value } => self.check_return(statement, result, *value, scopes),
        }
    }

    fn check_binding(
        &mut self,
        statement: Idx<Statement>,
        name: Spur,
        mutable: bool,
        annotation: Option<Idx<TypeAnnotation>>,
        initializer: Idx<Expression>,
        scopes: &mut ScopeStack<'_>,
    ) -> Result<(), Diagnostic> {
        let destination = annotation
            .map(|annotation| self.resolve_annotation(annotation, scopes))
            .transpose()?;
        let expression = self.check_expression(initializer, scopes, destination)?;
        if !mutable && expression.constant.is_none() {
            return Err(Diagnostic::new(
                self.syntax.statements[statement].span.clone(),
                "const initializer must be a constant expression",
            ));
        }
        let ty = expression.ty.clone();
        let constant = if mutable {
            None
        } else {
            expression.constant.clone()
        };
        self.expressions.insert(initializer, expression);
        self.record_binding(statement, name, mutable, ty, constant, scopes);
        Ok(())
    }

    fn record_binding(
        &mut self,
        statement: Idx<Statement>,
        name: Spur,
        mutable: bool,
        ty: Type,
        constant: Option<Constant>,
        scopes: &mut ScopeStack<'_>,
    ) {
        let binding = self.bindings.alloc(Binding {
            ty,
            mutable,
            constant,
        });
        self.declarations.insert(statement, binding);
        scopes.insert(name, binding);
    }

    /// Checks `target op= value` as the binary operation it stands for, with
    /// the target as the left operand.
    fn check_compound_assignment(
        &mut self,
        statement: Idx<Statement>,
        target: Idx<Expression>,
        operator: BinaryOperator,
        operator_span: &std::ops::Range<usize>,
        value: Idx<Expression>,
        scopes: &mut ScopeStack<'_>,
    ) -> Result<(), Diagnostic> {
        let target = self.assignment_target(target, scopes)?;
        let left = CheckedExpression {
            ty: target.ty.clone(),
            untyped: false,
            // This is never recorded or lowered: only its type participates
            // in the shared numeric-operator check below.
            value: ExpressionValue::CompoundAssignmentTarget,
            constant: None,
        };
        let right = self.check_expression(
            value,
            scopes,
            if operator.is_shift() {
                None
            } else {
                Some(target.ty.clone())
            },
        )?;
        let (_, right, _, _, _) = self.check_numeric_binary(
            operator,
            operator_span,
            &format!("{}=", operator.spelling()),
            CheckedBinaryOperand {
                id: None,
                expression: left,
            },
            CheckedBinaryOperand {
                id: Some(value),
                expression: right,
            },
        )?;
        self.expressions.insert(value, right.expression);
        self.assignments.insert(statement, target);
        Ok(())
    }

    fn check_for(
        &mut self,
        statement: Idx<Statement>,
        result: Option<&Type>,
        scopes: &mut ScopeStack<'_>,
        loops: &mut Vec<()>,
    ) -> Result<(), Diagnostic> {
        let StatementKind::For { header, body } = &self.syntax.statements[statement].kind else {
            unreachable!()
        };
        if let ForHeader::Condition(condition) = header {
            self.check_condition(*condition, scopes)?;
        }
        loops.push(());
        self.check_body(body, result, scopes, loops)?;
        loops.pop();
        Ok(())
    }
    fn check_loop_jump(
        &self,
        statement: Idx<Statement>,
        keyword: &str,
        loops: &[()],
    ) -> Result<(), Diagnostic> {
        if loops.is_empty() {
            let start = self.syntax.statements[statement].span.start;
            return Err(Diagnostic::new(
                start..start + keyword.len(),
                format!("`{keyword}` is not inside a loop"),
            ));
        }
        Ok(())
    }

    fn check_return(
        &mut self,
        statement: Idx<Statement>,
        result: Option<&Type>,
        value: Option<Idx<Expression>>,
        scopes: &mut ScopeStack<'_>,
    ) -> Result<(), Diagnostic> {
        match (result, value) {
            (None, None) => Ok(()),
            (None, Some(value)) => Err(Diagnostic::new(
                self.syntax.expressions[value].span.clone(),
                "cannot return a value from a `void` function",
            )),
            (Some(result), None) => Err(Diagnostic::new(
                self.syntax.statements[statement].span.clone(),
                format!("`return` must supply a value of type `{result}`"),
            )),
            (Some(result), Some(value)) => {
                let expression = self.check_expression(value, scopes, Some(result.clone()))?;
                self.expressions.insert(value, expression);
                Ok(())
            }
        }
    }

    /// Checks an assignable location.
    fn assignment_target(
        &mut self,
        target: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedTarget, Diagnostic> {
        let location = self.mutable_location(target, scopes, LocationUse::Assign)?;
        Ok(CheckedTarget {
            ty: location.ty.clone(),
            location,
        })
    }

    /// Checks a location that must be mutable: one reached through a
    /// pointer, or one whose root binding is mutable.
    pub(super) fn mutable_location(
        &mut self,
        target: Idx<Expression>,
        scopes: &ScopeStack<'_>,
        use_: LocationUse,
    ) -> Result<CheckedLocation, Diagnostic> {
        let location = self.check_location(target, scopes, use_)?;
        if self.immutable_root(&location).is_some() {
            return Err(Diagnostic::new(
                location.span.clone(),
                format!(
                    "cannot {} immutable binding `{}`",
                    use_.verb(),
                    self.syntax.names.resolve(
                        &self
                            .target_name(target)
                            .expect("a location originates at a reference")
                    )
                ),
            ));
        }
        Ok(location)
    }

    fn target_name(&self, target: Idx<Expression>) -> Option<lasso::Spur> {
        let mut current = target;
        loop {
            match &self.syntax.expressions[current].kind {
                crate::frontend::syntax::ExpressionKind::Reference(name) => return Some(name.name),
                crate::frontend::syntax::ExpressionKind::Grouping { expression }
                | crate::frontend::syntax::ExpressionKind::Index {
                    operand: expression,
                    ..
                }
                | crate::frontend::syntax::ExpressionKind::Field {
                    operand: expression,
                    ..
                } => current = *expression,
                _ => return None,
            }
        }
    }

    /// The immutable binding that makes this location immutable, or `None`
    /// when the location is mutable. A location reached through a pointer is
    /// mutable whatever binding holds the pointer.
    fn immutable_root(&self, location: &CheckedLocation) -> Option<Idx<Binding>> {
        match &location.kind {
            CheckedLocationKind::Binding(binding) => {
                (!self.bindings[*binding].mutable).then_some(*binding)
            }
            CheckedLocationKind::Index { operand, .. }
            | CheckedLocationKind::Field { operand, .. } => {
                if matches!(operand.ty, Type::Pointer { .. }) {
                    None
                } else {
                    self.immutable_root(operand)
                }
            }
            CheckedLocationKind::Dereference { .. } => None,
            CheckedLocationKind::Pointer(_) => {
                unreachable!("a pointer value is only the operand of a step or a dereference")
            }
        }
    }

    fn check_condition(
        &mut self,
        condition: Idx<Expression>,
        scopes: &ScopeStack<'_>,
    ) -> Result<(), Diagnostic> {
        let expression = self.check_expression(condition, scopes, Some(Scalar::Bool.into()))?;
        self.expressions.insert(condition, expression);
        Ok(())
    }

    pub(super) fn check_call(
        &mut self,
        call: &Call,
        scopes: &ScopeStack<'_>,
        value_context: bool,
    ) -> Result<(Idx<Function>, Option<Type>), Diagnostic> {
        let function = self.resolve_call(call, scopes)?;
        let signature = self.functions[function].clone();
        if call.arguments.len() != signature.parameters.len() {
            return Err(Diagnostic::new(
                call.target.span.clone(),
                format!(
                    "function `{}` expects {} argument{}, found {}",
                    self.syntax.names.resolve(&call.target.name),
                    signature.parameters.len(),
                    if signature.parameters.len() == 1 {
                        ""
                    } else {
                        "s"
                    },
                    call.arguments.len(),
                ),
            ));
        }
        for (&argument, &parameter) in call.arguments.iter().zip(&signature.parameters) {
            let expression =
                self.check_expression(argument, scopes, Some(self.bindings[parameter].ty.clone()))?;
            self.expressions.insert(argument, expression);
        }
        if value_context && signature.result.is_none() {
            return Err(Diagnostic::new(
                call.target.span.clone(),
                format!(
                    "void function `{}` cannot be used as a value",
                    self.syntax.names.resolve(&call.target.name)
                ),
            ));
        }
        Ok((function, signature.result))
    }
}

/// Reports whether every path through `body` ends in a `return` or `exit`, using
/// the structural rule under Function return in the specification.
pub(super) fn body_terminates(syntax: &Syntax, body: &[Idx<Statement>]) -> bool {
    body.iter()
        .any(|&statement| statement_terminates(syntax, statement))
}

fn statement_terminates(syntax: &Syntax, statement: Idx<Statement>) -> bool {
    match &syntax.statements[statement].kind {
        StatementKind::Return { .. } => true,
        StatementKind::Block { body } => body_terminates(syntax, body),
        StatementKind::If {
            then_body,
            else_branch,
            ..
        } => {
            else_branch.is_some_and(|else_branch| statement_terminates(syntax, else_branch))
                && body_terminates(syntax, then_body)
        }
        StatementKind::For { header, body } => {
            matches!(header, ForHeader::Infinite) && !body_breaks(syntax, body)
        }
        _ => false,
    }
}

/// Whether the body can break out of its immediately enclosing loop.
fn body_breaks(syntax: &Syntax, body: &[Idx<Statement>]) -> bool {
    for &statement in body {
        if statement_breaks(syntax, statement) {
            return true;
        }
        if statement_stops(syntax, statement) {
            return false;
        }
    }
    false
}
fn statement_breaks(syntax: &Syntax, statement: Idx<Statement>) -> bool {
    match &syntax.statements[statement].kind {
        StatementKind::Break { .. } => true,
        StatementKind::Block { body } => body_breaks(syntax, body),
        StatementKind::If {
            then_body,
            else_branch,
            ..
        } => {
            body_breaks(syntax, then_body)
                || else_branch.is_some_and(|branch| statement_breaks(syntax, branch))
        }
        _ => false,
    }
}

fn statement_stops(syntax: &Syntax, statement: Idx<Statement>) -> bool {
    match &syntax.statements[statement].kind {
        StatementKind::Break { .. } | StatementKind::Continue { .. } => true,
        StatementKind::Block { body } => body.iter().any(|&s| statement_stops(syntax, s)),
        StatementKind::If {
            then_body,
            else_branch,
            ..
        } => {
            then_body.iter().any(|&s| statement_stops(syntax, s))
                && else_branch.is_some_and(|s| statement_stops(syntax, s))
        }
        _ => statement_terminates(syntax, statement),
    }
}
