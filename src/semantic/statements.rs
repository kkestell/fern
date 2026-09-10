//! Statement, scope, control-flow, call, and assignment checking.

use crate::{
    diagnostic::Diagnostic,
    frontend::syntax::{
        AssignmentTarget, Call, Expression, ForHeader, Function, Label, Statement, StatementKind,
        Syntax, TargetStep, TypeAnnotation,
    },
    types::{BinaryOperator, Scalar, Type},
};
use la_arena::Idx;
use lasso::Spur;
use std::collections::HashMap;

use super::model::*;

impl CheckedProgram<'_> {
    pub(super) fn check_body(
        &mut self,
        body: &[Idx<Statement>],
        result: Option<&Type>,
        scopes: &mut ScopeStack<'_>,
        loops: &mut Vec<Option<Spur>>,
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
        loops: &mut Vec<Option<Spur>>,
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
                let target = self.assignment_target(target, scopes)?;
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
                target,
                *operator,
                operator_span,
                *value,
                scopes,
            ),
            StatementKind::Block { body } => self.check_body(body, result, scopes, loops),
            StatementKind::Exit { argument } => {
                let expression =
                    self.check_expression(*argument, scopes, Some(Scalar::Int.into()))?;
                self.expressions.insert(*argument, expression);
                Ok(())
            }
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
            StatementKind::Break { label } => {
                self.check_loop_jump(statement, "break", label.as_ref(), loops)
            }
            StatementKind::Continue { label } => {
                self.check_loop_jump(statement, "continue", label.as_ref(), loops)
            }
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
        let destination = match annotation {
            Some(annotation) => {
                Some(self.resolve_annotation(annotation, scopes, Some(initializer))?)
            }
            None => None,
        };
        let expression = self.check_expression(initializer, scopes, destination)?;
        let ty = expression.ty.clone();
        let constant = if mutable {
            None
        } else {
            expression.constant.clone()
        };
        self.expressions.insert(initializer, expression);
        let binding = self.bindings.alloc(Binding {
            ty,
            mutable,
            constant,
        });
        self.declarations.insert(statement, binding);
        scopes.insert(name, binding);
        Ok(())
    }

    /// Checks `target op= value` as the binary operation it stands for, with
    /// the target as the left operand.
    fn check_compound_assignment(
        &mut self,
        statement: Idx<Statement>,
        target: &AssignmentTarget,
        operator: BinaryOperator,
        operator_span: &std::ops::Range<usize>,
        value: Idx<Expression>,
        scopes: &mut ScopeStack<'_>,
    ) -> Result<(), Diagnostic> {
        let target = self.assignment_target(target, scopes)?;
        let left = CheckedExpression {
            ty: target.ty.clone(),
            untyped: false,
            value: ExpressionValue::Reference(target.binding),
            constant: None,
        };
        let right = self.infer_expression(value, scopes)?;
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
        loops: &mut Vec<Option<Spur>>,
    ) -> Result<(), Diagnostic> {
        let StatementKind::For {
            label,
            header,
            body,
        } = &self.syntax.statements[statement].kind
        else {
            unreachable!("a `for` statement is checked from its own syntax")
        };
        if let Some(label) = label
            && loops.iter().flatten().any(|name| *name == label.name)
        {
            return Err(Diagnostic::new(
                label.name_span.clone(),
                format!(
                    "duplicate enclosing loop label `{}`",
                    self.syntax.names.resolve(&label.name)
                ),
            ));
        }
        let has_header_scope = self.check_for_header(statement, header, result, scopes, loops)?;
        loops.push(label.as_ref().map(|label| label.name));
        self.check_body(body, result, scopes, loops)?;
        loops.pop();
        if has_header_scope {
            scopes.pop();
        }
        Ok(())
    }

    /// Checks a `for` header, reporting whether it opened a scope for the
    /// bindings its initializer declares.
    fn check_for_header(
        &mut self,
        statement: Idx<Statement>,
        header: &ForHeader,
        result: Option<&Type>,
        scopes: &mut ScopeStack<'_>,
        loops: &mut Vec<Option<Spur>>,
    ) -> Result<bool, Diagnostic> {
        match header {
            ForHeader::Infinite => Ok(false),
            ForHeader::Condition(condition) => {
                self.check_condition(*condition, scopes)?;
                Ok(false)
            }
            ForHeader::ThreeClause {
                initializer,
                condition,
                post,
            } => {
                scopes.push(HashMap::new());
                self.check_statement(*initializer, result, scopes, loops)?;
                self.check_condition(*condition, scopes)?;
                self.check_statement(*post, result, scopes, loops)?;
                Ok(true)
            }
            ForHeader::Iteration {
                value,
                index,
                operand,
                ..
            } => {
                self.check_iteration_header(statement, *value, index.as_ref(), *operand, scopes)?;
                Ok(true)
            }
        }
    }

    /// Checks `for v in a` and `for v, i in a`, opening the scope that holds
    /// the loop's bindings. They are immutable and live only inside the loop.
    fn check_iteration_header(
        &mut self,
        statement: Idx<Statement>,
        value: Spur,
        index: Option<&(Spur, std::ops::Range<usize>)>,
        operand: Idx<Expression>,
        scopes: &mut ScopeStack<'_>,
    ) -> Result<(), Diagnostic> {
        let checked = self.infer_expression(operand, scopes)?;
        let Type::Array { element, .. } = &checked.ty else {
            return Err(Diagnostic::new(
                self.syntax.expressions[operand].span.clone(),
                format!("`for … in` requires an array, found `{}`", checked.ty),
            ));
        };
        let element = (**element).clone();
        if let Some((name, span)) = index
            && *name == value
        {
            return Err(Diagnostic::new(
                span.clone(),
                "a `for` loop's value and index bindings must have different names",
            ));
        }
        self.expressions.insert(operand, checked);
        let mut loop_scope = HashMap::new();
        let mut bind = |program: &mut Self, name: Spur, ty: Type| {
            let binding = program.bindings.alloc(Binding {
                ty,
                mutable: false,
                constant: None,
            });
            loop_scope.insert(name, binding);
            binding
        };
        let value = bind(self, value, element);
        let index = index.map(|(name, _)| bind(self, *name, Scalar::Int.into()));
        self.iterations
            .insert(statement, IterationBindings { value, index });
        scopes.push(loop_scope);
        Ok(())
    }

    /// Checks that a `break` or `continue` names a loop it is inside.
    fn check_loop_jump(
        &self,
        statement: Idx<Statement>,
        keyword: &str,
        label: Option<&Label>,
        loops: &[Option<Spur>],
    ) -> Result<(), Diagnostic> {
        if loops.is_empty() {
            let start = self.syntax.statements[statement].span.start;
            return Err(Diagnostic::new(
                start..start + keyword.len(),
                format!("`{keyword}` is not inside a loop"),
            ));
        }
        if let Some(label) = label
            && !loops.iter().flatten().any(|name| *name == label.name)
        {
            return Err(Diagnostic::new(
                label.name_span.clone(),
                format!(
                    "unknown enclosing loop label `{}`",
                    self.syntax.names.resolve(&label.name)
                ),
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

    /// The binding an assignment stores into, its resolved steps, and the type
    /// they reach. An element or field of a `const` binding is rejected with
    /// the binding itself, so the mutability check comes first.
    fn assignment_target(
        &mut self,
        target: &AssignmentTarget,
        scopes: &ScopeStack<'_>,
    ) -> Result<CheckedTarget, Diagnostic> {
        let name = &target.name;
        let binding = self.resolve(name, scopes)?;
        if !self.bindings[binding].mutable {
            return Err(Diagnostic::new(
                name.span.clone(),
                format!(
                    "cannot assign to immutable binding `{}`",
                    self.syntax.names.resolve(&name.name)
                ),
            ));
        }
        // The steps are checked left to right, the order they are evaluated
        // in, and each one descends into the element or field type.
        let mut ty = self.bindings[binding].ty.clone();
        let mut steps = Vec::with_capacity(target.steps.len());
        for step in &target.steps {
            match step {
                TargetStep::Index(index) => {
                    ty = self.check_index_step(&ty, &name.span, *index, scopes)?;
                    steps.push(CheckedStep::Index(*index));
                }
                TargetStep::Field {
                    name: field,
                    name_span,
                } => {
                    let (ordinal, field_type) =
                        self.check_field_step(&ty, &name.span, *field, name_span)?;
                    ty = field_type;
                    steps.push(CheckedStep::Field(ordinal));
                }
            }
        }
        Ok(CheckedTarget { binding, steps, ty })
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
        StatementKind::Return { .. } | StatementKind::Exit { .. } => true,
        StatementKind::Block { body } => body_terminates(syntax, body),
        StatementKind::If {
            then_body,
            else_branch,
            ..
        } => {
            else_branch.is_some_and(|else_branch| statement_terminates(syntax, else_branch))
                && body_terminates(syntax, then_body)
        }
        StatementKind::For {
            label,
            header,
            body,
        } => {
            matches!(header, ForHeader::Infinite)
                && !body_breaks(syntax, body, label.as_ref().map(|label| label.name), false)
        }
        _ => false,
    }
}

/// Reports whether a reachable `break` in `body` targets the loop identified by
/// `label`. `nested` marks a body that lies inside a loop enclosed by that one,
/// where an unlabeled `break` targets the inner loop instead.
fn body_breaks(
    syntax: &Syntax,
    body: &[Idx<Statement>],
    label: Option<Spur>,
    nested: bool,
) -> bool {
    for &statement in body {
        if statement_breaks(syntax, statement, label, nested) {
            return true;
        }
        if statement_terminates(syntax, statement) {
            return false;
        }
    }
    false
}

fn statement_breaks(
    syntax: &Syntax,
    statement: Idx<Statement>,
    label: Option<Spur>,
    nested: bool,
) -> bool {
    match &syntax.statements[statement].kind {
        StatementKind::Break { label: target } => match target {
            Some(target) => label == Some(target.name),
            None => !nested,
        },
        StatementKind::Block { body } => body_breaks(syntax, body, label, nested),
        StatementKind::If {
            then_body,
            else_branch,
            ..
        } => {
            body_breaks(syntax, then_body, label, nested)
                || else_branch
                    .is_some_and(|else_branch| statement_breaks(syntax, else_branch, label, nested))
        }
        StatementKind::For { body, .. } => body_breaks(syntax, body, label, true),
        _ => false,
    }
}
