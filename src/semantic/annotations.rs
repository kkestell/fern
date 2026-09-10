//! Annotation resolution, array lengths, and element-type rules.

use crate::{
    diagnostic::Diagnostic,
    frontend::syntax::{
        AnnotationKind, Expression, ExpressionKind, Syntax, TypeAnnotation, find_call,
    },
    types::{Scalar, Type},
};

use la_arena::Idx;

use lasso::Spur;

use num_traits::ToPrimitive;

use std::collections::HashMap;

use super::model::*;

/// The length `[_]` takes from the declaration's initializer, which must be an
/// array literal that states its own length.
pub(super) fn inferred_length(
    syntax: &Syntax,
    span: &std::ops::Range<usize>,
    initializer: Option<Idx<Expression>>,
) -> Result<u64, Diagnostic> {
    let literal = initializer.map(|id| &syntax.expressions[id].kind);
    let Some(ExpressionKind::ArrayLiteral { elements, fill }) = literal else {
        return Err(Diagnostic::new(
            span.clone(),
            "`[_]` requires an array-literal initializer",
        ));
    };
    if fill.is_some() {
        return Err(Diagnostic::new(
            span.clone(),
            "`[_]` cannot take a length from a literal with a fill",
        ));
    }
    Ok(u64::try_from(elements.len()).expect("a source file holds fewer elements than u64::MAX"))
}
impl CheckedProgram<'_> {
    /// Resolves a written annotation to the type it names. `initializer` is the
    /// declaration's initializer, which is where a `[_]` length comes from.
    pub(super) fn resolve_annotation(
        &mut self,
        annotation: Idx<TypeAnnotation>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
        initializer: Option<Idx<Expression>>,
    ) -> Result<Type, Diagnostic> {
        let syntax = self.syntax;
        let written = &syntax.annotations[annotation];
        match &written.kind {
            AnnotationKind::Named(scalar) => Ok(Type::Scalar(*scalar)),
            AnnotationKind::Array { length, element } => {
                let (length, element) = (*length, *element);
                let length = match length {
                    Some(length) => self.array_length(length, scopes)?,
                    None => inferred_length(syntax, &written.span, initializer)?,
                };
                // Only the declaration's own annotation has an initializer, so
                // a nested `[_]` has nothing to take a length from.
                let element = self.resolve_annotation(element, scopes, None)?;
                Ok(Type::Array {
                    length,
                    element: Box::new(element),
                })
            }
        }
    }

    /// Evaluates a written array length, which is a constant `int` of at least
    /// one.
    fn array_length(
        &mut self,
        length: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<u64, Diagnostic> {
        let span = self.syntax.expressions[length].span.clone();
        // A length is resolved before the module's signatures are, so a call
        // here has no signature to check against. It is never constant anyway.
        let constant = |span: std::ops::Range<usize>| {
            Diagnostic::new(span, "array length must be a constant expression")
        };
        if let Some(call) = find_call(self.syntax, length) {
            return Err(constant(self.syntax.expressions[call].span.clone()));
        }
        let checked = self.check_expression(length, scopes, Some(Scalar::Int.into()))?;
        let Some(value) = checked.integer().cloned() else {
            return Err(constant(span));
        };
        self.expressions.insert(length, checked);
        value
            .to_u64()
            .filter(|length| *length >= 1)
            .ok_or_else(|| Diagnostic::new(span, "array length must be at least 1"))
    }
}

/// Checks a literal's element count against the array's length. A fill covers
/// the remaining elements, so it only bounds the count from above.
pub(super) fn check_element_count(
    span: std::ops::Range<usize>,
    ty: &Type,
    length: u64,
    count: usize,
    fill: bool,
) -> Result<(), Diagnostic> {
    let count = u64::try_from(count).expect("a source file holds fewer elements than u64::MAX");
    if if fill {
        count <= length
    } else {
        count == length
    } {
        return Ok(());
    }
    let bound = if fill { "at most " } else { "" };
    Err(Diagnostic::new(
        span,
        format!("expected {bound}{length} elements for `{ty}`, found {count}"),
    ))
}
